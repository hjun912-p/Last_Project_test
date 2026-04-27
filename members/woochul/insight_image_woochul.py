# ============================================================
# InSIGHT — AI 가상인물 판독기
# woochul ver. | 스켈레톤(dfk.py) 기반 이미지 버전 재설계
#
# 구조:
#   1차 필터: C2PA 메타데이터 + SynthID(FFT) → API 비용 없이 선검열
#   2차 필터: ViT + Gram Matrix + FFT → AI 모델 판별
#   성능 측정: 6가지 지표 (Accuracy, Precision, Recall, F1, FPR, Inference Time)
# ============================================================

# ── 라이브러리 임포트 ──────────────────────────────────────
import cv2
import torch
import numpy as np
import gradio as gr
from torchvision import transforms
from PIL import Image
import os
import time
from datetime import datetime
from transformers import pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix
)

# ── C2PA (1차 필터) ───────────────────────────────────────
try:
    import c2pa
    C2PA_AVAILABLE = True
except ImportError:
    C2PA_AVAILABLE = False

# ── ViT 모델 로드 (2차 필터) ──────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    vit_detector = pipeline(
        "image-classification",
        model="umm-maybe/AI-image-detector",
        device=0 if torch.cuda.is_available() else -1
    )
    VIT_AVAILABLE = True
except Exception:
    VIT_AVAILABLE = False

# ── 전처리 설정 ───────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ============================================================
# 공통 유틸
# ============================================================

def get_gram_matrix(img_tensor):
    """Gram Matrix 질감 복잡도 측정 — AI 이미지는 질감이 단순하거나 비정상적으로 일정함"""
    (b, c, h, w) = img_tensor.size()
    features = img_tensor.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)


# ============================================================
# 1차 필터 — C2PA + SynthID(FFT)
# AI API KEY 소모 없이 선검열, 여기서 판별되면 2차 필터 스킵
# ============================================================

def filter1_c2pa(image_path: str):
    """
    C2PA 메타데이터 검사
    반환: (결과: True=AI / False=실제 / None=판별불가, 메시지)
    """
    if not C2PA_AVAILABLE:
        return None, "C2PA 라이브러리 없음 — 건너뜀"
    try:
        reader = c2pa.Reader(image_path)
        manifest = reader.json()
        if manifest:
            if "ai" in str(manifest).lower():
                return True, "C2PA: AI 생성물 서명 감지"
            return False, "C2PA: 원본(실제) 서명 확인"
    except Exception:
        pass
    return None, "C2PA 데이터 없음"


def filter1_synthid_fft(pil_image: Image.Image):
    """
    SynthID 워터마크 패턴 탐지 — FFT 주파수 분석
    AI 생성 이미지는 특정 주파수에서 비정상적인 피크(Peak)를 가짐
    반환: (AI 확률 0~100, 피크 비율)
    """
    gray = np.array(pil_image.convert("L"))
    dft = np.fft.fft2(gray)
    fshift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    h, w = gray.shape
    crow, ccol = h // 2, w // 2

    # 저주파(중심부) 제외 — 고주파 영역만 마스킹
    mask = np.ones((h, w), np.uint8)
    cv2.circle(mask, (ccol, crow), 70, 0, -1)
    high_freq_area = magnitude_spectrum * mask

    values = high_freq_area[high_freq_area > 0]
    if len(values) == 0:
        return 0.0, 0.0

    avg_val = np.mean(values)
    max_val = np.max(values)
    peak_ratio = max_val / (avg_val + 1e-5)

    # 1.8 이상이면 인공 워터마크 의심 → 급격히 확률 상승
    synthid_score = float(min(99.9, max(0.0, (peak_ratio - 1.8) * 150)))
    return synthid_score, peak_ratio


# ============================================================
# 2차 필터 — ViT + Gram Matrix + FFT
# 1차에서 판별 불가한 경우에만 진입
# ============================================================

def filter2_ai_model(pil_image: Image.Image):
    """
    AI 모델 기반 이미지 판별
    반환: (AI 여부 bool, 종합 확률 0~100, 상세 리포트 str)
    """
    # 1. ViT 판독 (umm-maybe/AI-image-detector)
    vit_score = 50.0
    if VIT_AVAILABLE:
        res = vit_detector(pil_image)
        vit_score = sum(
            r["score"] for r in res
            if "ai" in r["label"].lower() or "artificial" in r["label"].lower()
        ) * 100

    # 2. Gram Matrix 질감 분석
    input_tensor = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        gram = get_gram_matrix(input_tensor)
        gram_std = torch.std(gram).item() * 1000

    # 3. SynthID(FFT) 재측정
    synthid_score, peak_ratio = filter1_synthid_fft(pil_image)

    # 4. Laplacian 선명도 (AI 이미지는 경계가 비정상적으로 선명하거나 흐릿함)
    gray = np.array(pil_image.convert("L"))
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 점수 환산
    p_vit   = max(0.0, min(99.0, vit_score * 1.1))
    p_gram  = max(0.0, min(98.0, (100.0 - gram_std) * 0.5 + 20.0))
    p_fft   = synthid_score
    p_blur  = max(0.0, min(99.0, 100.0 - (laplacian_var / 5.0)))

    total = float(np.mean([p_vit, p_gram, p_fft, p_blur]))
    is_ai = total > 50.0

    report = (
        f"  ① ViT 판독      : {'❌ AI 의심' if p_vit  > 70 else '✅ 정상'} ({p_vit:.1f}%)\n"
        f"  ② Gram 질감     : {'⚠️ 주의'   if p_gram > 60 else '✅ 정상'} ({p_gram:.1f}%)\n"
        f"  ③ SynthID(FFT)  : {'❌ 위험'   if p_fft  > 50 else '✅ 정상'} ({p_fft:.1f}%) — 피크비율 {peak_ratio:.2f}\n"
        f"  ④ 선명도 분석   : {'⚠️ 주의'   if p_blur > 70 else '✅ 정상'} ({p_blur:.1f}%)"
    )
    return is_ai, total, report


# ============================================================
# 통합 판독 함수 (단일 이미지 — 서비스 모드)
# ============================================================

def process_single_image(image):
    """
    단일 이미지 입력 → 1차 필터 → 2차 필터 → 판정 결과 반환
    """
    if image is None:
        return "⚠️ 이미지를 업로드해주세요.", "", None

    start_time = time.time()
    pil_image = Image.fromarray(image).convert("RGB")

    # C2PA 검사를 위한 임시 저장
    temp_path = "temp_insight_woochul.png"
    pil_image.save(temp_path)

    try:
        # ── 1차 필터: C2PA ────────────────────────────────
        c2pa_result, c2pa_msg = filter1_c2pa(temp_path)
        if c2pa_result is not None:
            elapsed = time.time() - start_time
            verdict  = "🤖 AI 생성 가상인물" if c2pa_result else "👤 실제 인물"
            result   = f"[1차 필터 — C2PA 판정]\n{verdict}\n근거: {c2pa_msg}"
            detail   = f"처리 시간: {elapsed:.2f}초\n1차 필터에서 판정 완료 → 2차 필터 생략 (비용 절감)"
            return result, detail, pil_image

        # ── 1차 필터: SynthID(FFT) ─────────────────────────
        synthid_score, peak_ratio = filter1_synthid_fft(pil_image)
        if synthid_score > 80:
            elapsed = time.time() - start_time
            result  = (
                f"[1차 필터 — SynthID 판정]\n🤖 AI 생성 가상인물 의심\n"
                f"근거: 워터마크 패턴 감지 ({synthid_score:.1f}%)"
            )
            detail  = (
                f"처리 시간: {elapsed:.2f}초\n"
                f"주파수 피크 비율: {peak_ratio:.2f}  (1.8 이상 → AI 의심)"
            )
            return result, detail, pil_image

        # ── 2차 필터: AI 모델 ──────────────────────────────
        is_ai, score, report = filter2_ai_model(pil_image)
        elapsed = time.time() - start_time

        verdict = "🤖 AI 생성 가상인물" if is_ai else "👤 실제 인물"
        result  = f"[2차 필터 — AI 모델 판정]\n{verdict}\nAI 확률: {score:.1f}%"
        detail  = f"처리 시간: {elapsed:.2f}초\n\n[상세 분석]\n{report}"
        return result, detail, pil_image

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# 베이스라인 성능 측정 (배치 평가 모드)
# 6가지 지표: Accuracy / Precision / Recall / F1 / FPR / Inference Time
# ============================================================

def run_baseline(ai_files, real_files):
    """
    레이블된 이미지 배치 → 6가지 성능 지표 측정
    ai_files  : AI 생성 이미지 경로 리스트 (Positive, label=1)
    real_files: 실제 인물 이미지 경로 리스트 (Negative, label=0)
    """
    if not ai_files and not real_files:
        return "⚠️ AI 이미지 또는 실제 이미지를 업로드해주세요.", ""

    dataset = []
    if ai_files:
        for f in ai_files:
            dataset.append((f.name, 1))
    if real_files:
        for f in real_files:
            dataset.append((f.name, 0))

    y_true, y_pred, times = [], [], []

    for img_path, true_label in dataset:
        try:
            pil_image = Image.open(img_path).convert("RGB")
            t0 = time.time()

            # 1차: C2PA
            c2pa_result, _ = filter1_c2pa(img_path)
            if c2pa_result is not None:
                times.append(time.time() - t0)
                y_true.append(true_label)
                y_pred.append(1 if c2pa_result else 0)
                continue

            # 1차: SynthID
            s_score, _ = filter1_synthid_fft(pil_image)
            if s_score > 80:
                times.append(time.time() - t0)
                y_true.append(true_label)
                y_pred.append(1)
                continue

            # 2차: AI 모델
            is_ai, _, _ = filter2_ai_model(pil_image)
            times.append(time.time() - t0)
            y_true.append(true_label)
            y_pred.append(1 if is_ai else 0)

        except Exception:
            continue

    if not y_true:
        return "❌ 처리 가능한 이미지가 없습니다.", ""

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 혼동 행렬
    if len(set(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    else:
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    # 6가지 지표 계산
    accuracy  = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, zero_division=0) * 100
    recall    = recall_score(y_true, y_pred, zero_division=0) * 100
    f1        = f1_score(y_true, y_pred, zero_division=0) * 100
    fpr       = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0
    avg_time  = float(np.mean(times)) if times else 0.0

    def chk(val, target, lower=False):
        return "✅" if (val <= target if lower else val >= target) else "❌"

    n_ai   = int(np.sum(y_true == 1))
    n_real = int(np.sum(y_true == 0))

    metrics_text = f"""========================================
실험 일시    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
실험자       : woochul
모델 버전    : woochul_baseline_v1
테스트 데이터: AI이미지 {n_ai}장 / 실제이미지 {n_real}장

[ 혼동 행렬 ]
TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}

[ 성능 지표 ]
Accuracy       : {accuracy:.2f}%   {chk(accuracy,  70)}  (목표 70% 이상)
Precision      : {precision:.2f}%   {chk(precision, 75)}  (목표 75% 이상)
Recall         : {recall:.2f}%   {chk(recall,    75)}  (목표 75% 이상) ★핵심
F1 Score       : {f1:.2f}%   {chk(f1,        70)}  (목표 70% 이상)
FPR            : {fpr:.2f}%   {chk(fpr,       20, lower=True)}  (목표 20% 이하)
Inference Time : {avg_time:.2f}초   {chk(avg_time,  3.0, lower=True)}  (목표 3초 이하)
========================================"""

    analysis_text = f"""[ 오류 케이스 분석 ]
FP (실제 → AI 오탐) : {fp}건 — 실제 이미지를 AI로 잘못 판정
FN (AI → 실제 미탐) : {fn}건 — AI 이미지를 실제로 놓침 ← 더 위험

[ 종합 의견 ]
{'✅ Recall 목표 달성' if recall >= 75 else '❌ Recall 목표 미달 — 데이터 보강 또는 모델 교체 필요'}
{'✅ Accuracy 목표 달성' if accuracy >= 70 else '❌ Accuracy 목표 미달'}
{'✅ 추론 속도 양호' if avg_time <= 3.0 else '⚠️  추론 속도 개선 필요'}

[ 다음 실험 제안 ]
{"- FN이 많음: AI 이미지 학습 데이터 다양성 확보 필요" if fn > fp else ""}
{"- FP가 많음: 오탐 줄이기 위한 임계값(threshold) 조정 고려" if fp > fn else ""}
{"- 모든 지표 목표 달성 → Phase 3 영상 확장 검토 가능" if all([accuracy>=70, recall>=75, f1>=70, fpr<=20]) else ""}"""

    return metrics_text, analysis_text


# ============================================================
# Gradio UI
# ============================================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 InSIGHT — AI 가상인물 판독기")
    gr.Markdown(
        "**woochul ver.** &nbsp;|&nbsp; "
        "1차 필터 (C2PA + SynthID) → 2차 필터 (ViT + Gram + FFT)\n\n"
        f"Device: `{device}` &nbsp;|&nbsp; "
        f"ViT 모델: `{'로드 완료' if VIT_AVAILABLE else '로드 실패'}` &nbsp;|&nbsp; "
        f"C2PA: `{'사용 가능' if C2PA_AVAILABLE else '라이브러리 없음'}`"
    )

    with gr.Tabs():

        # ── 탭 1: 단일 이미지 판독 ────────────────────────
        with gr.Tab("📷 이미지 판독 (서비스 모드)"):
            gr.Markdown("이미지를 업로드하면 AI 생성 가상인물 여부를 판별합니다.")

            with gr.Row():
                img_input   = gr.Image(label="분석할 이미지 업로드", type="numpy")
                img_preview = gr.Image(label="입력 이미지 미리보기", type="pil")

            analyze_btn = gr.Button("🔍 판독 시작", variant="primary", size="lg")

            with gr.Row():
                result_box = gr.Textbox(label="📋 판정 결과", interactive=False, lines=4)
                detail_box = gr.Textbox(label="🔬 상세 분석", interactive=False, lines=6)

            analyze_btn.click(
                fn=process_single_image,
                inputs=[img_input],
                outputs=[result_box, detail_box, img_preview]
            )

        # ── 탭 2: 베이스라인 성능 측정 ────────────────────
        with gr.Tab("📊 베이스라인 성능 측정"):
            gr.Markdown(
                "### 레이블된 이미지를 업로드하여 6가지 성능 지표를 측정합니다\n"
                "- **AI 생성 이미지** : Positive (label = 1)\n"
                "- **실제 인물 이미지** : Negative (label = 0)"
            )

            with gr.Row():
                ai_files_input   = gr.File(
                    label="🤖 AI 생성 이미지 (Positive)",
                    file_count="multiple",
                    file_types=["image"]
                )
                real_files_input = gr.File(
                    label="👤 실제 인물 이미지 (Negative)",
                    file_count="multiple",
                    file_types=["image"]
                )

            eval_btn = gr.Button("📊 성능 측정 시작", variant="primary", size="lg")

            with gr.Row():
                metrics_box  = gr.Textbox(label="📈 6가지 성능 지표", interactive=False, lines=16)
                analysis_box = gr.Textbox(label="🔎 오류 분석 및 종합 의견", interactive=False, lines=12)

            eval_btn.click(
                fn=run_baseline,
                inputs=[ai_files_input, real_files_input],
                outputs=[metrics_box, analysis_box]
            )

if __name__ == "__main__":
    print(f"[InSIGHT woochul ver.]")
    print(f"  Device    : {device}")
    print(f"  ViT 모델  : {'로드 완료' if VIT_AVAILABLE else '로드 실패 — transformers 확인 필요'}")
    print(f"  C2PA      : {'사용 가능' if C2PA_AVAILABLE else '없음 — pip install c2pa-python'}")
    demo.launch(debug=True, share=True)
