# ============================================================
# InSIGHT — 모델 자동 벤치마크 (ViT 순수 성능 비교)
# woochul ver.
#
# 목적: umm-maybe 대체 후보 5개 모델 성능 자동 측정
# 조건: 1차 필터(C2PA/FFT) 제외 — ViT 모델 성능만 순수 비교
# 지표: Accuracy / Precision / Recall / F1 / FPR / Inference Time
# ============================================================

import os
import time
import csv
from datetime import datetime

import numpy as np
from PIL import Image
from transformers import pipeline
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# ============================================================
# 설정 — 여기만 수정하면 됩니다
# ============================================================

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATASET_AI_DIR   = os.path.join(_ROOT, "test_dataset", "ai")
DATASET_REAL_DIR = os.path.join(_ROOT, "test_dataset", "real")

# 비교할 모델 5개 (umm-maybe 포함해서 기준선 함께 측정)
MODELS = [
    "umm-maybe/AI-image-detector",                    # 현재 기준 모델
    "Organika/sdxl-detector",                         # SDXL 특화
    "haywoodsloan/ai-image-detector-deploy",          # 다양한 생성 모델 학습
    "dima806/ai-generated-vs-real-image-detection",   # Real vs AI 직접 학습
    "ideepankarsharma2003/AI_Image_Classifier",       # ViT 기반 분류
    "prithivMLmods/Deep-Fake-Detector-Model",         # 딥페이크 통합 탐지
]

# 모델 출력 레이블에서 AI 확률을 추출하는 키워드
AI_LABEL_KEYWORDS = ["ai", "artificial", "fake", "generated", "synthetic", "deepfake", "FAKE"]

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEVICE = 0 if torch.cuda.is_available() else -1
THRESHOLD = 0.5  # AI 판정 임계값

# 성능 목표치 (달성 여부 표시용)
TARGET = {
    "accuracy": 70.0,
    "precision": 75.0,
    "recall": 75.0,
    "f1": 70.0,
    "fpr": 20.0,       # 이하
    "avg_time": 3.0,   # 이하
}


# ============================================================
# 데이터셋 로드
# ============================================================

def load_dataset(ai_dir: str, real_dir: str):
    """
    ai_dir  안의 이미지 → label=1 (Positive)
    real_dir 안의 이미지 → label=0 (Negative)
    반환: [(이미지경로, 레이블), ...]
    """
    dataset = []

    if not os.path.isdir(ai_dir):
        raise FileNotFoundError(f"AI 이미지 폴더를 찾을 수 없습니다: {ai_dir}")
    if not os.path.isdir(real_dir):
        raise FileNotFoundError(f"실제 이미지 폴더를 찾을 수 없습니다: {real_dir}")

    for fname in sorted(os.listdir(ai_dir)):
        if os.path.splitext(fname)[1].lower() in IMG_EXTENSIONS:
            dataset.append((os.path.join(ai_dir, fname), 1))

    for fname in sorted(os.listdir(real_dir)):
        if os.path.splitext(fname)[1].lower() in IMG_EXTENSIONS:
            dataset.append((os.path.join(real_dir, fname), 0))

    return dataset


# ============================================================
# 레이블 파싱 — 모델마다 출력 레이블 이름이 달라서 자동 감지
# ============================================================

def parse_ai_score(results: list) -> float:
    """
    pipeline 출력 리스트에서 AI 확률을 추출
    AI 관련 키워드가 있는 레이블 우선, 없으면 첫 번째 레이블 사용
    """
    for r in results:
        label = r["label"].lower()
        if any(kw.lower() in label for kw in AI_LABEL_KEYWORDS):
            return float(r["score"])
    # 키워드 미매칭: 레이블이 확인되지 않은 모델 — 첫 번째 레이블 점수 반환
    # (주의) 이 경우 로그를 확인하고 AI_LABEL_KEYWORDS 수동 추가 필요
    return float(results[0]["score"])


# ============================================================
# 단일 모델 벤치마크
# ============================================================

def benchmark_model(model_id: str, dataset: list) -> dict | None:
    """
    모델 하나를 전체 데이터셋에 대해 추론 후 6가지 지표 반환
    """
    print(f"\n{'─'*60}")
    print(f"  모델 로드 중: {model_id}")
    print(f"{'─'*60}")

    try:
        detector = pipeline(
            "image-classification",
            model=model_id,
            device=DEVICE,
        )
        print(f"  ✅ 로드 완료")
    except Exception as e:
        print(f"  ❌ 로드 실패: {e}")
        return None

    # 첫 번째 이미지로 레이블 이름 사전 확인
    first_img_path = dataset[0][0]
    try:
        probe = detector(Image.open(first_img_path).convert("RGB"))
        print(f"  레이블 확인: {[r['label'] for r in probe]}")
    except Exception:
        pass

    y_true, y_pred, times = [], [], []
    fail_count = 0

    for idx, (img_path, true_label) in enumerate(dataset, 1):
        try:
            pil_image = Image.open(img_path).convert("RGB")
            t0 = time.time()

            results = detector(pil_image)
            elapsed = time.time() - t0

            ai_score = parse_ai_score(results)
            pred = 1 if ai_score >= THRESHOLD else 0

            y_true.append(true_label)
            y_pred.append(pred)
            times.append(elapsed)

            # 진행 상황 (50장마다 출력)
            if idx % 50 == 0:
                print(f"  진행: {idx}/{len(dataset)}장 완료")

        except Exception as e:
            fail_count += 1
            print(f"  ⚠️  처리 실패 [{img_path}]: {e}")
            continue

    if not y_true:
        print("  ❌ 처리 가능한 이미지가 없습니다.")
        return None

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── 혼동 행렬 ────────────────────────────────────────────
    if len(set(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    else:
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    # ── 6가지 지표 ───────────────────────────────────────────
    accuracy  = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, zero_division=0) * 100
    recall    = recall_score(y_true, y_pred, zero_division=0) * 100
    f1        = f1_score(y_true, y_pred, zero_division=0) * 100
    fpr       = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0
    avg_time  = float(np.mean(times)) if times else 0.0

    return {
        "model":     model_id,
        "accuracy":  round(accuracy, 2),
        "precision": round(precision, 2),
        "recall":    round(recall, 2),
        "f1":        round(f1, 2),
        "fpr":       round(fpr, 2),
        "avg_time":  round(avg_time, 3),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_total":   len(y_true),
        "n_failed":  fail_count,
    }


# ============================================================
# 결과 출력
# ============================================================

def chk(val: float, target: float, lower: bool = False) -> str:
    return "✅" if (val <= target if lower else val >= target) else "❌"


def print_results(results: list):
    valid = [r for r in results if r is not None]
    if not valid:
        print("출력할 결과가 없습니다.")
        return

    n_ai   = sum(1 for _, l in dataset if l == 1)
    n_real = sum(1 for _, l in dataset if l == 0)

    print("\n" + "=" * 110)
    print(f"  InSIGHT 모델 벤치마크 결과")
    print(f"  실험 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  데이터셋: AI {n_ai}장 / Real {n_real}장  |  Device: {'CUDA' if DEVICE == 0 else 'CPU'}")
    print("=" * 110)

    header = f"{'모델':<48} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FPR':>7} {'Time':>7}  {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}"
    print(header)
    print("-" * 110)

    for r in valid:
        short = r["model"].split("/")[-1][:46]
        row = (
            f"{short:<48} "
            f"{r['accuracy']:>6.1f}% "
            f"{r['precision']:>6.1f}% "
            f"{r['recall']:>6.1f}% "
            f"{r['f1']:>6.1f}% "
            f"{r['fpr']:>6.1f}% "
            f"{r['avg_time']:>6.2f}s  "
            f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4} {r['tn']:>4}"
        )
        print(row)

    print("-" * 110)
    target_row = (
        f"{'[목표치]':<48} "
        f"{'70%+':>7} {'75%+':>7} {'75%+':>7} {'70%+':>7} {'20%-':>7} {'3.0s-':>7}"
    )
    print(target_row)
    print("=" * 110)

    # 목표 달성 체크
    print("\n[ 목표 달성 여부 ]")
    for r in valid:
        short = r["model"].split("/")[-1]
        flags = (
            f"  Acc {chk(r['accuracy'], TARGET['accuracy'])} "
            f"Prec {chk(r['precision'], TARGET['precision'])} "
            f"Rec {chk(r['recall'], TARGET['recall'])} "
            f"F1 {chk(r['f1'], TARGET['f1'])} "
            f"FPR {chk(r['fpr'], TARGET['fpr'], lower=True)} "
            f"Time {chk(r['avg_time'], TARGET['avg_time'], lower=True)}"
        )
        print(f"  {short:<46} {flags}")

    # 추천 모델 (Recall + F1 우선 순위)
    ranked = sorted(valid, key=lambda x: (x["recall"] + x["f1"]) / 2, reverse=True)
    print(f"\n  Recall+F1 기준 TOP 1: {ranked[0]['model']}")


# ============================================================
# CSV 저장
# ============================================================

def save_csv(results: list, filename: str):
    valid = [r for r in results if r is not None]
    if not valid:
        return

    fields = ["model", "accuracy", "precision", "recall", "f1",
              "fpr", "avg_time", "tp", "fp", "fn", "tn", "n_total", "n_failed"]

    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in valid:
            writer.writerow({k: r[k] for k in fields})

    print(f"\n  [CSV saved] {filename}")


# ============================================================
# 메인
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  InSIGHT — 모델 자동 벤치마크 시작")
    print("=" * 60)
    print(f"  Device     : {'CUDA' if DEVICE == 0 else 'CPU'}")
    print(f"  AI 폴더    : {DATASET_AI_DIR}")
    print(f"  Real 폴더  : {DATASET_REAL_DIR}")
    print(f"  비교 모델  : {len(MODELS)}개")
    print(f"  임계값     : {THRESHOLD}")

    # 데이터셋 로드
    try:
        dataset = load_dataset(DATASET_AI_DIR, DATASET_REAL_DIR)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("DATASET_AI_DIR / DATASET_REAL_DIR 경로를 확인해주세요.")
        exit(1)

    n_ai   = sum(1 for _, l in dataset if l == 1)
    n_real = sum(1 for _, l in dataset if l == 0)
    print(f"  데이터셋   : AI {n_ai}장 / Real {n_real}장  합계 {len(dataset)}장")

    # 모델별 벤치마크 실행
    all_results = []
    for model_id in MODELS:
        result = benchmark_model(model_id, dataset)
        all_results.append(result)

    # 결과 출력 + CSV 저장
    print_results(all_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(
        os.path.dirname(__file__),
        f"benchmark_results_{timestamp}.csv"
    )
    save_csv(all_results, csv_path)
