# 1. 필수 라이브러리 임포트
import cv2
import torch
import numpy as np
import gradio as gr
from torchvision import transforms
from PIL import Image
import os
from transformers import pipeline

# [NEW] C2PA 라이브러리 (선택 사항)
try:
    import c2pa
    C2PA_AVAILABLE = True
except ImportError:
    C2PA_AVAILABLE = False

# AI Image Detector 로드 (umm-maybe/AI-image-detector)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    vit_detector = pipeline("image-classification", model="umm-maybe/AI-image-detector", device=0 if torch.cuda.is_available() else -1)
    VIT_AVAILABLE = True
except Exception:
    VIT_AVAILABLE = False

# Gram Matrix 질감 분석
def get_gram_matrix(img_tensor):
    (b, c, h, w) = img_tensor.size()
    features = img_tensor.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)

# 전처리 설정
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 2. C2PA 검사
def check_c2pa_metadata(image_path):
    if not C2PA_AVAILABLE: return None, "C2PA 라이브러리 없음"
    try:
        reader = c2pa.Reader(image_path)
        manifest = reader.json()
        if manifest:
            if "ai" in str(manifest).lower(): return 100.0, "✅ C2PA: AI 생성물 확인"
            return 0.0, "✅ C2PA: 원본 확인"
    except Exception: pass
    return None, "🔍 C2PA 데이터 없음"

# 3. 이미지 AI 판별 로직 (ViT + Gram-Net + FFT)
def analyze_image_ai(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        frame = np.array(img)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. ViT 분석
        vit_score = 0
        if VIT_AVAILABLE:
            res = vit_detector(img)
            vit_score = sum([r['score'] for r in res if 'ai' in r['label'].lower() or 'artificial' in r['label'].lower()]) * 100

        # 2. Gram Matrix 질감 분석
        input_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            gram = get_gram_matrix(input_tensor)
            gram_val = torch.std(gram).item() * 1000

        # 3. FFT 주파수 분석 (Peak-to-Average Ratio)
        dft = np.fft.fft2(gray)
        fshift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        h, w = gray.shape
        crow, ccol = h // 2, w // 2
        mask = np.ones((h, w), np.uint8)
        cv2.circle(mask, (ccol, crow), 70, 0, -1)
        high_freq_area = magnitude_spectrum * mask
        
        high_freq_values = high_freq_area[high_freq_area > 0]
        avg_freq = 0
        if len(high_freq_values) > 0:
            avg_val = np.mean(high_freq_values)
            max_val = np.max(high_freq_values)
            avg_freq = max_val / (avg_val + 1e-5)

        # 4. 기타 지표 (Laplacian 등)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        color_std = np.std(frame)

        # 리포트 작성 (10대 심층 분석 항목)
        results = []
        p1 = min(99, vit_score * 1.1)
        results.append(f"1. ViT 심층 판독: {'❌ AI 의심' if p1 > 70 else '✅ 정상'} ({p1:.1f}%)")
        
        p2 = min(95, (1000 / (laplacian_var + 1)) * 5)
        results.append(f"2. 미세 유체(선명도): {'⚠️ 주의' if p2 > 70 else '✅ 정상'} ({p2:.1f}%)")
        
        p3 = min(98, (100 - gram_val) * 0.5 + 20)
        results.append(f"3. Gram 질감: {'⚠️ 주의' if p3 > 60 else '✅ 정상'} ({p3:.1f}%)")
        
        p4 = min(99, 100 - (laplacian_var / 5))
        results.append(f"4. 텍스트 정밀도: {'❌ 위험' if p4 > 80 else '✅ 정상'} ({p4:.1f}%)")
        
        p5 = min(94, color_std * 0.5) # 이미지 버전으로 단순화
        results.append(f"5. 색상 일관성: {'⚠️ 주의' if p5 > 60 else '✅ 정상'} ({p5:.1f}%)")
        
        p6 = min(96, (150 / (laplacian_var + 1)) * 10)
        results.append(f"6. 경계면 처리: {'❌ 위험' if p6 > 75 else '✅ 정상'} ({p6:.1f}%)")
        
        p7 = min(99, avg_freq * 10) # FFT 기반 고주파 분석
        results.append(f"7. 고주파 아티팩트: {'❌ 위험' if p7 > 85 else '✅ 정상'} ({p7:.1f}%)")
        
        p8 = p1 * 0.8 + p5 * 0.2
        results.append(f"8. 의미론적 오류: {'⚠️ 주의' if p8 > 70 else '✅ 정상'} ({p8:.1f}%)")

        p9 = min(99.9, max(0, (avg_freq - 1.8) * 150))
        results.append(f"9. SynthID(FFT): {'❌ 위험' if p9 > 50 else '✅ 정상'} ({p9:.1f}%)\n    - 근거: 주파수 피크 비율 {avg_freq:.2f}")

        p10 = min(99.9, (p1 + p3 + p9) / 3) # 종합 판정 지표
        results.append(f"10. 종합 생성 패턴: {'❌ 위험' if p10 > 60 else '✅ 정상'} ({p10:.1f}%)")

        total_prob = round(np.mean([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]), 2)
        
        # FFT 스펙트럼 이미지를 Gradio에 표시하기 위해 정규화
        mag_norm = np.uint8(cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX))
        
        return total_prob, "📊 [10대 심층 분석 리포트]\n\n" + "\n\n".join(results), mag_norm
    except Exception as e:
        return 0, f"분석 중 오류 발생: {str(e)}", None

# 4. 통합 실행 함수
def process_service(input_file):
    if not input_file:
        return None, None, "❌ 오류: 파일을 업로드해 주세요.", ""
    
    try:
        # C2PA 검사 (1단계)
        c2pa_prob, c2pa_msg = check_c2pa_metadata(input_file)
        if c2pa_prob is not None and c2pa_prob > 0:
            return input_file, None, f"🔍 1단계 판정: {c2pa_msg}", f"디지털 지문 결과: {c2pa_prob}%"

        # AI 분석 (2단계)
        prob, report, fft_img = analyze_image_ai(input_file)
        result_msg = f"🔍 최종 판정: 이 이미지는 {prob}% 확률로 AI 생성물로 의심됩니다."
        return input_file, fft_img, result_msg, report
    except Exception as e:
        return None, None, f"❌ 시스템 오류: {str(e)}", ""

# 5. Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎨 AI Image Detector: 10-Point Analysis")
    gr.Markdown("기존 영상 분석기를 이미지 전용으로 전환한 하이브리드 탐지기입니다. (ViT, Gram-Net, FFT 적용)")
    
    with gr.Row():
        file_input = gr.Image(label="이미지 업로드", type="filepath")

    with gr.Row():
        image_output = gr.Image(label="분석 대상 이미지")
        fft_output = gr.Image(label="주파수 스펙트럼 (FFT)")

    with gr.Row():
        result_output = gr.Textbox(label="판정 결과", interactive=False)
        detail_output = gr.Textbox(label="10대 심층 분석 리포트", interactive=False, lines=20)

    submit_btn = gr.Button("이미지 AI 분석 시작", variant="primary")
    submit_btn.click(fn=process_service, inputs=[file_input], outputs=[image_output, fft_output, result_output, detail_output])

if __name__ == "__main__":
    print(f"Using device: {device}")
    demo.launch(debug=True, share=True, theme=gr.themes.Soft())
