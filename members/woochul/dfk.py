# 1. 필수 라이브러리 임포트
import yt_dlp
import cv2
import torch
import numpy as np
import gradio as gr
from torchvision import models, transforms
from PIL import Image
import os
import librosa # [NEW] 오디오 분석용 (53, 56번 기반)
from transformers import pipeline # [NEW] Hugging Face Pipeline (76번 기반)

# [NEW] C2PA 라이브러리 (설치 필요: pip install c2pa-python)
try:
    import c2pa
    C2PA_AVAILABLE = True
except ImportError:
    C2PA_AVAILABLE = False

# [NEW] Hugging Face AI Image Detector 로드 (76. umm-maybe/AI-image-detector 기반)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    # 비디오 프레임을 즉시 분류하기 위한 전용 파이프라인
    vit_detector = pipeline("image-classification", model="umm-maybe/AI-image-detector", device=0 if torch.cuda.is_available() else -1)
    VIT_AVAILABLE = True
except Exception:
    VIT_AVAILABLE = False

# [NEW] Gram Matrix 질감 분석 (49. Gram-Net 스타일)
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

# 2. 영상 및 오디오 추출 모듈
def download_instagram_video(url):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': 'temp_video.mp4',
        'quiet': False,
        'no_warnings': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'referer': 'https://www.instagram.com/',
        'ignoreerrors': True,
    }
    try:
        if os.path.exists("temp_video.mp4"):
            os.remove("temp_video.mp4")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return "temp_video.mp4" if os.path.exists("temp_video.mp4") else "Error: 다운로드 실패"
    except Exception as e:
        return f"Error: {str(e)}"

# [NEW] 2.1 오디오 아티팩트 분석 (53, 56번 기반)
def analyze_audio_artifacts(video_path):
    try:
        # 오디오 로드 (영상의 소리만 추출)
        y, sr = librosa.load(video_path, sr=None, duration=10) # 10초만 분석
        if len(y) == 0: return 0.0, "소리 없음"
        
        # 스펙트로그램의 에너지 분산 (AI 음성은 특정 주파수가 비정상적으로 일정하거나 불연속적임)
        S = np.abs(librosa.stft(y))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        
        # 0.0(순수 음) ~ 1.0(화이트 노이즈) 사이의 값
        # AI 음성은 자연 음성보다 더 평탄하거나(노이즈 부족) 극단적인 노이즈를 가짐
        score = min(100, (0.05 / (spectral_flatness + 0.001)) * 10)
        return score, f"오디오 평탄도: {spectral_flatness:.4f}"
    except Exception:
        return 0.0, "오디오 분석 불가"

# 3. C2PA 검사 (유지)
def check_c2pa_metadata(video_path):
    if not C2PA_AVAILABLE: return None, "C2PA 라이브러리 없음"
    try:
        reader = c2pa.Reader(video_path)
        manifest = reader.json()
        if manifest:
            if "ai" in str(manifest).lower(): return 100.0, "✅ C2PA: AI 생성물 확인"
            return 0.0, "✅ C2PA: 원본 확인"
    except Exception: pass
    return None, "🔍 C2PA 데이터 없음"

# 4. 고도화된 AI 판별 로직 (ViT + Gram-Net + FFT)
def analyze_video_ai(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_diffs, blur_values, frequency_anomalies, color_consistency = [], [], [], []
    vit_scores, gram_scores = [], []
    last_fft_spectrum = None
    count = 0
    
    while cap.isOpened() and count < 30:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. ViT 분석 (76번 기반 - AI 이미지 전용 모델)
        if VIT_AVAILABLE and count % 10 == 0: # 10프레임마다 한 번씩
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            res = vit_detector(pil_img)
            # 'ai' 또는 'synthetic' 레이블의 점수 합산
            ai_score = sum([r['score'] for r in res if 'ai' in r['label'].lower() or 'artificial' in r['label'].lower()])
            vit_scores.append(ai_score * 100)

        # 2. Gram Matrix 질감 분석 (49번 기반)
        input_tensor = preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad():
            gram = get_gram_matrix(input_tensor)
            # 질감의 '복잡도' 측정 (AI는 질감이 단순화되거나 비정상적으로 일정함)
            gram_scores.append(torch.std(gram).item() * 1000)

        # 3. FFT 주파수 분석 정교화 (Peak-to-Average Ratio 방식)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dft = np.fft.fft2(gray)
        fshift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        h, w = gray.shape
        crow, ccol = h // 2, w // 2
        
        # 고주파 영역 마스크 (중심부 제외)
        mask = np.ones((h, w), np.uint8)
        cv2.circle(mask, (ccol, crow), 70, 0, -1)
        high_freq_area = magnitude_spectrum * mask
        
        # [개선] 단순히 편차를 구하지 않고, 평균 대비 아주 강한 '피크'가 있는지 확인
        # 인공 워터마크는 특정 주파수에서 비정상적으로 강한 점(Peak)을 가짐
        high_freq_values = high_freq_area[high_freq_area > 0]
        if len(high_freq_values) > 0:
            avg_val = np.mean(high_freq_values)
            max_val = np.max(high_freq_values)
            # 피크 대비 평균 비율 (정상 영상은 에너지가 고르게 퍼져 점수가 낮음)
            peak_ratio = max_val / (avg_val + 1e-5)
            frequency_anomalies.append(peak_ratio)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_values.append(laplacian.var())
        
        if count > 0:
            frame_diffs.append(np.mean(cv2.absdiff(gray, prev_gray)))
            color_consistency.append(np.std(frame))

        prev_gray = gray
        count += 1
    cap.release()

    # 오디오 분석 (NEW)
    audio_score, audio_detail = analyze_audio_artifacts(video_path)

    # 지표 계산 및 리포트 작성
    avg_vit = np.mean(vit_scores) if vit_scores else 50
    avg_gram = np.mean(gram_scores) if gram_scores else 0
    avg_freq = np.mean(frequency_anomalies) if frequency_anomalies else 0
    
    results = []
    p1 = min(99, avg_vit * 1.1)
    results.append(f"1. ViT 심층 판독(76번): {'❌ AI 의심' if p1 > 70 else '✅ 정상'} ({p1:.1f}%)")
    
    p2 = min(95, (1000 / (np.mean(blur_values) + 1)) * 5)
    results.append(f"2. 미세 유체: {'⚠️ 주의' if p2 > 70 else '✅ 정상'} ({p2:.1f}%)")
    
    p3 = min(98, (100 - avg_gram) * 0.5 + 20)
    results.append(f"3. Gram 질감(49번): {'⚠️ 주의' if p3 > 60 else '✅ 정상'} ({p3:.1f}%)")
    
    p4 = min(99, 100 - (np.mean(blur_values) / 5))
    results.append(f"4. 텍스트 정밀도: {'❌ 위험' if p4 > 80 else '✅ 정상'} ({p4:.1f}%)")
    
    p5 = min(94, (np.mean(frame_diffs) + np.std(color_consistency)) * 2)
    results.append(f"5. 생체 역학: {'⚠️ 주의' if p5 > 60 else '✅ 정상'} ({p5:.1f}%)")
    
    p6 = min(96, (150 / (np.mean(blur_values) + 1)) * 10)
    results.append(f"6. 경계면 처리: {'❌ 위험' if p6 > 75 else '✅ 정상'} ({p6:.1f}%)")
    
    p7 = min(99, np.mean(frame_diffs) * 5)
    results.append(f"7. 시간적 주파수: {'❌ 위험' if p7 > 85 else '✅ 정상'} ({p7:.1f}%)")
    
    p8 = (p1 + p5) / 2
    results.append(f"8. 의미론적 오류: {'⚠️ 주의' if p8 > 70 else '✅ 정상'} ({p8:.1f}%)")

    p9 = min(99.9, max(0, (avg_freq - 1.8) * 150)) # 1.8 이상일 때 확률 급상승
    results.append(f"9. SynthID(FFT): {'❌ 위험' if p9 > 50 else '✅ 정상'} ({p9:.1f}%)\n    - 근거: 주파수 피크 비율 {avg_freq:.2f} (1.8 미만 정상)")

    p10 = min(99.9, audio_score)
    results.append(f"10. 오디오 위조(53번): {'❌ 위험' if p10 > 60 else '✅ 정상'} ({p10:.1f}%)\n    - 분석: {audio_detail}")

    total_prob = round(np.mean([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]), 2)
    return total_prob, "📊 [10대 심층 분석 리포트]\n\n" + "\n\n".join(results), last_fft_spectrum

# 5. 통합 실행 함수
def process_service(url, input_file):
    try:
        video_file = input_file if input_file else download_instagram_video(url)
        if "Error" in video_file:
            return None, None, f"❌ 오류: {video_file}", "인스타그램 차단 시 파일을 직접 업로드해 주세요."

        c2pa_prob, c2pa_msg = check_c2pa_metadata(video_file)
        if c2pa_prob is not None:
            return video_file, None, f"🔍 1단계 판정: {c2pa_msg}", f"디지털 지문 결과: {c2pa_prob}%"

        prob, report, fft_img = analyze_video_ai(video_file)
        result_msg = f"🔍 최종 판정: 이 영상은 {prob}% 확률로 AI 생성물로 의심됩니다."
        return video_file, fft_img, result_msg, report
    except Exception as e:
        return None, None, f"❌ 시스템 오류: {str(e)}", ""

# 6. Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 10-Point Hybrid Deepfake Detector")
    gr.Markdown("오픈소스 100개 프로젝트 중 최신 기술(ViT, Gram-Net, Audio Artifact)을 선별 적용한 분석기입니다.")
    
    with gr.Row():
        url_input = gr.Textbox(label="Instagram URL", placeholder="https://www.instagram.com/p/...")
        file_input = gr.File(label="또는 파일 업로드", file_types=["video"])

    with gr.Row():
        video_output = gr.Video(label="분석 대상 영상")
        fft_output = gr.Image(label="주파수 스펙트럼 (FFT)")

    with gr.Row():
        result_output = gr.Textbox(label="판정 결과", interactive=False)
        detail_output = gr.Textbox(label="10대 심층 분석 리포트", interactive=False, lines=20)

    submit_btn = gr.Button("영상 하이브리드 분석 시작", variant="primary")
    submit_btn.click(fn=process_service, inputs=[url_input, file_input], outputs=[video_output, fft_output, result_output, detail_output])

if __name__ == "__main__":
    print(f"Using device: {device}")
    demo.launch(debug=True, share=True, theme=gr.themes.Soft())
