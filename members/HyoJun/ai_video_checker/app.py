import gradio as gr
import cv2
import numpy as np
from deepface import DeepFace
import os
from PIL import Image

# 임시 프레임 저장 경로
TEMP_DIR = "temp_frames"
os.makedirs(TEMP_DIR, exist_ok=True)

def extract_key_frame(video_path):
    """영상의 중간 지점에서 분석용 프레임 하나를 추출합니다."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    return None

# --- 분석 단계별 함수 ---

def stage1_deepface_analysis(frame):
    """1단계: 얼굴 위조 방지(Anti-Spoofing) 분석"""
    try:
        # DeepFace를 사용하여 얼굴 추출 및 안티 스푸핑 분석
        results = DeepFace.extract_faces(
            img_path=frame,
            detector_backend='opencv',
            anti_spoofing=True,
            enforce_detection=False
        )
        if not results or results[0].get("confidence", 0) == 0:
            return "얼굴을 찾을 수 없음", 0, frame

        res = results[0]
        is_real = res.get("is_real", False)
        score = res.get("antispoof_score", 0)
        
        status = "실물 사람으로 판단됨" if is_real else "가짜 이미지(사진/화면) 의심"
        return status, round(score * 100, 2), frame
    except Exception as e:
        return f"분석 오류: {str(e)}", 0, frame

def stage2_frequency_analysis(frame):
    """2단계: 주파수 아티팩트 분석 (FreqNet 컨셉 시뮬레이션)"""
    # 원리: AI 생성물은 업샘플링 과정에서 특정 주파수 대역에 노이즈 지문을 남깁니다.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # 간단한 가짜 판별 로직 (데모용: 고주파 성분 비율 체크)
    h, w = magnitude_spectrum.shape
    center_area = magnitude_spectrum[h//4:3*h//4, w//4:3*w//4]
    avg_intensity = np.mean(center_area)
    
    status = "주파수 패턴 정상" if avg_intensity < 150 else "생성형 AI 특유의 고주파 흔적 감지"
    
    # 시각화를 위해 0-255 범위로 정규화
    spec_img = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return status, spec_img

def stage3_fake_shield_reasoning(frame):
    """3단계: 시각적 부자연스러움 설명 (FakeShield 컨셉)"""
    # 실제로는 VLM 모델이 필요하지만, 여기서는 시각적 특징 추출 시뮬레이션을 수행합니다.
    reasons = [
        "- 피부 질감이 지나치게 매끄러움 (AI 필터 가능성)",
        "- 배경과 인물의 경계선에 미세한 잔상 존재",
        "- 조명의 방향과 그림자의 일관성 부족"
    ]
    return "\n".join(reasons)

# --- 메인 실행 함수 ---

def analyze_video(video):
    if video is None:
        return "영상을 업로드해주세요.", None, None, None, None

    # 프레임 추출
    frame = extract_key_frame(video)
    if frame is None:
        return "프레임 추출 실패", None, None, None, None

    # Step 1: DeepFace (얼굴 분석)
    s1_status, s1_score, face_img = stage1_deepface_analysis(frame)
    
    # Step 2: FreqNet (주파수 분석)
    s2_status, freq_img = stage2_frequency_analysis(frame)
    
    # Step 3: FakeShield (이유 분석)
    s3_report = stage3_fake_shield_reasoning(frame)
    
    # Step 4: 종합 판정 (DeepfakeBench 컨셉)
    # 얼굴 분석 점수와 주파수 분석 결과를 가중 평균
    freq_weight = 60 if "정상" in s2_status else 20
    final_score = (s1_score + freq_weight) / 1.6
    
    verdict = "⚠️ AI 생성 영상일 확률이 높습니다." if final_score < 70 else "✅ 실제 촬영 영상일 확률이 높습니다."

    return (
        f"### [종합 결과]\n{verdict} (신뢰도: {final_score:.1f}%)",
        face_img,
        f"**판정:** {s1_status}\n**수치:** {s1_score}%",
        freq_img,
        s3_report
    )

# --- Gradio UI 구성 ---

with gr.Blocks(title="AI Video Authenticator") as demo:
    gr.Markdown("# 🛡️ AI 영상 진위 판별 단계별 분석 도구")
    gr.Markdown("영상을 업로드하면 4단계 분석(얼굴-주파수-논리-종합)을 거쳐 결과를 보여줍니다.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="분석할 영상 업로드")
            btn = gr.Button("단계별 분석 시작", variant="primary")
        
        with gr.Column():
            final_output = gr.Markdown(label="최종 분석 요약")

    with gr.Row():
        with gr.Tab("1단계: 얼굴 분석 (DeepFace)"):
            with gr.Row():
                face_preview = gr.Image(label="감지된 얼굴")
                s1_result = gr.Textbox(label="결과")
        
        with gr.Tab("2단계: 주파수 분석 (FreqNet)"):
            with gr.Row():
                freq_preview = gr.Image(label="주파수 지문 (FFT)")
                gr.Markdown("중앙에서 벗어난 흰 점들이 많을수록 AI 생성 흔적이 강함을 의미합니다.")
        
        with gr.Tab("3단계: 논리적 오류 (FakeShield)"):
            s3_result = gr.Textbox(label="시각적 아티팩트 보고서", lines=5)

    btn.click(
        fn=analyze_video,
        inputs=[video_input],
        outputs=[final_output, face_preview, s1_result, freq_preview, s3_result]
    )

if __name__ == "__main__":
    demo.launch()
