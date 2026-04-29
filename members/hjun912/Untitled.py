import os
import logging
import cv2
import torch
import numpy as np
import time
import gradio as gr
from PIL import Image
from typing import Tuple, Dict, Optional, List, Any
from transformers import pipeline
from ultralytics import YOLO

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class AIDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = {
            "VIT_MAIN_WEIGHT": 0.6,
            "VIT_FACE_WEIGHT": 0.4,
            "AI_THRESHOLD": 60.0,
            "FFT_SENSITIVITY": 150,
            "FFT_BIAS": 1.8
        }
        self.MEDIAPIPE_AVAILABLE = False
        self.face_mesh = None
        self._init_mediapipe()
        self.models = self._load_models()

    def _init_mediapipe(self):
        try:
            import mediapipe as mp
            if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, 
                    max_num_faces=5, 
                    refine_landmarks=True
                )
                self.MEDIAPIPE_AVAILABLE = True
                logger.info("✅ MediaPipe FaceMesh 로드 성공")
        except Exception:
            logger.warning("⚠️ MediaPipe 분석 기능 비활성화")

    def _load_models(self) -> Dict[str, Any]:
        models = {}
        pipe_device = 0 if self.device == "cuda" else -1
        try:
            models['yolo'] = YOLO("yolo11n.pt").to(self.device)
        except Exception as e:
            logger.warning(f"⚠️ YOLO 로드 실패: {e}")

        model_names = {
            'vit_main': "umm-maybe/AI-image-detector",
            'vit_face': "prithivMLmods/Deep-Fake-Detector-Model"
        }

        for key, m_path in model_names.items():
            try:
                models[key] = pipeline(
                    "image-classification", 
                    model=m_path, 
                    device=pipe_device,
                    model_kwargs={"ignore_mismatched_sizes": True}
                )
                logger.info(f"✅ Model {key} 로드 완료")
            except Exception as e:
                logger.error(f"❌ 모델 {m_path} 로드 실패: {e}")
        return models

    def analyze_anatomy(self, frame_rgb: np.ndarray) -> Tuple[float, str]:
        if not self.MEDIAPIPE_AVAILABLE or not self.face_mesh:
            return -1.0, "N/A"
        try:
            results = self.face_mesh.process(frame_rgb)
            if not results or not results.multi_face_landmarks:
                return -1.0, "얼굴 미검출"
            pts = results.multi_face_landmarks[0].landmark
            upper_dist = abs(pts[1].y - (pts[33].y + pts[263].y) / 2)
            lower_dist = abs((pts[61].y + pts[291].y) / 2 - pts[1].y)
            ratio = upper_dist / (lower_dist + 1e-6)
            score = 80.0 if not (0.7 <= ratio <= 1.5) else abs(1.0 - ratio) * 100
            return float(score), f"{ratio:.2f}"
        except Exception:
            return -1.0, "분석 오류"

    def analyze_frequency(self, gray_img: np.ndarray) -> Tuple[float, np.ndarray]:
        f = np.fft.fft2(gray_img)
        fshift = np.fft.fftshift(f)
        mag = 20 * np.log(np.abs(fshift) + 1e-5)
        h, w = gray_img.shape
        mask = np.ones((h, w), np.uint8)
        cv2.circle(mask, (w//2, h//2), 70, 0, -1)
        hfv = mag[mask == 1]
        avg_freq = np.max(hfv) / (np.mean(hfv) + 1e-5) if len(hfv) > 0 else 0
        score = max(0, min(99.9, (avg_freq - self.config["FFT_BIAS"]) * self.config["FFT_SENSITIVITY"]))
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return float(score), mag_norm

    def process_image(self, image_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str, float]:
        if image_input is None: return None, None, "이미지 없음", 0.0
        
        img_pil = Image.fromarray(image_input).convert('RGB')
        frame_rgb = np.array(img_pil)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # 1. ViT 분석
        vit_score = -1.0
        if self.models.get('vit_main') and self.models.get('vit_face'):
            neg = {'ai', 'fake', 'synthetic', 'generated', 'modified', 'label_1'}
            res_m = self.models['vit_main'](img_pil)
            res_f = self.models['vit_face'](img_pil)
            s_m = sum(r['score'] for r in res_m if any(k in r['label'].lower() for k in neg)) * 100
            s_f = sum(r['score'] for r in res_f if any(k in r['label'].lower() for k in neg)) * 100
            vit_score = (s_m * self.config["VIT_MAIN_WEIGHT"] + s_f * self.config["VIT_FACE_WEIGHT"])

        # 2. 구조 및 주파수 분석
        anat_score, anat_msg = self.analyze_anatomy(frame_rgb)
        fft_score, fft_map = self.analyze_frequency(gray)

        # 3. 유효 지표 평균 산출
        valid_scores = [s for s in [vit_score, anat_score, fft_score] if s >= 0]
        total_prob = float(np.mean(valid_scores)) if valid_scores else 0.0
        
        # 4. 리포트 텍스트
        report = f"--- Deepfake Analysis Report ---\n"
        report += f"1. Ensemble ViT: {vit_score:.1f}%\n"
        report += f"2. Anatomy Ratio: {anat_score:.1f}% (Val: {anat_msg})\n"
        report += f"3. FFT Frequency: {fft_score:.1f}%\n"
        report += f"-------------------------------\n"
        report += f"FINAL AI RISK: {total_prob:.1f}%"

        # 5. 시각화 (YOLO)
        if 'yolo' in self.models:
            results = self.models['yolo'](frame_rgb, verbose=False)
            color = (0,255,0) if total_prob<=40 else (0,255,255) if total_prob<=70 else (0,0,255)
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 3)
                        label = f"Person (Risk: {total_prob:.1f}%)"
                        cv2.rectangle(frame_bgr, (x1, y1-25), (x1+240, y1), color, -1)
                        cv2.putText(frame_bgr, label, (x1+5, y1-7), 1, 1.2, (0,0,0), 2)

        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), fft_map, report, total_prob

# 6. Gradio 연동 부분
detector = AIDetector()

def service_wrap(img):
    if img is None: return None, None, "Please upload an image."
    res_img, fft_map, report, _ = detector.process_image(img)
    return res_img, fft_map, report

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI Image Forensic Pro")
    gr.Markdown("이전 버전보다 정돈된 클래스 기반의 정밀 분석 도구입니다.")
    
    with gr.Row():
        input_img = gr.Image(label="Upload Image")
        with gr.Column():
            output_img = gr.Image(label="Analysis Result")
            output_fft = gr.Image(label="FFT Spectrum")
    
    with gr.Row():
        output_report = gr.Textbox(label="Deepfake Report", lines=8)
    
    submit_btn = gr.Button("Analyze Now", variant="primary")
    submit_btn.click(fn=service_wrap, inputs=input_img, outputs=[output_img, output_fft, output_report])

if __name__ == "__main__":
    demo.launch(debug=True, share=True)
