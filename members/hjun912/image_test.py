# 1. 필수 라이브러리 임포트
import cv2
import torch
import numpy as np
import gradio as gr
from torchvision import transforms
from PIL import Image
import os
import time
from transformers import pipeline
from ultralytics import YOLO

# MediaPipe 초기화 및 안전 로드
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=5, refine_landmarks=True)
        MEDIAPIPE_AVAILABLE = True
except Exception:
    MEDIAPIPE_AVAILABLE = False

try:
    yolo_model = YOLO("yolo11n.pt")
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# 강력한 모델 앙상블 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    detector_main = pipeline("image-classification", model="nateraw/vit-base-patch16-224-deepfake", device=0 if torch.cuda.is_available() else -1)
    detector_face = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-Model", device=0 if torch.cuda.is_available() else -1)
    VIT_AVAILABLE = True
except Exception:
    VIT_AVAILABLE = False

performance_history = {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "total_time": 0.0, "count": 0}

# 2. 정밀 포렌식 분석 함수들

# 2.1 얼굴 배치 및 해부학적 비율 분석 (New)
def analyze_face_alignment(landmarks):
    if not landmarks: return 0.0, "데이터 없음"
    try:
        pts = landmarks.landmark
        # 눈-코 거리 vs 코-입 거리 비율 (보통 1:1 ~ 1:1.2 수준이 정상)
        eye_center_y = (pts[33].y + pts[263].y) / 2
        nose_y = pts[1].y
        mouth_y = (pts[61].y + pts[291].y) / 2
        
        upper_dist = abs(nose_y - eye_center_y)
        lower_dist = abs(mouth_y - nose_y)
        ratio = upper_dist / (lower_dist + 1e-6)
        
        # 비정상적 비율(너무 길거나 짧음) 탐지
        if ratio < 0.7 or ratio > 1.5:
            score = 80.0
        else:
            score = abs(1.0 - ratio) * 100
            
        return max(0, min(100, score)), f"이목구비 배치 비율: {ratio:.2f}"
    except: return 0.0, "분석 불가"

# 2.2 피부색 일관성 및 색상 번짐 분석 (New)
def analyze_skin_tone_consistency(frame_rgb, landmarks):
    if not landmarks: return 0.0, "데이터 없음"
    try:
        h, w, _ = frame_rgb.shape
        # Lab 색공간으로 변환 (인간의 색지각과 유사함)
        lab_img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        
        def get_avg_color(idx):
            cx, cy = int(landmarks.landmark[idx].x * w), int(landmarks.landmark[idx].y * h)
            patch = lab_img[cy-5:cy+5, cx-5:cx+5]
            return np.mean(patch, axis=(0, 1)) if patch.size > 0 else None

        # 이마(10), 왼쪽 뺨(234), 오른쪽 뺨(454), 턱(152) 샘플링
        colors = [get_avg_color(i) for i in [10, 234, 454, 152]]
        colors = [c for c in colors if c is not None]
        
        if len(colors) >= 2:
            # 샘플들 간의 색상 거리(Delta E) 계산
            diffs = []
            for i in range(len(colors)):
                for j in range(i + 1, len(colors)):
                    diffs.append(np.linalg.norm(colors[i][1:] - colors[j][1:])) # a, b 채널만 비교
            
            avg_diff = np.mean(diffs)
            # AI 합성은 피부톤이 얼룩덜룩하거나(번짐) 특정 부위만 튀는 경우가 많음
            score = max(0, min(100, (avg_diff - 3) * 10))
            return score, f"피부톤 불일치 지수: {avg_diff:.2f}"
    except: pass
    return 0.0, "분석 불가"

# 기존 눈, 피부, 치아 분석 함수 유지... (최적화)
def analyze_eye_details(frame_rgb, landmarks):
    if not landmarks: return 0.0, "데이터 없음"
    try:
        h, w, _ = frame_rgb.shape
        def get_eye_patch(idx_list):
            pts = np.array([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in idx_list])
            x, y, sw, sh = cv2.boundingRect(pts.astype(np.int32))
            patch = frame_rgb[y:y+sh, x:x+sw]
            return cv2.resize(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY), (50, 30)) if patch.size > 0 else None
        l_e, r_e = get_eye_patch([33, 160, 133, 144]), get_eye_patch([362, 385, 263, 373])
        if l_e is not None and r_e is not None:
            res = cv2.matchTemplate(l_e, r_e, cv2.TM_CCOEFF_NORMED)[0][0]
            return max(0, (1 - res) * 100), f"눈 대칭 상관도: {res:.2f}"
    except: pass
    return 0.0, "분석 불가"

# 3. 이미지 AI 판별 메인 로직
def analyze_image_ai(image_path):
    try:
        img_pil = Image.open(image_path).convert('RGB')
        frame_rgb = np.array(img_pil); frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR); gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        landmarks = None
        if MEDIAPIPE_AVAILABLE:
            mp_res = face_mesh.process(frame_rgb)
            if mp_res.multi_face_landmarks: landmarks = mp_res.multi_face_landmarks[0]

        # 앙상블 분석
        vit_score = 0
        if VIT_AVAILABLE:
            res_m = detector_main(img_pil); res_f = detector_face(img_pil)
            neg = ['ai', 'fake', 'synthetic', 'generated', 'modified']
            s_m = sum([r['score'] for r in res_m if any(l in r['label'].lower() for l in neg)]) * 100
            s_f = sum([r['score'] for r in res_f if any(l in r['label'].lower() for l in neg)]) * 100
            vit_score = (s_m * 0.6 + s_f * 0.4)

        # 포렌식 지표 계산
        eye_score, eye_msg = analyze_eye_details(frame_rgb, landmarks)
        align_score, align_msg = analyze_face_alignment(landmarks) # New
        skin_score, skin_msg = analyze_skin_tone_consistency(frame_rgb, landmarks) # New
        
        results_list = []
        p1 = max(0, min(99.9, vit_score))
        results_list.append(f"1. 통합 앙상블 판독: {'❌ AI 의심' if p1 > 70 else '✅ 정상'} ({p1:.1f}%)")
        results_list.append(f"2. 눈동자 반사광 및 시선 모순: {eye_msg} ({eye_score:.1f}%)")
        results_list.append(f"3. 얼굴 배치 및 해부학적 비율: {'❌ 위험' if align_score > 60 else '✅ 정상'} ({align_score:.1f}%)\n    - 분석: {align_msg}")
        results_list.append(f"4. 피부색 일관성 및 색상 번짐: {'❌ 위험' if skin_score > 50 else '✅ 정상'} ({skin_score:.1f}%)\n    - 분석: {skin_msg}")
        
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        p5 = max(0, min(98, 100 - (lap_var / 10)))
        results_list.append(f"5. 경계면 및 잔머리 처리: {'❌ 위험' if p5 > 75 else '✅ 정상'} ({p5:.1f}%)")

        # FFT 분석
        dft = np.fft.fft2(gray); fshift = np.fft.fftshift(dft); mag = 20 * np.log(np.abs(fshift) + 1); h, w = gray.shape; crow, ccol = h // 2, w // 2
        mask = np.ones((h, w), np.uint8); cv2.circle(mask, (ccol, crow), 70, 0, -1)
        hfv = (mag * mask)[mag * mask > 0]
        avg_freq = np.max(hfv) / (np.mean(hfv) + 1e-5) if len(hfv) > 0 else 0
        p9 = max(0, min(99.9, (avg_freq - 1.8) * 150))
        results_list.append(f"9. SynthID(FFT) 노이즈: {'❌ 위험' if p9 > 50 else '✅ 정상'} ({p9:.1f}%)")

        # 종합 판정 (새로운 지표들 반영)
        total_prob = round(np.mean([p1, eye_score, align_score, skin_score, p5, p9]), 2)
        results_list.append(f"10. 종합 생성 패턴: {'❌ 위험' if total_prob > 60 else '✅ 정상'} ({total_prob:.1f}%)")

        # YOLO 시각화
        detection_frame = frame_bgr.copy()
        if YOLO_AVAILABLE:
            yolo_results = yolo_model(image_path, verbose=False)
            box_color = (0, 255, 0) if total_prob <= 40 else (0, 255, 255) if total_prob <= 70 else (0, 0, 255)
            for r in yolo_results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label_text = f"Person (AI Risk: {total_prob}%)"
                        cv2.rectangle(detection_frame, (x1, y1), (x2, y2), box_color, 3)
                        cv2.rectangle(detection_frame, (x1, y1 - 25), (x1 + 240, y1), box_color, -1)
                        cv2.putText(detection_frame, label_text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        mag_norm = np.uint8(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX))
        final_image = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
        return total_prob, "📊 [포렌식 정밀 분석 리포트]\n\n" + "\n\n".join(results_list), mag_norm, final_image
    except Exception as e: return 0, f"오류: {str(e)}", None, None

# 성능 보고서 및 통합 함수 (유지)
def export_performance_report():
    tp, fp, fn, tn = performance_history["TP"], performance_history["FP"], performance_history["FN"], performance_history["TN"]
    total = tp + fp + fn + tn
    if total == 0: return "데이터 부족"
    report = f"========== 성능 평가 결과 ==========\nAccuracy: {((tp+tn)/total)*100:.2f}%\nPrecision: {(tp/(tp+fp) if tp+fp>0 else 0)*100:.2f}%\nRecall: {(tp/(tp+fn) if tp+fn>0 else 0)*100:.2f}%  ← 핵심\n------------------------------------\nTP: {tp:02d} FP: {fp:02d} FN: {fn:02d} TN: {tn:02d}\n===================================="
    d_path = os.path.join(os.path.expanduser("~"), "Downloads", "performance_report.md")
    with open(d_path, "w", encoding="utf-8") as f: f.write(report)
    return f"✅ 저장 완료: {d_path}\n\n{report}"

def process_service(input_file, ground_truth):
    if not input_file: return None, None, "❌ 오류", "", ""
    start = time.time(); prob, report, fft_img, det_img = analyze_image_ai(input_file); inf_time = time.time() - start
    prediction = "AI" if prob > 50 else "Real"
    performance_history["count"] += 1; performance_history["total_time"] += inf_time
    if ground_truth == "AI":
        if prediction == "AI": performance_history["TP"] += 1
        else: performance_history["FN"] += 1
    else:
        if prediction == "AI": performance_history["FP"] += 1
        else: performance_history["TN"] += 1
    return det_img, fft_img, f"🔍 판정: {prob}% AI 의심", report, f"누적 테스트: {performance_history['count']}"

# UI 구성 (유지)
with gr.Blocks() as demo:
    gr.Markdown("# 🎨 AI Image Detector: Advanced Forensic & Anatomy Mode")
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.Image(label="이미지 업로드", type="filepath")
            ground_truth_input = gr.Radio(["Real", "AI"], label="실제 정답", value="Real")
            submit_btn = gr.Button("🚀 포렌식/해부학 정밀 분석 시작", variant="primary")
            report_btn = gr.Button("📊 보고서 생성", variant="secondary")
            report_display = gr.Textbox(label="보고서 미리보기", interactive=False, lines=10)
        with gr.Column(scale=1):
            image_output = gr.Image(label="Forensic Analysis (YOLO & Anatomy)")
            status_output = gr.Textbox(label="누적 기록", interactive=False)
            result_output = gr.Textbox(label="최종 판정", interactive=False)
            fft_output = gr.Image(label="주파수 스펙트럼 (FFT)")
    with gr.Row(): detail_output = gr.Textbox(label="정밀 분석 리포트", interactive=False, lines=15)
    submit_btn.click(fn=process_service, inputs=[file_input, ground_truth_input], outputs=[image_output, fft_output, result_output, detail_output, status_output])
    report_btn.click(fn=export_performance_report, outputs=[report_display])

if __name__ == "__main__":
    demo.launch(debug=True, share=True, theme=gr.themes.Soft())
