import os
import numpy as np
from PIL import Image
import sys
import cv2
import matplotlib.pyplot as plt

# Add reverse_synthid to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, "reverse_synthid"))

try:
    from extraction.robust_extractor import RobustSynthIDExtractor
    from extraction.synthid_bypass_v4 import SpectralCodebookV4
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Import Error: {e}")
    IMPORT_SUCCESS = False

_EXTRACTOR = None
_CODEBOOK = None

def get_synthid_tools():
    global _EXTRACTOR, _CODEBOOK
    if not IMPORT_SUCCESS: return None, None
    if _EXTRACTOR is None:
        try:
            _EXTRACTOR = RobustSynthIDExtractor()
            _CODEBOOK = SpectralCodebookV4()
            codebook_path = os.path.join(project_root, "artifacts", "spectral_codebook_v4.npz")
            if os.path.exists(codebook_path): _CODEBOOK.load(codebook_path)
        except: return None, None
    return _EXTRACTOR, _CODEBOOK

def create_visualization(img_np, filename="synthid_viz.png"):
    """주파수 영역의 스펙트럼 이상 징후를 시각화합니다."""
    # Convert to grayscale and perform FFT
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(img_np), plt.title('Original Image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='magma'), plt.title('Spectral Signature')
    plt.xticks([]), plt.yticks([])
    
    viz_path = os.path.join(project_root, "temp", filename)
    plt.savefig(viz_path)
    plt.close()
    return viz_path

def check_synthid(file_path, media_type):
    if not os.path.exists(file_path): return False, "파일 없음", None
    extractor, codebook = get_synthid_tools()
    if extractor is None: return False, "도구 로드 실패", None

    try:
        viz_path = None
        if media_type == "image":
            img = Image.open(file_path).convert("RGB")
            img_np = np.array(img)
            result = extractor.detect_from_v4_codebook(img_np, codebook)
            is_found = result.get('is_watermarked', False) if isinstance(result, dict) else getattr(result, 'is_watermarked', False)
            conf = result.get('confidence', 0) if isinstance(result, dict) else getattr(result, 'confidence', 0)
            
            detail_msg = f"### [분석 상세]\n"
            detail_msg += f"- **탐지 결과:** {'성공' if is_found else '미탐지'}\n"
            detail_msg += f"- **신뢰도:** {conf:.2%}\n"
            detail_msg += f"- **기술적 근거:** 주파수 영역의 특정 Bin에서 Google Gemini 고유의 위상 일관성 패턴이 확인되었습니다." if is_found else "- **기술적 근거:** 주파수 영역에서 인위적인 워터마크 신호가 통계적 임계치를 넘지 않았습니다."
            
            if is_found:
                viz_path = create_visualization(img_np, "synthid_result.png")
            
            return is_found, detail_msg, viz_path

        elif media_type == "video":
            cap = cv2.VideoCapture(file_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # 20개의 균등한 지점에서 프레임 추출 (정밀도 상향)
            check_points = np.linspace(0, total_frames - 1, 20, dtype=int)

            best_conf = -1
            best_frame = None
            detected_frames_count = 0

            # Threshold for strict detection
            STRICT_THRESHOLD = 0.65

            for idx in check_points:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = extractor.detect_from_v4_codebook(frame_rgb, codebook)
                    conf = result.get('confidence', 0) if isinstance(result, dict) else getattr(result, 'confidence', 0)

                    if conf > best_conf:
                        best_conf = conf
                        best_frame = frame_rgb

                    if conf >= STRICT_THRESHOLD:
                        detected_frames_count += 1

            cap.release()

            # 20개 중 4개(20%) 이상 프레임에서 탐지될 경우 최종 판정
            is_found_final = detected_frames_count >= 4

            if best_frame is not None:
                detail_msg = f"### [비디오 정밀 분석 결과]\n"
                detail_msg += f"- **최종 판정:** {'AI 생성형 워터마크 탐지' if is_found_final else '실제 촬영 미디어(또는 워터마크 미감지)'}\n"
                detail_msg += f"- **최대 신뢰도:** {best_conf:.2%}\n"
                detail_msg += f"- **탐지된 프레임 수:** {detected_frames_count}/20\n"
                detail_msg += f"- **분석 근거:** 샘플링된 20개 프레임 중 {detected_frames_count}개에서 고신뢰도(0.65↑) 워터마크 신호가 확인되었습니다."

                
                viz_path = create_visualization(best_frame, "synthid_video_result.png")
                return is_found_final, detail_msg, viz_path
            
            return False, "비디오 프레임 추출 실패", None
    except Exception as e:
        return False, f"에러: {str(e)}", None
