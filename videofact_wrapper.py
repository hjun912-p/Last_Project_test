import sys
import torch
import yaml
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path

# VideoFact 경로 추가
ROOT = Path(__file__).resolve().parent
VIDEOFACT_DIR = ROOT / "external" / "videofact"
sys.path.insert(0, str(VIDEOFACT_DIR))

try:
    from model.common.videofact import VideoFACT as VideoFACT_Module
    from model.videofact_pl_wrapper import VideoFACTPLWrapper
    VIDEOFACT_AVAILABLE = True
except ImportError as e:
    print(f"VideoFact 임포트 오류: {e}")
    VIDEOFACT_AVAILABLE = False

class VideoFactDetector:
    def __init__(self, model_type="df"):
        if not VIDEOFACT_AVAILABLE:
            self.model = None
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config_path = VIDEOFACT_DIR / "configs" / "default.yaml"
        self.ckpt_path = VIDEOFACT_DIR / "weights" / f"videofact_{model_type}.ckpt"
        
        if not self.ckpt_path.exists():
            print(f"가중치 파일 없음: {self.ckpt_path}")
            self.model = None
            return

        with open(self.config_path, "r") as f:
            self.config = yaml.full_load(f)

        try:
            # VideoFACTPLWrapper.load_from_checkpoint 사용
            self.model = VideoFACTPLWrapper.load_from_checkpoint(
                str(self.ckpt_path), 
                model=VideoFACT_Module, 
                map_location=self.device, 
                **self.config
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"VideoFact 모델 로드 실패: {e}")
            self.model = None

    @torch.no_grad()
    def detect(self, image: Image.Image) -> float:
        if self.model is None:
            return 0.0

        img_np = np.array(image.convert("RGB")).astype(np.float32)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        # inference_single.py 기준 학습 해상도
        img_tensor = TF.resize(img_tensor, (1080, 1920), antialias=True)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        class_out, _ = self.model(img_tensor)
        probs = torch.softmax(class_out, dim=1)
        return float(probs[0, 1].item())

# 싱글톤 인스턴스 생성 (최초 호출 시 로드)
_detector = None

def detect_videofact_score(image: Image.Image) -> float:
    global _detector
    if _detector is None:
        _detector = VideoFactDetector(model_type="df")
    return _detector.detect(image)

if __name__ == "__main__":
    # 간단한 테스트 코드
    test_img = Image.new('RGB', (1920, 1080), color = (73, 109, 131))
    score = detect_videofact_score(test_img)
    print(f"Test Score: {score}")
