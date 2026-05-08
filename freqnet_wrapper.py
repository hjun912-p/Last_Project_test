import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import cv2

# FreqNet 경로 추가
ROOT = Path(__file__).resolve().parent
FREQNET_DIR = ROOT / "external" / "freqnet"
sys.path.insert(0, str(FREQNET_DIR))

try:
    from networks.freqnet import FreqNet, Bottleneck
    FREQNET_AVAILABLE = True
except ImportError as e:
    print(f"FreqNet 임포트 오류: {e}")
    FREQNET_AVAILABLE = False

class FreqNetDetector:
    def __init__(self):
        if not FREQNET_AVAILABLE:
            self.model = None
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ckpt_path = FREQNET_DIR / "4-classes-freqnet-v2.pth"
        
        if not self.ckpt_path.exists():
            print(f"FreqNet 가중치 파일 없음: {self.ckpt_path}")
            self.model = None
            return

        try:
            # FreqNet 소스코드에 .cuda()가 하드코딩되어 있어 CPU 환경에서 오류 발생 가능
            # 이를 해결하기 위해 torch.nn.Module.cuda를 더미 함수로 교체하거나 
            # 모델 로드 후 적절히 디바이스 이동
            
            self.model = FreqNet(block=Bottleneck, layers=[3, 4], num_classes=1)
            
            # 가중치 로드
            state_dict = torch.load(str(self.ckpt_path), map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            
            # 모델 전체를 타겟 디바이스로 이동 (하드코딩된 .cuda() 파라미터 덮어쓰기)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"FreqNet 모델 로드 실패: {e}")
            self.model = None

    @torch.no_grad()
    def detect(self, image: Image.Image) -> float:
        """
        이미지 1장에 대해 AI 생성 확률(0~1)을 반환합니다.
        """
        if self.model is None:
            return 0.0

        # 전처리
        img = image.convert("RGB")
        img_np = np.array(img)
        
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_res = cv2.resize(img_cv, (256, 256))
        
        img_res = img_res.astype(np.float32) / 255.0
        img_res = (img_res - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # float32 텐서로 명시적 변환 (RuntimeError 방지)
        img_tensor = torch.from_numpy(img_res).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
        
        # 추론
        try:
            output = self.model(img_tensor)
            fake_prob = torch.sigmoid(output).item()
            return float(fake_prob)
        except Exception as e:
            print(f"FreqNet 추론 오류: {e}")
            return 0.0

# 싱글톤
_detector = None

def detect_freqnet_score(image: Image.Image) -> float:
    global _detector
    if _detector is None:
        _detector = FreqNetDetector()
    return _detector.detect(image)

if __name__ == "__main__":
    test_img = Image.new('RGB', (256, 256), color = (128, 128, 128))
    score = detect_freqnet_score(test_img)
    print(f"FreqNet Score: {score}")
