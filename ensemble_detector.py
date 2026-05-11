"""
InSIGHT — Ensemble Detector
haywoodsloan/ai-image-detector-deploy (w=0.6)
+ Organika/sdxl-detector              (w=0.4)
소프트 보팅 (가중 평균)

사용 예:
    from ensemble_detector import EnsembleDetector
    detector = EnsembleDetector()
    score, pred, elapsed = detector.predict_path("image.jpg")
    # score: AI 확률 0~1 / pred: 1=AI, 0=Real / elapsed: 추론 시간(초)
"""

import time
from pathlib import Path

import torch
from PIL import Image
from transformers import pipeline

# ── 모델 설정 ──────────────────────────────────────────────────
DEPLOY_MODEL = "haywoodsloan/ai-image-detector-deploy"
SDXL_MODEL   = "Organika/sdxl-detector"

DEFAULT_WEIGHTS = {
    DEPLOY_MODEL: 0.6,   # F1 81.1% 기준 가중치
    SDXL_MODEL:   0.4,   # F1 63.0% 기준 가중치
}

DEFAULT_THRESHOLD = 0.5
DEVICE = 0 if torch.cuda.is_available() else -1

# 모델마다 AI 레이블 이름이 달라서 키워드로 감지
_AI_KEYWORDS = ["ai", "artificial", "fake", "generated", "synthetic", "deepfake"]


# ── 내부 유틸 ──────────────────────────────────────────────────

def _parse_ai_score(results: list[dict]) -> float:
    """pipeline 출력에서 AI 확률 추출. 키워드 미매칭 시 첫 번째 레이블 사용."""
    for r in results:
        if any(kw in r["label"].lower() for kw in _AI_KEYWORDS):
            return float(r["score"])
    return float(results[0]["score"])


# ── 앙상블 클래스 ───────────────────────────────────────────────

class EnsembleDetector:
    """
    두 ViT 모델의 AI 확률을 가중 평균해 최종 판정.

    Parameters
    ----------
    weights   : {model_id: float} — 기본값 deploy=0.6 / sdxl=0.4
    threshold : AI 판정 기준 (기본 0.5)
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.weights   = weights or DEFAULT_WEIGHTS.copy()
        self.threshold = threshold
        self._total    = sum(self.weights.values())
        self._pipes: dict[str, object] = {}
        self._load_models()

    # ── 모델 로드 ───────────────────────────────────────────────

    def _load_models(self) -> None:
        print("=" * 55)
        print("  EnsembleDetector — 모델 로드")
        print("=" * 55)
        for model_id in self.weights:
            print(f"  Loading : {model_id}")
            try:
                self._pipes[model_id] = pipeline(
                    "image-classification",
                    model=model_id,
                    device=DEVICE,
                )
                print(f"  ✅ OK    : {model_id.split('/')[-1]}")
            except Exception as exc:
                raise RuntimeError(f"모델 로드 실패 [{model_id}]: {exc}") from exc
        print(f"  Device  : {'CUDA' if DEVICE == 0 else 'CPU'}")
        print(f"  Weights : { {k.split('/')[-1]: v for k, v in self.weights.items()} }")
        print(f"  Threshold: {self.threshold}")
        print("=" * 55)

    # ── 추론 ────────────────────────────────────────────────────

    def predict(self, image: Image.Image) -> tuple[float, int]:
        """
        PIL Image 한 장 추론.
        반환: (ai_score 0~1, pred_label 0|1)
        """
        weighted_sum = 0.0
        for model_id, pipe in self._pipes.items():
            ai_score      = _parse_ai_score(pipe(image))
            weight        = self.weights[model_id] / self._total
            weighted_sum += ai_score * weight
        pred = 1 if weighted_sum >= self.threshold else 0
        return round(weighted_sum, 4), pred

    def predict_path(self, image_path: str | Path) -> tuple[float, int, float]:
        """
        파일 경로로 추론 (추론 시간 포함).
        반환: (ai_score, pred_label, elapsed_sec)
        """
        img = Image.open(image_path).convert("RGB")
        t0  = time.time()
        score, pred = self.predict(img)
        return score, pred, round(time.time() - t0, 3)

    def predict_scores(self, image: Image.Image) -> dict[str, float]:
        """
        각 모델의 개별 AI 확률과 앙상블 점수를 모두 반환.
        디버깅·분석용.
        """
        scores: dict[str, float] = {}
        weighted_sum = 0.0
        for model_id, pipe in self._pipes.items():
            ai_score      = _parse_ai_score(pipe(image))
            weight        = self.weights[model_id] / self._total
            weighted_sum += ai_score * weight
            scores[model_id.split("/")[-1]] = round(ai_score, 4)
        scores["ensemble"] = round(weighted_sum, 4)
        return scores
