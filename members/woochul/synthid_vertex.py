"""
Vertex AI 공식 SynthID 워터마크 탐지 모듈
프로젝트: insight-494801

기존 synthid_detector.py (역공학 근사)를 대체하거나 병행 사용
"""

import time
from io import BytesIO
from PIL import Image

PROJECT_ID = "insight-494801"
LOCATION   = "us-central1"

# Vertex AI SDK는 선택적 임포트 (미설치 환경 대비)
try:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel
    from vertexai.preview.vision_models import Image as VertexImage
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False

# 신뢰도 등급 → 내부 점수 매핑
_CONFIDENCE_SCORE = {
    "VERY_LIKELY":   0.95,
    "LIKELY":        0.75,
    "POSSIBLE":      0.50,
    "UNLIKELY":      0.25,
    "VERY_UNLIKELY": 0.05,
}

_CONFIDENCE_KR = {
    "VERY_LIKELY":   "매우 높음",
    "LIKELY":        "높음",
    "POSSIBLE":      "불확실",
    "UNLIKELY":      "낮음",
    "VERY_UNLIKELY": "매우 낮음",
}

# 이미지 1장당 예상 비용 (USD) — Vertex AI Imagen 기준
COST_PER_IMAGE_USD = 0.0002


def detect_synthid_vertex(image: Image.Image) -> tuple:
    """
    Vertex AI 공식 SynthID 이미지 워터마크 탐지

    Returns:
        (True | None, 결과 문자열, 소요시간(초), 예상비용(USD))
    """
    if not VERTEX_AVAILABLE:
        return None, "⚠️ google-cloud-aiplatform 미설치 — pip install google-cloud-aiplatform", 0.0, 0.0

    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)

        # PIL Image → bytes
        buf = BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # Vertex AI Image 객체 생성
        vertex_image = VertexImage(image_bytes=img_bytes)

        # 모델 로드 및 탐지
        model = ImageGenerationModel.from_pretrained("imagegeneration@006")

        t0 = time.time()
        response = model.detect_watermark(vertex_image)
        elapsed = round(time.time() - t0, 3)

        # 결과 파싱 — SDK 버전에 따라 enum 또는 string으로 반환되므로 둘 다 처리
        confidence_raw = response.watermark_detection_result.confidence
        # enum 객체면 .name 속성으로 문자열 추출, 아니면 str() 변환
        if hasattr(confidence_raw, "name"):
            confidence_str = confidence_raw.name
        else:
            confidence_str = str(confidence_raw)
        score = _CONFIDENCE_SCORE.get(confidence_str, 0.5)
        kr    = _CONFIDENCE_KR.get(confidence_str, confidence_str)

        detected = confidence_str in ("LIKELY", "VERY_LIKELY")

        if detected:
            msg = (
                f"❌ SynthID 워터마크 감지 (Vertex AI 공식)\n"
                f"   신뢰도: {confidence_str} ({kr})  |  점수: {score:.2f}\n"
                f"   처리 시간: {elapsed:.2f}초  |  예상 비용: ${COST_PER_IMAGE_USD:.4f}"
            )
            return True, msg, elapsed, COST_PER_IMAGE_USD

        elif confidence_str == "POSSIBLE":
            msg = (
                f"⚠️ SynthID 경계값 (불확실)\n"
                f"   신뢰도: {confidence_str} ({kr})  |  점수: {score:.2f}\n"
                f"   처리 시간: {elapsed:.2f}초  |  예상 비용: ${COST_PER_IMAGE_USD:.4f}"
            )
            return None, msg, elapsed, COST_PER_IMAGE_USD

        else:
            msg = (
                f"✅ SynthID 미감지\n"
                f"   신뢰도: {confidence_str} ({kr})  |  점수: {score:.2f}\n"
                f"   처리 시간: {elapsed:.2f}초  |  예상 비용: ${COST_PER_IMAGE_USD:.4f}"
            )
            return None, msg, elapsed, COST_PER_IMAGE_USD

    except Exception as e:
        return None, f"⚠️ Vertex AI SynthID 오류: {str(e)}", 0.0, 0.0
