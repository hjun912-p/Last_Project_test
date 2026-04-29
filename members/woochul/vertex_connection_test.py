"""
Vertex AI SynthID 연결 테스트
실제 이미지 없이 API 인증 및 모델 로드만 확인합니다.

실행:
    cd members/woochul
    python3 vertex_connection_test.py
"""

import sys
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

print("=" * 55)
print("Vertex AI SynthID 연결 테스트")
print("=" * 55)

# ── 1. SDK 임포트 확인 ──────────────────────────────────
print("\n[1] SDK 임포트 확인...")
try:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel
    from vertexai.preview.vision_models import Image as VertexImage
    print("    ✅ google-cloud-aiplatform 임포트 성공")
except ImportError as e:
    print(f"    ❌ 임포트 실패: {e}")
    print("       pip3 install google-cloud-aiplatform")
    sys.exit(1)

# ── 2. GCP 인증 및 프로젝트 초기화 ─────────────────────
print("\n[2] GCP 인증 확인 (project: insight-494801)...")
try:
    vertexai.init(project="insight-494801", location="us-central1")
    print("    ✅ vertexai.init() 성공")
except Exception as e:
    print(f"    ❌ 초기화 실패: {e}")
    print("       gcloud auth application-default login 을 먼저 실행하세요")
    sys.exit(1)

# ── 3. 모델 로드 확인 ───────────────────────────────────
print("\n[3] imagegeneration@006 모델 로드 확인...")
try:
    model = ImageGenerationModel.from_pretrained("imagegeneration@006")
    print("    ✅ 모델 로드 성공")
except Exception as e:
    print(f"    ❌ 모델 로드 실패: {e}")
    print("       Vertex AI API가 프로젝트에서 활성화되어 있는지 확인하세요")
    sys.exit(1)

# ── 4. 실제 탐지 호출 테스트 (테스트용 더미 이미지) ─────
print("\n[4] detect_watermark() 호출 테스트 (512x512 흰색 테스트 이미지)...")
try:
    # 순수 흰색 이미지로 테스트 — 워터마크 없음이 정상 결과
    dummy = Image.fromarray(np.ones((512, 512, 3), dtype=np.uint8) * 200)
    buf = BytesIO()
    dummy.save(buf, format="PNG")
    vertex_img = VertexImage(image_bytes=buf.getvalue())

    response = model.detect_watermark(vertex_img)

    # 응답 구조 출력 (처음 테스트 시 실제 포맷 확인용)
    conf_raw = response.watermark_detection_result.confidence
    print(f"    ✅ API 호출 성공")
    print(f"    응답 confidence 원본값  : {repr(conf_raw)}")
    print(f"    응답 confidence 타입    : {type(conf_raw).__name__}")

    if hasattr(conf_raw, "name"):
        print(f"    confidence.name (enum)  : {conf_raw.name}")
    else:
        print(f"    confidence str 변환     : {str(conf_raw)}")

except Exception as e:
    print(f"    ❌ API 호출 실패: {e}")
    sys.exit(1)

# ── 5. synthid_vertex 모듈 전체 실행 확인 ──────────────
print("\n[5] synthid_vertex.detect_synthid_vertex() 전체 실행...")
try:
    from synthid_vertex import detect_synthid_vertex
    result, msg, elapsed, cost = detect_synthid_vertex(dummy)
    print(f"    ✅ 정상 실행")
    print(f"    결과  : {result}")
    print(f"    메시지: {msg.splitlines()[0]}")
    print(f"    시간  : {elapsed:.2f}초")
    print(f"    비용  : ${cost:.4f}")
except Exception as e:
    print(f"    ❌ 오류: {e}")

print("\n" + "=" * 55)
print("모든 연결 테스트 완료 — batch_test.py 실행 가능")
print("=" * 55)
