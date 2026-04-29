import os
import re
import tempfile
import numpy as np
import gradio as gr
from PIL import Image
from PIL.ExifTags import TAGS
from io import BytesIO
from dotenv import load_dotenv

import requests
from google import genai
from google.genai import types
from synthid_detector import detect_synthid
from synthid_vertex import detect_synthid_vertex, VERTEX_AVAILABLE

load_dotenv()

# C2PA (Python 3.10+ 문법 사용으로 3.9에서 SyntaxError 발생 가능)
try:
    import c2pa
    C2PA_AVAILABLE = True
except (ImportError, SyntaxError):
    C2PA_AVAILABLE = False

# Instaloader (인스타그램 이미지 다운로드)
try:
    import instaloader
    INSTALOADER_AVAILABLE = True
except ImportError:
    INSTALOADER_AVAILABLE = False


# ────────────────────────────────────────────────
# 이미지 다운로드
# ────────────────────────────────────────────────

def extract_instagram_shortcode(url: str) -> str | None:
    m = re.search(r'/(?:p|reel|tv)/([A-Za-z0-9_-]+)', url)
    return m.group(1) if m else None

def download_instagram_image(url: str) -> Image.Image:
    if not INSTALOADER_AVAILABLE:
        raise RuntimeError("instaloader 미설치 — pip install instaloader")
    shortcode = extract_instagram_shortcode(url)
    if not shortcode:
        raise ValueError("URL에서 Instagram shortcode를 찾을 수 없습니다.")
    L = instaloader.Instaloader(download_pictures=False, quiet=True)
    # 저장된 세션 파일이 있으면 자동 로드 (instaloader --login 으로 생성)
    ig_user = os.getenv("INSTAGRAM_USERNAME", "")
    if ig_user:
        try:
            L.load_session_from_file(ig_user)
        except Exception:
            pass
    post = instaloader.Post.from_shortcode(L.context, shortcode)
    img_url = post.url
    resp = requests.get(img_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")

def download_image(url: str) -> Image.Image:
    if "instagram.com" in url:
        return download_instagram_image(url)
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


# ────────────────────────────────────────────────
# 1차 필터
# ────────────────────────────────────────────────

def check_c2pa(image_path: str) -> tuple[bool | None, str]:
    if not C2PA_AVAILABLE:
        return None, "⚠️ c2pa-python 미설치"
    try:
        reader = c2pa.Reader(image_path)
        manifest_json = reader.json()
        if manifest_json:
            text = str(manifest_json).lower()
            ai_keywords = ["generativeai", "dall", "midjourney", "stable diffusion",
                           "firefly", "imagen", "ai.generated", "trainedAlgorithmicMedia"]
            for kw in ai_keywords:
                if kw.lower() in text:
                    return True, f"❌ C2PA: AI 생성 서명 감지 ({kw})"
            return False, "✅ C2PA: 원본 콘텐츠 서명 확인"
    except Exception:
        pass
    return None, "🔍 C2PA: 메타데이터 없음"

def check_exif(image: Image.Image) -> tuple[bool | None, str]:
    ai_tools = ["dall-e", "midjourney", "stable diffusion", "adobe firefly",
                "imagen", "generative", "ai generated", "synthid"]
    try:
        exif_raw = image._getexif()
        if not exif_raw:
            return None, "🔍 EXIF: 메타데이터 없음 (AI 이미지 가능성)"
        exif = {TAGS.get(k, k): str(v) for k, v in exif_raw.items()}
        combined = " ".join(exif.values()).lower()
        for tool in ai_tools:
            if tool in combined:
                return True, f"❌ EXIF: AI 도구 흔적 발견 ({tool})"
        camera_make = exif.get("Make", "")
        camera_model = exif.get("Model", "")
        if camera_make or camera_model:
            return False, f"✅ EXIF: 카메라 정보 존재 ({camera_make} {camera_model})"
        return None, "🔍 EXIF: 카메라 정보 없음"
    except Exception:
        return None, "🔍 EXIF: 분석 불가"

def check_synthid(image: Image.Image) -> tuple[bool | None, str]:
    # Vertex AI 공식 SynthID 우선 시도, 실패 시 역공학 근사로 폴백
    if VERTEX_AVAILABLE:
        result, msg, _, _ = detect_synthid_vertex(image)
        if "오류" not in msg:
            return result, msg
    return detect_synthid(image)

def run_filter_1(image_path: str, image: Image.Image) -> tuple[bool, str]:
    lines = ["[1차 필터 결과]", ""]
    detected = False

    r, msg = check_c2pa(image_path)
    lines.append(msg)
    if r is True:
        detected = True

    r, msg = check_exif(image)
    lines.append(msg)
    if r is True:
        detected = True

    r, msg = check_synthid(image)
    lines.append(msg)
    if r is True:
        detected = True

    lines.append("")
    lines.append("→ AI 생성 감지됨" if detected else "→ 감지되지 않음 (2차 필터로 이동)")
    return detected, "\n".join(lines)


# ────────────────────────────────────────────────
# 2차 필터 — Gemini 2.5 Flash
# ────────────────────────────────────────────────

GEMINI_PROMPT = """이 이미지가 AI가 생성한 이미지인지 분석해주세요.

다음 항목을 검토하세요:
1. 피부·텍스처의 과도한 매끄러움 또는 부자연스러운 균일함
2. 손가락, 귀, 치아 등 세부 부위의 형태 이상
3. 배경의 비논리적 구조나 반복 패턴
4. 조명·그림자의 물리적 불일치
5. 눈동자 반사 또는 홍채의 비현실성
6. 텍스트·문자의 왜곡 또는 의미 없는 글자

반드시 아래 형식으로만 답하세요:
판정: AI 생성 / 실제 이미지 / 불확실
신뢰도: 0~100%
근거: (2~3문장 한국어로)"""

AVAILABLE_MODELS = ["gemini-2.5-flash", "gemini-3.1-flash"]

def run_filter_2(image: Image.Image, api_key: str, model: str = "gemini-2.5-flash") -> tuple[bool | None, str]:
    if not api_key.strip():
        return None, "❌ Gemini API Key를 입력해주세요."
    try:
        client = genai.Client(api_key=api_key.strip())
        buf = BytesIO()
        image.save(buf, format="JPEG")
        img_bytes = buf.getvalue()
        response = client.models.generate_content(
            model=model,
            contents=[
                GEMINI_PROMPT,
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            ],
        )
        text = response.text.strip()
        is_ai = "AI 생성" in text and "실제 이미지" not in text.split("판정:")[1][:20]
        return is_ai, f"[2차 필터 — {model}]\n\n{text}"
    except Exception as e:
        return None, f"[2차 필터 — {model}]\n\n❌ 오류: {str(e)}"


# ────────────────────────────────────────────────
# 메인 처리
# ────────────────────────────────────────────────

def process(url: str, uploaded_file, api_key: str, model: str = "gemini-2.5-flash"):
    image: Image.Image | None = None
    image_path: str | None = None
    tmp_path: str | None = None

    try:
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            image_path = uploaded_file
        elif url.strip():
            image = download_image(url.strip())
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            image.save(tmp.name, "JPEG")
            tmp.close()
            image_path = tmp.name
            tmp_path = tmp.name
        else:
            return "URL 또는 파일을 입력해주세요.", "", "", None

        # 1차 필터
        detected_1, report_1 = run_filter_1(image_path, image)

        if detected_1:
            verdict = "❌ AI 생성 이미지 확인 (1차 필터)"
            report_2 = "1차 필터에서 감지되어 2차 필터를 생략했습니다."
        else:
            is_ai_2, report_2 = run_filter_2(image, api_key, model)
            if is_ai_2 is True:
                verdict = "❌ AI 생성 이미지 확인 (2차 필터)"
            elif is_ai_2 is False:
                verdict = "✅ 실제 이미지로 판단됨"
            else:
                verdict = "❓ 판별 불확실 — 추가 검토 필요"

        return verdict, report_1, report_2, image

    except Exception as e:
        return f"❌ 오류: {str(e)}", "", "", None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ────────────────────────────────────────────────
# Gradio UI
# ────────────────────────────────────────────────

with gr.Blocks(title="InSIGHT", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 InSIGHT — AI 생성 이미지 탐지기")
    gr.Markdown("Instagram URL 또는 이미지 파일을 입력하면 1차(C2PA/EXIF/SynthID) → 2차(Gemini Flash) 순으로 분석합니다.")

    with gr.Row():
        with gr.Column(scale=1):
            url_input = gr.Textbox(
                label="Instagram URL 또는 이미지 직접 URL",
                placeholder="https://www.instagram.com/p/..."
            )
            file_input = gr.File(
                label="또는 이미지 파일 직접 업로드",
                file_types=["image"]
            )
            api_key_input = gr.Textbox(
                label="Gemini API Key (2차 필터용)",
                type="password",
                placeholder="AIza...",
                value=os.getenv("GEMINI_API_KEY", "")
            )
            model_radio = gr.Radio(
                choices=AVAILABLE_MODELS,
                value=AVAILABLE_MODELS[0],
                label="2차 필터 모델 선택",
            )
            submit_btn = gr.Button("분석 시작", variant="primary", size="lg")

        with gr.Column(scale=1):
            image_preview = gr.Image(label="분석 대상 이미지", type="pil")

    verdict_output = gr.Textbox(label="최종 판정", interactive=False, lines=2)

    with gr.Row():
        filter1_output = gr.Textbox(
            label="1차 필터 (C2PA / EXIF / SynthID 워터마크 탐지)",
            interactive=False, lines=10
        )
        filter2_output = gr.Textbox(
            label="2차 필터 (Gemini Flash)",
            interactive=False, lines=10
        )

    submit_btn.click(
        fn=process,
        inputs=[url_input, file_input, api_key_input, model_radio],
        outputs=[verdict_output, filter1_output, filter2_output, image_preview]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)
