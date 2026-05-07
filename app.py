"""
InSIGHT — AI 생성 이미지/영상 판별기 (로컬 데모 버전)
Instagram / YouTube 링크 또는 이미지 파일 → 3단계 분석 → 증거 기반 판정

실행:
    conda activate insight
    python app.py

환경변수: .env 파일에 GEMINI_API_KEY, GOOGLE_APPLICATION_CREDENTIALS 설정
"""

import os
import re
import sys
import time
import json
import tempfile
from io import BytesIO
from pathlib import Path

import numpy as np
import gradio as gr
from PIL import Image
from PIL.ExifTags import TAGS
from dotenv import load_dotenv

load_dotenv()

# ── 선택적 임포트 ──────────────────────────────────────────

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import yt_dlp
    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False

try:
    import instaloader
    INSTALOADER_AVAILABLE = True
except ImportError:
    INSTALOADER_AVAILABLE = False

try:
    import c2pa
    C2PA_AVAILABLE = True
except (ImportError, SyntaxError):
    C2PA_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part as VertexPart
    VERTEXAI_GENAI_AVAILABLE = True
except ImportError:
    VERTEXAI_GENAI_AVAILABLE = False

# ── 로컬 모듈 ─────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from synthid_detector import detect_synthid
from synthid_vertex import detect_synthid_vertex, VERTEX_AVAILABLE

# ── 설정 ──────────────────────────────────────────────────

PROJECT_ID = "insight-494801"
LOCATION   = "us-central1"

STAGE1_THRESHOLD = 0.80

SYNTHID_SCORE_MAP = {
    "VERY_LIKELY":   0.95,
    "LIKELY":        0.75,
    "POSSIBLE":      0.50,
    "UNLIKELY":      0.25,
    "VERY_UNLIKELY": 0.05,
}

MODEL_CONFIGS: dict[str, dict] = {
    "gemini_flash": {
        "display":        "Gemini 2.5 Flash",
        "model_id":       "gemini-2.5-flash",
        "api":            "google_genai",
        "cost_per_image": 0.0001,
    },
    "gemma4": {
        "display":        "Gemma 4",
        "model_id":       "gemma-3-27b-it",
        "api":            "vertexai",
        "cost_per_image": 0.0003,
    },
    "vertexai": {
        "display":        "VertexAI Gemini",
        "model_id":       "gemini-2.5-flash",
        "api":            "vertexai",
        "cost_per_image": 0.001,
    },
}


# ═══════════════════════════════════════════════════════════
# 다운로드
# ═══════════════════════════════════════════════════════════

def _extract_instagram_shortcode(url: str) -> str | None:
    m = re.search(r'/(?:p|reel|tv)/([A-Za-z0-9_-]+)', url)
    return m.group(1) if m else None

def _download_instagram(url: str) -> tuple[Image.Image, str]:
    if not INSTALOADER_AVAILABLE:
        raise RuntimeError("instaloader 미설치 — pip install instaloader")
    shortcode = _extract_instagram_shortcode(url)
    if not shortcode:
        raise ValueError("Instagram URL에서 shortcode를 추출하지 못했습니다.")
    L = instaloader.Instaloader(download_pictures=False, quiet=True)
    ig_user = os.getenv("INSTAGRAM_USERNAME", "")
    if ig_user:
        try:
            L.load_session_from_file(ig_user)
        except Exception:
            pass
    post = instaloader.Post.from_shortcode(L.context, shortcode)
    resp = requests.get(post.url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, "JPEG")
    tmp.close()
    return img, tmp.name

def _download_youtube_frame(url: str) -> tuple[Image.Image, str]:
    if not YTDLP_AVAILABLE:
        raise RuntimeError("yt-dlp 미설치 — pip install yt-dlp")
    tmp_dir = tempfile.mkdtemp(prefix="insight_yt_")
    ydl_opts = {
        "format":         "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "outtmpl":        os.path.join(tmp_dir, "%(id)s.%(ext)s"),
        "writethumbnail": True,
        "quiet":          True,
        "no_warnings":    True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    thumbs = [f for f in Path(tmp_dir).iterdir()
              if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if thumbs:
        img = Image.open(str(thumbs[0])).convert("RGB")
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(tmp.name, "JPEG")
        tmp.close()
        return img, tmp.name

    if CV2_AVAILABLE:
        videos = [f for f in Path(tmp_dir).iterdir()
                  if f.suffix.lower() in {".mp4", ".webm", ".mkv"}]
        if videos:
            cap = cv2.VideoCapture(str(videos[0]))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
            ret, frame = cap.read()
            cap.release()
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                img.save(tmp.name, "JPEG")
                tmp.close()
                return img, tmp.name

    raise RuntimeError("YouTube에서 이미지를 추출하지 못했습니다.")

def download_media(url: str) -> tuple[Image.Image, str, str]:
    url = url.strip()
    if "instagram.com" in url or "instagr.am" in url:
        img, path = _download_instagram(url)
        return img, path, "instagram"
    if "youtube.com" in url or "youtu.be" in url:
        img, path = _download_youtube_frame(url)
        return img, path, "youtube"
    if url.startswith("http"):
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(tmp.name, "JPEG")
        tmp.close()
        return img, tmp.name, "direct"
    raise ValueError("지원하지 않는 URL입니다. Instagram / YouTube URL을 입력하세요.")


# ═══════════════════════════════════════════════════════════
# Stage 1: 메타데이터 + 비가시성 워터마크
# ═══════════════════════════════════════════════════════════

_AI_TOOL_NAMES = [
    "dall-e", "midjourney", "stable diffusion", "adobe firefly",
    "imagen", "generative", "ai generated", "synthid",
    "runway", "pika", "kling", "wan", "comfyui", "novelai",
    "sora", "hailuo", "veo", "flux",
]

def _s1_exif(image: Image.Image) -> tuple[float, list[str]]:
    indicators: list[str] = []
    score = 0.0
    try:
        raw = image._getexif()
        if not raw:
            indicators.append("EXIF 없음 — AI 생성 이미지에서 흔히 관찰됨")
            return 0.25, indicators
        exif = {TAGS.get(k, k): str(v) for k, v in raw.items()}
        combined = " ".join(exif.values()).lower()
        for tool in _AI_TOOL_NAMES:
            if tool in combined:
                indicators.append(f"AI 도구 서명 발견: {tool}")
                score = max(score, 0.85)
        make  = exif.get("Make", "")
        model = exif.get("Model", "")
        if make or model:
            indicators.append(f"카메라 정보 존재: {make} {model}".strip())
            score = max(score, 0.05)
        else:
            indicators.append("카메라 Make/Model 없음")
            score = max(score, 0.20)
    except Exception as e:
        indicators.append(f"EXIF 파싱 오류: {e}")
    return min(score, 1.0), indicators

def _s1_c2pa(image_path: str) -> tuple[float, list[str]]:
    indicators: list[str] = []
    if not C2PA_AVAILABLE:
        indicators.append("c2pa-python 미설치 (건너뜀)")
        return 0.0, indicators
    try:
        reader = c2pa.Reader(image_path)
        manifest = reader.json()
        if manifest:
            text = str(manifest).lower()
            for kw in ["generativeai", "dall", "midjourney", "stable diffusion",
                       "firefly", "imagen", "ai.generated"]:
                if kw in text:
                    indicators.append(f"C2PA AI 서명: {kw}")
                    return 0.90, indicators
            indicators.append("C2PA 있음, AI 서명 미발견 (원본 가능성)")
            return 0.05, indicators
    except Exception:
        pass
    indicators.append("C2PA 메타데이터 없음")
    return 0.0, indicators

def _s1_watermark(image: Image.Image) -> tuple[float, list[str]]:
    indicators: list[str] = []
    detected, msg = detect_synthid(image)
    first = (msg or "").split("\n")[0]
    if detected is True:
        indicators.append(f"역공학 SynthID: 워터마크 감지 — {first}")
        return 0.75, indicators
    m = re.search(r'CVR[:\s=]+([0-9.]+)', msg or "")
    if m:
        cvr = float(m.group(1))
        indicators.append(f"역공학 SynthID: CVR={cvr:.3f} ({'워터마크 의심' if cvr > 0.7 else '미감지'})")
        return round(cvr * 0.4, 4), indicators
    indicators.append(f"역공학 SynthID: 미감지 — {first}")
    return 0.0, indicators

def run_stage1(image: Image.Image, image_path: str) -> dict:
    t0 = time.time()
    exif_score, exif_ind = _s1_exif(image)
    c2pa_score, c2pa_ind = _s1_c2pa(image_path)
    wm_score,   wm_ind   = _s1_watermark(image)

    score = exif_score * 0.40 + c2pa_score * 0.35 + wm_score * 0.25
    score = round(min(score, 1.0), 4)

    return {
        "score":        score,
        "verdict":      "AI 생성 확인" if score >= STAGE1_THRESHOLD else "불확실",
        "pass_to_next": score < STAGE1_THRESHOLD,
        "elapsed":      round(time.time() - t0, 3),
        "cost_usd":     0.0,
        "checks": {
            "exif":      {"score": exif_score, "indicators": exif_ind},
            "c2pa":      {"score": c2pa_score, "indicators": c2pa_ind},
            "watermark": {"score": wm_score,   "indicators": wm_ind},
        },
    }


# ═══════════════════════════════════════════════════════════
# Stage 2: Vertex AI SynthID Detector
# ═══════════════════════════════════════════════════════════

def run_stage2(image: Image.Image) -> dict:
    t0 = time.time()
    detected, msg, elapsed, cost = detect_synthid_vertex(image)

    if not VERTEX_AVAILABLE or "오류" in msg or "미설치" in msg or "스킵" in msg:
        return {
            "score":            0.0,
            "verdict":          "스킵 (Vertex AI 미연결)",
            "pass_to_next":     True,
            "elapsed":          round(time.time() - t0, 3),
            "cost_usd":         0.0,
            "confidence_level": "N/A",
            "raw_msg":          msg,
            "available":        False,
        }

    conf_level = "VERY_UNLIKELY"
    for level in SYNTHID_SCORE_MAP:
        if level in msg:
            conf_level = level
            break
    score = SYNTHID_SCORE_MAP[conf_level]
    verdict = "AI 생성 (SynthID)" if detected is True else (
              "불확실" if conf_level == "POSSIBLE" else "미감지")

    return {
        "score":            round(score, 4),
        "verdict":          verdict,
        "pass_to_next":     True,
        "elapsed":          elapsed,
        "cost_usd":         cost,
        "confidence_level": conf_level,
        "raw_msg":          msg,
        "available":        True,
    }


# ═══════════════════════════════════════════════════════════
# Stage 3: Gemini / Gemma / VertexAI 시각 분석
# ═══════════════════════════════════════════════════════════

_STAGE3_PROMPT = """당신은 AI 생성 이미지/영상 탐지 전문가입니다.
제공된 이미지를 분석하여 AI가 생성한 것인지 실제 촬영된 것인지 판단하세요.

반드시 아래 JSON 형식으로만 응답하세요 (다른 텍스트 없이):

{
  "verdict": "AI_GENERATED" | "REAL" | "UNCERTAIN",
  "confidence_pct": 0~100,
  "evidence": [
    {"item": "눈으로 직접 확인한 시각적 사실", "weight": "high|medium|low"}
  ],
  "inference": [
    {"item": "관찰 사실에서 도출한 해석", "basis": "그 해석의 근거"}
  ],
  "detail": {
    "texture":    "텍스처·피부 관찰",
    "lighting":   "조명·그림자 분석",
    "anatomy":    "손·귀·치아 등 해부학적 이상 여부",
    "background": "배경 일관성",
    "artifacts":  "AI 특유 아티팩트"
  }
}"""

def _call_google_genai(image: Image.Image, model_id: str, api_key: str) -> dict:
    if not GENAI_AVAILABLE:
        return {"error": "google-genai 미설치"}
    if not api_key.strip():
        return {"error": "GEMINI_API_KEY 없음 — .env 파일에 설정하거나 UI에 직접 입력하세요"}
    buf = BytesIO()
    image.save(buf, format="JPEG")
    try:
        client = genai.Client(api_key=api_key.strip())
        resp = client.models.generate_content(
            model=model_id,
            contents=[
                _STAGE3_PROMPT,
                genai_types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
            ],
        )
        return _parse_json(resp.text)
    except Exception as e:
        return {"error": str(e)}

def _call_vertexai(image: Image.Image, model_id: str) -> dict:
    if not VERTEXAI_GENAI_AVAILABLE:
        return {"error": "google-cloud-aiplatform 미설치"}
    buf = BytesIO()
    image.save(buf, format="JPEG")
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(model_id)
        resp = model.generate_content([
            VertexPart.from_text(_STAGE3_PROMPT),
            VertexPart.from_data(data=buf.getvalue(), mime_type="image/jpeg"),
        ])
        return _parse_json(resp.text)
    except Exception as e:
        return {"error": str(e)}

def _parse_json(text: str) -> dict:
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"raw_response": text, "parse_error": True}

def run_stage3(image: Image.Image, model_key: str, api_key: str) -> dict:
    cfg = MODEL_CONFIGS[model_key]
    t0  = time.time()

    if cfg["api"] == "google_genai":
        analysis = _call_google_genai(image, cfg["model_id"], api_key)
    else:
        analysis = _call_vertexai(image, cfg["model_id"])

    elapsed = round(time.time() - t0, 3)

    if "error" in analysis:
        return {
            "score":    0.0,
            "verdict":  f"오류: {analysis['error']}",
            "elapsed":  elapsed,
            "cost_usd": 0.0,
            "analysis": analysis,
            "model":    cfg["display"],
        }

    raw_verdict = analysis.get("verdict", "UNCERTAIN")
    conf        = analysis.get("confidence_pct", 50) / 100.0
    ai_score    = conf if raw_verdict == "AI_GENERATED" else (
                  1.0 - conf if raw_verdict == "REAL" else 0.5)

    verdict_map = {"AI_GENERATED": "AI 생성", "REAL": "실제 이미지", "UNCERTAIN": "불확실"}

    return {
        "score":    round(ai_score, 4),
        "verdict":  verdict_map.get(raw_verdict, "불확실"),
        "elapsed":  elapsed,
        "cost_usd": cfg["cost_per_image"],
        "analysis": analysis,
        "model":    cfg["display"],
    }


# ═══════════════════════════════════════════════════════════
# 결과 포매터
# ═══════════════════════════════════════════════════════════

def _bar(score: float, w: int = 20) -> str:
    n = int(round(score * w))
    return f"[{'█' * n}{'░' * (w - n)}] {score * 100:5.1f}%"

def fmt_stage1(r: dict) -> str:
    out = ["━━━  Stage 1 — 메타데이터 / 비가시성 워터마크  ━━━", ""]
    out.append(f"  종합 점수  {_bar(r['score'])}  →  {r['verdict']}")
    out.append(f"  소요 시간  {r['elapsed']}s  |  비용  $0.0000  (로컬 처리)")
    out.append("")
    label = {"exif": "EXIF 분석", "c2pa": "C2PA 출처", "watermark": "SynthID 역공학"}
    for key, data in r["checks"].items():
        out.append(f"  {label[key]:<14} {_bar(data['score'], 14)}")
        for ind in data["indicators"]:
            out.append(f"    • {ind}")
    out.append("")
    if r["pass_to_next"]:
        out.append(f"  → 점수 {r['score']*100:.1f}% < {STAGE1_THRESHOLD*100:.0f}%  :  Stage 2 로 진행")
    else:
        out.append(f"  → 점수 {r['score']*100:.1f}% ≥ {STAGE1_THRESHOLD*100:.0f}%  :  AI 생성 확인")
    return "\n".join(out)

def fmt_stage2(r: dict) -> str:
    out = ["━━━  Stage 2 — Vertex AI SynthID Detector  ━━━", ""]
    if not r.get("available"):
        out.append(f"  ⚠️  {r.get('verdict', 'Vertex AI 미연결')}")
        out.append(f"  {r.get('raw_msg', '')}")
        out.append("")
        out.append("  💡 GOOGLE_APPLICATION_CREDENTIALS 를 .env 에 설정하면 활성화됩니다.")
        return "\n".join(out)
    out.append(f"  종합 점수  {_bar(r['score'])}  →  {r['verdict']}")
    out.append(f"  신뢰 등급  {r['confidence_level']}")
    out.append(f"  소요 시간  {r['elapsed']}s  |  예상 비용  ${r['cost_usd']:.4f}")
    out.append("")
    for line in r.get("raw_msg", "").splitlines():
        out.append(f"  {line}")
    return "\n".join(out)

def fmt_stage3(r: dict) -> str:
    out = [f"━━━  Stage 3 — {r['model']} 시각 분석  ━━━", ""]
    if "오류" in r["verdict"]:
        out.append(f"  {r['verdict']}")
        return "\n".join(out)
    analysis = r.get("analysis", {})
    if analysis.get("parse_error"):
        out.append(analysis.get("raw_response", ""))
        return "\n".join(out)

    out.append(f"  판정       {r['verdict']}  (신뢰도 {analysis.get('confidence_pct', 0)}%)")
    out.append(f"  소요 시간  {r['elapsed']}s  |  예상 비용  ${r['cost_usd']:.4f}")
    out.append("")

    evidence = analysis.get("evidence", [])
    if evidence:
        out.append("  ┌ 증거 (직접 관찰) ─────────────────────────")
        for e in evidence:
            icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                e.get("weight", "") if isinstance(e, dict) else "", "•")
            text = e.get("item", str(e)) if isinstance(e, dict) else str(e)
            out.append(f"  │  {icon} {text}")
        out.append("  └────────────────────────────────────────────")
        out.append("")

    inference = analysis.get("inference", [])
    if inference:
        out.append("  ┌ 추론 (증거 기반 해석) ─────────────────────")
        for i in inference:
            if isinstance(i, dict):
                out.append(f"  │  → {i.get('item', '')}")
                if i.get("basis"):
                    out.append(f"  │      근거: {i['basis']}")
            else:
                out.append(f"  │  → {i}")
        out.append("  └────────────────────────────────────────────")
        out.append("")

    detail = analysis.get("detail", {})
    if detail:
        out.append("  ┌ 세부 분석 ─────────────────────────────────")
        for key, lbl in [("texture","텍스처/피부"), ("lighting","조명/그림자"),
                         ("anatomy","해부학적  "), ("background","배경      "),
                         ("artifacts","AI 아티팩트")]:
            val = detail.get(key, "")
            if val:
                out.append(f"  │  {lbl}  {val}")
        out.append("  └────────────────────────────────────────────")

    return "\n".join(out)

def fmt_summary(s1, s2, s3) -> str:
    out = ["━━━━━━━━━━━━━━━  종합 판정  ━━━━━━━━━━━━━━━", ""]
    scores: list[float] = []
    total_cost = 0.0

    if s1:
        scores.append(s1["score"])
        out.append(f"  Stage 1  {_bar(s1['score'], 14)}  {s1['verdict']}")
    if s2 and s2.get("available"):
        scores.append(s2["score"])
        total_cost += s2.get("cost_usd", 0)
        out.append(f"  Stage 2  {_bar(s2['score'], 14)}  {s2['verdict']}")
    if s3 and "오류" not in s3["verdict"]:
        scores.append(s3["score"])
        total_cost += s3.get("cost_usd", 0)
        out.append(f"  Stage 3  {_bar(s3['score'], 14)}  {s3['verdict']}  [{s3['model']}]")

    out.append("")
    if scores:
        avg = sum(scores) / len(scores)
        out.append(f"  평균 점수  {_bar(avg)}")
        if avg >= 0.75:
            final = "❌  AI 생성 콘텐츠로 판단됨"
        elif avg >= 0.45:
            final = "⚠️   판별 불확실 — 추가 검토 필요"
        else:
            final = "✅  실제 콘텐츠로 판단됨"
        out.append(f"  최종 판정  {final}")
    else:
        out.append("  실행된 단계 없음")

    out.append("")
    out.append(f"  총 예상 비용  ${total_cost:.4f} USD")
    return "\n".join(out)


# ═══════════════════════════════════════════════════════════
# 메인 처리
# ═══════════════════════════════════════════════════════════

def process(url, uploaded_file, api_key, active_model, use_s1, use_s2, use_s3):
    image      = None
    image_path = None
    tmp_path   = None
    s1 = s2 = s3 = None

    try:
        if uploaded_file:
            image      = Image.open(uploaded_file).convert("RGB")
            image_path = uploaded_file
        elif url.strip():
            image, image_path, _ = download_media(url.strip())
            tmp_path = image_path
        else:
            return "URL 또는 파일을 입력하세요.", "", "", "", "", None

        s1_text = s2_text = s3_text = ""

        if use_s1:
            s1      = run_stage1(image, image_path)
            s1_text = fmt_stage1(s1)
        else:
            s1_text = "(Stage 1 비활성화)"

        if use_s2:
            s2      = run_stage2(image)
            s2_text = fmt_stage2(s2)
        else:
            s2_text = "(Stage 2 비활성화)"

        if use_s3:
            s3      = run_stage3(image, active_model, api_key)
            s3_text = fmt_stage3(s3)
        else:
            s3_text = "(Stage 3 비활성화)"

        summary = fmt_summary(s1, s2, s3)
        return summary, s1_text, s2_text, s3_text, summary, image

    except Exception as e:
        err = f"❌ 오류: {e}"
        return err, err, "", "", err, None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════
# Gradio UI
# ═══════════════════════════════════════════════════════════

CSS = """
.mono textarea, .mono .wrap {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 13px !important;
    background: #1e1e2e !important;
    color: #cdd6f4 !important;
}
"""

with gr.Blocks(title="InSIGHT", theme=gr.themes.Soft(), css=CSS) as demo:

    active_model_state = gr.State("gemini_flash")

    gr.Markdown(
        "# 🔍 InSIGHT — AI 생성 이미지/영상 탐지기\n"
        "Instagram · YouTube URL 또는 이미지 파일 → 3단계 분석 → 증거 기반 판정"
    )

    with gr.Group():
        gr.Markdown("#### Stage 3 분석 모델 선택")
        with gr.Row(equal_height=True):
            btn_gemini = gr.Button("▶ Gemini 2.5 Flash", variant="primary",   min_width=150)
            btn_gemma  = gr.Button("   Gemma 4",          variant="secondary", min_width=150)
            btn_vertex = gr.Button("   VertexAI Gemini",  variant="secondary", min_width=150)
        model_label = gr.Markdown("*현재 모델: **Gemini 2.5 Flash***")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=3):
            url_input = gr.Textbox(
                label="Instagram / YouTube URL 또는 이미지 직접 URL",
                placeholder="https://www.instagram.com/p/...   |   https://youtu.be/...",
            )
            file_input = gr.File(
                label="또는 이미지 파일 직접 업로드",
                file_types=["image"],
            )
            api_key_input = gr.Textbox(
                label="Gemini API Key  (Stage 3 Gemini 모델 사용 시 필요 / .env에 설정하면 자동 입력)",
                type="password",
                placeholder="AIza...",
                value=os.getenv("GEMINI_API_KEY", ""),
            )

            gr.Markdown("#### 분석 단계 선택")
            with gr.Row():
                chk1 = gr.Checkbox(value=True,  label="Stage 1  메타데이터 + SynthID 역공학  (무료, 로컬)")
                chk2 = gr.Checkbox(value=True,  label="Stage 2  Vertex AI SynthID  (GCP 연동)")
                chk3 = gr.Checkbox(value=True,  label="Stage 3  AI 시각 분석  (Gemini / VertexAI)")

            analyze_btn = gr.Button("🔍  분석 시작", variant="primary", size="lg")

        with gr.Column(scale=2):
            image_preview = gr.Image(label="분석 대상 이미지", type="pil", height=320)

    summary_box = gr.Textbox(label="종합 판정", interactive=False, lines=9, elem_classes=["mono"])

    with gr.Tabs():
        with gr.Tab("Stage 1 — 메타데이터/워터마크"):
            s1_out = gr.Textbox(interactive=False, lines=20, elem_classes=["mono"])
        with gr.Tab("Stage 2 — SynthID (Vertex AI)"):
            s2_out = gr.Textbox(interactive=False, lines=20, elem_classes=["mono"])
        with gr.Tab("Stage 3 — AI 시각 분석"):
            s3_out = gr.Textbox(interactive=False, lines=20, elem_classes=["mono"])
        with gr.Tab("종합 (전체)"):
            summary_tab = gr.Textbox(interactive=False, lines=20, elem_classes=["mono"])

    def _set_model(key: str):
        labels   = {"gemini_flash": "Gemini 2.5 Flash", "gemma4": "Gemma 4", "vertexai": "VertexAI Gemini"}
        md_texts = {k: f"*현재 모델: **{v}***" for k, v in labels.items()}
        return (
            key,
            md_texts[key],
            gr.update(value=f"{'▶' if key=='gemini_flash' else '  '} {labels['gemini_flash']}",
                      variant="primary" if key=="gemini_flash" else "secondary"),
            gr.update(value=f"{'▶' if key=='gemma4'       else '  '} {labels['gemma4']}",
                      variant="primary" if key=="gemma4"       else "secondary"),
            gr.update(value=f"{'▶' if key=='vertexai'     else '  '} {labels['vertexai']}",
                      variant="primary" if key=="vertexai"     else "secondary"),
        )

    for btn, key in [(btn_gemini, "gemini_flash"), (btn_gemma, "gemma4"), (btn_vertex, "vertexai")]:
        btn.click(
            fn=lambda k=key: _set_model(k),
            outputs=[active_model_state, model_label, btn_gemini, btn_gemma, btn_vertex],
        )

    analyze_btn.click(
        fn=process,
        inputs=[url_input, file_input, api_key_input, active_model_state, chk1, chk2, chk3],
        outputs=[summary_box, s1_out, s2_out, s3_out, summary_tab, image_preview],
    )


if __name__ == "__main__":
    demo.launch(debug=True, share=False)
