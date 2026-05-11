"""
InSIGHT — AI 생성 이미지/영상 판별기 (로컬 데모 버전)
Instagram / YouTube 링크 또는 이미지 파일 → 2단계 분석 → 증거 기반 판정

실행:
    conda activate hnf
    python app.py

환경변수: .env 파일에 GEMINI_API_KEY 설정
"""

import os
import re
import sys
import time
import json
import base64
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

# ── 로컬 모듈 ─────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from videofact_wrapper import detect_videofact_score
from freqnet_wrapper import detect_freqnet_score

try:
    from ensemble_detector import EnsembleDetector
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

_ensemble_instance = None

# ── 설정 ──────────────────────────────────────────────────

STAGE1_THRESHOLD = 0.80
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

MODEL_CONFIGS: dict[str, dict] = {
    "gemini_flash": {
        "display":        "Gemini 2.5 Flash",
        "model_id":       "gemini-2.5-flash",
        "api":            "google_genai",
        "cost_per_image": 0.0001,
    },
    "gemini_flash3": {
        "display":        "Gemini 3 Flash",
        "model_id":       "gemini-3-flash-preview",
        "api":            "google_genai",
        "cost_per_image": 0.0001,
    },
    "gemma4": {
        "display":        "Gemma 4 (Ollama)",
        "model_id":       "gemma4:e4b",
        "api":            "ollama",
        "cost_per_image": 0.0,
    },
    "ensemble": {
        "display":        "앙상블 (ViT×2)",
        "model_id":       "ensemble",
        "api":            "ensemble",
        "cost_per_image": 0.0,
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
# Stage 1: 메타데이터 + 포렌식 분석
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
        raw = image.getexif()          # _getexif()는 JPEG 전용 private API → 공개 API로 교체
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

def _s1_videofact(image: Image.Image) -> tuple[float, list[str]]:
    indicators: list[str] = []
    try:
        import videofact_wrapper as _vfw
        score = detect_videofact_score(image)
        if _vfw._detector is None or _vfw._detector.model is None:
            indicators.append("VideoFact 모델 미로드 — 외부 가중치 파일 필요 (건너뜀)")
            return 0.0, indicators
        indicators.append(f"VideoFact (WACV 2024) 분석 완료: 스코어 {score:.4f}")
        indicators.append("디지털 포렌식 흔적(Forensic Traces) 및 장면 문맥(Scene Context) 분석")
        indicators.append("픽셀 노이즈, 압축 아티팩트, 미세 불일치 정밀 감지")
        return score, indicators
    except Exception as e:
        indicators.append(f"VideoFact 분석 중 오류 발생: {e}")
        return 0.0, indicators

def _s1_freqnet(image: Image.Image) -> tuple[float, list[str]]:
    indicators: list[str] = []
    try:
        import freqnet_wrapper as _fnw
        score = detect_freqnet_score(image)
        if _fnw._detector is None or _fnw._detector.model is None:
            indicators.append("FreqNet 모델 미로드 — 외부 가중치 파일 필요 (건너뜀)")
            return 0.0, indicators
        indicators.append(f"FreqNet (AAAI 2024) 분석 완료: 스코어 {score:.4f}")
        indicators.append("주파수 영역(Frequency Space) 도메인 학습 기반 탐지")
        indicators.append("눈에 보이지 않는 주파수 성분의 위조 흔적 정밀 분석")
        return score, indicators
    except Exception as e:
        indicators.append(f"FreqNet 분석 중 오류 발생: {e}")
        return 0.0, indicators

def run_stage1(image: Image.Image, image_path: str) -> dict:
    t0 = time.time()
    exif_score, exif_ind = _s1_exif(image)
    c2pa_score, c2pa_ind = _s1_c2pa(image_path)
    vf_score,   vf_ind   = _s1_videofact(image)
    fn_score,   fn_ind   = _s1_freqnet(image)

    # EXIF(0.15) + C2PA(0.15) + VideoFact(0.30) + FreqNet(0.40)
    score = (exif_score * 0.15 + c2pa_score * 0.15 +
             vf_score * 0.30 + fn_score * 0.40)
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
            "videofact": {"score": vf_score,   "indicators": vf_ind},
            "freqnet":   {"score": fn_score,   "indicators": fn_ind},
        },
    }


# ═══════════════════════════════════════════════════════════
# Stage 3: Gemini / Gemma4 / 앙상블 시각 분석
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

def _call_ollama(image: Image.Image, model_id: str) -> dict:
    if not REQUESTS_AVAILABLE:
        return {"error": "requests 미설치"}
    buf = BytesIO()
    image.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  model_id,
                "prompt": _STAGE3_PROMPT,
                "images": [img_b64],
                "stream": False,
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        return _parse_json(data.get("response", ""))
    except requests.exceptions.ConnectionError:
        return {"error": f"Ollama 서버 연결 실패 ({OLLAMA_BASE_URL}) — 'ollama serve' 실행 여부 확인"}
    except Exception as e:
        return {"error": str(e)}


def _ensemble_explain_prompt(score_pct: float) -> str:
    verdict = "AI 생성" if score_pct >= 50 else "실제 이미지"
    return f"""당신은 AI 이미지 탐지 전문가입니다.
딥러닝 앙상블 탐지 모델이 이 이미지를 AI 생성 확률 {score_pct:.1f}%로 분석했습니다 (판정: {verdict}).

이미지를 직접 시각적으로 분석하여 이 판정을 뒷받침하는 시각적 근거를 구체적으로 설명하세요.
각 특징이 얼마나 비자연스러운지 퍼센트로 추정하고, 구체적인 이유를 함께 작성하세요.

반드시 아래 JSON 형식으로만 응답하세요 (다른 텍스트 없이):
{{
  "evidence": [
    {{"item": "특징명: XX% — 구체적 설명", "weight": "high|medium|low"}},
    {{"item": "특징명: XX% — 구체적 설명", "weight": "high|medium|low"}}
  ],
  "detail": {{
    "texture":    "텍스처·피부·머리카락 관찰",
    "lighting":   "조명·그림자 분석",
    "anatomy":    "손·귀·치아 등 해부학적 이상 여부",
    "background": "배경 일관성",
    "artifacts":  "AI 특유 아티팩트"
  }}
}}

evidence는 3~6개 항목으로 구성하세요."""


def _call_ollama_explain(image: Image.Image, score_pct: float, model_id: str) -> dict:
    if not REQUESTS_AVAILABLE:
        return {"error": "requests 미설치"}
    buf = BytesIO()
    image.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  model_id,
                "prompt": _ensemble_explain_prompt(score_pct),
                "images": [img_b64],
                "stream": False,
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        return _parse_json(data.get("response", ""))
    except requests.exceptions.ConnectionError:
        return {"error": f"Ollama 서버 연결 실패 ({OLLAMA_BASE_URL}) — 'ollama serve' 실행 여부 확인"}
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

def run_ensemble(image_path: str, explain: bool = False, image: Image.Image = None) -> dict:
    global _ensemble_instance
    if not ENSEMBLE_AVAILABLE:
        return {
            "score":    0.0,
            "verdict":  "오류: ensemble_detector 로드 실패",
            "elapsed":  0.0,
            "cost_usd": 0.0,
            "analysis": {"error": "ensemble_detector 로드 실패"},
            "model":    "앙상블 (ViT×2)",
        }
    t0 = time.time()
    try:
        if _ensemble_instance is None:
            _ensemble_instance = EnsembleDetector()

        img     = image if image is not None else Image.open(image_path).convert("RGB")
        t_inf   = time.time()
        detail  = _ensemble_instance.predict_scores(img)   # 모델별 점수 + ensemble 키 포함
        elapsed = round(time.time() - t_inf, 3)

        score = detail.pop("ensemble")                     # 앙상블 최종 점수
        pred  = 1 if score >= _ensemble_instance.threshold else 0
        thr   = _ensemble_instance.threshold
        verdict = "AI 생성" if pred == 1 else "실제 이미지"

        # 가중치 맵 (short_name → weight)
        weight_map = {mid.split("/")[-1]: w
                      for mid, w in _ensemble_instance.weights.items()}

        # 각 모델 기여 계산
        evidence, calc_parts = [], []
        for short_name, m_score in detail.items():
            w = weight_map.get(short_name, 0.0)
            calc_parts.append(f"{m_score*100:.1f}%×{w}")
            evidence.append({
                "item":   f"{short_name}  →  AI 확률 {m_score*100:.1f}%  (가중치 {w})",
                "weight": "high" if w >= 0.5 else "medium",
            })

        calc_str = " + ".join(calc_parts)
        inference = [{
            "item":  f"소프트 보팅 = {calc_str} = {score*100:.1f}%",
            "basis": f"임계값 {thr} 초과 → {verdict}",
        }]

        # Gemma 시각 근거 설명 (옵션)
        gemma_explain = None
        if explain:
            t_exp = time.time()
            gemma_explain = _call_ollama_explain(img, score * 100, MODEL_CONFIGS["gemma4"]["model_id"])
            gemma_explain["_elapsed"] = round(time.time() - t_exp, 3)

        return {
            "score":    round(score, 4),
            "verdict":  verdict,
            "elapsed":  elapsed,
            "cost_usd": 0.0,
            "analysis": {
                "verdict":        "AI_GENERATED" if pred == 1 else "REAL",
                "confidence_pct": round(score * 100),
                "evidence":       evidence,
                "inference":      inference,
                "detail":         {},
                "gemma_explain":  gemma_explain,
            },
            "model": "앙상블 (ViT×2)",
        }
    except Exception as e:
        return {
            "score":    0.0,
            "verdict":  f"오류: {e}",
            "elapsed":  round(time.time() - t0, 3),
            "cost_usd": 0.0,
            "analysis": {"error": str(e)},
            "model":    "앙상블 (ViT×2)",
        }

def run_stage3(image: Image.Image, image_path: str, model_key: str, api_key: str, gemma_explain: bool = False) -> dict:
    cfg = MODEL_CONFIGS[model_key]

    if cfg["api"] == "ensemble":
        return run_ensemble(image_path, explain=gemma_explain, image=image)

    t0 = time.time()

    if cfg["api"] == "google_genai":
        analysis = _call_google_genai(image, cfg["model_id"], api_key)
    elif cfg["api"] == "ollama":
        analysis = _call_ollama(image, cfg["model_id"])
    else:
        analysis = {"error": f"알 수 없는 API: {cfg['api']}"}

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

_S1_WEIGHTS = {"exif": 0.15, "c2pa": 0.15, "videofact": 0.30, "freqnet": 0.40}
_S1_LABELS  = {
    "exif":      ("①", "EXIF 분석      "),
    "c2pa":      ("②", "C2PA 출처      "),
    "videofact": ("③", "VideoFact(WACV)"),
    "freqnet":   ("④", "FreqNet(AAAI)  "),
}

def _bar(score: float, w: int = 20) -> str:
    n = int(round(score * w))
    return f"[{'█' * n}{'░' * (w - n)}] {score * 100:5.1f}%"

def _suspect(score: float) -> str:
    """점수(AI 확률 0~1)를 '몇 %로 분석되어 **verdict**으로 의심됩니다.' 문장으로 변환."""
    if score >= 0.55:
        return f"{score * 100:.1f}%로 분석되어 **AI 생성**으로 의심됩니다."
    elif score <= 0.45:
        return f"{(1 - score) * 100:.1f}%로 분석되어 **실제 이미지**로 의심됩니다."
    else:
        return f"AI 생성 확률 {score * 100:.1f}% — 판별이 어렵습니다. (추가 분석 권장)"

def fmt_stage1(r: dict) -> str:
    out = ["━━━  Stage 1 — 메타데이터 / 포렌식 분석  ━━━", ""]
    out.append("  [추론 근거]")
    out.append("")

    for key, data in r["checks"].items():
        num, lbl = _S1_LABELS.get(key, ("•", key))
        w   = _S1_WEIGHTS.get(key, 0.0)
        s   = data["score"]
        con = s * w
        out.append(f"  {num} {lbl}  {s*100:5.1f}%  × {w:.2f}  →  기여 {con*100:5.2f}%")
        for ind in data["indicators"]:
            out.append(f"      └ {ind}")
        out.append("")

    sep = "  " + "─" * 52
    out.append(sep)
    out.append(f"  AI 생성 확률  {_bar(r['score'])}")
    out.append(f"  소요 시간  {r['elapsed']}s  |  비용  $0.0000  (로컬 처리)")
    out.append("")

    # 미로드·오류 키워드로 신뢰 가능한 컴포넌트 수 계산
    _skip_kw = ("미로드", "미설치", "오류", "실패")
    n_skipped = sum(
        1 for data in r["checks"].values()
        if any(kw in " ".join(data["indicators"]) for kw in _skip_kw)
    )
    n_total = len(r["checks"])

    if n_skipped >= n_total - 1:            # 분석 가능 컴포넌트가 1개 이하
        out.append("  → 주요 분석 모델 미로드로 1단계 스크리닝 불가합니다.")
        out.append("     2단계 결과만으로 판단하세요.")
    elif r["score"] > 0.5:
        out.append(f"  → {r['score']*100:.1f}% — 이상 신호 감지. 2단계 분석을 시행합니다.")
    else:
        out.append(f"  → {(1 - r['score'])*100:.1f}% — 이상 신호 없음. 2단계에서 최종 확인합니다.")
    return "\n".join(out)

def fmt_stage3(r: dict) -> str:
    out = [f"━━━  Stage 3 — {r['model']} 시각 분석  ━━━", ""]
    if "오류" in r["verdict"]:
        out.append(f"  {r['verdict']}")
        return "\n".join(out)

    analysis = r.get("analysis", {})
    if analysis.get("parse_error"):
        out.append("  [원본 응답]")
        out.append(f"  {analysis.get('raw_response', '')}")
        return "\n".join(out)

    out.append("  [추론 근거]")
    out.append("")

    # ① 직접 관찰 증거
    evidence = analysis.get("evidence", [])
    if evidence:
        weight_label = {"high": "강", "medium": "중", "low": "약"}
        weight_icon  = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        out.append("  ① 직접 관찰 (증거)")
        for e in evidence:
            if isinstance(e, dict):
                w    = e.get("weight", "")
                icon = weight_icon.get(w, "•")
                wlbl = weight_label.get(w, w)
                text = e.get("item", "")
            else:
                icon, wlbl, text = "•", "", str(e)
            out.append(f"      {icon} [{wlbl}]  {text}")
        out.append("")

    # ② 증거 기반 추론
    inference = analysis.get("inference", [])
    if inference:
        out.append("  ② 증거 기반 추론")
        for i in inference:
            if isinstance(i, dict):
                out.append(f"      → {i.get('item', '')}")
                if i.get("basis"):
                    out.append(f"          근거: {i['basis']}")
            else:
                out.append(f"      → {i}")
        out.append("")

    # ③ 세부 분석
    detail = analysis.get("detail", {})
    detail_rows = [(k, v) for k, v in [
        ("texture",    "텍스처/피부"),
        ("lighting",   "조명/그림자"),
        ("anatomy",    "해부학적  "),
        ("background", "배경      "),
        ("artifacts",  "AI 아티팩트"),
    ] if detail.get(k)]
    if detail_rows:
        out.append("  ③ 세부 분석")
        for k, lbl in detail_rows:
            out.append(f"      {lbl}  {detail[k]}")
        out.append("")

    # ③ Gemma 시각 근거 분석 (앙상블 + 설명 모드)
    gemma_exp = analysis.get("gemma_explain")
    if gemma_exp and not gemma_exp.get("error") and not gemma_exp.get("parse_error"):
        ge_elapsed = gemma_exp.get("_elapsed", 0)
        out.append(f"  ③ Gemma 시각 근거 분석  ({ge_elapsed:.1f}s)")
        weight_label = {"high": "강", "medium": "중", "low": "약"}
        weight_icon  = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        for e in gemma_exp.get("evidence", []):
            if isinstance(e, dict):
                w    = e.get("weight", "")
                icon = weight_icon.get(w, "•")
                wlbl = weight_label.get(w, w)
                text = e.get("item", "")
            else:
                icon, wlbl, text = "•", "", str(e)
            out.append(f"      {icon} [{wlbl}]  {text}")
        gd = gemma_exp.get("detail", {})
        gd_rows = [(k, v) for k, v in [
            ("texture",    "텍스처/피부  "),
            ("lighting",   "조명/그림자  "),
            ("anatomy",    "해부학적     "),
            ("background", "배경         "),
            ("artifacts",  "AI 아티팩트  "),
        ] if gd.get(k)]
        if gd_rows:
            out.append("")
            for k, lbl in gd_rows:
                out.append(f"      {lbl}  {gd[k]}")
        out.append("")
    elif gemma_exp and (gemma_exp.get("error") or gemma_exp.get("parse_error")):
        err_msg = gemma_exp.get("error") or "응답 파싱 실패"
        out.append(f"  ③ Gemma 시각 근거 분석  오류: {err_msg}")
        out.append("")

    sep = "  " + "─" * 52
    out.append(sep)
    conf_pct = analysis.get("confidence_pct", round(r["score"] * 100))
    out.append(f"  AI 생성 확률  {_bar(r['score'])}")
    out.append(f"  소요 시간  {r['elapsed']}s  |  예상 비용  ${r['cost_usd']:.4f}")
    out.append("")
    out.append(f"  → {_suspect(r['score'])}")
    return "\n".join(out)

def fmt_summary(s1, s3) -> str:
    out = ["━━━━━━━━━━━━━━━  종합 판정  ━━━━━━━━━━━━━━━", ""]
    total_cost = 0.0

    # 1단계: 이진 분류 결과만 표시
    if s1:
        s1_flag = "의심됨 →" if s1["score"] > 0.5 else "이상 없음"
        out.append(f"  1단계  {_bar(s1['score'], 14)}  {s1_flag}")

    # 2단계: 결과 표시
    if s3 and "오류" not in s3["verdict"]:
        total_cost += s3.get("cost_usd", 0)
        out.append(f"  2단계  {_bar(s3['score'], 14)}  ({s3['model']})")

    out.append("")
    sep = "  " + "─" * 52
    out.append(sep)

    # 최종 판정 = 2단계 단독 결과 (평균 내지 않음)
    if s3 and "오류" not in s3["verdict"]:
        out.append(f"  최종 분석  {_bar(s3['score'])}")
        out.append("")
        out.append(f"  → {_suspect(s3['score'])}")
    elif s1:
        # 2단계 미실행 시 1단계 이진 결과
        if s1["score"] > 0.5:
            out.append(f"  → {s1['score']*100:.1f}% — AI 생성으로 의심됩니다.")
        else:
            out.append(f"  → {(1 - s1['score'])*100:.1f}% — 실제 이미지로 의심됩니다.")
    else:
        out.append("  실행된 단계 없음")

    out.append("")
    out.append(f"  총 예상 비용  ${total_cost:.4f} USD")
    return "\n".join(out)


# ═══════════════════════════════════════════════════════════
# 메인 처리
# ═══════════════════════════════════════════════════════════

def process(url, uploaded_file, api_key, active_model, use_s1, use_s3, gemma_explain=False):
    image      = None
    image_path = None
    tmp_path   = None
    s1 = s3 = None

    try:
        if uploaded_file:
            image      = Image.open(uploaded_file).convert("RGB")
            image_path = uploaded_file
        elif url.strip():
            image, image_path, _ = download_media(url.strip())
            tmp_path = image_path
        else:
            return "URL 또는 파일을 입력하세요.", "", "", "", None

        s1_text = s3_text = ""

        if use_s1:
            s1      = run_stage1(image, image_path)
            s1_text = fmt_stage1(s1)
        else:
            s1_text = "(Stage 1 비활성화)"

        if use_s3:
            s3      = run_stage3(image, image_path, active_model, api_key, gemma_explain)
            s3_text = fmt_stage3(s3)
        else:
            s3_text = "(Stage 3 비활성화)"

        summary = fmt_summary(s1, s3)
        return summary, s1_text, s3_text, summary, image

    except Exception as e:
        err = f"❌ 오류: {e}"
        return err, err, "", err, None
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
        "Instagram · YouTube URL 또는 이미지 파일 → 2단계 분석 → 증거 기반 판정"
    )

    with gr.Group():
        gr.Markdown("#### Stage 3 분석 모델 선택")
        with gr.Row(equal_height=True):
            btn_gemini  = gr.Button("▶ Gemini 2.5 Flash", variant="primary",   min_width=150)
            btn_gemini3 = gr.Button("   Gemini 3 Flash",  variant="secondary", min_width=150)
            btn_gemma   = gr.Button("   Gemma 4",         variant="secondary", min_width=150)
            btn_ensemble = gr.Button("   앙상블 (ViT×2)", variant="secondary", min_width=150)
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
                label="Gemini API Key  (Gemini 모델 사용 시 필요 / .env에 설정하면 자동 입력)",
                type="password",
                placeholder="AIza...",
                value=os.getenv("GEMINI_API_KEY", ""),
            )

            gr.Markdown("#### 분석 단계 선택")
            with gr.Row():
                chk1 = gr.Checkbox(value=True, label="Stage 1  메타데이터 + 포렌식 분석  (무료, 로컬)")
                chk3 = gr.Checkbox(value=True, label="Stage 3  AI 시각 분석  (Gemini / Gemma4 / 앙상블)")
            chk_gemma_explain = gr.Checkbox(
                value=False,
                label="앙상블 선택 시 — Gemma 4 시각 근거 설명 추가  (Ollama 필요, 추가 30~60초 소요)",
            )

            analyze_btn = gr.Button("🔍  분석 시작", variant="primary", size="lg")

        with gr.Column(scale=2):
            image_preview = gr.Image(label="분석 대상 이미지", type="pil", height=320)

    summary_box = gr.Textbox(label="종합 판정", interactive=False, lines=9, elem_classes=["mono"])

    with gr.Tabs():
        with gr.Tab("Stage 1 — 메타데이터/포렌식"):
            s1_out = gr.Textbox(interactive=False, lines=20, elem_classes=["mono"])
        with gr.Tab("Stage 3 — AI 시각 분석"):
            s3_out = gr.Textbox(interactive=False, lines=20, elem_classes=["mono"])
        with gr.Tab("종합 (전체)"):
            summary_tab = gr.Textbox(interactive=False, lines=20, elem_classes=["mono"])

    _MODEL_LABELS = {
        "gemini_flash":  "Gemini 2.5 Flash",
        "gemini_flash3": "Gemini 3 Flash",
        "gemma4":        "Gemma 4",
        "ensemble":      "앙상블 (ViT×2)",
    }

    def _set_model(key: str):
        lbl = _MODEL_LABELS[key]
        def _btn(k):
            return gr.update(
                value=f"{'▶' if k == key else '  '} {_MODEL_LABELS[k]}",
                variant="primary" if k == key else "secondary",
            )
        return (
            key,
            f"*현재 모델: **{lbl}***",
            _btn("gemini_flash"),
            _btn("gemini_flash3"),
            _btn("gemma4"),
            _btn("ensemble"),
        )

    for btn, key in [
        (btn_gemini,   "gemini_flash"),
        (btn_gemini3,  "gemini_flash3"),
        (btn_gemma,    "gemma4"),
        (btn_ensemble, "ensemble"),
    ]:
        btn.click(
            fn=lambda k=key: _set_model(k),
            outputs=[active_model_state, model_label,
                     btn_gemini, btn_gemini3, btn_gemma, btn_ensemble],
        )

    analyze_btn.click(
        fn=process,
        inputs=[url_input, file_input, api_key_input, active_model_state, chk1, chk3, chk_gemma_explain],
        outputs=[summary_box, s1_out, s3_out, summary_tab, image_preview],
    )


if __name__ == "__main__":
    demo.launch(debug=True, share=False)
