"""
InSIGHT — Instagram URL 배치 테스트
1차 필터(C2PA/EXIF/Vertex SynthID) → 2차 필터(Gemini) 순차 실행
결과: 1차/2차 분리 리포트 + 성능 지표 + CSV 저장

사용법:
    python3 batch_test.py
    (아래 TEST_URLS 목록에 링크와 정답 레이블을 넣으세요)
"""

import os, sys, time, csv, tempfile, re, json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ExifTags
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

ROOT = Path(__file__).resolve().parent  # members/woochul/
sys.path.insert(0, str(ROOT))

from synthid_vertex import detect_synthid_vertex, VERTEX_AVAILABLE

# ════════════════════════════════════════════════════════════
#  설정 — 여기만 수정하세요
# ════════════════════════════════════════════════════════════

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")   # .env 또는 직접 입력

# 2차 필터 모델 선택
# gemini-2.5-flash        → 현재 안정 버전
# gemini-3.0-flash        → 최신 (Google AI Studio에서 확인)
# gemini-3.1-flash        → 최신 (Google AI Studio에서 확인)
# 확인 방법: aistudio.google.com → 모델 목록
FILTER2_MODEL = "gemini-2.5-flash"

# ────────────────────────────────────────────────────────────
#  테스트 URL 목록
#  형식: (URL, 정답레이블, 메모)
#    정답레이블: 1 = AI 생성,  0 = 실제 이미지
# ────────────────────────────────────────────────────────────
TEST_URLS = [
    # AI 생성 (Gemini/Imagen — SynthID 워터마크 있음)
    ("https://www.instagram.com/p/XXXXXXXXX/", 1, "Gemini 생성 AI 이미지"),

    # 실제 이미지 (카메라 촬영)
    ("https://www.instagram.com/p/YYYYYYYYY/", 0, "실제 인물 사진"),

    # 여기에 계속 추가:
    # ("https://www.instagram.com/p/...", 1, "AI"),
    # ("https://www.instagram.com/p/...", 0, "실제"),
]

# ════════════════════════════════════════════════════════════
#  이미지 다운로드
# ════════════════════════════════════════════════════════════

def download_image(url: str) -> tuple:
    """URL에서 이미지 다운로드 → (PIL.Image, 임시파일경로)"""
    try:
        import instaloader
        m = re.search(r'/(?:p|reel|tv)/([A-Za-z0-9_-]+)', url)
        if m and "instagram.com" in url:
            shortcode = m.group(1)
            L = instaloader.Instaloader(download_pictures=False, quiet=True)
            ig_user = os.getenv("INSTAGRAM_USERNAME", "")
            if ig_user:
                try: L.load_session_from_file(ig_user)
                except: pass
            post = instaloader.Post.from_shortcode(L.context, shortcode)
            img_url = post.url
            resp = requests.get(img_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        else:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(tmp.name, "JPEG")
        tmp.close()
        return img, tmp.name
    except Exception as e:
        return None, str(e)


# ════════════════════════════════════════════════════════════
#  1차 필터
# ════════════════════════════════════════════════════════════

def filter1_c2pa(path: str) -> tuple:
    try:
        import c2pa
        reader = c2pa.Reader(path)
        mj = reader.json()
        if mj:
            text = str(mj).lower()
            ai_kw = ["generativeai","dall","midjourney","stable diffusion",
                     "firefly","imagen","ai.generated","trainedalgorithmicmedia"]
            for kw in ai_kw:
                if kw in text:
                    return True, f"❌ C2PA: AI 서명 감지 ({kw})"
            return False, "✅ C2PA: 원본 서명 확인"
    except: pass
    return None, "🔍 C2PA: 메타데이터 없음"


def filter1_exif(img: Image.Image) -> tuple:
    ai_tools = ["dall-e","midjourney","stable diffusion","adobe firefly",
                "imagen","generative","ai generated","synthid"]
    try:
        exif_raw = img._getexif()
        if not exif_raw:
            return None, "🔍 EXIF: 없음"
        exif = {ExifTags.TAGS.get(k, k): str(v) for k, v in exif_raw.items()}
        blob = " ".join(exif.values()).lower()
        for t in ai_tools:
            if t in blob:
                return True, f"❌ EXIF: AI 흔적 ({t})"
        make  = exif.get("Make", "")
        model = exif.get("Model", "")
        if make or model:
            return False, f"✅ EXIF: 카메라 정보 ({make} {model})"
        return None, "🔍 EXIF: 카메라 없음"
    except:
        return None, "🔍 EXIF: 분석 불가"


def filter1_synthid(img: Image.Image) -> tuple:
    if VERTEX_AVAILABLE:
        result, msg, elapsed, cost = detect_synthid_vertex(img)
        return result, msg, elapsed, cost
    # 폴백: 역공학 근사
    try:
        from synthid_detector import detect_synthid
        result, msg = detect_synthid(img)
        return result, msg, 0.0, 0.0
    except:
        return None, "⚠️ SynthID 모듈 없음", 0.0, 0.0


def run_filter1(path: str, img: Image.Image) -> dict:
    results = {}
    detected = False

    r, msg = filter1_c2pa(path)
    results["c2pa"] = {"result": r, "msg": msg}
    if r is True: detected = True

    r, msg = filter1_exif(img)
    results["exif"] = {"result": r, "msg": msg}
    if r is True: detected = True

    r, msg, elapsed, cost = filter1_synthid(img)
    results["synthid"] = {"result": r, "msg": msg, "elapsed": elapsed, "cost": cost}
    if r is True: detected = True

    results["detected"] = detected
    return results


# ════════════════════════════════════════════════════════════
#  2차 필터 — Gemini
# ════════════════════════════════════════════════════════════

GEMINI_PROMPT = """이 이미지가 AI가 생성한 이미지인지 분석해주세요.

다음 항목을 검토하세요:
1. 피부·텍스처의 과도한 매끄러움
2. 손가락, 귀, 치아 등 세부 부위 형태 이상
3. 배경의 비논리적 구조 또는 반복 패턴
4. 조명·그림자의 물리적 불일치
5. 눈동자 반사 또는 홍채 비현실성
6. 텍스트·문자의 왜곡

반드시 아래 형식으로만 답하세요:
판정: AI 생성 / 실제 이미지 / 불확실
신뢰도: 0~100%
근거: (2~3문장 한국어로)"""


def run_filter2(img: Image.Image, api_key: str) -> dict:
    if not api_key.strip():
        return {"result": None, "msg": "❌ GEMINI_API_KEY 없음", "elapsed": 0.0}
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key.strip())
        buf = BytesIO()
        img.save(buf, format="JPEG")

        t0 = time.time()
        response = client.models.generate_content(
            model=FILTER2_MODEL,
            contents=[
                GEMINI_PROMPT,
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
            ],
        )
        elapsed = round(time.time() - t0, 3)
        text = response.text.strip()

        is_ai = None
        if "판정:" in text:
            verdict_line = text.split("판정:")[1].split("\n")[0].strip()
            if "AI 생성" in verdict_line:
                is_ai = True
            elif "실제 이미지" in verdict_line:
                is_ai = False

        return {"result": is_ai, "msg": text, "elapsed": elapsed}
    except Exception as e:
        return {"result": None, "msg": f"❌ 오류: {e}", "elapsed": 0.0}


# ════════════════════════════════════════════════════════════
#  메인 실행
# ════════════════════════════════════════════════════════════

def run_batch():
    print("=" * 65)
    print("InSIGHT — Instagram URL 배치 테스트")
    print(f"일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"2차 필터 모델: {FILTER2_MODEL}")
    print(f"테스트 URL 수: {len(TEST_URLS)}개")
    print("=" * 65)

    records = []
    f1_records = []   # 1차 필터에서 판정된 것
    f2_records = []   # 2차 필터까지 간 것

    for i, (url, true_label, memo) in enumerate(TEST_URLS):
        label_str = "AI생성" if true_label == 1 else "실제"
        print(f"\n[{i+1}/{len(TEST_URLS)}] {memo}  (정답: {label_str})")
        print(f"  URL: {url[:60]}...")

        # 이미지 다운로드
        img, path_or_err = download_image(url)
        if img is None:
            print(f"  ⚠️ 다운로드 실패: {path_or_err}")
            records.append({"url": url, "memo": memo, "true_label": true_label,
                            "error": path_or_err})
            continue

        tmp_path = path_or_err

        try:
            # ── 1차 필터 ──────────────────────────────────
            f1 = run_filter1(tmp_path, img)

            print(f"  [1차] C2PA   : {f1['c2pa']['msg']}")
            print(f"  [1차] EXIF   : {f1['exif']['msg']}")
            print(f"  [1차] SynthID: {f1['synthid']['msg'].splitlines()[0]}")

            f1_caught = f1["detected"]
            final_pred = 1 if f1_caught else None
            f2_result  = None
            f2_msg     = "—"
            f2_elapsed = 0.0

            # ── 2차 필터 (1차 통과한 경우만) ───────────────
            if not f1_caught:
                print(f"  → 1차 미감지, 2차 필터({FILTER2_MODEL}) 진행...")
                f2 = run_filter2(img, GEMINI_API_KEY)
                f2_result  = f2["result"]
                f2_msg     = f2["msg"]
                f2_elapsed = f2["elapsed"]
                final_pred = 1 if f2_result is True else (0 if f2_result is False else None)
                verdict_line = f2_msg.split("\n")[0] if f2_msg else ""
                print(f"  [2차] {verdict_line}")
            else:
                print(f"  → 1차 필터에서 AI 감지 — 2차 필터 생략")

            correct = (final_pred == true_label)
            mark = "✅" if correct else "❌"
            print(f"  최종 판정: {'AI생성' if final_pred==1 else '실제' if final_pred==0 else '불확실'}  {mark}")

            rec = {
                "url":          url,
                "memo":         memo,
                "true_label":   true_label,
                "f1_caught":    f1_caught,
                "f1_c2pa":      f1["c2pa"]["msg"],
                "f1_exif":      f1["exif"]["msg"],
                "f1_synthid":   f1["synthid"]["msg"].splitlines()[0],
                "synthid_cost": f1["synthid"]["cost"],
                "synthid_time": f1["synthid"]["elapsed"],
                "f2_model":     FILTER2_MODEL if not f1_caught else "—",
                "f2_result":    str(f2_result),
                "f2_elapsed":   f2_elapsed,
                "f2_verdict":   f2_msg.split("판정:")[-1].split("\n")[0].strip() if "판정:" in f2_msg else "—",
                "final_pred":   final_pred,
                "correct":      correct,
            }
            records.append(rec)

            if f1_caught:
                f1_records.append(rec)
            else:
                f2_records.append(rec)

        finally:
            try: os.remove(tmp_path)
            except: pass

    # ════════════════════════════════════════════════════════
    #  결과 리포트
    # ════════════════════════════════════════════════════════

    valid = [r for r in records if "error" not in r and r.get("final_pred") is not None]
    if not valid:
        print("\n[!] 유효한 결과 없음")
        return

    y_true = np.array([r["true_label"]  for r in valid])
    y_pred = np.array([r["final_pred"]  for r in valid])

    if len(set(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    else:
        tp = int(np.sum((y_true==1)&(y_pred==1)))
        fp = int(np.sum((y_true==0)&(y_pred==1)))
        fn = int(np.sum((y_true==1)&(y_pred==0)))
        tn = int(np.sum((y_true==0)&(y_pred==0)))

    accuracy  = accuracy_score(y_true, y_pred)  * 100
    precision = precision_score(y_true, y_pred, zero_division=0) * 100
    recall    = recall_score(y_true, y_pred, zero_division=0)    * 100
    f1        = f1_score(y_true, y_pred, zero_division=0)         * 100
    fpr       = (fp/(fp+tn)*100) if (fp+tn) > 0 else 0.0

    total_synthid_cost = sum(r.get("synthid_cost", 0) for r in valid)
    avg_synthid_time   = np.mean([r.get("synthid_time", 0) for r in valid if r.get("synthid_time",0)>0]) if valid else 0

    def chk(v, t, lo=False): return "✅" if (v<=t if lo else v>=t) else "❌"

    # ── 1차 필터 리포트 ──────────────────────────────────
    print("\n" + "=" * 65)
    print("  [1차 필터 결과 리포트]")
    print("=" * 65)
    print(f"  1차에서 AI 감지 : {len(f1_records)}건")
    print(f"  2차로 넘어감    : {len(f2_records)}건")
    print(f"  SynthID 총 비용 : ${total_synthid_cost:.4f}  (₩{total_synthid_cost*1400:,.0f})")
    print(f"  SynthID 평균 속도: {avg_synthid_time:.2f}초/장")

    if f1_records:
        print("\n  1차 감지 케이스:")
        for r in f1_records:
            lbl = "AI" if r["true_label"]==1 else "실제"
            mark = "✅ 정답" if r["correct"] else "❌ 오답"
            print(f"    [{mark}] {r['memo']} (정답:{lbl})")
            print(f"      SynthID: {r['f1_synthid']}")

    # ── 2차 필터 리포트 ──────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  [2차 필터 결과 리포트]  모델: {FILTER2_MODEL}")
    print("=" * 65)
    if f2_records:
        for r in f2_records:
            lbl  = "AI" if r["true_label"]==1 else "실제"
            pred = "AI생성" if r["final_pred"]==1 else "실제" if r["final_pred"]==0 else "불확실"
            mark = "✅" if r["correct"] else "❌"
            print(f"  {mark} {r['memo']}")
            print(f"     정답:{lbl}  |  판정:{pred}  |  {r['f2_verdict']}  ({r['f2_elapsed']:.2f}초)")
    else:
        print("  모든 이미지가 1차에서 처리됨 — 2차 필터 미실행")

    # ── 전체 지표 (투두리스트 기록용) ────────────────────
    print("\n" + "=" * 65)
    print("  [전체 성능 지표]  ← 투두리스트 기록용")
    print("=" * 65)
    print(f"  테스트 수    : {len(valid)}장  (AI {int(y_true.sum())}장 / 실제 {int((y_true==0).sum())}장)")
    print(f"  혼동 행렬    : TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Accuracy    : {accuracy:.1f}%   {chk(accuracy,  70)}")
    print(f"  Precision   : {precision:.1f}%   {chk(precision, 75)}")
    print(f"  Recall      : {recall:.1f}%   {chk(recall,    75)}  ★핵심")
    print(f"  F1 Score    : {f1:.1f}%   {chk(f1,        70)}")
    print(f"  FPR         : {fpr:.1f}%   {chk(fpr,       20, lo=True)}")
    print(f"  SynthID 비용: ${total_synthid_cost:.4f}/회  {chk(total_synthid_cost/max(len(valid),1), 0.01, lo=True)}")
    print(f"  SynthID 속도: {avg_synthid_time:.2f}초/장  {chk(avg_synthid_time, 3.0, lo=True)}")

    adopted = all([recall>=75, total_synthid_cost/max(len(valid),1)<=0.01, avg_synthid_time<=3.0])
    print(f"\n  1차 필터 채택: {'✅ 권고' if adopted else '❌ 미채택'}")

    # ── CSV 저장 ─────────────────────────────────────────
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(__file__).parent / f"batch_results_{ts}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        valid_rows = [r for r in records if "error" not in r]
        if valid_rows:
            writer = csv.DictWriter(f, fieldnames=valid_rows[0].keys())
            writer.writeheader()
            writer.writerows(valid_rows)
    print(f"\n  결과 저장: {out}")
    print("=" * 65)


if __name__ == "__main__":
    run_batch()
