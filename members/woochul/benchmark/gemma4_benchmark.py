"""
Gemma 4 12B 로컬 벤치마크 (Ollama)
====================================
실행 전 준비:
    ollama pull gemma4:12b

실행:
    python gemma4_benchmark.py              # 전체 350장
    python gemma4_benchmark.py --sample 50  # 빠른 테스트 (50장)
    python gemma4_benchmark.py --resume     # 중단된 곳부터 재개

결과:
    benchmark_gemma4_12b_<timestamp>.csv    # 지표 요약
    gemma4_raw_<timestamp>.csv              # 이미지별 상세 예측값
"""

import os
import re
import csv
import sys
import json
import time
import base64
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from io import BytesIO

import numpy as np
from PIL import Image

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix,
    )
except ImportError:
    print("sklearn 미설치: pip install scikit-learn")
    sys.exit(1)

try:
    import ollama as _ollama
except ImportError:
    print("ollama 미설치: pip install ollama")
    sys.exit(1)

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
_ROOT           = Path(__file__).resolve().parents[3]
DATASET_AI_DIR  = _ROOT / "data" / "test_dataset" / "ai"
DATASET_REAL_DIR= _ROOT / "data" / "test_dataset" / "real"
RESULT_DIR      = Path(__file__).resolve().parent
CHARTS_DIR      = RESULT_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
MODEL_NAME  = "gemma4:e4b"
THRESHOLD   = 0.50        # confidence_pct / 100 ≥ threshold → AI 판정
IMG_SIZE    = (512, 512)  # Ollama에 전달할 이미지 리사이즈 (속도 vs 품질)
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

TARGET = {
    "accuracy":  70.0,
    "precision": 75.0,
    "recall":    75.0,
    "f1":        70.0,
    "fpr":       20.0,   # 이하
    "avg_time":  30.0,   # LLM은 ViT보다 느림 — 30s 이하 목표
}

# ──────────────────────────────────────────────
# 프롬프트
# ──────────────────────────────────────────────
PROMPT = """이 이미지가 AI가 생성한 것인지, 실제로 촬영된 것인지 판단하세요.

판단 기준:
- 피부/털/직물 텍스처의 과도한 균일함
- 손가락·귀·치아 등 해부학적 이상
- 배경의 비현실적 흐림 또는 반복 패턴
- 조명·그림자 방향 불일치
- AI 생성 특유의 수채화풍 또는 과도하게 매끄러운 질감

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 절대 쓰지 마세요:
{"verdict": "AI_GENERATED", "confidence_pct": 85, "reason": "한 줄 근거"}

verdict 값은 반드시 "AI_GENERATED" 또는 "REAL" 중 하나여야 합니다."""


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────

def encode_image(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail(IMG_SIZE, Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def parse_response(text: str) -> tuple[str, float, str]:
    """
    LLM 응답에서 verdict / confidence_pct / reason 추출.
    JSON 파싱 실패 시 키워드 기반 폴백.
    반환: (verdict, confidence 0~1, reason)
    """
    text = text.strip()

    # JSON 블록 추출
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            verdict = str(obj.get("verdict", "")).upper()
            conf    = float(obj.get("confidence_pct", 50)) / 100.0
            reason  = str(obj.get("reason", ""))
            if verdict in ("AI_GENERATED", "REAL"):
                return verdict, min(max(conf, 0.0), 1.0), reason
        except (json.JSONDecodeError, ValueError):
            pass

    # 폴백: 텍스트 키워드
    lower = text.lower()
    if "ai_generated" in lower or "ai generated" in lower:
        return "AI_GENERATED", 0.70, "키워드 폴백"
    if "real" in lower:
        return "REAL", 0.70, "키워드 폴백"

    return "UNCERTAIN", 0.50, "파싱 실패"


def check_model_installed() -> bool:
    try:
        models = [m.model for m in _ollama.list().models]
        return any(MODEL_NAME in m for m in models)
    except Exception:
        return False


def pull_model():
    print(f"\n  {MODEL_NAME} 다운로드 시작 (약 8 GB)...")
    print("  (중단하려면 Ctrl+C, 나중에 다시 실행하면 이어받기 됩니다)\n")
    subprocess.run(["/opt/homebrew/bin/ollama", "pull", MODEL_NAME], check=True)


# ──────────────────────────────────────────────
# 데이터셋 로드
# ──────────────────────────────────────────────

def load_dataset(sample: int | None = None) -> list[tuple[Path, int]]:
    dataset: list[tuple[Path, int]] = []

    for p in sorted(DATASET_AI_DIR.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            dataset.append((p, 1))

    for p in sorted(DATASET_REAL_DIR.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            dataset.append((p, 0))

    if sample and sample < len(dataset):
        # AI / Real 비율 유지하며 샘플링
        ai   = [(p, l) for p, l in dataset if l == 1]
        real = [(p, l) for p, l in dataset if l == 0]
        ratio = len(ai) / len(dataset)
        n_ai  = round(sample * ratio)
        n_real= sample - n_ai
        import random
        random.seed(42)
        dataset = random.sample(ai, min(n_ai, len(ai))) + \
                  random.sample(real, min(n_real, len(real)))

    return dataset


# ──────────────────────────────────────────────
# 재개용 중간 결과 로드
# ──────────────────────────────────────────────

def load_partial(raw_path: Path) -> dict[str, dict]:
    """이미지 경로 → 예측 결과 딕셔너리 반환"""
    done: dict[str, dict] = {}
    if not raw_path.exists():
        return done
    with open(raw_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done[row["image_path"]] = row
    print(f"  [재개] {len(done)}개 이미지 기존 결과 로드됨")
    return done


# ──────────────────────────────────────────────
# 추론 루프
# ──────────────────────────────────────────────

def run_inference(
    dataset: list[tuple[Path, int]],
    raw_path: Path,
    resume: bool = False,
) -> list[dict]:
    done = load_partial(raw_path) if resume else {}

    raw_fields = ["image_path", "true_label", "verdict", "confidence", "pred_label", "reason", "elapsed", "error"]
    write_header = not raw_path.exists() or not resume

    raw_file = open(raw_path, "a" if resume else "w", newline="", encoding="utf-8-sig")
    writer   = csv.DictWriter(raw_file, fieldnames=raw_fields)
    if write_header:
        writer.writeheader()

    rows: list[dict] = list(done.values())
    n = len(dataset)

    try:
        for idx, (img_path, true_label) in enumerate(dataset, 1):
            key = str(img_path)
            if key in done:
                print(f"  [{idx:>3}/{n}] 스킵 (기존 결과): {img_path.name}")
                continue

            print(f"  [{idx:>3}/{n}] 분석 중: {img_path.name} (label={'AI' if true_label else 'Real'})")
            t0    = time.time()
            error = ""

            try:
                img_b64 = encode_image(img_path)
                resp    = _ollama.chat(
                    model=MODEL_NAME,
                    messages=[{
                        "role":    "user",
                        "content": PROMPT,
                        "images":  [img_b64],
                    }],
                    options={"temperature": 0.1},
                )
                raw_text = resp.message.content
                verdict, conf, reason = parse_response(raw_text)
            except Exception as e:
                verdict, conf, reason = "UNCERTAIN", 0.5, ""
                error = str(e)[:200]

            elapsed    = round(time.time() - t0, 2)
            pred_label = 1 if (verdict == "AI_GENERATED" and conf >= THRESHOLD) else 0

            row = {
                "image_path": key,
                "true_label": true_label,
                "verdict":    verdict,
                "confidence": round(conf, 4),
                "pred_label": pred_label,
                "reason":     reason[:200],
                "elapsed":    elapsed,
                "error":      error,
            }
            writer.writerow(row)
            raw_file.flush()
            rows.append(row)

            status = "✅ AI" if pred_label == 1 else "✅ Real" if verdict == "REAL" else "⚠️ 불확실"
            print(f"         → {status}  신뢰도 {conf*100:.0f}%  {elapsed:.1f}s")

    finally:
        raw_file.close()

    return rows


# ──────────────────────────────────────────────
# 지표 계산
# ──────────────────────────────────────────────

def calc_metrics(rows: list[dict]) -> dict:
    valid = [r for r in rows if r.get("verdict") != "UNCERTAIN" or float(r.get("confidence", 0)) != 0.5]

    y_true = np.array([int(r["true_label"]) for r in valid])
    y_pred = np.array([int(r["pred_label"]) for r in valid])
    times  = [float(r["elapsed"]) for r in valid]

    if len(set(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    else:
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    accuracy  = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, zero_division=0) * 100
    recall    = recall_score(y_true, y_pred, zero_division=0) * 100
    f1        = f1_score(y_true, y_pred, zero_division=0) * 100
    fpr       = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0
    avg_time  = float(np.mean(times)) if times else 0.0

    uncertain = sum(1 for r in rows if r.get("verdict") == "UNCERTAIN")

    return {
        "model":     MODEL_NAME,
        "accuracy":  round(accuracy, 2),
        "precision": round(precision, 2),
        "recall":    round(recall, 2),
        "f1":        round(f1, 2),
        "fpr":       round(fpr, 2),
        "avg_time":  round(avg_time, 3),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_total":    len(rows),
        "n_valid":    len(valid),
        "n_uncertain":uncertain,
        "n_failed":   sum(1 for r in rows if r.get("error")),
    }


# ──────────────────────────────────────────────
# 결과 출력
# ──────────────────────────────────────────────

def chk(val: float, target: float, lower: bool = False) -> str:
    return "✅" if (val <= target if lower else val >= target) else "❌"


def print_summary(m: dict):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Gemma 4 12B (로컬) — 벤치마크 결과")
    print(f"  측정 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)
    print(f"  처리 이미지  : {m['n_total']}장  (불확실 {m['n_uncertain']}장, 오류 {m['n_failed']}장)")
    print(f"  혼동 행렬    : TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")
    print(sep)
    print(f"  Accuracy   {m['accuracy']:>6.1f}%   {chk(m['accuracy'],  TARGET['accuracy'])}")
    print(f"  Precision  {m['precision']:>6.1f}%   {chk(m['precision'], TARGET['precision'])}")
    print(f"  Recall     {m['recall']:>6.1f}%   {chk(m['recall'],    TARGET['recall'])}")
    print(f"  F1-score   {m['f1']:>6.1f}%   {chk(m['f1'],        TARGET['f1'])}")
    print(f"  FPR        {m['fpr']:>6.1f}%   {chk(m['fpr'],       TARGET['fpr'],  lower=True)}")
    print(f"  Avg Time   {m['avg_time']:>6.1f}s   {chk(m['avg_time'],  TARGET['avg_time'], lower=True)}")
    print(sep)


def save_summary_csv(m: dict, out_path: Path):
    fields = ["model", "accuracy", "precision", "recall", "f1",
              "fpr", "avg_time", "tp", "fp", "fn", "tn",
              "n_total", "n_valid", "n_uncertain", "n_failed"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({k: m[k] for k in fields})
    print(f"  [저장] 요약 CSV: {out_path.name}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gemma 4 12B 로컬 벤치마크")
    parser.add_argument("--sample", type=int, default=None,
                        help="테스트할 이미지 수 (기본: 전체, 예: --sample 50)")
    parser.add_argument("--resume", action="store_true",
                        help="이전 실행에서 중단된 곳부터 재개")
    args = parser.parse_args()

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = RESULT_DIR / f"gemma4_raw_{ts}.csv"
    sum_path = RESULT_DIR / f"benchmark_gemma4_12b_{ts}.csv"

    # 재개 시 최신 raw 파일 사용
    if args.resume:
        raws = sorted(RESULT_DIR.glob("gemma4_raw_*.csv"), reverse=True)
        if raws:
            raw_path = raws[0]
            print(f"  [재개] {raw_path.name} 이어서 실행")
        else:
            print("  재개할 파일이 없습니다. 새로 시작합니다.")

    print("=" * 70)
    print("  Gemma 4 12B 로컬 벤치마크")
    print("=" * 70)
    print(f"  모델      : {MODEL_NAME}")
    print(f"  AI 폴더   : {DATASET_AI_DIR}")
    print(f"  Real 폴더 : {DATASET_REAL_DIR}")
    print(f"  임계값    : {THRESHOLD}")

    # 모델 설치 확인
    if not check_model_installed():
        ans = input(f"\n  {MODEL_NAME} 미설치. 지금 다운로드할까요? (y/N): ").strip().lower()
        if ans == "y":
            pull_model()
        else:
            print("  ollama pull gemma4:12b 를 먼저 실행하세요.")
            sys.exit(1)

    # 데이터셋 로드
    dataset = load_dataset(sample=args.sample)
    n_ai    = sum(1 for _, l in dataset if l == 1)
    n_real  = sum(1 for _, l in dataset if l == 0)
    print(f"  데이터셋  : AI {n_ai}장 / Real {n_real}장  합계 {len(dataset)}장")
    if args.sample:
        print(f"  [샘플 모드] {args.sample}장 기준 비율 유지 샘플링")
    print()

    # 추론 실행
    t_start = time.time()
    rows    = run_inference(dataset, raw_path, resume=args.resume)
    total_t = round(time.time() - t_start, 1)

    if not rows:
        print("처리된 이미지가 없습니다.")
        sys.exit(1)

    # 지표 계산 및 출력
    metrics = calc_metrics(rows)
    print_summary(metrics)
    print(f"  총 소요 시간: {total_t:.0f}초 ({total_t/60:.1f}분)")

    # CSV 저장
    save_summary_csv(metrics, sum_path)
    print(f"  [저장] 상세 CSV: {raw_path.name}")
    print(f"\n  시각화 실행:")
    print(f"  python gemma4_visualization.py --result {sum_path.name} --raw {raw_path.name}")


if __name__ == "__main__":
    main()
