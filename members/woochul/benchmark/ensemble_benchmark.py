"""
InSIGHT — Ensemble Benchmark
EnsembleDetector를 350장 데이터셋에 돌려 6가지 지표를 측정합니다.

실행:
    python ensemble_benchmark.py            # 전체 350장
    python ensemble_benchmark.py --sample 50  # 빠른 테스트

결과:
    ensemble_results_<timestamp>.csv
"""

import os
import csv
import sys
import time
import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

from ensemble_detector import EnsembleDetector, DEPLOY_MODEL, SDXL_MODEL

# ── 경로 설정 ───────────────────────────────────────────────────
_ROOT            = Path(__file__).resolve().parents[3]
DATASET_AI_DIR   = _ROOT / "data" / "test_dataset" / "ai"
DATASET_REAL_DIR = _ROOT / "data" / "test_dataset" / "real"
RESULT_DIR       = Path(__file__).resolve().parent
IMG_EXTS         = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ── 목표치 ──────────────────────────────────────────────────────
TARGET = {
    "accuracy":  70.0,
    "precision": 75.0,
    "recall":    75.0,
    "f1":        70.0,
    "fpr":       20.0,
    "avg_time":  3.0,
}

# ── 단일 모델 기준값 (비교용) ────────────────────────────────────
BASELINE = {
    "deploy":    {"accuracy": 88.25, "precision": 75.86, "recall": 87.13,
                  "f1": 81.11, "fpr": 11.29, "avg_time": 0.828},
    "sdxl":      {"accuracy": 77.08, "precision": 59.13, "recall": 67.33,
                  "f1": 62.96, "fpr": 18.95, "avg_time": 0.449},
}


# ── 데이터셋 로드 ────────────────────────────────────────────────

def load_dataset(sample: int | None = None) -> list[tuple[Path, int]]:
    dataset: list[tuple[Path, int]] = []
    for p in sorted(DATASET_AI_DIR.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            dataset.append((p, 1))
    for p in sorted(DATASET_REAL_DIR.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            dataset.append((p, 0))

    if sample and sample < len(dataset):
        ai   = [(p, l) for p, l in dataset if l == 1]
        real = [(p, l) for p, l in dataset if l == 0]
        ratio = len(ai) / len(dataset)
        n_ai  = round(sample * ratio)
        n_real = sample - n_ai
        random.seed(42)
        dataset = random.sample(ai, min(n_ai, len(ai))) + \
                  random.sample(real, min(n_real, len(real)))
    return dataset


# ── 추론 루프 ────────────────────────────────────────────────────

def run_benchmark(
    detector: EnsembleDetector,
    dataset: list[tuple[Path, int]],
) -> tuple[list[int], list[int], list[float], int]:

    y_true, y_pred, times = [], [], []
    fail_count = 0
    n = len(dataset)

    for idx, (img_path, true_label) in enumerate(dataset, 1):
        try:
            score, pred, elapsed = detector.predict_path(img_path)
            y_true.append(true_label)
            y_pred.append(pred)
            times.append(elapsed)

            tag = "AI" if pred == 1 else "Real"
            ok  = "✅" if pred == true_label else "❌"
            print(f"  [{idx:>3}/{n}] {ok} pred={tag:<4}  score={score:.3f}  {elapsed:.2f}s  {img_path.name}")

        except Exception as exc:
            fail_count += 1
            print(f"  [{idx:>3}/{n}] ⚠️  실패: {img_path.name} — {exc}")

    return y_true, y_pred, times, fail_count


# ── 지표 계산 ────────────────────────────────────────────────────

def calc_metrics(
    y_true: list[int],
    y_pred: list[int],
    times: list[float],
    fail_count: int,
    model_name: str = "Ensemble(deploy+sdxl)",
) -> dict:

    yt = np.array(y_true)
    yp = np.array(y_pred)

    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()

    return {
        "model":     model_name,
        "accuracy":  round(accuracy_score(yt, yp) * 100, 2),
        "precision": round(precision_score(yt, yp, zero_division=0) * 100, 2),
        "recall":    round(recall_score(yt, yp, zero_division=0) * 100, 2),
        "f1":        round(f1_score(yt, yp, zero_division=0) * 100, 2),
        "fpr":       round((fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0, 2),
        "avg_time":  round(float(np.mean(times)) if times else 0.0, 3),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_total":  len(y_true),
        "n_failed": fail_count,
    }


# ── 결과 출력 ────────────────────────────────────────────────────

def chk(val: float, target: float, lower: bool = False) -> str:
    return "✅" if (val <= target if lower else val >= target) else "❌"


def print_summary(m: dict) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  InSIGHT — Ensemble Benchmark 결과")
    print(f"  측정 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)
    print(f"  모델      : {m['model']}")
    print(f"  처리      : {m['n_total']}장  (실패 {m['n_failed']}장)")
    print(f"  혼동 행렬 : TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")
    print(sep)

    metrics = [
        ("Accuracy",   "accuracy",  False, "%"),
        ("Precision",  "precision", False, "%"),
        ("Recall ★",   "recall",    False, "%"),
        ("F1-score",   "f1",        False, "%"),
        ("FPR",        "fpr",       True,  "%"),
        ("Avg Time",   "avg_time",  True,  "s"),
    ]
    for label, key, lower, unit in metrics:
        val = m[key]
        tgt = TARGET[key]
        b_deploy = BASELINE["deploy"][key]
        b_sdxl   = BASELINE["sdxl"][key]
        diff     = val - b_deploy
        sign     = f"({diff:+.1f} vs deploy)"
        print(f"  {label:<12}: {val:>6.1f}{unit}  {chk(val, tgt, lower)}  target={tgt}{unit}  "
              f"deploy={b_deploy}{unit}  sdxl={b_sdxl}{unit}  {sign}")

    pass_count = sum(
        1 for _, key, lower, _ in metrics
        if (m[key] <= TARGET[key] if lower else m[key] >= TARGET[key])
    )
    print(sep)
    print(f"  목표 달성: {pass_count}/6")
    print(sep)


# ── CSV 저장 ──────────────────────────────────────────────────────

def save_csv(m: dict, out_path: Path) -> None:
    fields = ["model", "accuracy", "precision", "recall", "f1",
              "fpr", "avg_time", "tp", "fp", "fn", "tn", "n_total", "n_failed"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({k: m[k] for k in fields})
    print(f"\n  [저장] {out_path.name}")


# ── 메인 ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble Benchmark")
    parser.add_argument("--sample", type=int, default=None,
                        help="테스트 이미지 수 (기본: 전체)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="AI 판정 임계값 (기본: 0.5)")
    parser.add_argument("--w-deploy", type=float, default=0.6,
                        help="deploy 모델 가중치 (기본: 0.6)")
    parser.add_argument("--w-sdxl",   type=float, default=0.4,
                        help="sdxl 모델 가중치 (기본: 0.4)")
    args = parser.parse_args()

    weights = {DEPLOY_MODEL: args.w_deploy, SDXL_MODEL: args.w_sdxl}

    # 데이터셋 로드
    try:
        dataset = load_dataset(sample=args.sample)
    except FileNotFoundError as exc:
        print(f"\n❌ {exc}")
        print("DATASET_AI_DIR / DATASET_REAL_DIR 경로를 확인해주세요.")
        sys.exit(1)

    n_ai   = sum(1 for _, l in dataset if l == 1)
    n_real = sum(1 for _, l in dataset if l == 0)

    print("=" * 65)
    print("  InSIGHT — Ensemble Benchmark")
    print("=" * 65)
    print(f"  모델      : deploy(w={args.w_deploy}) + sdxl(w={args.w_sdxl})")
    print(f"  Threshold : {args.threshold}")
    print(f"  데이터셋  : AI {n_ai}장 / Real {n_real}장  합계 {len(dataset)}장")
    if args.sample:
        print(f"  [샘플 모드] {args.sample}장 비율 유지 샘플링")

    # 앙상블 모델 로드
    detector = EnsembleDetector(weights=weights, threshold=args.threshold)

    # 추론 실행
    t_start = time.time()
    y_true, y_pred, times, fail_count = run_benchmark(detector, dataset)
    total_t = round(time.time() - t_start, 1)

    if not y_true:
        print("처리된 이미지가 없습니다.")
        sys.exit(1)

    # 지표 계산 & 출력
    model_name = f"Ensemble(deploy w={args.w_deploy} + sdxl w={args.w_sdxl}, thr={args.threshold})"
    metrics    = calc_metrics(y_true, y_pred, times, fail_count, model_name)
    print_summary(metrics)
    print(f"  총 소요 시간: {total_t:.0f}초 ({total_t/60:.1f}분)")

    # CSV 저장
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULT_DIR / f"ensemble_results_{ts}.csv"
    save_csv(metrics, csv_path)

    print(f"\n  시각화 실행:")
    print(f"  노트북 ensemble_report.ipynb 에서 {csv_path.name} 로드")


if __name__ == "__main__":
    main()
