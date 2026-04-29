"""
Vertex AI SynthID 1차 필터 검증 실험
- 성능 측정: Accuracy / Precision / Recall / F1 / FPR
- 비용 측정: 장당 비용 / 총 비용
- 속도 측정: 장당 평균 추론 시간

사용법:
    python synthid_vertex_benchmark.py \
        --ai_dir  /path/to/ai_images \
        --real_dir /path/to/real_images \
        --max 50
"""

import os
import sys
import csv
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# 프로젝트 루트를 경로에 추가
ROOT = Path(__file__).resolve().parents[1]  # members/woochul/
sys.path.insert(0, str(ROOT))

from synthid_vertex import detect_synthid_vertex, COST_PER_IMAGE_USD, VERTEX_AVAILABLE

# ──────────────────────────────────────────────
# 채택 기준 (Track B 목표치)
# ──────────────────────────────────────────────
TARGET_RECALL   = 75.0   # %
TARGET_COST     = 0.01   # USD / 장
TARGET_TIME     = 3.0    # 초 / 장

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_images(directory: str, max_n: int) -> list[Path]:
    d = Path(directory)
    if not d.exists():
        print(f"[!] 폴더 없음: {directory}")
        return []
    files = [f for f in d.iterdir() if f.suffix.lower() in IMG_EXTS]
    files.sort()
    return files[:max_n]


def run_benchmark(ai_dir: str, real_dir: str, max_per_class: int = 50):
    if not VERTEX_AVAILABLE:
        print("[ERROR] google-cloud-aiplatform 미설치")
        print("        pip install google-cloud-aiplatform")
        return

    print("=" * 60)
    print("InSIGHT — Vertex AI SynthID 1차 필터 검증 실험")
    print(f"실험 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    ai_files   = load_images(ai_dir,   max_per_class)
    real_files = load_images(real_dir, max_per_class)

    if not ai_files and not real_files:
        print("[!] 이미지 파일이 없습니다.")
        print(f"    AI 이미지 폴더  : {ai_dir}")
        print(f"    실제 이미지 폴더: {real_dir}")
        return

    print(f"\nAI 이미지  : {len(ai_files)}장  ({ai_dir})")
    print(f"실제 이미지: {len(real_files)}장  ({real_dir})")

    dataset = [(f, 1) for f in ai_files] + [(f, 0) for f in real_files]

    y_true, y_pred = [], []
    times, costs   = [], []
    rows = []  # CSV용

    for i, (img_path, true_label) in enumerate(dataset):
        label_str = "AI" if true_label == 1 else "실제"
        print(f"\n[{i+1:3d}/{len(dataset)}] {img_path.name}  (정답: {label_str})")

        try:
            pil = Image.open(img_path).convert("RGB")
            result, msg, elapsed, cost = detect_synthid_vertex(pil)

            pred_label = 1 if result is True else 0
            correct    = "✅" if pred_label == true_label else "❌"

            print(f"         {msg.splitlines()[0]}  {correct}")

            y_true.append(true_label)
            y_pred.append(pred_label)
            times.append(elapsed)
            costs.append(cost)

            rows.append({
                "file":       img_path.name,
                "true_label": true_label,
                "pred_label": pred_label,
                "result":     str(result),
                "elapsed":    round(elapsed, 3),
                "cost_usd":   cost,
                "message":    msg.splitlines()[0],
            })

        except Exception as e:
            print(f"         [SKIP] 오류: {e}")
            continue

    if not y_true:
        print("\n[!] 처리된 이미지가 없습니다.")
        return

    # ── 지표 계산 ──────────────────────────────────────────
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(set(y_true)) > 1:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    else:
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    accuracy  = accuracy_score(y_true, y_pred)  * 100
    precision = precision_score(y_true, y_pred, zero_division=0) * 100
    recall    = recall_score(y_true, y_pred, zero_division=0) * 100
    f1        = f1_score(y_true, y_pred, zero_division=0) * 100
    fpr       = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0
    avg_time  = float(np.mean(times)) if times else 0.0
    total_cost = float(np.sum(costs))
    avg_cost   = float(np.mean(costs)) if costs else 0.0

    def chk(val, target, lower=False):
        return "✅" if (val <= target if lower else val >= target) else "❌"

    # ── 결과 출력 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[ 혼동 행렬 ]")
    print(f"  TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")

    print("\n[ 성능 지표 ]")
    print(f"  Accuracy       : {accuracy:.2f}%  {chk(accuracy,  70)}  (목표 70%+)")
    print(f"  Precision      : {precision:.2f}%  {chk(precision, 75)}  (목표 75%+)")
    print(f"  Recall         : {recall:.2f}%  {chk(recall, TARGET_RECALL)}  (목표 {TARGET_RECALL}%+) ★핵심")
    print(f"  F1 Score       : {f1:.2f}%  {chk(f1,        70)}  (목표 70%+)")
    print(f"  FPR            : {fpr:.2f}%  {chk(fpr,       20, lower=True)}  (목표 20%-)")

    print("\n[ 비용 · 속도 ]")
    print(f"  장당 평균 비용  : ${avg_cost:.4f}  {chk(avg_cost,  TARGET_COST,  lower=True)}  (목표 ${TARGET_COST})")
    print(f"  총 실험 비용    : ${total_cost:.4f}  (₩{total_cost*1400:,.0f} 환산)")
    print(f"  장당 평균 시간  : {avg_time:.2f}초  {chk(avg_time,  TARGET_TIME,  lower=True)}  (목표 {TARGET_TIME}초-)")

    adopted = all([
        recall    >= TARGET_RECALL,
        avg_cost  <= TARGET_COST,
        avg_time  <= TARGET_TIME,
    ])
    print("\n[ 1차 필터 채택 판정 ]")
    print(f"  {'✅ 채택 권고 — 1차 필터로 통합 가능' if adopted else '❌ 미채택 — 팀원 대안 방법 검토 필요'}")

    print("\n[ 오류 케이스 ]")
    print(f"  FP (실제→AI 오탐): {fp}건  — 실제 이미지를 AI로 잘못 판정")
    print(f"  FN (AI→실제 미탐): {fn}건  — AI 이미지를 놓침 ← 더 위험")
    print("=" * 60)

    # ── CSV 저장 ───────────────────────────────────────────
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent
    csv_path = out_dir / f"synthid_vertex_results_{ts}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # 요약 행 추가
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        f.write("\n요약\n")
        f.write(f"Accuracy,{accuracy:.2f}%\n")
        f.write(f"Precision,{precision:.2f}%\n")
        f.write(f"Recall,{recall:.2f}%\n")
        f.write(f"F1,{f1:.2f}%\n")
        f.write(f"FPR,{fpr:.2f}%\n")
        f.write(f"avg_time,{avg_time:.3f}초\n")
        f.write(f"avg_cost_usd,${avg_cost:.4f}\n")
        f.write(f"total_cost_usd,${total_cost:.4f}\n")
        f.write(f"채택,{'예' if adopted else '아니오'}\n")

    print(f"\n결과 저장: {csv_path}")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vertex AI SynthID 벤치마크")
    parser.add_argument("--ai_dir",   default=str(ROOT / "test_dataset" / "ai"),
                        help="AI 생성 이미지 폴더")
    parser.add_argument("--real_dir", default=str(ROOT / "test_dataset" / "real"),
                        help="실제 이미지 폴더")
    parser.add_argument("--max",      type=int, default=50,
                        help="클래스당 최대 이미지 수 (기본 50)")
    args = parser.parse_args()

    run_benchmark(args.ai_dir, args.real_dir, args.max)
