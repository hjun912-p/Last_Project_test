"""
Gemma 4 12B 벤치마크 시각화
=============================
기존 ViT 모델 결과 + Gemma 4 12B 결과를 합쳐 비교 차트 생성

실행:
    python gemma4_visualization.py
    python gemma4_visualization.py --result benchmark_gemma4_12b_xxx.csv --raw gemma4_raw_xxx.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

matplotlib.rcParams["font.family"] = ["AppleGothic", "Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ──────────────────────────────────────────────
# 경로
# ──────────────────────────────────────────────
RESULT_DIR = Path(__file__).resolve().parent
CHARTS_DIR = RESULT_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

# 기존 ViT 벤치마크 결과
EXISTING_CSV = RESULT_DIR / "benchmark_results_20260426_184945.csv"

# 목표치
TARGET = {
    "accuracy":  70.0,
    "precision": 75.0,
    "recall":    75.0,
    "f1":        70.0,
    "fpr":       20.0,
}

# 색상 팔레트
COLORS = {
    "gemma4":     "#4CAF50",   # 초록 — Gemma 4 강조
    "vit_best":   "#2196F3",   # 파랑 — 기존 최고 모델
    "vit_others": "#90CAF9",   # 연파랑 — 기타 ViT
    "target":     "#FF5722",   # 주황 — 목표선
    "bg":         "#FAFAFA",
}

SHORT_NAMES = {
    "umm-maybe/AI-image-detector":                  "umm-maybe",
    "Organika/sdxl-detector":                       "SDXL-detector",
    "haywoodsloan/ai-image-detector-deploy":        "haywoodsloan ★",
    "dima806/ai-generated-vs-real-image-detection": "dima806",
    "ideepankarsharma2003/AI_Image_Classifier":     "ideepankarsharma",
    "prithivMLmods/Deep-Fake-Detector-Model":       "DeepFake-Detector",
    "gemma4:12b":                                   "Gemma 4 12B ★",
}


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_raw(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def as_float(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return 0.0


def get_display_name(model_id: str) -> str:
    for k, v in SHORT_NAMES.items():
        if k in model_id:
            return v
    return model_id.split("/")[-1][:24]


# ──────────────────────────────────────────────
# 차트 1 — 핵심 지표 막대 그래프 (Gemma 4 포함)
# ──────────────────────────────────────────────

def chart_core_metrics(all_rows: list[dict]):
    metrics   = ["accuracy", "precision", "recall", "f1"]
    titles    = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-score (%)"]
    target_v  = [TARGET["accuracy"], TARGET["precision"], TARGET["recall"], TARGET["f1"]]

    names  = [get_display_name(r["model"]) for r in all_rows]
    colors = [COLORS["gemma4"] if "gemma4" in r["model"].lower() else
              COLORS["vit_best"] if "haywoodsloan" in r["model"] else
              COLORS["vit_others"] for r in all_rows]

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Gemma 4 12B vs ViT 기반 모델 — 핵심 지표 비교", fontsize=15, fontweight="bold", y=1.02)

    for ax, metric, title, tv in zip(axes, metrics, titles, target_v):
        vals = [as_float(r, metric) for r in all_rows]
        bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=0.8)

        ax.axhline(y=tv, color=COLORS["target"], linestyle="--", linewidth=1.5,
                   label=f"목표 {tv:.0f}%")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.set_ylabel("%")
        ax.tick_params(axis="x", rotation=35, labelsize=9)
        ax.legend(fontsize=8)
        ax.set_facecolor(COLORS["bg"])

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=COLORS["gemma4"],     label="Gemma 4 12B (LLM)"),
        mpatches.Patch(color=COLORS["vit_best"],   label="최고 ViT 모델"),
        mpatches.Patch(color=COLORS["vit_others"], label="기타 ViT 모델"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.06), fontsize=10)

    fig.tight_layout()
    out = CHARTS_DIR / "gemma4_core_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  [차트] {out.name}")


# ──────────────────────────────────────────────
# 차트 2 — Gemma 4 혼동 행렬
# ──────────────────────────────────────────────

def chart_confusion_gemma4(gemma4_row: dict):
    tp = int(as_float(gemma4_row, "tp"))
    fp = int(as_float(gemma4_row, "fp"))
    fn = int(as_float(gemma4_row, "fn"))
    tn = int(as_float(gemma4_row, "tn"))

    cm    = np.array([[tn, fp], [fn, tp]])
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    im = ax.imshow(cm, cmap="Blues")

    labels_x = ["예측: Real", "예측: AI"]
    labels_y = ["실제: Real", "실제: AI"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels_x, fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels_y, fontsize=11)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / total * 100
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                    color=color, fontsize=13, fontweight="bold")

    ax.set_title(f"Gemma 4 12B — 혼동 행렬\n(N={total})", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    acc = (tp + tn) / total * 100
    f1  = as_float(gemma4_row, "f1")
    fig.text(0.5, -0.04,
             f"Accuracy {acc:.1f}%  |  F1 {f1:.1f}%  |  TP={tp} FP={fp} FN={fn} TN={tn}",
             ha="center", fontsize=10, color="#555")

    fig.tight_layout()
    out = CHARTS_DIR / "gemma4_confusion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  [차트] {out.name}")


# ──────────────────────────────────────────────
# 차트 3 — FPR / 속도 산점도
# ──────────────────────────────────────────────

def chart_fpr_time(all_rows: list[dict]):
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(COLORS["bg"])

    for row in all_rows:
        name  = get_display_name(row["model"])
        fpr   = as_float(row, "fpr")
        t     = as_float(row, "avg_time")
        f1    = as_float(row, "f1")
        is_g4 = "gemma4" in row["model"].lower()

        color = COLORS["gemma4"] if is_g4 else COLORS["vit_best"] if "haywoodsloan" in row["model"] else COLORS["vit_others"]
        size  = 220 if is_g4 else 120

        ax.scatter(fpr, t, s=size, color=color, zorder=3, edgecolors="white", linewidth=1.5)
        ax.annotate(f"{name}\nF1={f1:.1f}%", (fpr, t),
                    textcoords="offset points", xytext=(8, 4), fontsize=8.5)

    ax.axvline(x=TARGET["fpr"], color=COLORS["target"], linestyle="--",
               linewidth=1.5, label=f"FPR 목표 ≤{TARGET['fpr']:.0f}%")
    ax.set_xlabel("FPR — False Positive Rate (낮을수록 좋음) %", fontsize=11)
    ax.set_ylabel("이미지당 평균 추론 시간 (초)", fontsize=11)
    ax.set_title("FPR vs 추론 속도 (좌하단이 최적)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_facecolor(COLORS["bg"])
    ax.grid(True, alpha=0.3)

    out = CHARTS_DIR / "gemma4_fpr_time.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  [차트] {out.name}")


# ──────────────────────────────────────────────
# 차트 4 — 레이더 차트 (Gemma 4 vs 기존 최고)
# ──────────────────────────────────────────────

def chart_radar(gemma4_row: dict, best_vit_row: dict):
    categories = ["Accuracy", "Precision", "Recall", "F1", "FPR 역전\n(100-FPR)"]
    N = len(categories)

    def row_to_vals(r: dict) -> list[float]:
        return [
            as_float(r, "accuracy"),
            as_float(r, "precision"),
            as_float(r, "recall"),
            as_float(r, "f1"),
            100 - as_float(r, "fpr"),
        ]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(COLORS["bg"])

    for row, color, label in [
        (gemma4_row,  COLORS["gemma4"],   f"Gemma 4 12B"),
        (best_vit_row,COLORS["vit_best"], f"haywoodsloan (최고 ViT)"),
    ]:
        vals = row_to_vals(row) + row_to_vals(row)[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=label)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7)
    ax.set_title("Gemma 4 12B vs 기존 최고 ViT — 레이더 비교", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=10)

    out = CHARTS_DIR / "gemma4_radar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  [차트] {out.name}")


# ──────────────────────────────────────────────
# 차트 5 — 신뢰도 분포 (raw 데이터 필요)
# ──────────────────────────────────────────────

def chart_confidence_dist(raw_rows: list[dict]):
    if not raw_rows:
        print("  [스킵] 상세 원본 데이터 없음 — 신뢰도 분포 차트 생략")
        return

    ai_conf_correct   = []
    ai_conf_wrong     = []
    real_conf_correct = []
    real_conf_wrong   = []

    for r in raw_rows:
        true_label = int(r.get("true_label", -1))
        pred_label = int(r.get("pred_label", -1))
        conf       = float(r.get("confidence", 0.5))
        if true_label == -1 or pred_label == -1:
            continue

        if true_label == 1:   # 실제 AI
            (ai_conf_correct if pred_label == 1 else ai_conf_wrong).append(conf)
        else:                  # 실제 Real
            (real_conf_correct if pred_label == 0 else real_conf_wrong).append(conf)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Gemma 4 12B — 신뢰도 분포 (정답 vs 오답)", fontsize=13, fontweight="bold")

    for ax, correct, wrong, title in [
        (axes[0], ai_conf_correct,   ai_conf_wrong,   "실제 AI 이미지"),
        (axes[1], real_conf_correct, real_conf_wrong, "실제 Real 이미지"),
    ]:
        bins = np.linspace(0, 1, 21)
        ax.hist(correct, bins=bins, alpha=0.7, color=COLORS["gemma4"],   label=f"정답 ({len(correct)}장)")
        ax.hist(wrong,   bins=bins, alpha=0.7, color="#EF5350",          label=f"오답 ({len(wrong)}장)")
        ax.axvline(x=0.5, color=COLORS["target"], linestyle="--", linewidth=1.5, label="임계값 0.5")
        ax.set_xlabel("모델 신뢰도", fontsize=10)
        ax.set_ylabel("이미지 수", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_facecolor(COLORS["bg"])

    fig.tight_layout()
    out = CHARTS_DIR / "gemma4_confidence_dist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  [차트] {out.name}")


# ──────────────────────────────────────────────
# 차트 6 — 종합 요약 테이블
# ──────────────────────────────────────────────

def chart_summary_table(all_rows: list[dict]):
    fig, ax = plt.subplots(figsize=(14, len(all_rows) * 0.7 + 2))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.axis("off")

    headers = ["모델", "Accuracy", "Precision", "Recall", "F1", "FPR", "Avg Time"]
    col_widths = [0.32, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]

    table_data = []
    for r in all_rows:
        table_data.append([
            get_display_name(r["model"]),
            f"{as_float(r, 'accuracy'):.1f}%",
            f"{as_float(r, 'precision'):.1f}%",
            f"{as_float(r, 'recall'):.1f}%",
            f"{as_float(r, 'f1'):.1f}%",
            f"{as_float(r, 'fpr'):.1f}%",
            f"{as_float(r, 'avg_time'):.2f}s",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # 헤더 스타일
    for j in range(len(headers)):
        table[(0, j)].set_facecolor("#37474F")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Gemma 4 행 강조
    for i, row in enumerate(all_rows, 1):
        if "gemma4" in row["model"].lower():
            for j in range(len(headers)):
                table[(i, j)].set_facecolor("#E8F5E9")
                table[(i, j)].set_text_props(fontweight="bold")

    ax.set_title("Gemma 4 12B vs ViT 모델 — 전체 성능 요약",
                 fontsize=13, fontweight="bold", pad=20)

    out = CHARTS_DIR / "gemma4_summary_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  [차트] {out.name}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=str, default=None,
                        help="Gemma 4 요약 CSV 파일명 (없으면 최신 자동 탐색)")
    parser.add_argument("--raw", type=str, default=None,
                        help="Gemma 4 상세 raw CSV 파일명 (신뢰도 분포 차트용)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Gemma 4 12B 벤치마크 시각화")
    print("=" * 60)

    # Gemma 4 요약 CSV 탐색
    if args.result:
        g4_path = RESULT_DIR / args.result
    else:
        candidates = sorted(RESULT_DIR.glob("benchmark_gemma4_12b_*.csv"), reverse=True)
        if not candidates:
            print("  Gemma 4 결과 CSV를 찾을 수 없습니다.")
            print("  먼저 gemma4_benchmark.py 를 실행하세요.")
            return
        g4_path = candidates[0]
        print(f"  [자동 탐색] {g4_path.name}")

    g4_rows = load_csv(g4_path)
    if not g4_rows:
        print(f"  CSV 데이터가 비어 있습니다: {g4_path}")
        return

    # 기존 ViT 결과 로드
    vit_rows = load_csv(EXISTING_CSV)
    if not vit_rows:
        print(f"  기존 벤치마크 CSV 없음: {EXISTING_CSV}")
        print("  Gemma 4 단독 차트만 생성합니다.")

    all_rows = vit_rows + g4_rows

    # raw 데이터 로드
    raw_path = None
    if args.raw:
        raw_path = RESULT_DIR / args.raw
    else:
        raws = sorted(RESULT_DIR.glob("gemma4_raw_*.csv"), reverse=True)
        if raws:
            raw_path = raws[0]
    raw_rows = load_raw(raw_path)

    # 기존 최고 ViT 모델
    best_vit = max(vit_rows, key=lambda r: as_float(r, "f1")) if vit_rows else None
    gemma4   = g4_rows[0]

    print(f"\n  차트 생성 중...")
    chart_core_metrics(all_rows)
    chart_confusion_gemma4(gemma4)
    chart_fpr_time(all_rows)
    if best_vit:
        chart_radar(gemma4, best_vit)
    chart_confidence_dist(raw_rows)
    chart_summary_table(all_rows)

    print(f"\n  모든 차트 저장 완료: {CHARTS_DIR}/")
    print("  생성 파일:")
    for f in sorted(CHARTS_DIR.glob("gemma4_*.png")):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
