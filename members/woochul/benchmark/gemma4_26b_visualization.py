"""
Gemma 4 26B 벤치마크 시각화
=============================
raw CSV에서 지표를 직접 계산 후 12B · ViT 모델과 비교 차트 생성

실행:
    python gemma4_26b_visualization.py
    python gemma4_26b_visualization.py --raw gemma4_26b_raw_xxx.csv
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

RESULT_DIR = Path(__file__).resolve().parent
CHARTS_DIR = RESULT_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

EXISTING_CSV   = RESULT_DIR / "benchmark_results_20260426_184945.csv"
GEMMA4_12B_CSV = RESULT_DIR / "benchmark_gemma4_12b_20260508_131040.csv"

TARGET = {
    "accuracy":  70.0,
    "precision": 75.0,
    "recall":    75.0,
    "f1":        70.0,
    "fpr":       20.0,
}

COLORS = {
    "gemma4_26b":  "#FF6F00",   # 주황 — 26B 강조
    "gemma4_12b":  "#4CAF50",   # 초록 — 12B
    "vit_best":    "#2196F3",   # 파랑 — 기존 최고 ViT
    "vit_others":  "#90CAF9",   # 연파랑 — 기타 ViT
    "target":      "#E53935",   # 빨강 — 목표선
    "bg":          "#FAFAFA",
}

SHORT_NAMES = {
    "umm-maybe/AI-image-detector":                  "umm-maybe",
    "Organika/sdxl-detector":                       "SDXL-detector",
    "haywoodsloan/ai-image-detector-deploy":        "haywoodsloan ★",
    "dima806/ai-generated-vs-real-image-detection": "dima806",
    "ideepankarsharma2003/AI_Image_Classifier":     "ideepankarsharma",
    "prithivMLmods/Deep-Fake-Detector-Model":       "DeepFake-Detector",
    "gemma4:e4b":                                   "Gemma 4 12B",
    "gemma4:26b":                                   "Gemma 4 26B ★",
}


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    if not path.exists():
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


def compute_metrics_from_raw(raw_rows: list[dict]) -> dict:
    """raw CSV에서 지표를 직접 계산해 요약 dict 반환"""
    valid = [r for r in raw_rows if r.get("verdict", "") != "UNCERTAIN"]
    if not valid:
        valid = raw_rows

    y_true = np.array([int(r["true_label"]) for r in valid])
    y_pred = np.array([int(r["pred_label"]) for r in valid])
    times  = [float(r["elapsed"]) for r in valid if r.get("elapsed")]

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    total = tp + fp + fn + tn
    accuracy  = (tp + tn) / total * 100 if total else 0
    precision = tp / (tp + fp) * 100   if (tp + fp) else 0
    recall    = tp / (tp + fn) * 100   if (tp + fn) else 0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    fpr       = fp / (fp + tn) * 100   if (fp + tn) else 0
    avg_time  = float(np.mean(times))  if times else 0.0

    return {
        "model":    "gemma4:26b",
        "accuracy": round(accuracy, 2),
        "precision":round(precision, 2),
        "recall":   round(recall, 2),
        "f1":       round(f1, 2),
        "fpr":      round(fpr, 2),
        "avg_time": round(avg_time, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_total": len(raw_rows),
    }


# ──────────────────────────────────────────────
# 차트 1 — 핵심 지표 그룹 막대 (전 모델 비교)
# ──────────────────────────────────────────────

def chart_core_metrics(all_rows: list[dict]):
    metrics  = ["accuracy", "precision", "recall", "f1"]
    titles   = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-score (%)"]
    targets  = [TARGET["accuracy"], TARGET["precision"], TARGET["recall"], TARGET["f1"]]

    names  = [get_display_name(r["model"]) for r in all_rows]
    colors = []
    for r in all_rows:
        m = r["model"].lower()
        if "26b" in m:
            colors.append(COLORS["gemma4_26b"])
        elif "gemma4" in m or "e4b" in m:
            colors.append(COLORS["gemma4_12b"])
        elif "haywoodsloan" in m:
            colors.append(COLORS["vit_best"])
        else:
            colors.append(COLORS["vit_others"])

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Gemma 4 26B vs 12B vs ViT 모델 — 핵심 지표 비교",
                 fontsize=15, fontweight="bold", y=1.02)

    for ax, metric, title, tv in zip(axes, metrics, titles, targets):
        vals = [as_float(r, metric) for r in all_rows]
        bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax.axhline(y=tv, color=COLORS["target"], linestyle="--",
                   linewidth=1.5, label=f"목표 {tv:.0f}%")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 115)
        ax.set_ylabel("%")
        ax.tick_params(axis="x", rotation=38, labelsize=8.5)
        ax.legend(fontsize=8)
        ax.set_facecolor(COLORS["bg"])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color=COLORS["gemma4_26b"], label="Gemma 4 26B (LLM)"),
        mpatches.Patch(color=COLORS["gemma4_12b"], label="Gemma 4 12B (LLM)"),
        mpatches.Patch(color=COLORS["vit_best"],   label="최고 ViT 모델"),
        mpatches.Patch(color=COLORS["vit_others"], label="기타 ViT 모델"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.07), fontsize=10)
    fig.tight_layout()

    out = CHARTS_DIR / "gemma4_26b_core_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  [차트] {out.name}")


# ──────────────────────────────────────────────
# 차트 2 — 혼동 행렬 (26B)
# ──────────────────────────────────────────────

def chart_confusion(row: dict, model_label: str, filename: str):
    tp = int(as_float(row, "tp"))
    fp = int(as_float(row, "fp"))
    fn = int(as_float(row, "fn"))
    tn = int(as_float(row, "tn"))

    cm    = np.array([[tn, fp], [fn, tp]])
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    im = ax.imshow(cm, cmap="Oranges")

    ax.set_xticks([0, 1]); ax.set_xticklabels(["예측: Real", "예측: AI"], fontsize=11)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["실제: Real", "실제: AI"], fontsize=11)

    for i in range(2):
        for j in range(2):
            val   = cm[i, j]
            pct   = val / total * 100
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                    color=color, fontsize=13, fontweight="bold")

    ax.set_title(f"{model_label} — 혼동 행렬\n(N={total})",
                 fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    acc = (tp + tn) / total * 100
    f1  = as_float(row, "f1")
    fig.text(0.5, -0.04,
             f"Accuracy {acc:.1f}%  |  F1 {f1:.1f}%  |  TP={tp} FP={fp} FN={fn} TN={tn}",
             ha="center", fontsize=10, color="#555")

    fig.tight_layout()
    out = CHARTS_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  [차트] {out.name}")


# ──────────────────────────────────────────────
# 차트 3 — 12B vs 26B 나란히 혼동 행렬
# ──────────────────────────────────────────────

def chart_confusion_compare(row_12b: dict, row_26b: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Gemma 4 12B vs 26B — 혼동 행렬 비교",
                 fontsize=14, fontweight="bold")

    for ax, row, label, cmap in [
        (axes[0], row_12b, "Gemma 4 12B", "Greens"),
        (axes[1], row_26b, "Gemma 4 26B", "Oranges"),
    ]:
        tp = int(as_float(row, "tp"))
        fp = int(as_float(row, "fp"))
        fn = int(as_float(row, "fn"))
        tn = int(as_float(row, "tn"))
        cm    = np.array([[tn, fp], [fn, tp]])
        total = cm.sum()

        im = ax.imshow(cm, cmap=cmap)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["예측: Real", "예측: AI"], fontsize=10)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["실제: Real", "실제: AI"], fontsize=10)

        for i in range(2):
            for j in range(2):
                val   = cm[i, j]
                pct   = val / total * 100
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                        color=color, fontsize=12, fontweight="bold")

        acc = (tp + tn) / total * 100
        f1  = as_float(row, "f1")
        ax.set_title(f"{label}\nAcc={acc:.1f}%  F1={f1:.1f}%",
                     fontsize=12, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out = CHARTS_DIR / "gemma4_26b_confusion_compare.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  [차트] {out.name}")


# ──────────────────────────────────────────────
# 차트 4 — FPR vs 속도 산점도
# ──────────────────────────────────────────────

def chart_fpr_time(all_rows: list[dict]):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS["bg"])

    for row in all_rows:
        name = get_display_name(row["model"])
        fpr  = as_float(row, "fpr")
        t    = as_float(row, "avg_time")
        f1   = as_float(row, "f1")
        m    = row["model"].lower()

        if "26b" in m:
            color, size = COLORS["gemma4_26b"], 280
        elif "gemma4" in m or "e4b" in m:
            color, size = COLORS["gemma4_12b"], 220
        elif "haywoodsloan" in m:
            color, size = COLORS["vit_best"], 120
        else:
            color, size = COLORS["vit_others"], 90

        ax.scatter(fpr, t, s=size, color=color, zorder=3,
                   edgecolors="white", linewidth=1.5)
        ax.annotate(f"{name}\nF1={f1:.1f}%", (fpr, t),
                    textcoords="offset points", xytext=(8, 4), fontsize=8.5)

    ax.axvline(x=TARGET["fpr"], color=COLORS["target"], linestyle="--",
               linewidth=1.5, label=f"FPR 목표 ≤{TARGET['fpr']:.0f}%")
    ax.set_xlabel("FPR — False Positive Rate (낮을수록 좋음) %", fontsize=11)
    ax.set_ylabel("이미지당 평균 추론 시간 (초)", fontsize=11)
    ax.set_title("FPR vs 추론 속도 — 좌하단이 최적", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_facecolor(COLORS["bg"])
    ax.grid(True, alpha=0.3)

    out = CHARTS_DIR / "gemma4_26b_fpr_time.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  [차트] {out.name}")


# ──────────────────────────────────────────────
# 차트 5 — 레이더 (26B vs 12B vs 최고 ViT)
# ──────────────────────────────────────────────

def chart_radar(row_26b: dict, row_12b: dict | None, row_vit: dict | None):
    categories = ["Accuracy", "Precision", "Recall", "F1", "FPR 역전\n(100-FPR)"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    def to_vals(r: dict) -> list[float]:
        return [
            as_float(r, "accuracy"),
            as_float(r, "precision"),
            as_float(r, "recall"),
            as_float(r, "f1"),
            100 - as_float(r, "fpr"),
        ]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(COLORS["bg"])

    entries = [(row_26b, COLORS["gemma4_26b"], "Gemma 4 26B")]
    if row_12b:
        entries.append((row_12b, COLORS["gemma4_12b"], "Gemma 4 12B"))
    if row_vit:
        entries.append((row_vit, COLORS["vit_best"], "haywoodsloan (최고 ViT)"))

    for row, color, label in entries:
        vals = to_vals(row) + to_vals(row)[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=label)
        ax.fill(angles, vals, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7)
    ax.set_title("Gemma 4 26B vs 12B vs 최고 ViT — 레이더 비교",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)

    out = CHARTS_DIR / "gemma4_26b_radar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  [차트] {out.name}")


# ──────────────────────────────────────────────
# 차트 6 — 신뢰도 분포 (26B raw)
# ──────────────────────────────────────────────

def chart_confidence_dist(raw_rows: list[dict]):
    if not raw_rows:
        print("  [스킵] raw 데이터 없음")
        return

    ai_ok, ai_ng, real_ok, real_ng = [], [], [], []
    for r in raw_rows:
        try:
            tl   = int(r["true_label"])
            pl   = int(r["pred_label"])
            conf = float(r["confidence"])
        except (KeyError, ValueError):
            continue
        if tl == 1:
            (ai_ok   if pl == 1 else ai_ng).append(conf)
        else:
            (real_ok if pl == 0 else real_ng).append(conf)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Gemma 4 26B — 신뢰도 분포 (정답 vs 오답)",
                 fontsize=13, fontweight="bold")

    bins = np.linspace(0, 1, 21)
    for ax, correct, wrong, title in [
        (axes[0], ai_ok,   ai_ng,  "실제 AI 이미지"),
        (axes[1], real_ok, real_ng,"실제 Real 이미지"),
    ]:
        ax.hist(correct, bins=bins, alpha=0.7, color=COLORS["gemma4_26b"],
                label=f"정답 ({len(correct)}장)")
        ax.hist(wrong,   bins=bins, alpha=0.7, color="#EF5350",
                label=f"오답 ({len(wrong)}장)")
        ax.axvline(x=0.5, color=COLORS["target"], linestyle="--",
                   linewidth=1.5, label="임계값 0.5")
        ax.set_xlabel("모델 신뢰도", fontsize=10)
        ax.set_ylabel("이미지 수", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_facecolor(COLORS["bg"])

    fig.tight_layout()
    out = CHARTS_DIR / "gemma4_26b_confidence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  [차트] {out.name}")


# ──────────────────────────────────────────────
# 차트 7 — 12B vs 26B 직접 비교 (그룹 막대)
# ──────────────────────────────────────────────

def chart_12b_vs_26b(row_12b: dict, row_26b: dict):
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1"]
    v12 = [as_float(row_12b, m) for m in metrics]
    v26 = [as_float(row_26b, m) for m in metrics]
    tgt = [TARGET[m] for m in metrics]

    x   = np.arange(len(metrics))
    w   = 0.32

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS["bg"])

    b12 = ax.bar(x - w/2, v12, w, label="Gemma 4 12B",
                 color=COLORS["gemma4_12b"], edgecolor="white")
    b26 = ax.bar(x + w/2, v26, w, label="Gemma 4 26B",
                 color=COLORS["gemma4_26b"], edgecolor="white")

    for bars in (b12, b26):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{bar.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    for tv, xi in zip(tgt, x):
        ax.plot([xi - w, xi + w], [tv, tv], color=COLORS["target"],
                linestyle="--", linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_ylabel("%", fontsize=11)
    ax.set_title("Gemma 4 12B vs 26B — 핵심 지표 직접 비교\n(점선: 목표치)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_facecolor(COLORS["bg"])
    ax.grid(axis="y", alpha=0.3)

    # 개선율 표시
    for xi, v1, v2 in zip(x, v12, v26):
        delta = v2 - v1
        sign  = "+" if delta >= 0 else ""
        color = "#2E7D32" if delta >= 0 else "#C62828"
        ax.text(xi, max(v1, v2) + 7, f"{sign}{delta:.1f}%p",
                ha="center", va="bottom", fontsize=9,
                color=color, fontweight="bold")

    fig.tight_layout()
    out = CHARTS_DIR / "gemma4_26b_vs_12b.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  [차트] {out.name}")


# ──────────────────────────────────────────────
# 차트 8 — 전체 요약 테이블
# ──────────────────────────────────────────────

def chart_summary_table(all_rows: list[dict]):
    fig, ax = plt.subplots(figsize=(15, len(all_rows) * 0.7 + 2.5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.axis("off")

    headers = ["모델", "Accuracy", "Precision", "Recall", "F1", "FPR", "Avg Time", "TP", "FP", "FN", "TN"]
    table_data = []
    for r in all_rows:
        table_data.append([
            get_display_name(r["model"]),
            f"{as_float(r, 'accuracy'):.1f}%",
            f"{as_float(r, 'precision'):.1f}%",
            f"{as_float(r, 'recall'):.1f}%",
            f"{as_float(r, 'f1'):.1f}%",
            f"{as_float(r, 'fpr'):.1f}%",
            f"{as_float(r, 'avg_time'):.1f}s",
            str(int(as_float(r, "tp"))),
            str(int(as_float(r, "fp"))),
            str(int(as_float(r, "fn"))),
            str(int(as_float(r, "tn"))),
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.8)

    for j in range(len(headers)):
        table[(0, j)].set_facecolor("#37474F")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    for i, row in enumerate(all_rows, 1):
        m = row["model"].lower()
        if "26b" in m:
            bg = "#FFF3E0"
        elif "gemma4" in m or "e4b" in m:
            bg = "#E8F5E9"
        else:
            bg = "white"
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(bg)
            if "26b" in m or "e4b" in m or "gemma4" in m:
                table[(i, j)].set_text_props(fontweight="bold")

    ax.set_title("전체 모델 성능 요약 — Gemma 4 26B · 12B · ViT",
                 fontsize=13, fontweight="bold", pad=20)

    out = CHARTS_DIR / "gemma4_26b_summary_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  [차트] {out.name}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default=None,
                        help="Gemma 4 26B raw CSV 파일명")
    args = parser.parse_args()

    print("=" * 65)
    print("  Gemma 4 26B 벤치마크 시각화")
    print("=" * 65)

    # 26B raw CSV
    if args.raw:
        raw_path = RESULT_DIR / args.raw
    else:
        candidates = sorted(RESULT_DIR.glob("gemma4_26b_raw_*.csv"), reverse=True)
        if not candidates:
            print("  gemma4_26b_raw_*.csv 파일을 찾을 수 없습니다.")
            return
        raw_path = candidates[0]
        print(f"  [자동 탐색] {raw_path.name}")

    raw_rows_26b = load_csv(raw_path)
    if not raw_rows_26b:
        print(f"  raw 데이터가 비어 있습니다: {raw_path}")
        return
    print(f"  26B 샘플 수: {len(raw_rows_26b)}장")

    row_26b = compute_metrics_from_raw(raw_rows_26b)
    print(f"  26B 지표 계산 완료 — Acc={row_26b['accuracy']}%  F1={row_26b['f1']}%  FPR={row_26b['fpr']}%")

    # 기존 ViT 결과
    vit_rows = load_csv(EXISTING_CSV)
    if not vit_rows:
        print(f"  [경고] ViT 벤치마크 CSV 없음: {EXISTING_CSV}")

    # 12B 결과
    g12b_rows = load_csv(GEMMA4_12B_CSV)
    row_12b   = g12b_rows[0] if g12b_rows else None

    best_vit = max(vit_rows, key=lambda r: as_float(r, "f1")) if vit_rows else None

    # 전체 비교용 행 목록 (ViT + 12B + 26B)
    all_rows = vit_rows + (g12b_rows or []) + [row_26b]

    print(f"\n  차트 생성 중...")
    chart_core_metrics(all_rows)
    chart_confusion(row_26b, "Gemma 4 26B", "gemma4_26b_confusion.png")
    if row_12b:
        chart_confusion_compare(row_12b, row_26b)
        chart_12b_vs_26b(row_12b, row_26b)
    chart_fpr_time(all_rows)
    chart_radar(row_26b, row_12b, best_vit)
    chart_confidence_dist(raw_rows_26b)
    chart_summary_table(all_rows)

    print(f"\n  완료! 저장 위치: {CHARTS_DIR}/")
    print("  생성 파일:")
    for f in sorted(CHARTS_DIR.glob("gemma4_26b_*.png")):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
