"""
Generate charts for multi-run averaging results (Section 9.8).

Run: venv/bin/python3 scripts/generate_multirun_charts.py
Output: docs/figures/chart19_multirun_*.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = (
    Path(__file__).parent.parent
    / "data" / "evaluation" / "results" / "multi_run_20260409_230502.json"
)

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

BLUE = "#4A90D9"
GREEN = "#2ECC71"
RED = "#E74C3C"
ORANGE = "#F39C12"
PURPLE = "#9B59B6"
TEAL = "#1ABC9C"
DARK = "#2C3E50"
GRAY = "#95A5A6"


def load_data():
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


# ── Chart 19a: Aggregate Metrics with Error Bars ─────────────────────

def chart_aggregate_metrics(data):
    """Bar chart of aggregate metrics with 95% CI error bars."""
    agg = data["aggregate"]

    metrics = ["faithfulness", "relevancy", "retrieval_coverage",
               "generation_coverage", "generation_coverage_given_retrieval"]
    labels = ["Faithfulness", "Relevancy", "Retrieval\nCoverage",
              "Generation\nCoverage", "Gen|Ret\nCoverage"]
    colors = [BLUE, GREEN, TEAL, ORANGE, PURPLE]

    means = [agg[m]["mean"] for m in metrics]
    ci_low = [agg[m]["mean"] - agg[m]["ci_95_low"] for m in metrics]
    ci_high = [agg[m]["ci_95_high"] - agg[m]["mean"] for m in metrics]
    stds = [agg[m]["std"] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics))
    bars = ax.bar(x, means, color=colors, edgecolor="white", linewidth=1.5,
                  width=0.6, zorder=3)
    ax.errorbar(x, means, yerr=[ci_low, ci_high], fmt="none",
                ecolor=DARK, elinewidth=2, capsize=8, capthick=2, zorder=4)

    # Annotate with mean +/- std
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + ci_high[i] + 0.02, f"{m:.3f}\n\u00b1{s:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Multi-Run Aggregate Metrics (GPT-4o + Rerank, k=10, 89q, 3 runs)")
    ax.axhline(y=1.0, color=GRAY, linestyle="--", alpha=0.5, zorder=1)

    # Iteration 9 comparison markers
    iter9 = {"Faithfulness": 0.854, "Relevancy": 1.000,
             "Generation\nCoverage": 0.430}
    for i, label in enumerate(labels):
        if label in iter9:
            ax.scatter(i, iter9[label], marker="D", color=RED, s=60, zorder=5,
                      edgecolors="white", linewidth=1)
    # Legend for iter9
    ax.scatter([], [], marker="D", color=RED, s=60, edgecolors="white",
              label="Iteration 9 (k=5)")
    ax.legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    out = OUTPUT_DIR / "chart19a_multirun_aggregate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Chart 19b: Per-Question Variance ─────────────────────────────────

def chart_per_question_variance(data):
    """Scatter plot of per-question std for retrieval vs generation."""
    pq = data["per_question"]

    ret_stds = [q["retrieval_coverage"]["std"] for q in pq]
    gen_stds = [q["generation_correctness"]["std"] for q in pq]
    unstable = [q["unstable"] for q in pq]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Stable questions
    stable_ret = [r for r, u in zip(ret_stds, unstable) if not u]
    stable_gen = [g for g, u in zip(gen_stds, unstable) if not u]
    ax.scatter(stable_ret, stable_gen, c=BLUE, alpha=0.6, s=50,
              edgecolors="white", linewidth=0.5, label=f"Stable ({len(stable_ret)})", zorder=3)

    # Unstable questions
    unstable_ret = [r for r, u in zip(ret_stds, unstable) if u]
    unstable_gen = [g for g, u in zip(gen_stds, unstable) if u]
    ax.scatter(unstable_ret, unstable_gen, c=RED, alpha=0.8, s=70,
              edgecolors="white", linewidth=0.5, label=f"Unstable ({len(unstable_ret)})", zorder=4)

    # Label the most unstable questions
    for q in pq:
        gen_s = q["generation_correctness"]["std"]
        ret_s = q["retrieval_coverage"]["std"]
        if gen_s > 0.25 or ret_s > 0.15:
            ax.annotate(q["question_id"], (ret_s, gen_s),
                       fontsize=7, alpha=0.8,
                       xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Retrieval Coverage Std Dev")
    ax.set_ylabel("Generation Correctness Std Dev")
    ax.set_title("Per-Question Variance (89 Questions, 3 Runs)")
    ax.axhline(y=0.1, color=GRAY, linestyle="--", alpha=0.5, label="Unstable threshold")
    ax.axvline(x=0.1, color=GRAY, linestyle="--", alpha=0.5)
    ax.set_xlim(-0.02, max(ret_stds) + 0.05)
    ax.set_ylim(-0.02, max(gen_stds) + 0.05)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_aspect("equal")

    plt.tight_layout()
    out = OUTPUT_DIR / "chart19b_multirun_variance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Chart 19c: Faithfulness Distribution Across Runs ─────────────────

def chart_faithfulness_comparison(data):
    """Bar chart comparing faithfulness: Iteration 9 vs multi-run with CI."""
    iter9_data = {
        "GPT-4o\n(Iter 9, k=5)": 0.854,
        "Mistral\n(Iter 9, k=5)": 0.798,
        "Llama\n(Iter 9, k=5)": 0.730,
        "Qwen3 Base\n(Iter 9, k=5)": 0.427,
    }

    agg = data["aggregate"]
    multirun_mean = agg["faithfulness"]["mean"]
    multirun_ci_low = agg["faithfulness"]["ci_95_low"]
    multirun_ci_high = agg["faithfulness"]["ci_95_high"]

    labels = list(iter9_data.keys()) + ["GPT-4o\n(3-run avg, k=10)"]
    values = list(iter9_data.values()) + [multirun_mean]
    colors = [BLUE, GREEN, ORANGE, TEAL, PURPLE]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=1.5,
                  width=0.6, zorder=3)

    # Error bar only on the multi-run result
    ax.errorbar(x[-1], multirun_mean,
                yerr=[[multirun_mean - multirun_ci_low],
                      [multirun_ci_high - multirun_mean]],
                fmt="none", ecolor=DARK, elinewidth=2, capsize=8, capthick=2, zorder=4)

    # Annotate
    for i, v in enumerate(values):
        if i == len(values) - 1:
            ax.text(i, v + (multirun_ci_high - multirun_mean) + 0.02,
                    f"{v:.3f}\n[{multirun_ci_low:.3f}, {multirun_ci_high:.3f}]",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        else:
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Faithfulness")
    ax.set_title("Faithfulness: Iteration 9 (Single Run) vs Multi-Run Average")

    plt.tight_layout()
    out = OUTPUT_DIR / "chart19c_multirun_faithfulness.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    print("Generating multi-run charts...")
    data = load_data()
    chart_aggregate_metrics(data)
    chart_per_question_variance(data)
    chart_faithfulness_comparison(data)
    print("Done!")
