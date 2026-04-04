"""
Generate evaluation charts for the Renters Legal Assistance Chatbot report.
Consolidated to 10 clean charts covering all 5 models:
  GPT-4o, Llama 3.3 70B, Mistral Small 24B, Qwen3 4B Base, Qwen3 4B Fine-tuned.

Charts 3, 4, 6, 10 read live from eval_20260404_182956.json.
Charts 1, 2, 5, 7, 8, 9 use hardcoded data from earlier eval iterations.

Run:    venv\Scripts\python scripts\generate_charts.py
Output: docs/figures/*.png
"""

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVAL_PATH = Path(__file__).parent.parent / "data" / "evaluation" / "results" / "eval_20260404_231818.json"

def _load_eval():
    with open(EVAL_PATH, encoding="utf-8") as f:
        d = json.load(f)
    return d["summary"], d["retrieval_metrics"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": "#333333",
    "axes.grid": True, "grid.alpha": 0.3,
})

BLUE="#4A90D9"; GREEN="#2ECC71"; RED="#E74C3C"; ORANGE="#F39C12"
PURPLE="#9B59B6"; TEAL="#1ABC9C"; DARK="#2C3E50"; PINK="#E91E63"

def save(fig, name):
    fig.savefig(OUTPUT_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ======================================================================
# Chart 1: Iteration Progression
# ======================================================================
def chart1():
    iters = ["Iter 1\n(1068ch)", "Iter 2\n(617ch)", "Iter 4\n(871ch)", "Iter 5\n(967ch)"]
    faith = [0.780, 0.900, 0.762, 0.725]
    mrr   = [0.480, 0.568, 0.243, 0.220]
    x = np.arange(len(iters))
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    l1, = ax1.plot(x, faith, "o-",  color=BLUE, linewidth=2.5, markersize=8, label="Faithfulness (RAG)")
    l2, = ax2.plot(x, mrr,   "s--", color=RED,  linewidth=2.5, markersize=8, label="MRR (rerank)")
    for i, (f, m) in enumerate(zip(faith, mrr)):
        ax1.annotate(f"{f:.3f}", (i, f), textcoords="offset points", xytext=(0, 12),  ha="center", fontsize=10, fontweight="bold", color=BLUE)
        ax2.annotate(f"{m:.3f}", (i, m), textcoords="offset points", xytext=(0, -18), ha="center", fontsize=10, fontweight="bold", color=RED)
    ax1.axvline(x=1.5, color="#999", linestyle=":", linewidth=1.2)
    ax1.annotate("Judge changed:\nGPT-4o -> Claude Sonnet 4\n(stricter grading)",
                 xy=(1.5, 0.83), xytext=(2.5, 0.95), arrowprops=dict(arrowstyle="->", color="#666"),
                 fontsize=9, color="#666", ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#ccc"))
    ax1.set_ylabel("Faithfulness", color=BLUE); ax2.set_ylabel("MRR", color=RED)
    ax1.set_ylim(0.0, 1.1); ax2.set_ylim(0.0, 0.7)
    ax1.set_xticks(x); ax1.set_xticklabels(iters)
    ax1.set_title("Chart 1 - Iteration Progression: Faithfulness & MRR")
    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="lower left")
    fig.tight_layout(); save(fig, "chart01_iteration_progression.png")


# ======================================================================
# Chart 2: Corpus Growth
# ======================================================================
def chart2():
    iters  = ["Iter 1", "Iter 2", "Iter 4", "Iter 5"]
    docs   = [112, 112, 146, 249]
    chunks = [1068, 617, 871, 967]
    x = np.arange(len(iters)); width = 0.35
    fig, ax1 = plt.subplots(figsize=(8, 5)); ax2 = ax1.twinx()
    bars1 = ax1.bar(x - width/2, docs,   width, label="Documents", color=BLUE,   edgecolor="white")
    bars2 = ax2.bar(x + width/2, chunks, width, label="Chunks",    color=ORANGE, edgecolor="white")
    for b in bars1: ax1.text(b.get_x()+b.get_width()/2, b.get_height()+3,  str(int(b.get_height())), ha="center", fontsize=10, fontweight="bold", color=BLUE)
    for b in bars2: ax2.text(b.get_x()+b.get_width()/2, b.get_height()+12, str(int(b.get_height())), ha="center", fontsize=10, fontweight="bold", color=ORANGE)
    ax1.set_ylabel("Documents", color=BLUE); ax2.set_ylabel("Chunks", color=ORANGE)
    ax1.set_ylim(0, 320); ax2.set_ylim(0, 1300)
    ax1.set_xticks(x); ax1.set_xticklabels(iters)
    ax1.set_title("Chart 2 - Corpus Growth Across Iterations")
    l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper left")
    fig.tight_layout(); save(fig, "chart02_corpus_growth.png")


# ======================================================================
# Chart 3: Retriever Comparison (Generation + Retrieval, live data)
# ======================================================================
def chart3():
    SUMMARY, RET = _load_eval()
    configs   = ["Baseline\n(no RAG)", "Vector", "BM25", "Hybrid", "Rerank"]
    keys      = ["baseline", "rag_vector", "rag_bm25", "rag_hybrid", "rag_rerank"]
    faith     = [SUMMARY[k]["faithfulness_mean"] for k in keys]
    relevancy = [SUMMARY[k]["relevancy_mean"]    for k in keys]
    correct   = [SUMMARY[k]["correctness_mean"]  for k in keys]
    ret_names = ["BM25", "Hybrid", "Rerank"]
    ret_keys  = ["bm25", "hybrid", "rerank"]
    mrr      = [RET[k]["mrr"]         for k in ret_keys]
    hit_rate = [RET[k]["hit_rate"]    for k in ret_keys]
    recall   = [RET[k]["recall_at_k"] for k in ret_keys]
    ndcg     = [RET[k]["ndcg_at_k"]   for k in ret_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    x = np.arange(len(configs)); w = 0.25
    b1 = ax1.bar(x - w, faith,     w, label="Faithfulness", color=BLUE)
    b2 = ax1.bar(x,     relevancy, w, label="Relevancy",    color=GREEN)
    b3 = ax1.bar(x + w, correct,   w, label="Correctness",  color=ORANGE)
    for bars in [b1, b3]:
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x()+b.get_width()/2, h+0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1.set_ylabel("Score"); ax1.set_ylim(0, 1.15)
    ax1.set_xticks(x); ax1.set_xticklabels(configs, fontsize=9)
    ax1.set_title("Generation Metrics\n(GPT-4o, 89 questions)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.axhline(y=1.0, color="#999", linestyle="--", linewidth=0.8, alpha=0.5)

    xr = np.arange(len(ret_names)); wr = 0.18
    b4 = ax2.bar(xr - 1.5*wr, mrr,      wr, label="MRR@5",      color=RED)
    b5 = ax2.bar(xr - 0.5*wr, hit_rate, wr, label="Hit Rate@5", color=TEAL)
    b6 = ax2.bar(xr + 0.5*wr, recall,   wr, label="Recall@5",   color=BLUE)
    b7 = ax2.bar(xr + 1.5*wr, ndcg,     wr, label="NDCG@5",     color=PURPLE)
    for bars in [b4, b5, b6, b7]:
        for b in bars:
            h = b.get_height()
            ax2.text(b.get_x()+b.get_width()/2, h+0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax2.set_ylabel("Score"); ax2.set_ylim(0, 0.55)
    ax2.set_xticks(xr); ax2.set_xticklabels(ret_names)
    ax2.set_title("Retrieval Metrics\n(89 QA pairs, k=5)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.text(0.5, -0.12, "Vector excluded: chunk ID mismatch artifact (MRR=0.000)",
             transform=ax2.transAxes, ha="center", fontsize=8, color="#888", style="italic")
    fig.suptitle("Chart 3 - Retriever Comparison: Generation & Retrieval Metrics (Final Evaluation)", fontsize=13, fontweight="bold")
    fig.tight_layout(); save(fig, "chart03_retriever_comparison.png")


# ======================================================================
# Chart 4: Multi-Model Full Comparison (live data, all 5 models)
# ======================================================================
def chart4():
    SUMMARY, _ = _load_eval()
    models        = ["GPT-4o", "Llama 3.3\n(70B)", "Mistral\n(24B)", "Qwen3 4B\n(Base)", "Qwen3 4B\n(Fine-tuned)"]
    baseline_keys = ["baseline",   "llama_baseline",   "mistral_baseline",   "qwen3_base_baseline",      "qwen3_finetuned_baseline"]
    rerank_keys   = ["rag_rerank", "llama_rerank",     "mistral_rerank",     "qwen3_base_rerank",        "qwen3_finetuned_rerank"]
    colors        = [BLUE, ORANGE, GREEN, PURPLE, PINK]
    base_f = [SUMMARY[k]["faithfulness_mean"] for k in baseline_keys]
    rnk_f  = [SUMMARY[k]["faithfulness_mean"] for k in rerank_keys]
    base_c = [SUMMARY[k]["correctness_mean"]  for k in baseline_keys]
    rnk_c  = [SUMMARY[k]["correctness_mean"]  for k in rerank_keys]
    rnk_r  = [SUMMARY[k]["relevancy_mean"]    for k in rerank_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5.5))
    x = np.arange(len(models)); w = 0.3
    b1 = ax1.bar(x - w/2, base_f, w, label="Baseline",    color="#95A5A6", alpha=0.8)
    b2 = ax1.bar(x + w/2, rnk_f,  w, label="RAG+Rerank",  color=colors,   alpha=0.9)
    for b in b1:
        h = b.get_height()
        ax1.text(b.get_x()+b.get_width()/2, h+0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8, color="#666", fontweight="bold")
    for i, (b, col) in enumerate(zip(b2, colors)):
        h = b.get_height()
        ax1.text(b.get_x()+b.get_width()/2, h+0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color=col)
        lbl = f"+{(rnk_f[i]-base_f[i])/base_f[i]*100:.0f}%" if base_f[i] > 0 else "base~0"
        ax1.annotate(lbl, xy=(b.get_x()+b.get_width()/2, h), xytext=(0, 13),
                     textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color=DARK)
    ax1.set_ylabel("Faithfulness"); ax1.set_ylim(0, 1.15)
    ax1.set_xticks(x); ax1.set_xticklabels(models, fontsize=9)
    ax1.set_title("Faithfulness: Baseline vs RAG+Rerank")
    ax1.legend(loc="upper right")
    ax1.axhline(y=1.0, color="#999", linestyle="--", linewidth=0.8, alpha=0.3)

    w2 = 0.22
    b3 = ax2.bar(x - w2, base_c, w2, label="Correctness (baseline)", color="#95A5A6", alpha=0.8)
    b4 = ax2.bar(x,      rnk_c,  w2, label="Correctness (rerank)",   color=colors,   alpha=0.9)
    b5 = ax2.bar(x + w2, rnk_r,  w2, label="Relevancy (rerank)",     color=colors,   alpha=0.4, hatch="//")
    for b in b3:
        h = b.get_height()
        ax2.text(b.get_x()+b.get_width()/2, h+0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color="#666")
    for i, (b, col) in enumerate(zip(b4, colors)):
        h = b.get_height()
        ax2.text(b.get_x()+b.get_width()/2, h+0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=col)
    ax2.set_ylabel("Score"); ax2.set_ylim(0, 1.15)
    ax2.set_xticks(x); ax2.set_xticklabels(models, fontsize=9)
    ax2.set_title("Correctness & Relevancy")
    ax2.axhline(y=1.0, color="#999", linestyle="--", linewidth=0.8, alpha=0.3)
    from matplotlib.patches import Patch
    ax2.legend(handles=[
        Patch(facecolor=BLUE,   label="GPT-4o (cloud)"),
        Patch(facecolor=ORANGE, label="Llama 3.3 70B"),
        Patch(facecolor=GREEN,  label="Mistral Small 24B"),
        Patch(facecolor=PURPLE, label="Qwen3 4B Base (local)"),
        Patch(facecolor=PINK,   label="Qwen3 4B FT (local)"),
    ], loc="lower right", fontsize=8.5)
    fig.suptitle("Chart 4 - Multi-Model Comparison: All 5 Models x Baseline & RAG+Rerank\n(89 questions, Claude Sonnet 4 judge)", fontsize=13, fontweight="bold")
    fig.tight_layout(); save(fig, "chart04_multimodel_comparison.png")


# ======================================================================
# Chart 5: Ablation Experiments (top_k + structured prompt)
# ======================================================================
def chart5():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    models = ["GPT-4o", "Llama 3.3", "Mistral Small"]
    faith_k5  = [0.857, 0.500, 0.643]; faith_k10 = [0.857, 0.607, 0.750]
    corr_k5   = [0.357, 0.357, 0.393]; corr_k10  = [0.464, 0.357, 0.393]
    x = np.arange(len(models)); w = 0.18
    b1 = ax1.bar(x - 1.5*w, faith_k5,  w, label="Faith k=5",  color=BLUE,   alpha=0.7)
    b2 = ax1.bar(x - 0.5*w, faith_k10, w, label="Faith k=10", color=BLUE)
    b3 = ax1.bar(x + 0.5*w, corr_k5,   w, label="Corr k=5",   color=ORANGE, alpha=0.7)
    b4 = ax1.bar(x + 1.5*w, corr_k10,  w, label="Corr k=10",  color=ORANGE)
    for bars in [b1, b2, b3, b4]:
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x()+b.get_width()/2, h+0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax1.set_ylabel("Score"); ax1.set_ylim(0, 1.1)
    ax1.set_xticks(x); ax1.set_xticklabels(models)
    ax1.set_title("top_k=5 vs top_k=10\n(rerank, 28 questions, GPT-4o/Llama/Mistral)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.text(0.5, -0.12, "Note: Qwen3 not tested in top_k experiment (added in final eval)",
             transform=ax1.transAxes, ha="center", fontsize=8, color="#888", style="italic")

    configs = ["Old prompt\nk=5", "Old prompt\nk=10", "Structured\nk=5", "Structured\nk=10"]
    faith_gpt = [0.857, 0.857, 0.893, 0.929]; corr_gpt = [0.357, 0.464, 0.321, 0.321]
    xr = np.arange(len(configs)); w2 = 0.3
    b5 = ax2.bar(xr - w2/2, faith_gpt, w2, label="Faithfulness", color=BLUE)
    b6 = ax2.bar(xr + w2/2, corr_gpt,  w2, label="Correctness",  color=ORANGE)
    for bars in [b5, b6]:
        for b in bars:
            h = b.get_height()
            ax2.text(b.get_x()+b.get_width()/2, h+0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.annotate("Best faithfulness\nin project (0.929)",
                 xy=(3 - w2/2, 0.929), xytext=(1.5, 1.02),
                 arrowprops=dict(arrowstyle="->", color=DARK), fontsize=9, fontweight="bold", color=DARK, ha="center")
    ax2.set_ylabel("Score"); ax2.set_ylim(0, 1.15)
    ax2.set_xticks(xr); ax2.set_xticklabels(configs)
    ax2.set_title("Structured Prompt Experiment\n(GPT-4o + Rerank, 28 questions)")
    ax2.legend(loc="lower right")
    ax2.axhline(y=1.0, color="#999", linestyle="--", linewidth=0.8, alpha=0.3)
    fig.suptitle("Chart 5 - Ablation Experiments: top_k & Structured Prompt", fontsize=13, fontweight="bold")
    fig.tight_layout(); save(fig, "chart05_experiments.png")


# ======================================================================
# Chart 6: Faithfulness vs Correctness Scatter (live data, all models)
# ======================================================================
def chart6():
    SUMMARY, _ = _load_eval()
    configs = [
        ("GPT-4o Baseline",      "baseline",                 "$3/1M",    BLUE,   80),
        ("GPT-4o + Rerank",      "rag_rerank",               "$3/1M",    BLUE,   200),
        ("Llama + Rerank",       "llama_rerank",             "$0.10/1M", ORANGE, 180),
        ("Mistral + Rerank",     "mistral_rerank",           "$0.14/1M", GREEN,  180),
        ("Qwen3 Base Baseline",  "qwen3_base_baseline",      "local",    PURPLE, 80),
        ("Qwen3 Base + Rerank",  "qwen3_base_rerank",        "local",    PURPLE, 200),
        ("Qwen3 FT Baseline",    "qwen3_finetuned_baseline", "local",    PINK,   80),
        ("Qwen3 FT + Rerank",    "qwen3_finetuned_rerank",   "local",    PINK,   160),
    ]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for label, key, cost, color, size in configs:
        faith = SUMMARY[key]["faithfulness_mean"]; corr = SUMMARY[key]["correctness_mean"]
        ax.scatter(faith, corr, s=size, color=color, edgecolor="white", linewidth=1.5, zorder=5)
        ax.annotate(f"{label} ({cost})", (faith, corr), textcoords="offset points",
                    xytext=(10, -4), fontsize=8, ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#f8f8f8", edgecolor="#ddd"))
    ax.set_xlabel("Faithfulness (higher = better grounding)")
    ax.set_ylabel("Correctness (higher = better key-fact recall)")
    ax.set_title("Chart 6 - All Configurations: Faithfulness vs Correctness\n(89 questions; dot size = larger means RAG enabled)")
    ax.set_xlim(-0.05, 1.0); ax.set_ylim(0.26, 0.54)
    ax.axhline(y=0.43, color="#eee", linewidth=0.8); ax.axvline(x=0.5, color="#eee", linewidth=0.8)
    ax.text(0.85, 0.535, "High faith\nHigh correct", fontsize=8, color="#aaa", ha="center", style="italic")
    ax.text(0.15, 0.535, "Low faith\nHigh correct",  fontsize=8, color="#aaa", ha="center", style="italic")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0],marker="o",color="w",markerfacecolor=BLUE,  markersize=10,label="GPT-4o (cloud)"),
        Line2D([0],[0],marker="o",color="w",markerfacecolor=ORANGE,markersize=10,label="Llama 3.3 70B"),
        Line2D([0],[0],marker="o",color="w",markerfacecolor=GREEN, markersize=10,label="Mistral Small 24B"),
        Line2D([0],[0],marker="o",color="w",markerfacecolor=PURPLE,markersize=10,label="Qwen3 4B Base (local)"),
        Line2D([0],[0],marker="o",color="w",markerfacecolor=PINK,  markersize=10,label="Qwen3 4B FT (local)"),
    ], loc="upper left", fontsize=8.5)
    fig.tight_layout(); save(fig, "chart06_best_configs_scatter.png")


# ======================================================================
# Chart 7: Corpus Composition
# ======================================================================
def chart7():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    src_labels = ["masslegalhelp.org", "mass.gov", "boston.gov", "bostonhousing.org", "Other"]
    src_chunks = [422, 228, 162, 59, 96]
    src_colors = [PINK, DARK, TEAL, PURPLE, ORANGE]
    w1, t1, at1 = ax1.pie(src_chunks, labels=src_labels, autopct="%1.0f%%",
        colors=src_colors, startangle=90, textprops={"fontsize": 9.5}, pctdistance=0.78)
    for at in at1: at.set_fontweight("bold")
    ax1.set_title("Chunks by Source", fontweight="bold")
    type_labels = ["Legal Tactics\n(guides)", "Statutes", "FAQ", "Regulations", "Court/Procedural", "Other guides"]
    type_chunks = [422, 180, 59, 148, 78, 80]
    type_colors = [BLUE, RED, GREEN, ORANGE, PURPLE, TEAL]
    w2, t2, at2 = ax2.pie(type_chunks, labels=type_labels, autopct="%1.0f%%",
        colors=type_colors, startangle=90, textprops={"fontsize": 9.5}, pctdistance=0.78)
    for at in at2: at.set_fontweight("bold")
    ax2.set_title("Chunks by Content Type", fontweight="bold")
    fig.suptitle("Chart 7 - Current Corpus Composition (249 docs, 967 chunks)", fontsize=13, fontweight="bold")
    fig.tight_layout(); save(fig, "chart07_corpus_composition.png")


# ======================================================================
# Chart 8: Judge Methodology Validation
# ======================================================================
def chart8():
    methods = ["Our Method\n(single-call)", "LlamaIndex-style\n(iterative refine)"]
    faith = [0.964, 0.964]; relev = [1.000, 1.000]
    x = np.arange(len(methods)); width = 0.28
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    b1 = ax1.bar(x - width/2, faith, width, label="Faithfulness", color=BLUE)
    b2 = ax1.bar(x + width/2, relev, width, label="Relevancy",    color=GREEN)
    for bars in [b1, b2]:
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x()+b.get_width()/2, h+0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Score"); ax1.set_title("Aggregate Scores")
    ax1.set_xticks(x); ax1.set_xticklabels(methods, fontsize=10)
    ax1.set_ylim(0, 1.15); ax1.legend(loc="lower right")
    ax1.axhline(y=1.0, color="#999", linestyle="--", linewidth=0.8, alpha=0.3)
    categories = ["Faithfulness\nAgreement", "Relevancy\nAgreement"]; agreement = [93, 100]
    bars2 = ax2.bar(categories, agreement, color=[BLUE, GREEN], width=0.5, alpha=0.8)
    for b, val in zip(bars2, agreement):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{val}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax2.annotate("Our method: 2 API calls/question\nLlamaIndex: ~10 API calls/question\n-> 5x cheaper, same quality",
                 xy=(0.5, 50), fontsize=9, ha="center",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f8ff", edgecolor=BLUE, alpha=0.8))
    ax2.set_ylabel("Per-Question Agreement (%)"); ax2.set_title("Per-Question Agreement"); ax2.set_ylim(0, 115)
    fig.suptitle("Chart 8 - Judge Methodology Validation: Custom vs LlamaIndex Evaluators\n(28 questions, GPT-4o structured prompt, Claude Sonnet 4 judge)",
                 fontsize=12, fontweight="bold", y=1.04)
    fig.tight_layout(); save(fig, "chart08_judge_methodology.png")


# ======================================================================
# Chart 9: Negative Results (prompt completeness + self-eval bias)
# ======================================================================
def chart9():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    questions = ["g017", "g025", "g026", "g039", "g041", "g047", "g050", "r026"]
    base_gen = [0.500, 0.333, 0.333, 0.333, 0.667, 1.000, 1.000, 0.000]
    new_gen  = [1.000, 0.667, 0.667, 0.000, 0.333, 0.500, 0.667, 0.500]
    x = np.arange(len(questions)); width = 0.35
    b1 = ax1.bar(x - width/2, base_gen, width, label="Baseline prompt",     color=BLUE,   alpha=0.8)
    b2 = ax1.bar(x + width/2, new_gen,  width, label="Completeness prompt", color=ORANGE, alpha=0.8)
    for i in range(len(questions)):
        delta = new_gen[i] - base_gen[i]
        if delta != 0:
            sym = "^" if delta > 0 else "v"
            col = GREEN if delta > 0 else RED
            ax1.annotate(sym, (x[i], 1.08), ha="center", fontsize=10, color=col, fontweight="bold")
    ax1.set_ylabel("Correctness"); ax1.set_title("Prompt Completeness Experiment\n(8/28 questions changed - net zero effect)")
    ax1.set_xticks(x); ax1.set_xticklabels(questions, fontsize=8, rotation=30, ha="right")
    ax1.set_ylim(0, 1.2); ax1.legend(loc="upper right")
    ax1.axhline(y=1.0, color="#999", linestyle="--", linewidth=0.8, alpha=0.3)

    models = ["GPT-4o", "Llama 3.3", "Mistral Small"]
    claude_faith = [0.857, 0.500, 0.643]; self_faith = [0.964, 0.929, 0.857]
    xr = np.arange(len(models)); w2 = 0.3
    b3 = ax2.bar(xr - w2/2, claude_faith, w2, label="Claude Sonnet 4 judge", color=BLUE)
    b4 = ax2.bar(xr + w2/2, self_faith,   w2, label="Self-judge",            color=RED, alpha=0.7)
    for bars in [b3, b4]:
        for b in bars:
            h = b.get_height()
            ax2.text(b.get_x()+b.get_width()/2, h+0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for i in range(len(models)):
        delta = self_faith[i] - claude_faith[i]
        ax2.annotate(f"+{delta:.3f}", (xr[i] + w2/2 + 0.05, (claude_faith[i]+self_faith[i])/2),
                     fontsize=8, color=RED, fontweight="bold", ha="left")
    ax2.set_ylabel("Faithfulness"); ax2.set_title("Self-Evaluation Bias\n(Claude vs self-judge, 28 questions)")
    ax2.set_xticks(xr); ax2.set_xticklabels(models, fontsize=10); ax2.set_ylim(0, 1.15); ax2.legend(fontsize=9)
    ax2.text(0.5, -0.12, "Note: Qwen3 not tested in self-eval experiment (added in final eval)",
             transform=ax2.transAxes, ha="center", fontsize=8, color="#888", style="italic")
    fig.suptitle("Chart 9 - Negative Results & Ablations: Prompt Completeness + Self-Eval Bias", fontsize=13, fontweight="bold")
    fig.tight_layout(); save(fig, "chart09_negative_results.png")


# ======================================================================
# Chart 10: Final Summary - All Models x All Metrics (live data)
# ======================================================================
def chart10():
    SUMMARY, _ = _load_eval()
    configs = [
        "GPT-4o\nBaseline", "GPT-4o\n+Rerank",
        "Llama\n+Rerank",   "Mistral\n+Rerank",
        "Qwen3 Base\nBaseline", "Qwen3 Base\n+Rerank",
        "Qwen3 FT\nBaseline",   "Qwen3 FT\n+Rerank",
    ]
    keys = ["baseline", "rag_rerank", "llama_rerank", "mistral_rerank",
            "qwen3_base_baseline", "qwen3_base_rerank",
            "qwen3_finetuned_baseline", "qwen3_finetuned_rerank"]
    colors  = [BLUE, BLUE, ORANGE, GREEN, PURPLE, PURPLE, PINK, PINK]
    hatches = ["", "///", "///", "///", "", "///", "", "///"]
    faith   = [SUMMARY[k]["faithfulness_mean"] for k in keys]
    correct = [SUMMARY[k]["correctness_mean"]  for k in keys]
    relev   = [SUMMARY[k]["relevancy_mean"]    for k in keys]

    x = np.arange(len(configs)); w = 0.22
    fig, ax = plt.subplots(figsize=(16, 5.5))
    for i, (f, c, r, col, hat) in enumerate(zip(faith, correct, relev, colors, hatches)):
        ax.bar(x[i] - w, f, w, color=col,   alpha=0.9, hatch=hat, edgecolor="white")
        ax.bar(x[i],     r, w, color=GREEN,  alpha=0.5, hatch=hat, edgecolor="white")
        ax.bar(x[i] + w, c, w, color=ORANGE, alpha=0.9, hatch=hat, edgecolor="white")
    for i, (f, c) in enumerate(zip(faith, correct)):
        ax.text(x[i] - w, f + 0.01, f"{f:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        ax.text(x[i] + w, c + 0.01, f"{c:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=BLUE,   label="GPT-4o (cloud)"),
        Patch(facecolor=ORANGE, label="Llama 3.3 70B"),
        Patch(facecolor=GREEN,  label="Mistral Small 24B"),
        Patch(facecolor=PURPLE, label="Qwen3 4B Base (local)"),
        Patch(facecolor=PINK,   label="Qwen3 4B FT (local)"),
        Patch(facecolor="#aaa", hatch="///", label="RAG+Rerank enabled"),
        Patch(facecolor=BLUE,   alpha=0.5,   label="Left bar = Faithfulness"),
        Patch(facecolor=GREEN,  alpha=0.5,   label="Mid bar = Relevancy"),
        Patch(facecolor=ORANGE, alpha=0.9,   label="Right bar = Correctness"),
    ], fontsize=7.5, loc="upper right", ncol=3)
    ax.set_ylabel("Score")
    ax.set_title("Chart 10 - Final Summary: All Models x All Metrics\n(89 questions, Claude Sonnet 4 judge; hatch = RAG+Rerank enabled)")
    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=8.5); ax.set_ylim(0, 1.15)
    ax.axhline(y=1.0, color="#999", linestyle="--", linewidth=0.8, alpha=0.3)
    fig.tight_layout(); save(fig, "chart10_final_summary.png")


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    print("Generating evaluation charts...")
    chart1(); chart2(); chart3(); chart4(); chart5()
    chart6(); chart7(); chart8(); chart9(); chart10()
    print(f"\nAll 10 charts saved to {OUTPUT_DIR}/")
