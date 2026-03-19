"""
Evaluation Results — Baseline vs. RAG
Generates charts and exports a summary report.
Run: venv/bin/python3 evaluation_results.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

PROJECT_ROOT = Path(__file__).parent
RESULTS_PATH = PROJECT_ROOT / "data" / "evaluation" / "results" / "eval_20260317_051538.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "results" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load results
with open(RESULTS_PATH) as f:
    data = json.load(f)

summary = data["summary"]
retrieval = data["retrieval_metrics"]
results = data["results"]

# ============================================================
# Style setup
# ============================================================
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

COLORS = {"baseline": "#4A90D9", "rag": "#2ECC71"}

# ============================================================
# Chart 1: Faithfulness & Relevancy — Baseline vs RAG
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

metrics = ["Faithfulness", "Relevancy"]
baseline_vals = [summary["baseline"]["faithfulness_mean"],
                 summary["baseline"]["relevancy_mean"]]
rag_vals = [summary["rag"]["faithfulness_mean"],
            summary["rag"]["relevancy_mean"]]

x = np.arange(len(metrics))
width = 0.32

bars1 = ax.bar(x - width/2, baseline_vals, width, label="Baseline (GPT-4o, no retrieval)",
               color=COLORS["baseline"], edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x + width/2, rag_vals, width, label="RAG (GPT-4o + ChromaDB retrieval)",
               color=COLORS["rag"], edgecolor="white", linewidth=0.5)

# Value labels
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("Score (0 = fail, 1 = pass)")
ax.set_title("Chart 1 — Generation Quality: Baseline vs. RAG\n(50 fixed questions, GPT-4o LLM judge)")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.15)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax.legend(loc="lower right")
ax.axhline(y=1.0, color="#999999", linestyle="--", linewidth=0.8, alpha=0.5)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "chart1_faithfulness_relevancy.png", dpi=200, bbox_inches="tight")
print(f"Saved chart1_faithfulness_relevancy.png")

# ============================================================
# Chart 2: Retrieval Metrics (RAG only)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

ret_metrics = ["MRR", "Hit Rate", "Precision", "Recall"]
ret_vals = [retrieval["mrr"], retrieval["hit_rate"],
            retrieval["precision"], retrieval["recall"]]

bars = ax.bar(ret_metrics, ret_vals, color=["#E74C3C", "#F39C12", "#9B59B6", "#1ABC9C"],
              edgecolor="white", linewidth=0.5, width=0.55)

for bar, val in zip(bars, ret_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("Score")
ax.set_title("Chart 2 — Retrieval Metrics (RAG, top_k=5)\n(50 auto-generated QA pairs from corpus chunks)")
ax.set_ylim(0, 1.0)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

# Annotation for precision ceiling
ax.annotate("Max possible precision\nwith 1 ground-truth & top_k=5 = 0.200",
            xy=(2, 0.2), xytext=(2.5, 0.45),
            arrowprops=dict(arrowstyle="->", color="#666666"),
            fontsize=9, color="#666666", ha="center")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "chart2_retrieval_metrics.png", dpi=200, bbox_inches="tight")
print(f"Saved chart2_retrieval_metrics.png")

# ============================================================
# Chart 3: Per-question Faithfulness breakdown by source type
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_idx, config_name in enumerate(["baseline", "rag"]):
    ax = axes[ax_idx]
    config_results = [r for r in results if r["configuration"] == config_name]

    # Split by question source
    reddit_scores = [r["scores"]["faithfulness"] for r in config_results
                     if r["question_id"].startswith("reddit")]
    faq_scores = [r["scores"]["faithfulness"] for r in config_results
                  if r["question_id"].startswith("faq")]

    categories = ["Reddit-style\n(30 questions)", "BHA FAQ\n(20 questions)"]
    means = [np.mean(reddit_scores), np.mean(faq_scores)]
    counts_pass = [sum(1 for s in reddit_scores if s == 1.0),
                   sum(1 for s in faq_scores if s == 1.0)]
    counts_total = [len(reddit_scores), len(faq_scores)]

    bars = ax.bar(categories, means, color=COLORS[config_name],
                  edgecolor="white", linewidth=0.5, width=0.5)

    for bar, mean, cp, ct in zip(bars, means, counts_pass, counts_total):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{mean:.3f}\n({cp}/{ct} pass)", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Faithfulness Score")
    ax.set_title(f"{'Baseline' if config_name == 'baseline' else 'RAG'}")
    ax.set_ylim(0, 1.25)
    ax.axhline(y=1.0, color="#999999", linestyle="--", linewidth=0.8, alpha=0.5)

fig.suptitle("Chart 3 — Faithfulness by Question Source (Baseline vs. RAG)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "chart3_faithfulness_by_source.png", dpi=200, bbox_inches="tight")
print(f"Saved chart3_faithfulness_by_source.png")

# ============================================================
# Chart 4: Corpus composition
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# By content type
type_labels = ["Guide", "FAQ", "Regulation", "Statute"]
type_counts = [291, 305, 346, 126]
type_colors = ["#3498DB", "#2ECC71", "#E74C3C", "#F39C12"]
wedges1, texts1, autotexts1 = ax1.pie(
    type_counts, labels=type_labels, autopct="%1.0f%%",
    colors=type_colors, startangle=90, textprops={"fontsize": 10}
)
ax1.set_title("By Content Type", fontweight="bold")

# By source
src_labels = ["mass.gov", "bostonhousing.org", "boston.gov"]
src_counts = [508, 305, 255]
src_colors = ["#2C3E50", "#8E44AD", "#16A085"]
wedges2, texts2, autotexts2 = ax2.pie(
    src_counts, labels=src_labels, autopct="%1.0f%%",
    colors=src_colors, startangle=90, textprops={"fontsize": 10}
)
ax2.set_title("By Source", fontweight="bold")

fig.suptitle("Chart 4 — Corpus Composition (112 documents, 1,068 chunks)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "chart4_corpus_composition.png", dpi=200, bbox_inches="tight")
print(f"Saved chart4_corpus_composition.png")

# ============================================================
# Export summary report as text
# ============================================================
report = f"""{'='*70}
EVALUATION REPORT — Renters Legal Assistance Chatbot
Baseline vs. RAG Comparison
Date: 2026-03-17
{'='*70}

1. METHODOLOGY
{'─'*70}

1.1 Pipeline Architecture
  - Generator Model: GPT-4o (via OpenRouter)
  - Retriever: ChromaDB vector store (cosine similarity, top_k=5)
  - Embedding Model: all-MiniLM-L6-v2 (ChromaDB default, 384-dim)
  - Configurations tested:
    * Baseline: GPT-4o with no retrieval context
    * RAG: GPT-4o with ChromaDB retrieval + citation verification

1.2 Chunking Strategy
  - Method: Recursive character splitting with Markdown-aware boundaries
  - Chunk size: 800 tokens, 200 token overlap
  - Tokenizer: tiktoken (cl100k_base)
  - Separators (priority order): ## headings > ### headings > paragraphs > sentences > words
  - Special handling:
    * FAQ documents: kept as single chunks (never split)
    * Statutes: split at section/subsection boundaries first
    * Generic docs: recursive splitting with overlap

1.3 Corpus
  - 112 documents, 1,068 chunks total
  - Sources: mass.gov (508 chunks), bostonhousing.org (305), boston.gov (255)
  - Content types: Regulations (346), FAQ (305), Guides (291), Statutes (126)

1.4 Evaluation Questions
  - 50 total: 30 Reddit-style tenant questions + 20 BHA FAQ questions
  - Same questions used for both Baseline and RAG (fair cross-strategy comparison)

1.5 Evaluation Metrics (following HW4 methodology)
  Generation-level (LLM Judge — GPT-4o):
    - Faithfulness (binary 0/1): Is the response grounded in source context?
      For baseline: does it avoid fabricating specific statutes/URLs?
    - Relevancy (binary 0/1): Does the response address the question asked?

  Retrieval-level (RAG only, 50 auto-generated QA pairs):
    - MRR (Mean Reciprocal Rank): Position of correct chunk in ranked results
    - Hit Rate: Whether correct chunk appears in top_k results
    - Precision: Fraction of retrieved chunks that are relevant
    - Recall: Fraction of relevant chunks retrieved

2. RESULTS
{'─'*70}

2.1 Generation Quality (50 fixed questions)

  Metric          Baseline    RAG         Delta
  ──────────────  ──────────  ──────────  ──────────
  Faithfulness    {summary['baseline']['faithfulness_mean']:.3f}       {summary['rag']['faithfulness_mean']:.3f}       {summary['rag']['faithfulness_mean'] - summary['baseline']['faithfulness_mean']:+.3f}
  Relevancy       {summary['baseline']['relevancy_mean']:.3f}       {summary['rag']['relevancy_mean']:.3f}       {summary['rag']['relevancy_mean'] - summary['baseline']['relevancy_mean']:+.3f}

2.2 Retrieval Metrics (RAG, top_k=5, 50 QA pairs)

  Metric          Score
  ──────────────  ──────────
  MRR             {retrieval['mrr']:.3f}
  Hit Rate        {retrieval['hit_rate']:.3f}
  Precision       {retrieval['precision']:.3f}
  Recall          {retrieval['recall']:.3f}

  Note: Precision max is 0.200 (1 ground-truth / top_k=5).

3. ANALYSIS
{'─'*70}

3.1 Key Findings
  - RAG improves Faithfulness by +{summary['rag']['faithfulness_mean'] - summary['baseline']['faithfulness_mean']:.1%} over baseline, demonstrating that
    grounding responses in retrieved source documents reduces hallucination.
  - Both configurations achieve perfect Relevancy (1.000), indicating GPT-4o
    consistently addresses the question asked regardless of retrieval context.
  - Retrieval quality has room for improvement: MRR of {retrieval['mrr']:.3f} means the correct
    chunk is often not the top-ranked result, and Hit Rate of {retrieval['hit_rate']:.3f} means
    {1 - retrieval['hit_rate']:.1%} of queries fail to retrieve the correct chunk at all.

3.2 Improvement Opportunities
  - Embedding model: all-MiniLM-L6-v2 (384-dim) is lightweight; larger models
    like text-embedding-3-large or BGE-large may improve retrieval accuracy.
  - Hybrid retrieval: Combining vector search with BM25 keyword matching
    (as demonstrated in HW4) could improve Hit Rate significantly.
  - Chunking optimization: Current 800-token chunks may be too large for
    precise retrieval; experimenting with sentence window or smaller chunks
    could improve MRR.
  - Cross-encoder reranking: A reranker (e.g., ms-marco-MiniLM-L-6-v2) could
    improve MRR by jointly scoring query-passage pairs.

{'='*70}
"""

report_path = OUTPUT_DIR / "evaluation_report.txt"
with open(report_path, "w") as f:
    f.write(report)
print(f"\nSaved evaluation_report.txt")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
