# Renters Legal Assistance Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about **Massachusetts tenant law**, with a focus on the Boston area. Built for CS6180 (Advanced NLP) at Northeastern University.

**Team 27:** En-Ping Su & Maxwell Berry

---

## Overview

Large language models hallucinate frequently on legal questions -- studies show 58-80% hallucination rates on legal tasks for general-purpose LLMs. This project builds a RAG system that grounds responses in an authoritative corpus of Massachusetts housing statutes, regulations, and tenant resources, improving faithfulness from **0.640 (baseline) to 0.900 (RAG with corpus cleanup)**.

The system retrieves relevant legal text chunks from a vector store, feeds them as context to an LLM, and generates cited, grounded answers. It explicitly provides legal *information*, not legal *advice*.

## Architecture

```
User Question
     |
     v
 +-----------+     +------------------+     +----------------+
 | Retriever | --> | Prompt Assembly  | --> | LLM Generator  |
 | (5 types) |     | (system prompt   |     | (GPT-4o /      |
 |           |     |  + top-k chunks) |     |  Llama / Mistral)
 +-----------+     +------------------+     +----------------+
     ^                                             |
     |                                             v
 +----------+                              +-----------------+
 | ChromaDB |                              | Citation Check  |
 | (967     |                              | + Sources Footer|
 | chunks)  |                              +-----------------+
 +----------+
```

### Retriever Strategies

| Retriever | Method |
|-----------|--------|
| `vector` | ChromaDB cosine similarity (all-MiniLM-L6-v2, 384-dim) |
| `bm25` | BM25Okapi keyword matching with Snowball stemming |
| `hybrid` | Reciprocal Rank Fusion of vector (0.6) + BM25 (0.4) |
| `rerank` | Hybrid candidates re-scored by cross-encoder (ms-marco-MiniLM-L-6-v2) |
| `parent_child` | Rerank + neighbor chunk expansion for broader context |

### Generator Models

All LLM calls route through [OpenRouter](https://openrouter.ai):

- `openai/gpt-4o`
- `meta-llama/llama-3.3-70b-instruct`
- `mistralai/mistral-small-3.1-24b-instruct`

### Evaluation

- **LLM Judge:** `anthropic/claude-sonnet-4` (separate model family from all generators to avoid self-evaluation bias)
- **Generation metrics:** Faithfulness, Relevancy, Correctness (binary, via LLM judge)
- **Retrieval metrics:** MRR, Hit Rate, Precision, Recall
- **Test sets:** 20 curated golden QA pairs + 30 Reddit-style tenant questions

## Project Structure

```
src/
  scraping/           # Web scrapers
    scrape_mass_gov.py        # mass.gov statutes & guides
    scrape_boston_gov.py       # boston.gov housing pages
    scrape_bha_faq.py         # Boston Housing Authority FAQ
    scrape_masslegalhelp.py   # MassLegalHelp Legal Tactics chapters
    scrape_gbls.py            # Greater Boston Legal Services
    collect_reddit.py         # r/legaladvice, r/bostonhousing
    scrape_mass_gov_browser.py  # Browser-based fallback for 403s
  processing/
    chunker.py                # Markdown-aware recursive splitting
    corpus_cleaner.py         # Dedup, nav removal, off-topic filtering
  rag/
    pipeline.py               # ask() -> retrieve -> generate -> cite
    retrievers.py             # Vector, BM25, hybrid, rerank, parent_child
  rag_llamaindex/             # LlamaIndex implementation (for comparison)
    pipeline.py, retrievers.py, index.py, llm.py, nodes.py, prompts.py
  evaluation/
    scorer.py                 # 9-config evaluation with LLM judge
    generate_golden_qa.py     # Golden QA pair generation
    enrich_reddit_qa.py       # Reddit question enrichment
data/
  raw/                  # Scraped HTML/PDF (gitignored)
  processed/            # Cleaned Markdown documents (249 docs)
  chunks/               # all_chunks.json (967 chunks)
  chroma_db/            # ChromaDB vector store (gitignored)
  evaluation/
    golden_qa.json      # 20 curated Q&A pairs with key facts
    reddit_questions.json  # 30 Reddit-style questions
    results/            # Evaluation run outputs (JSON)
    results/figures/    # Charts + evaluation_report.txt
docs/
  literature_review.md  # RAG literature review (12 papers)
  scraped_pages.md      # Index of all scraped source pages
```

## Setup

### Prerequisites

- Python 3.11+
- API keys for [OpenRouter](https://openrouter.ai) and optionally [Reddit](https://www.reddit.com/prefs/apps) (for scraping only)

### Installation

```bash
# Clone the repo
git clone https://github.com/En-PingSu/renters-legal-chatbot.git
cd renters-legal-chatbot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### Building the Vector Store

The processed documents and chunks are included in the repo. To build the ChromaDB index:

```bash
venv/bin/python3 -m src.rag.pipeline index
```

This indexes all 967 chunks into ChromaDB (~30 seconds).

### Sanity Check

```bash
venv/bin/python3 -m src.rag.pipeline sanity
```

Runs 5 test queries and prints the top-3 retrieved chunks for each.

## Usage

### Interactive Q&A

```bash
venv/bin/python3 -m src.rag.pipeline
```

Type a question about Massachusetts tenant law and get a cited response.

### Programmatic Usage

```python
from src.rag.pipeline import ask

result = ask(
    "Can my landlord keep my security deposit?",
    top_k=5,
    model="openai/gpt-4o",
    retriever="rerank",
)
print(result["response"])
```

### Run Evaluation

```bash
venv/bin/python3 -m src.evaluation.scorer
```

Runs all 9 configurations (3 models x 3 retriever settings) against the test set using Claude Sonnet 4 as the LLM judge. Results are saved to `data/evaluation/results/`.

## Results Summary

### Iteration 1: Baseline vs RAG (GPT-4o, vector retrieval)

| Metric | Baseline | RAG | Delta |
|--------|----------|-----|-------|
| Faithfulness | 0.640 | 0.780 | +0.140 |
| Relevancy | 1.000 | 1.000 | +0.000 |

### Iteration 2: After Corpus Cleanup (removed 42% junk chunks)

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Faithfulness | 0.780 | 0.900 | +0.120 |
| MRR | 0.480 | 0.568 | +0.088 |

### Iteration 4: Multi-Retriever (GPT-4o, Claude Sonnet 4 judge)

| Retriever | Faithfulness | Relevancy | MRR | Hit Rate |
|-----------|-------------|-----------|-----|----------|
| Baseline | 0.100 | 1.000 | -- | -- |
| Vector | 0.714 | 1.000 | 0.185 | 0.257 |
| BM25 | 0.738 | 1.000 | 0.186 | 0.271 |
| Hybrid | 0.738 | 1.000 | 0.229 | 0.314 |
| Rerank | 0.762 | 1.000 | 0.243 | 0.328 |

Full results and analysis: [`data/evaluation/results/figures/evaluation_report.txt`](data/evaluation/results/figures/evaluation_report.txt)

## Corpus

249 documents / 967 chunks sourced from:

- **mass.gov** -- MGL statutes (c.186, c.239, c.93A, c.111), AG's guide to landlord/tenant rights, sanitary code, tenant screening regulations
- **boston.gov** -- Housing department pages, eviction guides, tenant rights, housing inspections, Office of Housing Stability
- **bostonhousing.org** -- Boston Housing Authority FAQ (60 Q&A pairs)
- **masslegalhelp.org** -- Legal Tactics for tenants (18 chapters covering security deposits, evictions, repairs, discrimination, utilities, court procedures)
- **gbls.org** -- Greater Boston Legal Services housing resources
- **Reddit** -- r/legaladvice, r/renting, r/bostonhousing (used for evaluation questions, not retrieval corpus)

## License

This project is for academic purposes (CS6180, Northeastern University). The scraped legal documents are public government resources.
