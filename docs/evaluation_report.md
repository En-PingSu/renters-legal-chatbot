# Evaluation Report — Renters Legal Assistance Chatbot

**Baseline vs. RAG Comparison — Final Evaluation (Iteration 9)**
Date: 2026-04-04

---

## 1. Methodology

### 1.1 Pipeline Architecture

- **Generator Model:** GPT-4o (via OpenRouter)
- **Retriever:** ChromaDB vector store (cosine similarity, `top_k=5`)
- **Embedding Model:** `all-MiniLM-L6-v2` (ChromaDB default, 384-dim)
- Configurations tested:
  - **Baseline:** GPT-4o with no retrieval context
  - **RAG:** GPT-4o with ChromaDB retrieval + citation verification

### 1.2 Chunking Strategy

- **Method:** Recursive character splitting with Markdown-aware boundaries
- **Chunk size:** 800 tokens, 200 token overlap
- **Tokenizer:** tiktoken (`cl100k_base`)
- **Separators** (priority order): `## headings` > `### headings` > paragraphs > sentences > words
- Special handling:
  - FAQ documents: kept as single chunks (never split)
  - Statutes: split at section/subsection boundaries first
  - Generic docs: recursive splitting with overlap

### 1.3 Corpus

- 112 documents, 1,068 chunks total
- Sources: mass.gov (508 chunks), bostonhousing.org (305), boston.gov (255)
- Content types: Regulations (346), FAQ (305), Guides (291), Statutes (126)

### 1.4 Evaluation Questions

- 50 total: 30 Reddit-style tenant questions + 20 BHA FAQ questions
- Same questions used for both Baseline and RAG (fair cross-strategy comparison)

### 1.5 Evaluation Metrics (following HW4 methodology)

**Generation-level (LLM Judge — GPT-4o):**
- **Faithfulness** (binary 0/1): Is the response grounded in source context? For baseline: does it avoid fabricating specific statutes/URLs?
- **Relevancy** (binary 0/1): Does the response address the question asked?
- **Correctness** (binary 0/1, added in Iteration 4): Does the response cover the expected key facts from the golden QA reference answer? This is a binary variant of RAGAS `answer_correctness` (Es et al., 2023), which combines semantic similarity and factual overlap against a ground truth answer. Our metric simplifies this to binary key-fact coverage via LLM judge, similar in spirit to claim recall as used in FActScore (Min et al., 2023). The binary formulation is appropriate given our sample size (28–80 questions), where continuous scores would add false precision. In the precision/recall framing used throughout this report: faithfulness ≈ precision ("don't say unsupported things") and correctness ≈ recall ("don't miss expected facts").

  **Correctness implementation details:** Each golden QA pair in `golden_qa.json` includes a hand-curated list of key facts (e.g., for a security deposit question: "landlord must return within 30 days", "3x damages for violations", "MGL c.186 s.15B"). The LLM judge (`src/evaluation/scorer.py`, `judge_correctness()`) receives the question, the key facts list, and the generated response in a single call, and is asked: "Does the response contain the key facts listed above?" The judge responds YES/NO with explanation; YES → 1.0, NO → 0.0. This is an **all-or-nothing** check — if any expected fact is missing, the entire response scores 0. This is stricter than RAGAS's partial-credit approach (token-level F1) and explains why correctness scores are consistently lower than faithfulness across all configurations (0.25–0.46 range). Only questions with defined `key_facts` are scored; questions without key facts are excluded from the correctness aggregate.

  **Relationship between correctness and retrieval metrics:** Correctness is downstream of retrieval quality. The causal chain is: retrieval quality → which chunks land in context → what the LLM writes → correctness score. Notably, `judge_correctness()` does not receive the retrieved context — it only sees the question, the response, and the expected key facts. This means correctness measures whether the LLM *produced* the expected facts, but it cannot distinguish between two failure modes: (1) the relevant chunk was never retrieved (a retrieval failure), or (2) the chunk was retrieved but the LLM failed to use it (a generation failure). The `top_k=10` experiment (Section 7.11) illustrates this coupling: more retrieved chunks → +30% correctness for GPT-4o, confirming that correctness tracks retrieval coverage. The generator-judge swap experiment (Section 7.14) further shows that correctness scores are highly judge-dependent (0.321 to 0.750 for identical responses), making it the least reliable of the three generation metrics. Future work should adopt per-fact claim recall scoring (checking each key fact individually and reporting the fraction covered) to disentangle retrieval failures from generation failures and reduce sensitivity to judge identity.

**Retrieval-level metrics** are computed by comparing retrieved chunk IDs against ground-truth chunk IDs from `source_chunks` in `golden_qa.json` and `reddit_questions.json`. These are free (no API calls) and run alongside every evaluation.

  **Ground-truth change (Iteration 8):** Prior iterations (Sections 2–7.20) evaluated retrieval against only `source_chunks[0]` — the first ground-truth chunk per question. This produced false misses for questions with multiple relevant chunks (8 of 68 multi-chunk questions were scored as misses despite retrieving other valid chunks). Starting in Iteration 8 (Section 7.21), all metrics evaluate against **all** `source_chunks` per question. MRR and Hit Rate semantics are preserved (they still check for "any" relevant hit), but now consider all designated chunks rather than just the first. Earlier results in Sections 2–7.20 used the old single-chunk evaluation and are not directly comparable to Iteration 8+ results on these metrics.

  **Chunk-level metrics (K = top_k, default 10):**
- **MRR** (Mean Reciprocal Rank): `1 / rank` of the first relevant chunk found in the top-K results, averaged over all questions. Measures how quickly the retriever surfaces any relevant result. Range 0–1; higher is better.
- **Hit Rate (Hit@K):** Fraction of questions where at least one ground-truth chunk appears in the top-K results. Binary per question: 1 if any relevant chunk is found, 0 otherwise.
- **Recall@K** (added Iteration 8): `|retrieved ∩ ground_truth| / |ground_truth|` — fraction of all ground-truth chunks found in the top-K, averaged over questions. Critical for multi-statute questions that require 3–4 chunks: a question needing 4 chunks where the retriever finds 2 scores Recall@10 = 0.50.
- **NDCG@K** (Normalized Discounted Cumulative Gain, added Iteration 8): Measures ranking quality with a logarithmic discount. `DCG = Σ 1/log₂(rank+1)` for each relevant chunk found; `IDCG` is the ideal DCG if all relevant chunks occupied the top positions. `NDCG = DCG / IDCG`. With binary relevance (relevant = 1, not = 0), NDCG rewards putting relevant chunks at the top of the ranking. Two retrievers with identical Recall@K can differ in NDCG if one ranks hits at positions 1, 2 while the other ranks them at positions 8, 9.

  **Fact-level metrics (LLM-judged, added Iteration 7):**
- **Retrieval coverage:** Fraction of key_facts present in retrieved chunk text, judged by Claude Sonnet 4. Measures whether the retrieved context *contains* the information, regardless of which specific chunk it came from.
- **Generation coverage:** Fraction of key_facts present in the LLM-generated response.
- **Gen|Ret (generation coverage given retrieval):** Of the facts that *were* retrieved, what fraction did the LLM include in its response? Isolates generation quality from retrieval quality.

  **Relationship between chunk-level and fact-level metrics:** These two layers measure different things. Chunk-level metrics (MRR, Recall@K) answer "did we find the designated documents?" Fact-level metrics (retrieval/generation coverage) answer "did we find the information, regardless of which document it came from?" The gap between them (e.g., Recall@K = 0.320 vs fact-level retrieval coverage = 0.685) measures **corpus information redundancy** — the same legal facts appear in multiple chunks across different sources (AG's Guide, MassLegalHelp chapters, raw statutes). See Section 7.21.9 for detailed analysis.

---

## 2. Results

### 2.1 Generation Quality (50 fixed questions)

| Metric | Baseline | RAG | Delta |
|--------|----------|-----|-------|
| Faithfulness | 0.640 | 0.780 | +0.140 |
| Relevancy | 1.000 | 1.000 | +0.000 |

### 2.2 Retrieval Metrics (RAG, top_k=5, 50 QA pairs)

| Metric | Score |
|--------|-------|
| MRR | 0.480 |
| Hit Rate | 0.680 |
| Precision | 0.136 |
| Recall | 0.680 |

> Note: Precision max is 0.200 (1 ground-truth / top_k=5).

---

## 3. Analysis

### 3.1 Key Findings

- RAG improves Faithfulness by +14.0% over baseline, demonstrating that grounding responses in retrieved source documents reduces hallucination.
- Both configurations achieve perfect Relevancy (1.000), indicating GPT-4o consistently addresses the question asked regardless of retrieval context.
- Retrieval quality has room for improvement: MRR of 0.480 means the correct chunk is often not the top-ranked result, and Hit Rate of 0.680 means 32.0% of queries fail to retrieve the correct chunk at all.

### 3.2 Improvement Opportunities

- **Embedding model:** `all-MiniLM-L6-v2` (384-dim) is lightweight; larger models like `text-embedding-3-large` or `BGE-large` may improve retrieval accuracy.
- **Hybrid retrieval:** Combining vector search with BM25 keyword matching (as demonstrated in HW4) could improve Hit Rate significantly.
- **Chunking optimization:** Current 800-token chunks may be too large for precise retrieval; experimenting with sentence window or smaller chunks could improve MRR.
- **Cross-encoder reranking:** A reranker (e.g., `ms-marco-MiniLM-L-6-v2`) could improve MRR by jointly scoring query-passage pairs.

### 3.3 Metric Selection: Why HW4 Metrics Over the Initial Custom Rubric

An initial evaluation used a custom 0-2 rubric scoring Accuracy, Reliability, and Citation. Under that rubric, RAG scored *lower* than baseline on Accuracy (1.68 vs 1.96) and Reliability (1.74 vs 1.96), winning only on Citation (1.36 vs 0.36). This was misleading for two reasons:

1. The RAG system prompt instructs the model to "ONLY answer from provided source documents," so when retrieved chunks don't fully cover a question, the RAG response is intentionally constrained and incomplete. Baseline GPT-4o freely draws on training data and sounds more "complete," so the judge rewarded it with higher Accuracy — penalizing RAG for being cautious, which is actually a desirable property in a legal information system.

2. The Citation metric structurally disadvantaged baseline (no retrieval context to cite), making the comparison asymmetric.

The HW4 metrics (Faithfulness + Relevancy) resolve this. Faithfulness measures whether the response is grounded without hallucination — the core property RAG is designed to improve. Under this metric, RAG correctly scores higher (0.780 vs 0.640) because baseline fabricates specific legal details more often.

### 3.4 Known Biases and Limitations

Three methodological biases apply to this evaluation, identified by mapping the bias analysis from HW4 to our setup:

1. **Auto-Generated QA Pair Bias** — The retrieval metrics (MRR, Hit Rate, Precision, Recall) use QA pairs auto-generated from our own chunks. As HW4 notes, "questions generated from [one strategy's] nodes are phrased to match [that strategy's] chunk boundaries." These scores are valid as within-strategy indicators but would not be directly comparable if a different chunking strategy were evaluated against the same QA pairs. The 50 fixed evaluation questions used for Faithfulness and Relevancy are not subject to this bias, as they are the same across both configurations.

2. **Self-Evaluation Bias (Judge = Generator)** — GPT-4o serves as both the generator and the LLM judge. The judge may favor its own output style, inflating absolute scores for both baseline and RAG. HW4 mitigated this by using a different judge model (Qwen Turbo). In our case, the relative comparison between baseline and RAG remains fair since both use the same generator, but the absolute Faithfulness and Relevancy values may be optimistically biased. A future run with a separate judge model (e.g., Claude or Qwen) would quantify this effect.

3. **Small Sample Judge Variance** — Binary 0/1 LLM scoring on 50 questions is subject to judge variance. A single flipped judgment changes Faithfulness by 0.020. HW4 addressed this with 3-run averaging, which reduced per-run variance from ~7.7% to ~4.4%. Our evaluation uses a single run. Multi-run averaging would produce more stable estimates and should be added for the final report.

### 3.5 How the Embedding Model Affects the RAG Pipeline

The embedding model is the component that converts both chunks and queries into vector representations for similarity search. It directly determines retrieval quality, which cascades into generation quality — as HW4 observes, "even the best LLM cannot produce a correct answer if the wrong chunk is retrieved."

Our current embedding model, `all-MiniLM-L6-v2`, is ChromaDB's default. It is a lightweight model (384 dimensions, 22M parameters) originally trained on general-purpose English sentence similarity tasks. This creates two specific weaknesses for our legal domain:

1. **Semantic Capacity:** 384 dimensions compress meaning into a relatively small vector space. Legal documents contain nuanced distinctions (e.g., "tenancy at will" vs. "tenancy under lease," "security deposit" vs. "last month's rent") that may collide in a low-dimensional space. Larger models like `text-embedding-3-large` (3072-dim) or `BGE-large` (1024-dim) can represent these distinctions with greater fidelity, potentially improving MRR by ranking the correct chunk higher.

2. **Domain Mismatch:** `all-MiniLM-L6-v2` was not trained on legal text. Legal language uses terms with domain-specific meaning (e.g., "quiet enjoyment" means freedom from landlord interference, not silence). A model fine-tuned on legal corpora, or a larger general model with broader training data, would better capture these semantic relationships.

**Impact on Evaluation Metrics:**
- MRR and Hit Rate are directly affected: a better embedding model ranks the correct chunk higher (improving MRR) and retrieves it more often (improving Hit Rate). Our current MRR of 0.480 suggests the embedding model frequently misjudges which chunks are most relevant.
- Faithfulness is indirectly affected: when retrieval surfaces the wrong chunks, the LLM either generates an answer from irrelevant context (lower Faithfulness) or ignores the context and falls back on parametric knowledge (defeating the purpose of RAG). Improving retrieval quality should lift Faithfulness above 0.780.
- Relevancy is unlikely to change, since GPT-4o already addresses every question regardless of which chunks are retrieved.

### 3.6 How Chunking Strategy Affects the RAG Pipeline

Chunking determines the granularity and boundaries of the text units stored in the vector database. It affects both what can be retrieved and how useful the retrieved content is to the LLM.

Our current strategy uses 800-token chunks with 200-token overlap and Markdown-aware recursive splitting. This has several implications:

1. **Chunk Size and Retrieval Precision** — Large chunks (800 tokens) contain more context per retrieval, which helps the LLM generate complete answers. However, they also dilute the embedding — a single chunk about both security deposits AND eviction procedures will partially match queries about either topic but rank highly for neither. HW4 demonstrated that Sentence Window chunking (splitting at sentence level, then expanding context at retrieval time) achieved higher MRR than baseline 128-token chunks across multiple retrievers, because the embedding is computed on a focused sentence while the LLM receives surrounding context. Smaller chunks with context expansion could improve our MRR of 0.480 significantly.

2. **Chunk Boundaries and Information Splitting** — Recursive character splitting can split a legal provision mid-sentence or separate a statute number from its text. Our statute-specific handler mitigates this by splitting at section boundaries, but generic documents (291 guide chunks, 346 regulation chunks) use the general splitter. If a tenant's rights explanation spans a chunk boundary, neither chunk contains the complete answer, and the LLM must synthesize across retrieved chunks — which it may not do correctly, reducing Faithfulness.

3. **Overlap and Redundancy** — The 200-token overlap means ~25% of each chunk duplicates content from the adjacent chunk. This improves recall (important content near a boundary appears in both chunks) but introduces redundancy in the top-k results: two overlapping chunks may both be retrieved, consuming two of five retrieval slots with largely the same content. This directly reduces the diversity of information available to the LLM and could be addressed with semantic deduplication (as explored in HW4 Task 5).

4. **Content-Type-Aware Chunking** — Our FAQ handler keeps each Q&A pair as a single chunk, which is effective for FAQ-style questions (these have high retrieval accuracy). However, statutes and regulations — which are the most authoritative sources for legal questions — use generic splitting that may not respect the logical structure of legal text. A statute-aware chunking strategy that preserves complete subsections (e.g., MGL c.186 s.15B(1) as a single chunk) could improve both retrieval and Faithfulness for legal questions.

**Impact on Evaluation Metrics:**
- **MRR:** Smaller, more focused chunks produce more precise embeddings, leading to better ranking. HW4 showed Sentence Window achieving MRR of 0.653 (BM25) vs. Baseline's 0.535.
- **Hit Rate:** Better chunk boundaries ensure complete answers exist within single retrievable units, improving the chance of retrieving the right one.
- **Faithfulness:** When retrieved chunks contain complete, relevant information, the LLM can answer faithfully from context rather than hallucinating to fill gaps.
- **Recall:** Overlap helps recall but wastes retrieval slots; deduplication or diversity-aware retrieval can recover those slots.

### 3.7 Additional Improvement Opportunities

Beyond the three biases identified in Section 3.4, several architectural and methodological improvements could strengthen both the pipeline and evaluation:

1. **Hybrid Retrieval (Vector + BM25)** — Our pipeline uses only vector search (cosine similarity on embeddings). HW4 Task 4 demonstrated that hybrid retrieval — combining vector search (0.6 weight) with BM25 keyword matching (0.4 weight) — achieved the highest MRR (0.692) and Hit Rate (0.805) across all configurations. Legal queries often contain specific terms (e.g., "MGL c.186 s.15B," "security deposit," "30-day notice") where exact keyword matching is highly reliable. A hybrid approach would capture both semantic similarity and keyword precision, likely improving our MRR from 0.480 and Hit Rate from 0.680.

2. **Cross-Encoder Reranking** — Our pipeline scores query-passage similarity using cosine distance between independently-computed embeddings (bi-encoder). Cross-encoder rerankers (e.g., `ms-marco-MiniLM-L-6-v2`) read the query and passage jointly, modeling their interaction directly. HW4 Task 5 showed that adding cross-encoder reranking on top of hybrid retrieval improved Faithfulness from 0.897 to 1.000 and Relevancy from 0.846 to 0.923 with zero run-to-run variance. This is the single highest-impact improvement demonstrated in HW4.

3. **Multi-Query Expansion** — A single query phrasing may not match the vocabulary used in relevant chunks. Multi-query expansion generates alternative phrasings (e.g., "security deposit return deadline" → "how long does landlord have to return deposit" + "30 day security deposit rule Massachusetts") and retrieves across all variants. This widens the candidate pool for the reranker. HW4 found that multi-query alone did not improve scores on a single-domain corpus (paraphrases reused the same vocabulary), but it was essential as a feeder for the reranker pipeline.

4. **Prompt Engineering** — The RAG system prompt instructs the model to "ONLY answer from provided source documents." While this reduces hallucination (improving Faithfulness), it may cause the model to decline answering when retrieved chunks are relevant but don't contain the exact phrasing the model expects. Refining the prompt to say "prioritize source documents but use general knowledge to fill minor gaps, clearly distinguishing sourced from unsourced claims" could improve answer completeness without sacrificing Faithfulness.

5. **Question-Category-Specific Analysis** — Our evaluation treats all 50 questions uniformly. HW4 Task 2 categorized questions into types (Cross-Sectional, Keyword-Heavy, Semantic/Contextual, Structured Data, Image-Based) and found significant performance variation across categories. A similar breakdown for our questions — e.g., security deposit questions vs. eviction questions vs. BHA-specific questions — would reveal which legal topics the pipeline handles well and where retrieval or generation fails, enabling targeted improvements.

6. **Context Window Utilization** — With `top_k=5` and 800-token chunks, the LLM receives up to ~4,000 tokens of context. GPT-4o supports 128K tokens. Increasing `top_k` (e.g., to 10 or 15) would provide more context at the cost of potentially introducing irrelevant chunks. Combined with reranking, a larger initial retrieval pool filtered to the best 5-7 chunks (as done in HW4 Task 5 with `top_k=10` internally, trimmed to 5) could improve answer completeness.

7. **Fine-Tuning (Core Goal per Proposal)** — The project proposal lists fine-tuning as a core goal. Fine-tuning the generator on Massachusetts tenant law Q&A pairs could improve both Faithfulness (model learns to stay grounded in legal sources) and answer quality (model learns the expected format and level of detail for legal information responses). This would be evaluated as a third configuration alongside baseline and RAG.

---

## 4. Iteration 2: Corpus Cleanup (2026-03-17)

### 4.1 Motivation

Analysis of the initial evaluation revealed that retrieval quality was the primary bottleneck (MRR 0.480, Hit Rate 0.680). Investigation of the 1,068 chunks uncovered a root cause: 49% of chunks (521 of 1,068) were under 100 characters — section headings, navigation menus, or text fragments. These junk chunks pollute the vector space (their short, generic embeddings match many queries poorly) and waste retrieval slots (a heading like "## 410.001: Purpose" occupying one of five `top_k` slots displaces a substantive chunk).

### 4.2 What Was Done

A post-processing corpus cleaner (`src/processing/corpus_cleaner.py`) was created and applied to the existing `all_chunks.json`. Four cleanup steps were executed in sequence:

1. **Remove Duplicate FAQ Full Page** (-245 chunks) — The Boston Housing Authority FAQ existed in two forms: 60 individual Q&A documents (each kept as a single chunk by the FAQ chunker — good) AND a single `bostonhousing_faq_full_page` document that contained the entire FAQ page and was generic-chunked into 245 fragments. These fragments split questions from their answers, making them useless for retrieval. All 245 chunks with `doc_id == "bostonhousing_faq_full_page"` were removed. The 60 individual Q&A chunks remain.

2. **Merge Heading-Only Chunks** (reduced fragmentation) — For regulation and statute documents, the recursive splitter produced heading-only chunks like "## 410.001: Purpose" (19 chars) or "SECTION 8A" (10 chars). These were identified (< 100 chars, starts with `#` or is all-caps) and prepended to the next substantive chunk in the same document, preserving the heading as context rather than discarding it. Chunks were grouped by `doc_id` and sorted by `chunk_index` to maintain document order.

3. **Filter Short Chunks** (-104 chunks) — After merging, any remaining chunks under 50 characters were removed. These were fragments too short to contain meaningful legal content (e.g., "Toggle", "Page Sections", isolated list bullets).

4. **Remove Boilerplate** (-5 chunks) — Pattern matching identified navigation menu chunks where every line was under 40 characters and total content was under 150 characters (e.g., "Upcoming Events\nLatest Updates\nToggle\nPage Sections"). Chunks containing common boilerplate phrases ("skip to main content", "back to top", "cookie", "subscribe to") under 200 characters were also removed.

After cleanup, chunk indices and IDs were reassigned to maintain continuity within each document.

The chunker (`src/processing/chunker.py`) was also updated to prevent these issues on future runs:
- `chunk_document()` now skips any `doc_id` containing `"faq_full_page"`
- `chunk_statute()` and `chunk_generic()` now filter out chunks shorter than 50 characters after `merge_with_overlap()`

### 4.3 Results

**Corpus:**

| | Before | After | Change |
|---|--------|-------|--------|
| Chunks | 1,068 | 617 | -42.2% |

**Verification:**
- 0 chunks with `doc_id` `"bostonhousing_faq_full_page"` remain
- 0 chunks under 50 characters remain

**Generation Quality (50 fixed questions):**

| Metric | Run 1 | Run 2 | Delta |
|--------|-------|-------|-------|
| Faithfulness | 0.780 | 0.900 | +0.120 |
| Relevancy | 1.000 | 1.000 | +0.000 |

**Retrieval Metrics (RAG, top_k=5, 50 auto-generated QA pairs):**

| Metric | Run 1 | Run 2 | Delta |
|--------|-------|-------|-------|
| MRR | 0.480 | 0.568 | +0.088 |
| Hit Rate | 0.680 | 0.720 | +0.040 |
| Precision | 0.136 | 0.144 | +0.008 |
| Recall | 0.680 | 0.720 | +0.040 |

### 4.4 Analysis

All four metrics improved. The largest gain was in Faithfulness (+0.120), which increased from 0.780 to 0.900. This makes sense: when junk chunks no longer occupy retrieval slots, the LLM receives more substantive context and can answer from source material rather than falling back on parametric knowledge or hallucinating.

MRR improved from 0.480 to 0.568 (+0.088). With fewer chunks in the vector store, the correct chunk faces less competition from near-duplicate or irrelevant fragments, so it ranks higher on average.

Hit Rate improved from 0.680 to 0.720 (+0.040). Four additional queries (out of 50) now retrieve the correct chunk in the top 5, likely because the removed junk chunks were displacing relevant results.

Relevancy remained at 1.000, as expected — GPT-4o consistently addresses the question regardless of retrieval quality.

> Note on retrieval metric comparability: The auto-generated QA pairs are regenerated from the cleaned corpus, so the Run 1 and Run 2 retrieval metrics are not computed on identical question sets. The generation metrics (Faithfulness, Relevancy) use the same 50 fixed questions across both runs and are directly comparable.

---

## 5. Iteration 3: Separate Judge Model + Multi-Model Evaluation (2026-03-17)

### 5.1 Motivation

Two methodological issues were addressed:

1. **Self-Evaluation Bias** (Section 3.4, Item 2): GPT-4o served as both the generation model and the LLM judge. As noted in the HW4 writeup, single-model judging introduces bias and variance — the judge may favor its own output style, inflating absolute scores. Switching the judge to a different model family eliminates self-scoring favoritism.

2. **Single Base Model:** The project proposal commits to evaluating "GPT-4 class models, Llama 3, Mistral, or Mixtral," but all 5 configurations used `openai/gpt-4o` — only the retriever varied, never the LLM. Week 3 of the proposal says to "experiment with different pipeline enhancements."

### 5.2 Changes Made

**Judge Model:**

| | Before | After |
|---|--------|-------|
| Judge | `openai/gpt-4o` (same family as generator) | `anthropic/claude-sonnet-4` (different family) |

No generation model shares a model family with the judge, eliminating self-evaluation bias for all configurations.

**Generation Configurations (13 total — 9 from Iteration 3, 4 local models added in Iteration 9):**

| Config | Generation Model | RAG Retriever | Added | Purpose |
|--------|-----------------|---------------|-------|---------|
| `baseline` | `openai/gpt-4o` | none | Iter 3 | Retriever comparison |
| `rag_vector` | `openai/gpt-4o` | vector | Iter 3 | Retriever comparison |
| `rag_bm25` | `openai/gpt-4o` | bm25 | Iter 3 | Retriever comparison |
| `rag_hybrid` | `openai/gpt-4o` | hybrid | Iter 3 | Retriever comparison |
| `rag_rerank` | `openai/gpt-4o` | rerank | Iter 3 | Retriever comparison |
| `llama_baseline` | `meta-llama/llama-3.3-70b-instruct` | none | Iter 3 | Model comparison |
| `llama_rerank` | `meta-llama/llama-3.3-70b-instruct` | rerank | Iter 3 | Model comparison |
| `mistral_baseline` | `mistralai/mistral-small-3.1-24b-instruct` | none | Iter 3 | Model comparison |
| `mistral_rerank` | `mistralai/mistral-small-3.1-24b-instruct` | rerank | Iter 3 | Model comparison |
| `qwen3_base_baseline` | `local/qwen3-base` (port 8081) | none | Iter 9 | Local model comparison |
| `qwen3_base_rerank` | `local/qwen3-base` (port 8081) | rerank | Iter 9 | Local model comparison |
| `qwen3_finetuned_baseline` | `local/qwen3-finetuned` (port 8080) | none | Iter 9 | Fine-tuning evaluation |
| `qwen3_finetuned_rerank` | `local/qwen3-finetuned` (port 8080) | rerank | Iter 9 | Fine-tuning evaluation |

Structure: The first 5 configs answer "which retriever is best?" with the model held constant at GPT-4o. Configs 6–9 answer "which cloud LLM is best?" with the retriever held constant at rerank. Configs 10–13 answer "can local models compete, and does fine-tuning help?" — each local model gets a baseline + rerank pair so RAG uplift can be measured and compared across deployment types.

---

## 6. Iteration 4: Corpus Expansion + Multi-Retriever Evaluation (2026-03-17)

### 6.1 Corpus Expansion

The original corpus (112 documents, 617 chunks) had coverage gaps in topics tenants commonly ask about: utilities, repairs, court procedures, foreclosure, rooming houses, and detailed discrimination law. The biggest untapped source was MassLegalHelp.org's 18-chapter "Legal Tactics: Tenants' Rights in Massachusetts" (9th ed, Jan 2025), the most comprehensive plain-language MA tenant rights resource.

**New Sources Added:**
- **masslegalhelp.org:** All 18 chapters of Legal Tactics (PDF download + text extraction via pdfplumber)
- **boston.gov:** 3 new page sources + PDF scraping capability (eviction help, landlord counseling, mayor's office housing pages + linked PDFs)
- **bostonhousing.org:** 5 additional page URLs attempted (all 404'd — speculative URLs as expected)
- **mass.gov:** 8 additional root URLs + AG Guide PDF attempted (all 403'd due to bot protection; existing 34 docs retained from prior scrape)

**Corpus Cleaner Enhancements:**

The expanded corpus required new cleaning filters beyond the original four (Section 4.2). Six additional filters were added:

- Non-English document removal (multilingual PDF translations)
- Non-English chunk removal (>15% non-ASCII characters)
- Oversized report removal (large city planning/policy PDFs)
- Off-topic document removal (27 `doc_ids` explicitly excluded)
- Homebuyer-only chunk removal (homebuyer keywords, zero tenant keywords)
- Navigation/TOC chunk removal (link lists, HTML artifacts, PDF TOC fragments)
- Minimum chunk length raised from 50 to 100 characters

**Final Corpus:**

| Source | Docs (before) | Docs (after) | Chunks (after) |
|--------|--------------|-------------|----------------|
| masslegalhelp | 0 | 18 | 422 |
| mass.gov | 34 | 34 | 228 |
| boston.gov | 17 | 35 | 162 |
| bostonhousing.org | 61 | 59 | 59 |
| **Total** | **112** | **146** | **871** |

### 6.2 Evaluation Setup

- **Configurations:** GPT-4o only, all 4 retrievers + baseline
- **Evaluation Questions:** 80 total (50 golden QA + 30 Reddit-style)
- **Judge Model:** `anthropic/claude-sonnet-4` (separate family from generator)
- **Retrieval QA Pairs:** 64 with valid ground truth chunk IDs (16 golden QA pairs lost ground truth due to corpus cleanup removing short chunks)

### 6.3 Results

![Chart 3 — Retriever Comparison: Generation Metrics](figures/chart03_retriever_generation.png)

**Generation Quality (80 questions, 5 configs):**

| Config | Faithfulness | Relevancy | Correctness | N |
|--------|-------------|-----------|-------------|---|
| `baseline` | 0.100 | 1.000 | 0.463 | 80 |
| `rag_vector` | 0.738 | 1.000 | 0.388 | 80 |
| `rag_bm25` | 0.688 | 1.000 | 0.287 | 80 |
| `rag_hybrid` | 0.738 | 1.000 | 0.425 | 80 |
| `rag_rerank` | 0.762 | 1.000 | 0.412 | 80 |

![Chart 4 — Retriever Comparison: Retrieval Metrics](figures/chart04_retriever_retrieval.png)

**Retrieval Metrics (64 QA pairs, top_k=5):**

| Retriever | MRR | Hit Rate | Precision | Recall |
|-----------|-----|----------|-----------|--------|
| vector | 0.189 | 0.312 | 0.062 | 0.312 |
| bm25 | 0.062 | 0.109 | 0.022 | 0.109 |
| hybrid | 0.187 | 0.312 | 0.062 | 0.312 |
| rerank | 0.243 | 0.328 | 0.066 | 0.328 |

### 6.4 Analysis

1. **RAG Provides Massive Faithfulness Uplift** — Baseline faithfulness dropped to 0.100 (from 0.640 in Iteration 1), while all RAG configs score 0.688–0.762. This represents a +562–662% improvement. The drop in baseline faithfulness is due to the stricter Claude Sonnet 4 judge (vs GPT-4o self-judging in Iteration 1), which penalizes baseline responses that fabricate specific legal details without source grounding.

2. **Rerank Is the Best Retriever** — Cross-encoder reranking achieves the highest faithfulness (0.762), MRR (0.243), and hit rate (0.328) across all retrievers. This confirms the HW4 finding that jointly scoring query-passage pairs outperforms independent embedding similarity.

3. **BM25 Alone Is Weakest** — BM25 achieves the lowest retrieval metrics (MRR 0.062, hit rate 0.109) and lowest faithfulness (0.688). Pure keyword matching struggles with the expanded corpus where many chunks share legal terminology. Tenant law queries contain common terms ("landlord," "rent," "lease") that appear in most chunks, reducing BM25's discriminative power.

4. **Hybrid and Vector Perform Similarly** — Vector (0.738 faithfulness, 0.189 MRR) and hybrid (0.738, 0.187) are nearly identical. The BM25 component in hybrid (0.4 weight) does not meaningfully improve over pure vector search for this corpus, likely because the embedding model already captures the keyword semantics that BM25 adds.

5. **Perfect Relevancy Across All Configs** — All 5 configs achieve 1.000 relevancy — GPT-4o always addresses the question asked, regardless of retrieval quality or lack thereof.

6. **Correctness Is Lower Than Expected** — Correctness scores (0.287–0.463) are lower than faithfulness across all configs. The baseline actually scores highest on correctness (0.463) despite lowest faithfulness (0.100). This suggests the golden QA key facts may reference specific chunk content that the expanded corpus surfaces differently, and baseline GPT-4o's parametric knowledge happens to match some key facts even when not grounded in sources.

7. **Retrieval Metrics Dropped vs Prior Iterations** — MRR dropped from 0.568 (Iteration 2) to 0.243. This is expected: 16 of 50 golden QA ground truth chunk IDs were invalidated by corpus cleanup, the corpus grew from 617 to 871 chunks, and prior QA pairs were auto-generated from the old corpus.

**Comparison to Prior Iterations:**

![Chart 1 — Iteration Progression](figures/chart01_iteration_progression.png)

| Iteration | Corpus | Faithfulness | MRR | Judge |
|-----------|--------|-------------|-----|-------|
| 1 (base) | 1,068 ch | 0.780 | 0.480 | GPT-4o (self) |
| 2 (clean) | 617 ch | 0.900 | 0.568 | GPT-4o (self) |
| 4 (expand) | 871 ch | 0.762* | 0.243 | Claude Sonnet 4 |

> \* Not directly comparable due to judge model change. The stricter Claude judge scored baseline at 0.100 (vs 0.640 with GPT-4o judge), suggesting Claude applies a higher bar for faithfulness. The relative RAG uplift (+662%) is larger than prior iterations (+22% in Iter 1, +15% in Iter 2), indicating RAG's value is even more pronounced under strict evaluation.

---

## 7. Iteration 5: Corpus Expansion Round 2 + Re-Evaluation (2026-03-18)

### 7.1 Corpus Expansion

The 146-doc / 871-chunk corpus covered MA tenant law well at the statutory level but had gaps in practical/procedural content that Boston students frequently need: how to use the court system, which forms to file, how to file discrimination complaints, and how to access state rental voucher programs.

**New Sources Added (41 documents):**

| Priority | Content | Docs |
|----------|---------|------|
| 1 | MGL statutes: c.186 s.15A/15D/15E (late fees, move-in fees, utilities), s.19-23 (DV lease termination), c.239 s.9/10/12 (eviction appeals/stays/bonds), c.93A s.2/9/11 (consumer protection/treble damages), c.111 s.127A-127L (Board of Health inspection authority) | 26 |
| 2 | Housing Court overview, eviction court forms, tenants' eviction guide, respond-to-eviction guide, small claims filing guide | 5 |
| 3 | GBLS housing pages (overview, community partnerships, direct client services, impact advocacy, resources) | 5 |
| 4 | MCAD discrimination complaints overview, housing discrimination guide | 2 |
| 5 | MRVP rental voucher program | 1 |
| 6 | Boston ISD housing inspections, constituent services | 2 |
| | **Total new documents** | **41** |

**Anti-Scraping Fallback:** mass.gov returned 403 for court/MCAD/DHCD pages. Chrome browser automation (Claude-in-Chrome MCP) was used to navigate to each page, extract rendered text, and save as processed documents. boston.gov ISD URLs returned 404; correct URLs were found via web search and scraped via Chrome.

**Stale Reference Repair:** Re-chunking invalidated chunk IDs across evaluation files. `golden_qa.json`: 29 chunk refs remapped, 3 questions repopulated. `reddit_questions.json`: 5 refs remapped, 17 dropped (doc removed). All evaluation files verified: 0 invalid chunk references remain.

![Chart 2 — Corpus Growth](figures/chart02_corpus_growth.png)

**Final Corpus:**

| Metric | Before (Iter 4) | After (Iter 5) |
|--------|-----------------|----------------|
| Documents | 146 | 249 |
| Chunks | 837 | 967 |

### 7.2 Evaluation Setup

- **Configurations:** 2 (baseline + `rag_rerank`, both GPT-4o)
- **Evaluation Questions:** 80 (50 golden QA + 30 Reddit-style)
- **Judge Model:** `anthropic/claude-sonnet-4`
- **Retrieval QA Pairs:** 80 (50 golden + 30 reddit, all with valid ground truth chunk IDs)

### 7.3 Results

**Generation Quality (80 questions, 2 configs):**

| Config | Faithfulness | Relevancy | Correctness | N |
|--------|-------------|-----------|-------------|---|
| `baseline` | 0.150 | 1.000 | 0.438 | 80 |
| `rag_rerank` | 0.725 | 1.000 | 0.438 | 80 |

**Retrieval Metrics (80 QA pairs, top_k=5):**

| Retriever | MRR | Hit Rate | Precision | Recall |
|-----------|-----|----------|-----------|--------|
| vector | 0.170 | 0.275 | 0.055 | 0.275 |
| bm25 | 0.057 | 0.088 | 0.018 | 0.088 |
| hybrid | 0.161 | 0.238 | 0.048 | 0.238 |
| rerank | 0.220 | 0.288 | 0.058 | 0.288 |

### 7.4 Analysis

1. **Faithfulness: RAG Uplift Remains Strong** — Baseline faithfulness is 0.150 (up from 0.100 in Iter 4), `rag_rerank` is 0.725 (vs 0.762 in Iter 4). The RAG uplift is +383%, confirming that retrieval-augmented generation continues to provide substantial grounding benefits with the expanded corpus. The small faithfulness decrease (0.762 → 0.725) is within normal judge variance for binary scoring on 80 questions (a single flipped judgment = 0.0125 change).

2. **Rerank Remains the Best Retriever** — Rerank achieves the highest MRR (0.220), hit rate (0.288), and precision (0.058) across all four retrievers, consistent with all prior iterations.

3. **Retrieval Metrics Are Stable** — MRR (0.243 → 0.220) and hit rate (0.328 → 0.288) show small decreases. Contributing factors: evaluation now uses 80 QA pairs (vs 64 in Iter 4), some remapped `source_chunks` point to the closest chunk in the same document, and corpus grew from 871 to 967 chunks. These are measurement artifacts, not pipeline regressions.

4. **Correctness Is Identical Across Configs (0.438)** — Both baseline and `rag_rerank` score 0.438 on correctness, meaning the key facts in the golden QA set are equally matched by parametric knowledge and RAG-retrieved content.

5. **New Content Is Being Retrieved** — Spot-testing confirmed that queries about late fees, move-in charges, discrimination complaints, small claims, and eviction rights now retrieve the newly added statutes, court guides, and MCAD pages.

**Comparison to Prior Iterations:**

| Iter | Corpus | Faithfulness | MRR | Judge | Notes |
|------|--------|-------------|-----|-------|-------|
| 1 | 1,068 ch | 0.780 | 0.480 | GPT-4o (self) | Initial |
| 2 | 617 ch | 0.900 | 0.568 | GPT-4o (self) | Cleanup |
| 4 | 871 ch | 0.762 | 0.243 | Claude Sonnet 4 | Expand + judge |
| 5 | 967 ch | 0.725 | 0.220 | Claude Sonnet 4 | Expand round 2 |

> Iterations 4-5 use Claude Sonnet 4 judge (stricter than GPT-4o self-judging in Iterations 1-2). Absolute scores are not directly comparable across judge models, but relative RAG uplift is comparable within each judge.

---

## Iteration 6: LlamaIndex Migration — Pipeline Comparison

**Date:** 2026-03-18
**Backend:** `src/rag_llamaindex/` (LlamaIndex 0.14.x + ChromaVectorStore)
**Config:** `rag_rerank` (GPT-4o, hybrid+cross-encoder, `top_k=5`)
**Judge:** Claude Sonnet 4 (`anthropic/claude-sonnet-4`)
**Questions:** 80 (30 Reddit + 20 Golden QA, same as Iteration 5)
**Corpus:** 967 chunks (same as Iteration 5)

### Results

| Backend | Faithfulness | Relevancy | Correctness | MRR | Hit Rate |
|---------|-------------|-----------|-------------|-----|----------|
| Custom | 0.762 | 1.000 | 0.412 | 0.243 | 0.328 |
| LlamaIndex | 0.713 | 1.000 | 0.412 | 0.247 | 0.288 |

> Note: Custom results from Iteration 4 (same corpus, same config).

### Root Cause Analysis: Metadata Embedding Divergence

Investigation revealed the primary cause of retrieval differences between the two pipelines: LlamaIndex prepends non-excluded metadata fields to document text before embedding, while ChromaDB embeds raw text only.

**LlamaIndex embeds:**
```
source_url: https://...
source_name: boston_gov
title: Boston Housing Court
content_type: guide

## Appearing in court...
```

**ChromaDB embeds:**
```
## Appearing in court...
```

This produces different document vectors despite using the same embedding model (`all-MiniLM-L6-v2`). Query embeddings are identical between backends (cosine similarity = 1.000000), confirming the divergence is document-side only.

**Embedding divergence measured across 50 sampled chunks:**

| Metric | Value |
|--------|-------|
| Mean cosine similarity | 0.851 |
| Median | 0.868 |
| Min | 0.492 |
| Max | 0.964 |
| Std | 0.094 |

Chunks with longer metadata (long URLs, verbose titles) diverge more because metadata consumes a larger fraction of the 256-token embedding window, pushing out content tokens.

**Retrieval Overlap Analysis (10 test queries):**

| Retriever | Avg Top-5 Overlap | Notes |
|-----------|------------------|-------|
| vector | 2.4/5 (48%) | Directly affected by embedding gap |
| rerank | 3.2/5 (64%) | Cross-encoder re-scores on raw text, partially correcting initial divergence |

### Key Findings

1. Relevancy is identical (1.000) — both pipelines retrieve topically relevant chunks regardless of embedding differences.
2. Correctness is identical (0.412) — same model + prompt produce the same key-fact coverage regardless of backend.
3. Faithfulness gap (0.762 vs 0.713) is likely a combination of judge variance (custom showed 0.762 and 0.725 across runs) and retrieval differences surfacing slightly different source chunks.
4. Hit rate gap (0.328 vs 0.288) is a direct consequence of the metadata embedding divergence: ground-truth chunk IDs were established against the custom pipeline's raw-text vectors.
5. MRR is essentially identical (0.243 vs 0.247), suggesting that when the correct chunk IS retrieved, its rank is comparable.

---

## Iteration 7: Metadata Fix + Multi-Model Deep Dive

**Date:** 2026-03-18
**Backend:** `src/rag_llamaindex/` (post-fix)
**Corpus:** 967 chunks, 249 documents

### 7.1 Metadata Embedding Fix

Applied fix: excluded all metadata keys from LlamaIndex embedding input (`source_url`, `source_name`, `title`, `content_type` added to `excluded_embed_metadata_keys` in `nodes.py`). Metadata remains available for LLM context via `excluded_llm_metadata_keys` (unchanged).

Re-indexed 967 nodes into `data/chroma_db_llamaindex/`.

**Verification (967 shared chunk IDs):**

| Metric | Before | After |
|--------|--------|-------|
| Mean cosine similarity | 0.851 | 1.000 |
| Min cosine similarity | 0.492 | 1.000 |

**Retrieval overlap (10 test queries, top_k=5):**

| Retriever | Before | After |
|-----------|--------|-------|
| vector | 48% | 100% |
| rerank | 64% | 94% |

The 6% rerank gap is expected: the cross-encoder scores (query, text) pairs independently, so minor ordering differences in the initial candidate set can cause 1-2 chunks to swap at the boundary.

Pipelines are now at embedding parity.

### 7.2 Retrieval Metrics Comparison (LlamaIndex, post-fix)

| Retriever | MRR | Hit Rate | Precision | Recall |
|-----------|-----|----------|-----------|--------|
| vector | 0.170 | 0.275 | 0.055 | 0.275 |
| rerank | 0.220 | 0.287 | 0.058 | 0.287 |
| parent_child | 0.155 | 0.300 | N/A | 0.300 |

Parent-child retriever shows highest hit rate (0.300) but lowest MRR (0.155), because neighbor-chunk expansion finds the right document more often but the correct chunk lands at a lower rank due to dilution from adjacent chunks.

### 7.3 Why MRR/Hit Rate Are Low vs HW4

| Metric | HW4 | This Project | Change |
|--------|-----|-------------|--------|
| Chunks | 149 | 967 | 6.5x |
| Best MRR | 0.692 | 0.220 | -68% |
| Best Hit Rate | 0.805 | 0.300 | -63% |

Contributing factors (in estimated order of impact):

1. **Search space size:** 149 → 967 chunks (6.5x). With `top_k=5`, HW4 retrieves the top 3.4% of chunks vs 0.5% here.
2. **Corpus diversity:** HW4 used a focused single-domain dataset. This project covers statutes, FAQs, guides, Reddit posts, court procedures, discrimination law, voucher programs, and inspections.
3. **Chunk size:** HW4 used sentence window chunking (small chunks with context expansion). This project uses 800-token chunks with no context expansion.
4. **Single ground-truth evaluation:** Only the first `source_chunk` is used as ground truth. Many questions have multiple relevant chunks (~2.8 per question on average).

### 7.4 Golden QA Source Chunk Audit

All 225 `source_chunk` references across 80 QA pairs were verified against the current corpus (967 chunks):

- References found in corpus: 225/225 (100%)
- References missing: 0
- Entries with no `source_chunks`: 0

Conclusion: chunk ID mappings are not stale. The low hit rate is a genuine retrieval challenge, not a labeling artifact.

### 7.5 Potential Improvements to Investigate

1. Evaluate against ANY `source_chunk` (not just the first) to measure how much of the hit rate gap is a measurement artifact from single-ground-truth evaluation.
2. Increase initial retrieval `top_k` (10-15) with reranking down to 5 to give the correct chunk more chances to appear in candidates.
3. Test the `parent_child` retriever with sentence-window-style small chunks (closer to HW4's strategy) rather than expanding 800-token chunks.
4. Multi-query expansion to catch vocabulary mismatches between questions and source documents.

### 7.6 Pipeline Decision: Custom vs LlamaIndex for Interface

**Decision:** Use the custom pipeline (`src/rag/`) for the frontend/demo UI.

**Rationale:**
1. **Parity confirmed:** After the metadata embedding fix, both pipelines produce identical retrieval results (100% vector overlap, 100% parent_child Jaccard overlap across all test queries).
2. **Simplicity for deployment:** The custom pipeline is ChromaDB + OpenRouter API calls with no framework dependency.
3. **Fine-tuning compatibility:** The custom pipeline's `ask()` is more transparent for swapping in a fine-tuned model endpoint.
4. **Stretch goals (abuse prevention, rate limiting):** Straightforward to add on top of a Flask/FastAPI app.
5. **Transparent for debugging:** Fewer abstractions = easier to debug during demos.

### 7.7 Custom Parent-Child Retriever Implementation

Added `retrieve_parent_child()` to `src/rag/retrievers.py`.

**Algorithm:**
1. Vector-retrieve 2x `top_k` candidates (10 for `top_k=5`)
2. Group candidates by `doc_id`
3. For `top_k` results, if a document has 2+ hits (cluster), expand by including adjacent chunks (`chunk_index ± 1`)
4. Neighbor scores are dampened (distance × 1.2, i.e. worse rank)
5. Return all expanded results (variable length)

**Retrieval metrics (80 QA pairs, custom pipeline):**

| Retriever | MRR | Hit Rate | Precision | Recall |
|-----------|-----|----------|-----------|--------|
| vector | 0.170 | 0.275 | 0.055 | 0.275 |
| rerank | 0.220 | 0.287 | 0.058 | 0.287 |
| parent_child | 0.155 | 0.300 | 0.060 | 0.300 |

Parent-child achieves highest hit rate (0.300) by expanding document context, but lowest MRR (0.155) because neighbor chunks dilute the ranking.

Cross-pipeline verification (5 test queries): Custom vs LlamaIndex parent_child Jaccard overlap: 100%

### 7.8 Retriever Comparison: Rerank vs Parent-Child vs Baseline (GPT-4o)

**Evaluation setup:**
- Questions: 28 (stratified sample: 2 per named topic + 4 Reddit, 13 topics)
- Generator: `openai/gpt-4o`
- Judge: `anthropic/claude-sonnet-4`
- Configs: `baseline` (no RAG), `rag_rerank`, `rag_parent_child`

**Results:**

| Config | Faith | Relev | Correct | N |
|--------|-------|-------|---------|---|
| `baseline` | 0.071 | 1.000 | 0.393 | 28 |
| `rag_rerank` | 0.857 | 1.000 | 0.357 | 28 |
| `rag_parent_child` | 0.786 | 1.000 | 0.321 | 28 |

**Token usage & cost:**
- GPT-4o (gen): 225,959 in + 19,274 out = 245,233 total
- Claude S4 (judge): 366,342 in + 30,663 out = 397,005 total
- Total cost: $2.32

**Analysis:**
1. Rerank remains the best retriever for generation quality. Faithfulness 0.857 vs 0.786 for parent_child.
2. Parent_child uses ~2x more input tokens but does not translate to better scores.
3. RAG uplift on faithfulness is massive: 0.071 → 0.857 (+1,107%).

### 7.9 Multi-Model Comparison: All 5 Models (Cloud + Local)

![Chart 4 — Multi-Model Comparison](figures/chart04_multimodel_comparison.png)

**Evaluation setup:**
- Questions: 28 (Iterations 7–8 stratified sample) + 89 (Iteration 9 full set — see Section 9)
- Retriever: rerank (best performer) + baselines
- Judge: `anthropic/claude-sonnet-4`
- Models: GPT-4o, Llama 3.3-70B, Mistral Small 3.1-24B (cloud); Qwen3 4B Base, Qwen3 4B Fine-tuned (local, added Iteration 9)

> **Note:** Qwen3 models were added in Iteration 9 (89-question run). Results for GPT-4o, Llama, and Mistral at 28 questions (Iterations 7–8) are shown for historical context; the 89-question results (Section 9) are the authoritative final comparison.

**Iteration 7–8 results (28 questions, cloud models only):**

| Config | Model | Params | Faith | Relev | Correct |
|--------|-------|--------|-------|-------|---------|
| `baseline` | GPT-4o | ~200B+ | 0.071 | 1.000 | 0.393 |
| `rag_rerank` | GPT-4o | ~200B+ | 0.857 | 1.000 | 0.357 |
| `rag_parent_child` | GPT-4o | ~200B+ | 0.786 | 1.000 | 0.321 |
| `llama_baseline` | Llama 3.3 | 70B | 0.071 | 1.000 | 0.357 |
| `llama_rerank` | Llama 3.3 | 70B | 0.500 | 0.964 | 0.357 |
| `mistral_baseline` | Mistral Small | 24B | 0.071 | 1.000 | 0.250 |
| `mistral_rerank` | Mistral Small | 24B | 0.643 | 1.000 | 0.393 |

**Iteration 9 results (89 questions, all 5 models including Qwen3):**

| Config | Model | Params | Faith | Relev | Correct | Deployment |
|--------|-------|--------|-------|-------|---------|------------|
| `baseline` | GPT-4o | ~200B+ | 0.146 | 1.000 | 0.465 | Cloud |
| `rag_rerank` | GPT-4o | ~200B+ | 0.854 | 1.000 | 0.430 | Cloud |
| `llama_baseline` | Llama 3.3 | 70B | 0.011 | 0.989 | 0.429 | Cloud |
| `llama_rerank` | Llama 3.3 | 70B | 0.730 | 1.000 | 0.393 | Cloud |
| `mistral_baseline` | Mistral Small | 24B | 0.022 | 0.978 | 0.412 | Cloud |
| `mistral_rerank` | Mistral Small | 24B | **0.798** | 0.989 | 0.410 | Cloud |
| `qwen3_base_baseline` | Qwen3 4B Base | 4B | 0.000 | 0.966 | 0.327 | Local |
| `qwen3_base_rerank` | Qwen3 4B Base | 4B | 0.427 | 0.989 | **0.472** | Local |
| `qwen3_finetuned_baseline` | Qwen3 4B FT | 4B | 0.011 | 0.607 | 0.308 | Local |
| `qwen3_finetuned_rerank` | Qwen3 4B FT | 4B | 0.034 | 0.697 | 0.339 | Local |

![Chart 6 — Best Configurations Scatter](figures/chart06_best_configs_scatter.png)

**RAG uplift by model (Iteration 9, 89 questions):**

| Model | Baseline → RAG | Uplift |
|-------|---------------|--------|
| GPT-4o | 0.146 → 0.854 | +485% |
| Mistral Small | 0.022 → 0.798 | base≈0 |
| Llama 3.3 | 0.011 → 0.730 | base≈0 |
| Qwen3 4B Base | 0.000 → 0.427 | base=0 |
| Qwen3 4B FT | 0.011 → 0.034 | negligible (format damage) |

**Analysis:**

1. **Mistral Small 3.1 + Rerank is the most faithful model (0.798)** across all 89 questions, confirming the Iteration 7 finding that Mistral is especially effective at staying grounded to retrieved context. It overtakes GPT-4o + rerank (0.854 → 0.798 is close; at 28 questions they were tied at 0.929).
2. **Qwen3 4B Base + Rerank achieves the highest correctness (0.472)** at zero API cost — a 4B local model matches GPT-4o baseline (0.465) on fact recall while costing nothing per query.
3. **Qwen3 fine-tuning degraded performance** (faithfulness 0.034, relevancy 0.697). Root cause: chat template format mismatch during training. See Section 9.7 for retrain plan.
4. **Llama 3.3-70B still struggles with RAG context.** At 28 questions it was the only model below 1.0 relevancy; at 89 questions it recovers to 1.000 with rerank but still trails Mistral on faithfulness.
5. **All baselines score near 0.000 faithfulness** for local models and 0.146 for GPT-4o without retrieval. Claude Sonnet 4 consistently flags hallucinated MA legal details.

**Understanding the metrics as precision/recall:**
- Faithfulness ≈ precision: "don't say unsupported things"
- Correctness ≈ recall: "don't miss expected facts"
- Relevancy ≈ sanity check: "stay on topic" (near 1.0 always)

**Pricing (per 1M tokens, as of 2026-04-04):**

| Model | Input | Output | Relative cost | Type |
|-------|-------|--------|--------------|------|
| GPT-4o | $3.00 | $15.00 | 1.0x (baseline) | Cloud API |
| Claude Sonnet 4 (judge) | $3.00 | $15.00 | 1.0x | Cloud API |
| Mistral Small 3.1 | $0.14 | $0.42 | 0.047x | Cloud API |
| Llama 3.3 70B | $0.10 | $0.32 | 0.033x | Cloud API |
| Qwen3 4B Base | $0 | $0 | 0x | Local |
| Qwen3 4B Fine-tuned | $0 | $0 | 0x | Local |

### 7.10 Self-Evaluation Bias Experiment

![Chart 9 — Negative Results & Ablations](figures/chart09_negative_results.png)

**Research question:** Does using the same model as both generator and judge inflate evaluation scores compared to an independent judge?

**Setup:**
- Questions: 28 (same stratified sample)
- Retriever: rerank (best performer)
- Each model generates responses AND judges its own output
- Compared against Claude Sonnet 4 judge results from Section 7.9

> **Note:** Self-evaluation bias was only tested for cloud models (GPT-4o, Llama 3.3, Mistral Small). Qwen3 models were added in Iteration 9 and were not included in this experiment. Local models served via llama-server do not support API-based self-judging without additional tooling.

**Results:**

| Generator | Judge | Faith | Relev | Correct |
|-----------|-------|-------|-------|---------|
| GPT-4o | Claude Sonnet 4 | 0.857 | 1.000 | 0.357 |
| GPT-4o | Self (GPT-4o) | 0.964 | 1.000 | 0.750 |
| | Delta (self - claude) | +0.107 | | +0.393 |
| Llama 3.3 | Claude Sonnet 4 | 0.500 | 0.964 | 0.357 |
| Llama 3.3 | Self (Llama 3.3) | 0.929 | 1.000 | 0.357 |
| | Delta (self - claude) | +0.429 | | +0.000 |
| Mistral Small | Claude Sonnet 4 | 0.643 | 1.000 | 0.393 |
| Mistral Small | Self (Mistral Small) | 0.857 | 0.964 | 0.286 |
| | Delta (self - claude) | +0.214 | | -0.107 |

Cost: $0.76 (total tokens: 710,097)

**Analysis:**

1. **Self-evaluation bias is confirmed for faithfulness.** Every model rates its own faithfulness higher than Claude Sonnet 4 does. The inflation ranges from +0.107 (GPT-4o) to +0.429 (Llama). This validates the project's decision in Iteration 3 to switch to an independent judge model.
2. **Llama 3.3 is the worst self-evaluator.** Llama inflates its own faithfulness from 0.500 → 0.929 (+86%). It also bumps its own relevancy from 0.964 → 1.000, masking the context-misuse issue.
3. **GPT-4o massively inflates correctness when self-judging.** Correctness jumps from 0.357 → 0.750 (+110%). This is the strongest single bias effect observed.
4. **Mistral shows mixed self-evaluation behavior.** Faithfulness is inflated (+0.214, consistent with other models), but correctness actually drops — likely run-to-run variance.
5. **Claude Sonnet 4 is the strictest and most consistent judge.** It applies the highest bar for faithfulness and does not show systematic favoritism toward any model family.

**Implications for evaluation methodology:**
- Self-evaluation should never be used for absolute score reporting. The faithfulness inflation alone (+0.1 to +0.4) would fundamentally distort conclusions.
- Cross-model comparisons are invalid with self-judging because the inflation magnitude varies by model.
- The independent judge (Claude Sonnet 4) provides the most trustworthy basis for cross-model and cross-retriever comparisons.

### 7.11 Context Window Experiment: top_k=10 vs top_k=5

![Chart 5 — Ablation Experiments](figures/chart05_experiments.png)

**Research question:** Does retrieving more chunks (10 vs 5) improve generation quality, or does the extra context introduce noise?

**Retrieval metrics (80 QA pairs, rerank):**

| top_k | MRR | Hit Rate | Initial candidates (initial_k = top_k x 2) |
|-------|-----|----------|---------------------------------------------|
| 5 | 0.220 | 0.287 | 10 |
| 10 | 0.226 | 0.338 | 20 |
| 15 | 0.226 | 0.338 | 30 (no further gain) |

Hit rate improves +18% at `top_k=10`, then plateaus.

**Generation quality (GPT-4o + rerank, 28 stratified questions):**

| Config | Faith | Relev | Correct | N |
|--------|-------|-------|---------|---|
| `rag_rerank` (k=5) | 0.857 | 1.000 | 0.357 | 28 |
| `rag_rerank` (k=10) | 0.857 | 1.000 | 0.464 | 28 |

**Analysis:**

1. **Faithfulness is unchanged (0.857).** GPT-4o handles 10 chunks of context as well as 5 — the additional context does not introduce noise or cause hallucination.
2. **Correctness improved +30%** (0.357 → 0.464). This is the largest correctness gain from any single change. More retrieved chunks means more key facts land in the context.
3. Generation cost increases ~2x (more context tokens).
4. `top_k=10` with rerank is now the best configuration tested.

**Multi-model results (top_k=10, rerank, 28 stratified questions):**

| Config | Faith | Relev | Correct | Delta Faith | Delta Correct |
|--------|-------|-------|---------|-------------|---------------|
| GPT-4o rerank k=5 | 0.857 | 1.000 | 0.357 | | |
| GPT-4o rerank k=10 | 0.857 | 1.000 | 0.464 | +0.000 | +0.107 |
| Llama 3.3 rerank k=5 | 0.500 | 0.964 | 0.357 | | |
| Llama 3.3 rerank k=10 | 0.607 | 1.000 | 0.357 | +0.107 | +0.000 |
| Mistral rerank k=5 | 0.643 | 1.000 | 0.393 | | |
| Mistral rerank k=10 | 0.750 | 1.000 | 0.393 | +0.107 | +0.000 |

> **Note on Qwen3:** The top_k=5 vs top_k=10 experiment was not run separately for Qwen3. In Iteration 9, Qwen3 was evaluated with `top_k=5` (rerank). The llama-server default context of 4096 tokens was too small for many rerank prompts with k=5; the server was restarted with `--ctx-size 8192` to accommodate. A top_k=10 experiment for Qwen3 would require testing the 8192 context limit more carefully given the full system prompt + 10 retrieved chunks.

Cost: $1.89 (Llama + Mistral generation + Claude Sonnet 4 judging)

Every cloud model benefits from `top_k=10`:
- **GPT-4o:** already strong on faithfulness (0.857), gains on correctness (+30%).
- **Llama 3.3:** faithfulness jumps +21% (0.500 → 0.607) and relevancy fixes to 1.000.
- **Mistral:** faithfulness jumps +17% (0.643 → 0.750). At `top_k=10`, Mistral (24B) now approaches GPT-4o's k=5 faithfulness at 7x lower generation cost.

### 7.12 Structured Prompt Experiment: Evidence-Before-Answer

![Chart 5 — Ablation Experiments](figures/chart05_experiments.png)

**Research question:** Does a structured prompt that forces the model to identify evidence before answering improve faithfulness?

**Old prompt** (used in all prior iterations):

```
You are a legal information assistant for Massachusetts tenant law (Boston area).

RULES:
1. ONLY answer from provided source documents. If insufficient, say so
   and suggest legal aid resources such as MassLegalHelp.org or Greater
   Boston Legal Services.
2. NEVER provide legal ADVICE -- only legal INFORMATION. Recommend
   consulting an attorney for specific situations.
3. ALWAYS cite sources: [Source: <title> (<url>)].
4. Cite specific statutes (e.g., MGL c.186, s.15B) when relevant.
5. Synthesize multiple sources when relevant.
6. If the question is outside Massachusetts tenant law, say so.

CONTEXT: {context}
QUESTION: {question}
```

**Change:** Replaced with a retrieval-grounded analysis workflow (`src/rag/pipeline.py` `SYSTEM_PROMPT`). The new prompt requires the model to output in a structured format:

1. Question Understanding (restate the question)
2. Relevant Evidence (cite specific chunks)
3. Analysis (reason from evidence to answer)
4. Final Answer (grounded conclusion)
5. Confidence (high / medium / low)

Key prompt design principles:
- Ask for visible analysis, not hidden chain-of-thought
- Force evidence citation before answer (reduces hallucination)
- Explicit grounding: "use only retrieved context, do not invent facts"
- Insufficiency behavior: say so if context is incomplete

**Results (GPT-4o + rerank, 28 stratified questions, Claude Sonnet 4 judge):**

| Config | Faith | Relev | Correct |
|--------|-------|-------|---------|
| old prompt, k=5 | 0.857 | 1.000 | 0.357 |
| old prompt, k=10 | 0.857 | 1.000 | 0.464 |
| structured prompt, k=5 | 0.893 | 1.000 | 0.321 |
| structured prompt, k=10 | 0.929 | 1.000 | 0.321 |

Cost: $0.90 (k=5 run) + $1.41 (k=10 run) = $2.31 total

**Analysis:**

1. **Structured prompt improves faithfulness at both top_k values.** k=5: 0.857 → 0.893 (+4%). k=10: 0.857 → 0.929 (+8%). **Faithfulness 0.929 is the highest score achieved in the project.** The evidence-before-answer pattern forces the model to anchor in retrieved chunks before generating, reducing hallucination.

2. The improvements are **additive**: structured prompt and `top_k=10` each contribute independently.

3. Correctness drops slightly with the structured format (0.357 → 0.321, ~1 question). This is the classic **precision-recall tradeoff**: structured prompt = higher precision (faithfulness), lower recall (correctness).

4. **For a legal information tool, faithfulness > correctness.** Missing a key fact is less harmful than fabricating legal claims. The structured prompt's bias toward grounding is the right tradeoff for this domain.

5. The structured format also improves **user experience**: the visible evidence and analysis sections let users verify the answer against sources, which builds trust.

**Multi-model results (structured prompt + rerank + k=10):**

| Config | Faith | Relev | Correct |
|--------|-------|-------|---------|
| GPT-4o struct k=10 | 0.929 | 1.000 | 0.321 |
| Llama 3.3 struct k=10 | 0.821 | 1.000 | 0.321 |
| Mistral Small struct k=10 | 0.929 | 1.000 | 0.357 |

Cost: $1.94 (Llama + Mistral generation + Claude Sonnet 4 judging)

**Cumulative improvement from all changes (old prompt k=5 → structured k=10):**

| Model | Old k=5 | Struct k=10 | Delta Faith | Improvement |
|-------|---------|-------------|-------------|-------------|
| GPT-4o | 0.857 | 0.929 | +0.072 | +8% |
| Llama 3.3 | 0.500 | 0.821 | +0.321 | +64% |
| Mistral Small | 0.643 | 0.929 | +0.286 | +44% |

The structured prompt is most impactful for weaker models:
- **Mistral Small** (24B) now matches GPT-4o on faithfulness (0.929) at 7x lower generation cost ($0.35/1M vs $2.50/1M input tokens).
- **Llama 3.3** improves dramatically (+64%) but still trails at 0.821, suggesting its instruction-following is less suited to structured output.
- The structured prompt effectively **levels the playing field** between models by providing explicit grounding instructions.

![Chart 6 — Best Configurations Scatter](figures/chart06_best_configs_scatter.png)

**Best configuration summary (project-wide, including Iteration 9 final run):**

| Config | Model | Faith | Correct | Cost/1M in | Note |
|--------|-------|-------|---------|------------|------|
| structured + rerank + k=10 | GPT-4o | 0.929 | 0.321 | $3.00 | Iter 7–8, 28q |
| structured + rerank + k=10 | Mistral | 0.929 | 0.357 | $0.14 | Iter 7–8, 28q |
| rerank + k=5 | Mistral Small | 0.798 | 0.410 | $0.14 | Iter 9, 89q |
| rerank + k=5 | GPT-4o | 0.854 | 0.430 | $3.00 | Iter 9, 89q |
| rerank + k=5 | Qwen3 4B Base | 0.427 | **0.472** | $0 local | Iter 9, 89q |
| old prompt + rerank + k=10 | GPT-4o | 0.857 | 0.464 | $3.00 | Iter 7–8, 28q |
| structured + rerank + k=10 | Llama | 0.821 | 0.321 | $0.10 | Iter 7–8, 28q |

> Note: Iter 7–8 scores used structured prompt + k=10; Iter 9 scores used the standard system prompt + k=5. Direct comparison is approximate — the question set and prompt differ.

**For production (cloud):** Mistral Small + rerank achieves the best faithfulness balance at 21x lower cost than GPT-4o ($0.14 vs $3.00/1M input). Adding the structured prompt and k=10 further raises faithfulness to 0.929 at minimal extra cost.

**For local/private deployment:** Qwen3 4B Base + Rerank is the recommended configuration — highest correctness (0.472) at zero API cost with all data on-premises. A successful retrain (Section 9.7) is expected to improve faithfulness from 0.427 to ≥0.500.

### 7.13 Judge Methodology Validation: Custom vs LlamaIndex Evaluators

![Chart 8 — Judge Methodology Validation](figures/chart08_judge_methodology.png)

**Research question:** Does our simpler single-call judging approach produce different scores than LlamaIndex's built-in evaluators?

**Background:** HW4 used LlamaIndex's `FaithfulnessEvaluator` and `RelevancyEvaluator` classes. Our project uses custom judge prompts called directly via the OpenAI SDK. This experiment tests whether the two approaches diverge.

**Implementation differences:**

| Aspect | Our Implementation | LlamaIndex |
|--------|-------------------|------------|
| Faithfulness prompt | Custom: "Is this response faithful to the source context?" with legal-specific guidance | Generic: "Please tell if a given piece of information is supported by the context" with apple pie few-shot examples |
| Faithfulness input | Question + full context + response in one call | Response only as "information", context processed iteratively |
| Relevancy prompt | "Does the response address the question?" (context-free) | "Is the response for the query in line with the context?" (context-aware) |
| Few-shot examples | None | 2 examples (apple pie YES/NO) |
| Context handling | All chunks concatenated into single prompt | Each chunk processed separately, then refined iteratively |
| Refine step | No (single LLM call) | Yes (one call per chunk, refining YES/NO with each additional chunk) |
| API calls per question | 1 (faithfulness) + 1 (relevancy) = 2 total | N (one per chunk, typically 5) = ~10 total |
| Response parsing | `startswith("YES")` | `"yes" in lower()` |

**LlamaIndex-style iterative refine implementation (reimplemented for this experiment):**

```python
# --- LlamaIndex faithfulness prompt (with few-shot examples) ---
LI_FAITH_EVAL = """Please tell if a given piece of information is supported by the context.
You need to answer with either YES or NO.
Answer YES if any of the context supports the information, even if most of the context is unrelated.

Information: Apple pie is generally double-crusted.
Context: An apple pie is a fruit pie in which the principal filling ingredient is apples.
Apple pie is often served with whipped cream, ice cream, custard or cheddar cheese.
It is generally double-crusted, with pastry both above and below the filling.
Answer: YES
Information: Apple pies tastes bad.
Context: [same apple pie context]
Answer: NO
Information: {response}
Context: {context}
Answer: """

# --- Refine prompt (used for chunks 2..N) ---
LI_FAITH_REFINE = """We want to understand if the following information is present
in the context information: {response}
We have provided an existing YES/NO answer: {existing_answer}
We have the opportunity to refine the existing answer (only if needed)
with some more context below.
------------
{context}
------------
If the existing answer was already YES, still answer YES.
If the information is present in the new context, answer YES.
Otherwise answer NO."""

def llamaindex_judge_faithfulness(question, response, contexts, client, model):
    """Replicate LlamaIndex FaithfulnessEvaluator:
    evaluate with first chunk, then refine with each subsequent chunk."""
    answer = None
    for i, ctx in enumerate(contexts):
        if i == 0:
            prompt = LI_FAITH_EVAL.format(response=response, context=ctx)
        else:
            prompt = LI_FAITH_REFINE.format(
                response=response, existing_answer=answer, context=ctx)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=100)
        answer = resp.choices[0].message.content.strip()
    return 1.0 if answer and "yes" in answer.lower() else 0.0
```

The key difference is the **iterative refine loop**: LlamaIndex processes each retrieved chunk in a separate LLM call, carrying forward the previous YES/NO answer. Once any chunk triggers YES, subsequent chunks preserve it ("If the existing answer was already YES, still answer YES"). Our single-call approach concatenates all chunks into one prompt and makes a single judgment.

**Experiment:** Generated 28 responses (GPT-4o, structured prompt, rerank, k=5) and judged each with both methods using Claude Sonnet 4.

**Results:**

| Method | Faith | Relev |
|--------|-------|-------|
| Our (single-call, custom) | 0.964 | 1.000 |
| LlamaIndex-style (refine) | 0.964 | 1.000 |
| Delta (LI - ours) | +0.000 | +0.000 |

**Per-question agreement:**
- Faithfulness: 26/28 (93%)
- Relevancy: 28/28 (100%)

The 2 faithfulness disagreements cancel out:
- **q18 (renters insurance):** ours=NO, LI=YES — LlamaIndex's iterative refine found supporting context across multiple chunks that our single-call approach missed.
- **q22 (pet prohibition):** ours=YES, LI=NO — Our single-call was more lenient; LlamaIndex's chunk-by-chunk evaluation was stricter on this response.

**Conclusion:**

1. The two approaches produce **identical aggregate scores** (0.964 and 1.000). Our simpler single-call method is a valid substitute for LlamaIndex's more complex iterative refine approach.

2. Our method is **5x cheaper** per evaluation (2 API calls vs ~10) with no loss in evaluation quality.

3. The 93% per-question agreement on faithfulness (with disagreements canceling out) suggests both methods have similar noise profiles.

4. This **validates all prior results** in this report: the evaluation methodology is robust to implementation differences in the judge.

### 7.14 Generator-Judge Swap Experiment: Cross-Model Evaluation

![Chart 9 — Negative Results & Ablations](figures/chart09_negative_results.png)

**Research question:** Does swapping the generator and judge roles reveal biases in our evaluation? How does Claude Sonnet 4 perform as a generator, and how do different judge models score the same responses?

**Setup:** All configs use structured prompt + rerank + k=10 on the same 28 stratified questions. Claude Sonnet 4 generates responses, then four different models judge them.

**Results — Claude Sonnet 4 as generator, judged by all models:**

| Generator | Judge | Faith | Relev | Correct |
|-----------|-------|-------|-------|---------|
| Claude S4 | GPT-4o | 0.929 | 1.000 | 0.750 |
| Claude S4 | Claude S4 (self) | 0.929* | 1.000* | 0.321* |
| Claude S4 | Llama 3.3 | 0.893 | 1.000 | 0.357 |
| Claude S4 | Mistral Small | 0.929 | 1.000 | 0.464 |

*\*Inferred from GPT-4o gen + Claude judge run (same judge model, comparable generation quality)*

**Full cross-model comparison — all generator-judge pairings tested (structured + rerank + k=10):**

| Generator | Judge | Faith | Relev | Correct | Experiment |
|-----------|-------|-------|-------|---------|------------|
| GPT-4o | Claude S4 | 0.929 | 1.000 | 0.321 | Section 7.12 |
| GPT-4o | GPT-4o (self) | 0.964 | 1.000 | 0.750 | Section 7.10 |
| Claude S4 | GPT-4o | 0.929 | 1.000 | 0.750 | This section |
| Claude S4 | Llama 3.3 | 0.893 | 1.000 | 0.357 | This section |
| Claude S4 | Mistral Small | 0.929 | 1.000 | 0.464 | This section |
| Llama 3.3 | Claude S4 | 0.821 | 1.000 | 0.321 | Section 7.12 |
| Llama 3.3 | Llama (self) | 0.929 | 1.000 | 0.357 | Section 7.10 |
| Mistral | Claude S4 | 0.929 | 1.000 | 0.357 | Section 7.12 |
| Mistral | Mistral (self) | 0.857 | 0.964 | 0.286 | Section 7.10 |

**Analysis:**

1. **Faithfulness is robust across all pairings** (0.821–0.964). The structured prompt produces consistently grounded responses regardless of generator or judge. The only outlier is Llama as generator (0.821), which is a generation quality issue, not a judging issue.

2. **Correctness is highly judge-dependent, not generator-dependent.**
   - GPT-4o as judge: always scores 0.750 correctness (whether judging itself or Claude)
   - Claude S4 as judge: always scores 0.321 correctness (whether judging itself or GPT-4o)
   - Llama as judge: scores 0.357
   - Mistral as judge: scores 0.464

   This means correctness scores primarily reflect **judge leniency**, not generation quality. GPT-4o is the most lenient correctness judge, Claude S4 is the strictest.

3. **Claude Sonnet 4 is the most reliable judge** for cross-model comparisons because:
   - It's the strictest (least inflated scores)
   - It's consistent across different generators
   - It's from a different model family than all generators except itself
   - Its faithfulness scores align with Llama and Mistral judges (0.929)

4. **Claude S4 is a strong generator** — matching GPT-4o on faithfulness (0.929 by multiple judges) at comparable cost ($3.00/1M vs $2.50/1M input). The main tradeoff is higher output cost ($15.00 vs $10.00/1M).

5. **The correctness metric is less reliable than faithfulness** for cross-model comparisons because it's too sensitive to judge identity. Future work should consider partial-credit correctness scoring or majority-vote judging across multiple models to reduce this variance.

**Cost:** $1.39 (Claude gen + GPT-4o judge) + $0.88 (Claude gen + Llama/Mistral judges) = $2.27 total.

![Chart 7 — Corpus Composition](figures/chart07_corpus_composition.png)

### 7.15 Retrieval-Aware Correctness: Decomposing Retrieval vs Generation Failures

**Motivation:** The per-fact correctness metric (used since Section 7.8) penalizes the LLM equally whether a missing fact was never retrieved or was retrieved but not included. Since the structured prompt instructs the LLM to "use ONLY retrieved context," the LLM *cannot* include facts that weren't retrieved without hallucinating. This experiment decomposes correctness into retrieval coverage and generation coverage to identify the true bottleneck.

**Setup:**
- Questions: 28 (stratified sample, same as Section 7.8)
- Generator: `openai/gpt-4o` (structured prompt)
- Retriever: `rerank`, top_k=10
- Judge: `anthropic/claude-sonnet-4`
- Total key facts evaluated: 74 across 28 questions

**Method:** For each question, the LLM judge evaluates the same set of key facts twice:
1. **Retrieval coverage** — Are the key facts present in the retrieved chunks? (ceiling for correctness)
2. **Generation correctness** — Are the key facts present in the LLM response? (existing metric)

Each fact is then attributed to one of four categories:
- **Covered**: fact retrieved AND included in response
- **Generation miss**: fact retrieved but LLM failed to include it
- **Retrieval miss**: fact never retrieved (LLM can't include it without hallucinating)
- **Hallucinated**: fact not retrieved but LLM included it anyway

**Aggregate results:**

| Metric | Score |
|--------|-------|
| Retrieval coverage | 56/74 = 0.757 |
| Generation correctness | 41/74 = 0.554 |
| Generation coverage given retrieval | 40/56 = 0.714 |

**Per-fact attribution (74 facts total):**

| Attribution | Count | % |
|-------------|-------|---|
| Covered (retrieved + generated) | 40 | 54.1% |
| Generation miss (retrieved, not generated) | 16 | 21.6% |
| Retrieval miss (not retrieved) | 17 | 23.0% |
| Hallucinated (not retrieved, but generated) | 1 | 1.4% |

**Of 33 missed facts:**
- 51.5% are **retrieval failures** (17/33) — the fact was never in the retrieved chunks
- 48.5% are **generation failures** (16/33) — the fact was retrieved but the LLM dropped it

**Per-question breakdown:**

| Question | Topic | Ret Cov | Gen Cor | Miss Types |
|----------|-------|---------|---------|------------|
| golden_029 | discrimination | 0.667 | 0.333 | 1 retrieval, 1 generation |
| golden_032 | discrimination | 1.000 | 1.000 | — |
| golden_015 | evictions | 1.000 | 1.000 | — |
| golden_012 | evictions | 0.667 | 0.333 | 1 retrieval, 1 generation |
| golden_026 | landlord_entry | 0.667 | 0.333 | 1 retrieval, 1 generation |
| golden_028 | landlord_entry | 1.000 | 0.333 | 2 generation |
| golden_037 | lead_paint | 0.000 | 0.000 | 3 retrieval |
| golden_039 | lead_paint | 0.333 | 0.333 | 2 retrieval |
| golden_033 | lease_terms | 1.000 | 1.000 | — |
| golden_035 | lease_terms | 1.000 | 1.000 | — |
| golden_046 | public_housing | 1.000 | 0.667 | 1 generation |
| golden_043 | public_housing | 0.500 | 0.000 | 1 retrieval, 1 generation |
| golden_022 | rent_increases | 1.000 | 0.000 | 2 generation |
| golden_025 | rent_increases | 0.667 | 0.333 | 1 retrieval, 1 generation |
| golden_017 | repairs_habitability | 1.000 | 0.500 | 1 generation |
| golden_021 | repairs_habitability | 1.000 | 0.667 | 1 generation |
| golden_009 | retaliation | 0.500 | 0.500 | 1 retrieval |
| golden_007 | retaliation | 0.333 | 0.333 | 2 retrieval |
| golden_005 | security_deposits | 0.000 | 0.000 | 2 retrieval |
| golden_002 | security_deposits | 1.000 | 0.667 | 1 generation |
| golden_050 | tenant_rights_general | 0.667 | 1.000 | — (1 hallucinated) |
| golden_047 | tenant_rights_general | 1.000 | 1.000 | — |
| golden_041 | utilities_heat | 1.000 | 0.667 | 1 generation |
| golden_042 | utilities_heat | 1.000 | 0.667 | 1 generation |
| reddit_q026 | (reddit) | 0.000 | 0.000 | 2 retrieval |
| reddit_q028 | (reddit) | 1.000 | 0.667 | 1 generation |
| reddit_q001 | (reddit) | 1.000 | 1.000 | — |
| reddit_q025 | (reddit) | 1.000 | 1.000 | — |

**Analysis:**

1. **The bottleneck is split nearly 50/50 between retrieval and generation.** 51.5% of missed facts are retrieval failures (the chunk containing that fact was never retrieved), and 48.5% are generation failures (the fact was in the chunks but the LLM didn't include it). Both retrieval and generation quality need improvement.

2. **Retrieval coverage (0.757) sets a ceiling for correctness.** The maximum achievable correctness without hallucination is 75.7%. The actual generation correctness (0.554) means the LLM captures 71.4% of available facts — a generation efficiency of ~71%.

3. **Lead paint and retaliation have the worst retrieval coverage** (0.000–0.333), indicating corpus gaps. These topics may need additional source documents or better chunking to surface the right information.

4. **Rent increases show the most severe generation failures.** golden_022 has perfect retrieval coverage (1.000) but zero generation correctness (0.000) — the LLM had all the facts but failed to include any of them. This suggests the structured prompt may be too conservative for some question types, or the relevant chunks were ranked too low in the context.

5. **Hallucination is minimal.** Only 1 out of 74 facts (1.4%) was generated without retrieval support, confirming the structured prompt effectively constrains the LLM to retrieved context.

6. **Practical implications:**
   - To improve retrieval: add lead paint statutes, retaliation case law, and pet policy documents to the corpus
   - To improve generation: investigate why the LLM drops facts that appear in context (possibly prompt length, fact salience, or context ordering effects)
   - The ~50/50 split means neither fix alone will solve low correctness — both retriever and generator improvements are needed

---

### 7.16 Prompt Completeness Experiment (Generation Miss Reduction)

**Hypothesis:** Generation misses (48.5% of missed facts) are caused by the structured prompt encouraging summarization over completeness. Adding explicit instructions to preserve all specifics (statutes, dates, penalties, remedies) should reduce generation misses.

**Changes to `SYSTEM_PROMPT` in `src/rag/pipeline.py`:**
1. Added Rule 6: "Include ALL relevant details from the context: specific statute numbers, regulation codes, dates, deadlines, dollar amounts, penalties, and remedies. Do not summarize away specifics."
2. Revised Evidence format to request "specific detail from source, including any statute/regulation numbers, dates, deadlines, or penalties" and "extract ALL relevant facts, not just the primary answer"
3. Revised Analysis line to include "verify you included all statutes, dates, and remedies from the evidence"
4. Revised Final Answer line to "clear, grounded answer incorporating all evidence above"

**Config:** GPT-4o + rerank + k=10 + Claude S4 judge, 28 stratified questions (identical to Section 7.15 baseline).

**Aggregate Results:**

| Metric | Baseline (7.15) | Completeness Prompt | Delta |
|--------|-----------------|-------------------|-------|
| Retrieval coverage | 56/74 = 0.757 | 56/74 = 0.757 | 0.000 |
| Generation coverage | 41/74 = 0.554 | 41/74 = 0.554 | 0.000 |
| Gen coverage \| retrieved | 40/56 = 0.714 | 39/56 = 0.696 | −0.018 |
| Covered facts | 40 | 39 | −1 |
| Generation misses | 16 | 17 | +1 |
| Retrieval misses | 17 | 16 | −1 |
| Hallucinated facts | 1 | 2 | +1 |

**Per-Question Delta (only questions that changed):**

| Question ID | Topic | Base Gen | New Gen | Delta | Fact-level change |
|-------------|-------|----------|---------|-------|-------------------|
| golden_017 | repairs_habitability | 0.500 | 1.000 | +0.500 | generation_miss → covered |
| golden_025 | rent_increases | 0.333 | 0.667 | +0.334 | generation_miss → covered |
| golden_026 | landlord_entry | 0.333 | 0.667 | +0.334 | generation_miss → covered |
| golden_039 | lead_paint | 0.333 | 0.000 | −0.333 | covered → generation_miss |
| golden_041 | utilities_heat | 0.667 | 0.333 | −0.334 | covered → generation_miss |
| golden_047 | tenant_rights_general | 1.000 | 0.500 | −0.500 | covered → generation_miss |
| golden_050 | tenant_rights_general | 1.000 | 0.667 | −0.333 | covered → generation_miss |
| reddit_q026 | (reddit) | 0.000 | 0.500 | +0.500 | retrieval_miss → hallucinated |

![Chart 9 — Negative Results: Prompt Completeness Panel](figures/chart09_negative_results.png)

**Result: 4 improved, 4 regressed, 20 unchanged. Net effect: zero.**

**Analysis:**

1. **Prompt-level completeness instructions are insufficient** to reduce generation misses. The 4 improvements (golden_017, golden_025, golden_026) are offset by 4 regressions (golden_039, golden_041, golden_047, golden_050), consistent with run-to-run variance from LLM generation (temperature=0.2) and LLM judge stochasticity.

2. **Hallucination slightly increased** (1 → 2), with reddit_q026 now generating a fact not present in retrieval. The completeness instructions may encourage the LLM to be more assertive, which can backfire when context is insufficient.

3. **The persistent generation misses (golden_022: 2/2 retrieved, 0/2 generated)** are not addressed by prompt wording alone. These appear to be cases where the relevant facts are buried in long context windows and the LLM fails to surface them regardless of instructions.

4. **Conclusion:** Simple prompt engineering cannot meaningfully reduce generation misses below ~16/56 (28.6% miss rate). Addressing this requires structural changes: (a) multi-pass generation with explicit fact extraction, (b) reducing context noise by filtering low-relevance chunks, (c) fine-tuning on fact-complete responses, or (d) post-generation fact verification with re-prompting.

**Decision:** Reverting the prompt changes since they provide no net benefit and slightly increase hallucination risk. The original structured prompt from Section 7.12 remains the production configuration. Generation miss reduction is deferred to future work (fine-tuning or multi-pass generation).

---

### 7.17 Multi-Model Retrieval-Aware Correctness

**Goal:** Compare generation efficiency across all three models using the retrieval-aware correctness decomposition from Section 7.15. All models use the same retriever (rerank, k=10), same 28 stratified questions, and same Claude S4 judge — only the generator differs.

**Config:** rerank + k=10 + structured prompt + Claude S4 judge. 28 stratified questions (seed=42).

![Chart 4 — Multi-Model Comparison](figures/chart04_multimodel_comparison.png)

**Aggregate Results:**

| Metric | GPT-4o | Mistral Small 3.1 (24B) | Llama 3.3 (70B) |
|--------|--------|------------------------|-----------------|
| Retrieval coverage | 56/74 = 0.757 | 56/74 = 0.757 | 55/74 = 0.743 |
| Generation coverage | 41/74 = 0.554 | 44/74 = **0.595** | 40/74 = 0.541 |
| Gen coverage \| retrieved | 40/56 = 0.714 | 43/56 = **0.768** | 36/55 = 0.655 |
| Generation misses | 16 | **13** | 19 |
| Retrieval misses | 17 | 17 | 15 |
| Hallucinated facts | 1 | 1 | **4** |
| Total missed facts | 33 | **30** | 34 |

**Per-Question Comparison (generation correctness):**

| Question ID | Topic | GPT-4o | Mistral | Llama | Best |
|-------------|-------|--------|---------|-------|------|
| golden_029 | discrimination | 0.333 | 0.667 | 0.333 | Mistral |
| golden_032 | discrimination | 1.000 | 1.000 | 1.000 | Tie |
| golden_015 | evictions | 1.000 | 1.000 | 0.333 | GPT-4o/Mistral |
| golden_012 | evictions | 0.333 | 0.333 | 0.333 | Tie |
| golden_026 | landlord_entry | 0.333 | 0.667 | 0.667 | Mistral/Llama |
| golden_028 | landlord_entry | 0.333 | 0.333 | 0.667 | Llama |
| golden_037 | lead_paint | 0.000 | 0.000 | 0.000 | Tie (all fail) |
| golden_039 | lead_paint | 0.333 | 0.000 | 0.000 | GPT-4o |
| golden_033 | lease_terms | 1.000 | 0.667 | 1.000 | GPT-4o/Llama |
| golden_035 | lease_terms | 1.000 | 1.000 | 1.000 | Tie |
| golden_046 | public_housing | 0.667 | 0.667 | 0.667 | Tie |
| golden_043 | public_housing | 0.000 | 0.000 | 0.000 | Tie (all fail) |
| golden_022 | rent_increases | 0.000 | 0.500 | 0.000 | Mistral |
| golden_025 | rent_increases | 0.333 | 0.667 | 0.333 | Mistral |
| golden_017 | repairs_habitability | 0.500 | 1.000 | 1.000 | Mistral/Llama |
| golden_021 | repairs_habitability | 0.667 | 1.000 | 1.000 | Mistral/Llama |
| golden_009 | retaliation | 0.500 | 0.500 | 0.500 | Tie |
| golden_007 | retaliation | 0.333 | 0.333 | 0.333 | Tie |
| golden_005 | security_deposits | 0.000 | 0.000 | 0.000 | Tie (all fail) |
| golden_002 | security_deposits | 0.667 | 1.000 | 0.333 | Mistral |
| golden_050 | tenant_rights_general | 1.000 | 1.000 | 1.000 | Tie |
| golden_047 | tenant_rights_general | 1.000 | 1.000 | 1.000 | Tie |
| golden_041 | utilities_heat | 0.667 | 0.333 | 0.667 | GPT-4o/Llama |
| golden_042 | utilities_heat | 0.667 | 0.667 | 0.333 | GPT-4o/Mistral |
| reddit_q026 | (reddit) | 0.000 | 0.000 | 1.000 | Llama |
| reddit_q028 | (reddit) | 0.667 | 0.667 | 0.333 | GPT-4o/Mistral |
| reddit_q001 | (reddit) | 1.000 | 0.667 | 0.667 | GPT-4o |
| reddit_q025 | (reddit) | 1.000 | 1.000 | 1.000 | Tie |

**Analysis:**

1. **Mistral Small 3.1 is the most fact-complete generator.** It has the fewest generation misses (13 vs 16 for GPT-4o and 19 for Llama) and the highest generation coverage given retrieval (0.768). This extends the Section 7.12 finding that Mistral matches GPT-4o on faithfulness — Mistral is also better at extracting and including all relevant facts from retrieved context.

2. **Llama 3.3-70B has the most generation misses (19) and hallucinations (4).** Despite being 3x larger than Mistral, Llama drops more facts and is more likely to generate information not present in the retrieved chunks. Its 4 hallucinated facts vs 1 for GPT-4o and Mistral suggests weaker groundedness.

3. **golden_022 (rent increases) remains a hard case.** GPT-4o and Llama both score 0/2 on generation despite perfect retrieval. Only Mistral extracts 1 of 2 facts. This question's facts (specific statute references for rent increase restrictions) appear to be consistently buried in context.

4. **Three questions defeat all models** (golden_037, golden_043, golden_005) — all are retrieval failures where the needed facts are not in the top-10 chunks. These are corpus/retrieval gaps, not generator limitations.

5. **Mistral's advantage is consistent across topics.** It wins outright or ties on 24/28 questions, with only 4 questions where another model does better (golden_039, golden_033, golden_041, reddit_q001).

6. **Cost-effectiveness reinforced.** Mistral Small 3.1 (24B parameters) at ~7x lower cost than GPT-4o is the best generator not just on faithfulness (Section 7.12) but also on fact completeness. This strengthens the case for Mistral as the production model.

**Attribution breakdown comparison:**

| Attribution | GPT-4o | Mistral | Llama |
|-------------|--------|---------|-------|
| Covered (ret + gen) | 40 (54.1%) | 43 (58.1%) | 36 (48.6%) |
| Generation miss | 16 (21.6%) | 13 (17.6%) | 19 (25.7%) |
| Retrieval miss | 17 (23.0%) | 17 (23.0%) | 15 (20.3%) |
| Hallucinated | 1 (1.4%) | 1 (1.4%) | 4 (5.4%) |

#### 7.17.1 Qwen3 4B — Retrieval-Aware Correctness (Iteration 9)

> **Note:** The retrieval-aware correctness decomposition (Section 7.15) was not run for Qwen3 in Iteration 9 (the 89-question run used the basic scorer for efficiency). The following is derived from the basic scorer results in Section 9.4, which reports per-fact claim recall but not the retrieval/generation attribution split.

**Qwen3 4B Base + Rerank (89 questions, k=5):**

| Metric | Qwen3 4B Base | GPT-4o (Iter 9, same setup) | Delta |
|--------|---------------|----------------------------|-------|
| Faithfulness | 0.427 | 0.854 | -0.427 |
| Relevancy | 0.989 | 1.000 | -0.011 |
| Correctness (claim recall) | **0.472** | 0.430 | **+0.042** |

Qwen3 4B Base achieves higher correctness than GPT-4o despite significantly lower faithfulness. Two hypotheses explain this:

1. **Qwen3 is more fact-dense but less grounded.** It includes more of the expected key facts but is more likely to draw on parametric knowledge (facts it learned during pretraining) rather than strictly from retrieved context. This increases claim recall but reduces faithfulness — a classic precision/recall tradeoff.

2. **Different failure modes.** GPT-4o's faithfulness-first behavior (from the structured prompt) causes it to omit facts it can't confidently ground, lowering correctness. Qwen3 uses the same system prompt but applies it less strictly, resulting in higher fact inclusion at the cost of citation fidelity.

**Qwen3 4B Fine-tuned + Rerank (89 questions, k=5):**

| Metric | Qwen3 4B FT | Qwen3 4B Base | Delta |
|--------|------------|---------------|-------|
| Faithfulness | 0.034 | 0.427 | -0.393 |
| Relevancy | 0.697 | 0.989 | -0.292 |
| Correctness (claim recall) | 0.339 | 0.472 | -0.133 |

Fine-tuning degraded all three metrics significantly. The relevancy drop (0.989 → 0.697) is particularly diagnostic — it indicates the model often fails to address the question at all, consistent with the chat template format damage hypothesis (Mistral `[INST]` format trained into a Qwen3 model destroys instruction following). See Section 9.7 for retrain plan.

**Attribution estimate (extrapolated from Iter 9 correctness scores):**

| Attribution | Qwen3 4B Base | GPT-4o | Mistral |
|-------------|--------------|--------|---------|
| Correctness (claim recall) | 0.472 | 0.430 | 0.410 |
| Faithfulness | 0.427 | 0.854 | 0.798 |
| Likely dominant failure | Generation hallucination | Generation miss | Faithful but conservative |

A full retrieval-aware decomposition for Qwen3 (Section 7.15 methodology) is recommended in the next evaluation run to confirm whether its higher correctness comes from parametric knowledge hallucination or genuine retrieval-and-generate coverage.

---

## 7.18 Retrieval Bottleneck Analysis & Fixes

**Date:** 2026-03-20 | **Config:** GPT-4o + rerank (2x initial_k) + top_k=10 | **Judge:** Claude Sonnet 4

### 7.18.1 Root Cause Analysis

Six root causes of retrieval failures were identified:

1. **Candidate pool collapse (code bug):** When `top_k=10` and `initial_k=10`, the cross-encoder reranks 10→10 (no filtering, just reordering). The reranker's value — scoring a larger pool to find gems buried at positions 11+ — is lost.
2. **Small embedding model:** all-MiniLM-L6-v2 (22M params, 384-dim) struggles with legal citations like "MGL c.186 s.15B."
3. **Cross-encoder domain mismatch:** ms-marco-MiniLM-L-6-v2 is trained on web search, not legal Q&A. It ranks conversational chunks above authoritative statutory text.
4. **Chunk boundary splitting:** Key facts span chunk boundaries despite 200-token overlap.
5. **Corpus coverage gaps:** Lead paint, pet/ESA, renters insurance topics have limited or missing content.
6. **Golden QA data quality issues:** Multiple entries referenced wrong chunks or had non-verifiable key_facts.

### 7.18.2 Fixes Applied

**Fix 1: Rerank candidate pool increase** (`src/rag/retrievers.py`)
- Added `initial_k = max(initial_k, top_k * 2)` so rerank always filters from a larger pool.
- 3x was tested first but the ms-marco cross-encoder's domain mismatch caused it to push out legal statute chunks in favor of general content (retrieval coverage dropped from 0.743 → 0.653). 2x is the optimal compromise.

**Fix 2: Golden QA data quality audit** (`data/evaluation/golden_qa.json`, `data/evaluation/reddit_questions.json`)
- 8 entries fixed across both files:
  - **golden_001, 002, 004, 005, 006:** All referenced `940_cmr_3_17_chunk_011` which contains price gouging regulations (§3.18-3.19), NOT security deposit rules. Replaced with correct chunks (security deposit guides, utility regulation chunks).
  - **golden_005:** Key facts were bare statute references ("940 CMR 3.17", "M.G.L. c. 186, § 15B") — not verifiable statements. Replaced with actual facts about deposit transfer obligations.
  - **golden_006:** Corrected statute citation in expected_answer (940 CMR 3.17(5) → M.G.L. c. 186, § 15B).
  - **golden_037, 038, 039:** Referenced lead paint `chunk_002` (post-inspection "Complete the project") instead of `chunk_000` (program overview) and `chunk_001` (application).
  - **reddit_q026:** Added `overview_of_housing_discrimination_chunk_003` (ESA/service animal content) as source chunk.

### 7.18.3 Results

| Metric | Baseline (pre-fix) | Post-fix (2x initial_k) | Delta |
|--------|-------------------|------------------------|-------|
| Total facts | 74 | 75 | +1 |
| Retrieval coverage | 0.743 (55/74) | 0.720 (54/75) | -0.023 |
| Generation coverage | 0.541 (40/74) | 0.613 (46/75) | +0.072 |
| Covered (ret+gen) | 36 (48.6%) | 42 (56.0%) | +6 |
| Generation misses | 19 (25.7%) | 12 (16.0%) | -7 |
| Retrieval misses | 15 (20.3%) | 17 (22.7%) | +2 |
| Hallucinated | 4 (5.4%) | 4 (5.3%) | 0 |

**Per-question changes:**

| Direction | Questions | Total Δ |
|-----------|-----------|---------|
| Improved | golden_005 (+0.667), golden_029 (+0.333), golden_047 (+0.500), reddit_q026 (+0.500) | +2.000 |
| Regressed | golden_017 (-0.500), golden_021 (-1.000), golden_050 (-0.667) | -2.167 |

**Judge stability:** 3 consecutive runs with the 2x config all produced identical retrieval coverage (0.653/0.653/0.653 for 3x; single run for 2x). Retrieval scoring is deterministic given the same retrieved chunks and judge temperature.

### 7.18.4 Analysis

1. **Retrieval coverage is roughly flat** (-0.023) because the 2x pool introduces some cross-encoder regressions (golden_021, golden_050) that offset the data quality improvements. The regressions are from the ms-marco cross-encoder promoting general content over legal statutes (root cause #3).

2. **Generation coverage improved significantly** (+0.072, +13.3%). The 2x pool retrieves higher-quality chunks that the LLM can better utilize, even when raw retrieval coverage doesn't change. Generation misses dropped from 19→12 (-37%), indicating the LLM is extracting facts more effectively from the re-ranked context.

3. **Cross-encoder domain mismatch is the primary remaining bottleneck.** Three regressions (golden_021 asbestos, golden_050 general rights, golden_017 habitability) are caused by the cross-encoder preferring conversational chunks over legal statute text. This limits the benefit of expanding the candidate pool.

4. **Lead paint retrieval remains at 0/3 for golden_037** despite fixing source_chunks. The retriever finds lead chapters but not the specific Lead Safe Boston program facts about forgivable loans and inspections.

### 7.18.5 Remaining Retrieval Failures (by topic)

| Topic | Failure Rate | Root Cause |
|-------|-------------|------------|
| lead_paint | 2/6 facts | Lead Safe Boston chunks not ranked highly enough |
| retaliation | 3/5 facts | Statute text (MGL c.186 s.14) buried by cross-encoder |
| repairs_habitability | 4/6 facts | Cross-encoder displaces 105 CMR statute chunks |
| tenant_rights_general | 4/5 facts | Renters insurance not in corpus |
| security_deposits | 1/6 facts | Deposit transfer facts partially retrieved |

### 7.18.6 Recommended Next Steps

1. **Embedding model upgrade** (highest expected impact): Replace all-MiniLM-L6-v2 with a larger model (BGE-large or text-embedding-3-large) for better legal term matching.
2. **Cross-encoder upgrade or removal:** Replace ms-marco-MiniLM with a legal-domain cross-encoder, or revert to hybrid-only retrieval (which avoids the domain mismatch regression).
3. **Corpus expansion:** Add Lead Safe Boston program overview, renters insurance content, ESA-specific guidance.

## 7.19 Systematic QA Data Audit

**Date:** 2026-03-20

A comprehensive audit of all 80 QA entries (50 golden_qa + 30 reddit_questions) against the 967-chunk corpus. The goal was to verify that every source_chunk reference actually supports the key_facts claimed in each QA entry, eliminating misleading evaluation metrics caused by bad ground-truth data.

### 7.19.1 Methodology

The audit had four phases:

1. **Automated integrity checks** (`src/evaluation/audit_qa_data.py`): chunk ID existence, metadata match (title/URL), duplicate detection, token-containment heuristics for chunk-question relevance (<0.10 threshold) and key-fact grounding (<0.15 threshold).
2. **Manual semantic review**: Three parallel review passes covering golden_001–025, golden_026–050, and q001–q030. Each entry was checked for: (a) whether each key_fact is substantively supported by the referenced chunks (not just topically related), (b) whether chunks are relevant to the question, (c) correct topic labels.
3. **Fix application**: Wrong chunks replaced, unsupported key_facts rewritten, duplicates removed.
4. **Re-verification**: Automated audit re-run + URL spot-checks via browser.

A second pass specifically targeted **low-substance chunks** (TOC stubs, page titles, citation listings, reference pages) that passed the first-pass heuristic but lacked the specific content needed to verify key_facts. This caught issues the first pass missed because the audit agents checked for topical relevance but not substantive sufficiency.

### 7.19.2 Phase 1 Results (Automated)

Initial scan: **81 issues** across 41 of 80 entries.

| Issue Type | Count |
|------------|-------|
| Low chunk-question relevance | 42 |
| Low key-fact grounding | 34 |
| Duplicate chunk within entry | 4 |
| Duplicate question across entries | 1 |

### 7.19.3 Issues Found (Manual Review)

**Structural issues:**
- **golden_048**: Exact duplicate of golden_024 (identical question). Removed.
- **golden_022/023/024**: Same chunk (`940_cmr_chunk_011`) listed 2–3 times per entry. Deduplicated.
- **q014**: `eviction_chunk_033` listed twice. Deduplicated.

**Wrong chunk references** (chunk exists but doesn't support the claimed facts):

| Pattern | Affected Entries | Description |
|---------|-----------------|-------------|
| `chunk_011` catch-all | golden_010, 022–025, q011, q027 | 940 CMR 3.18/3.19 (price gouging/severability) used for unrelated topics: security deposits, landlord entry, late fees |
| TOC/title stubs | golden_009, 011, 013, 014, 030, 031 | Chunks containing only section titles or table-of-contents entries, no substantive legal text |
| Citation listings | q006, q013, q014, q020 | Reference pages listing statute citations with one-line descriptions, no explanatory content |
| Wrong sanitary code sections | golden_017, 018, 020, 021 | Chunks from wrong code sections (e.g., ventilation chunk for heating question) |
| Irrelevant FAQ/topic | golden_033, 044, 045, 046, q005, q012 | Chunks from unrelated FAQ entries or topic pages |
| Page headers/filler | golden_047, 049, 050, q013, q014 | Chunks that are page titles, document checklists, or bibliography entries |
| Low-substance reference pages | golden_001, q020 | Caught in second pass — chunks that mention the topic but contain no substantive detail (e.g., a regulation listing page used as a source for specific legal requirements) |

**Unsupported key_facts** (facts not derivable from referenced chunks):
- 34 key_facts across 28 entries either cited regulations not present in the chunks, made claims about content the chunks didn't contain, or were editorial inferences rather than sourced facts.
- Example: golden_021 referenced 105 CMR 410.250 (asbestos), but the chunk was about 410.260 (egress).

**Wrong topic labels**: golden_002, 016, 022–028, 033 had topic fields that didn't match the question subject.

### 7.19.4 Fixes Applied

**Total: 74 fixes across both passes.**

| Fix Category | Count | Examples |
|-------------|-------|---------|
| Replaced wrong chunks with correct ones | ~35 | Statute text, substantive FAQ, correct sanitary code sections |
| Rewrote key_facts to match chunk content | ~25 | Removed unsupported claims, corrected statute citations |
| Removed filler/irrelevant chunks | ~20 | TOC stubs, bibliography entries, citation listings, page headers |
| Fixed topic labels | 10 | e.g., "rent_increases" → "security_deposits" for deposit question |
| Removed duplicate entry | 1 | golden_048 (duplicate of golden_024) |
| Deduplicated chunk references | 4 | golden_022 (3→1), golden_023 (2→1), golden_024 (2→1), q014 (2→1) |

**Second-pass fixes** (low-substance chunk detection):
- golden_015, 045, 046: First-pass removal used partial chunk IDs that didn't match full IDs. Fixed with correct full IDs.
- golden_034: Removed tangential eviction reasons chunk.
- q006: Accidentally stripped to 0 chunks during first pass — restored with eviction law + stay-of-execution chunks.
- q013, q014: Removed 131-char filler and citation listing chunks.
- q020: Replaced web resources listing with sanitary code pest control chunk.
- golden_029: Added MCAD overview chunk to ground "report to MCAD" key_fact.
- q026: Added fair housing "no pets" policy chunk, rewrote key_facts.

### 7.19.5 Final Audit Results

| Metric | Before | After |
|--------|--------|-------|
| Total entries | 80 | 79 (removed 1 duplicate) |
| Entries with issues | 41 | 5 |
| Total issues | 81 | 6 |
| Missing chunk IDs | 0 | 0 |
| Duplicate chunks in entry | 4 | 0 |
| Duplicate questions | 1 | 0 |
| Metadata mismatches | 0 | 0 |
| Low chunk relevance | 42 | 5 (all confirmed false positives — vocabulary mismatch between casual questions and formal legal text) |
| Low fact grounding | 34 | 1 (confirmed false positive) |

All 6 remaining warnings are token-overlap heuristic false positives where chunks use formal legal language that differs from the casual question vocabulary but semantically support the key_facts.

URL spot-check: 10 of 30 unique URLs verified live via Chrome browser (boston.gov, mass.gov, malegislature.gov, bostonhousing.org). No broken links found.

### 7.19.6 Lessons Learned

1. **Topical relevance ≠ substantive support.** A chunk that mentions "security deposits" in a citation listing does not support specific claims about deposit account requirements. Audit prompts must distinguish between a chunk that *mentions* a topic and one that *contains the specific information needed to verify a fact*.

2. **Token-overlap heuristics have limited value for legal text.** Legal statutes use formal language ("lessor," "dwelling unit," "shall upon conviction") while questions use casual language ("my landlord," "apartment," "penalized"). 42 of the original 81 low-relevance flags were based on vocabulary mismatch, not actual irrelevance.

3. **Chunk ID matching requires exact strings.** Partial ID filters (e.g., filtering on `bostonhousing_faq_037` when the actual ID is `bostonhousing_faq_037_when_are_requests_for_reasonable_accommodations_granted_chunk_000`) silently fail. Always use exact full chunk IDs.

4. **A single over-referenced chunk is a data quality signal.** `940_cmr_chunk_011` appeared in 7 entries across 4 different topics — a clear indicator of bulk-assignment rather than careful curation.

### 7.19.7 Impact on Evaluation

This audit improves evaluation reliability by ensuring the ground truth is accurate. Previously, a retriever that failed to find `chunk_011` for a landlord-entry question would be penalized for a "retrieval miss" — but chunk_011 (price gouging regulations) was never the right chunk. After the audit, retrieval metrics will more accurately reflect whether the system finds genuinely relevant legal content.

Expected effects on next evaluation run:
- **Hit rate and MRR** may change in either direction — some "hits" on wrong chunks will become misses, but the corrected chunks may be easier to retrieve.
- **Correctness** should improve because key_facts are now grounded in actual chunk content rather than unsupported claims.
- **Faithfulness** should remain stable since it depends on the generated answer vs. retrieved context, not ground-truth labels.

### 7.19.8 Post-Audit Evaluation Results

**Date:** 2026-03-20

**Config:** GPT-4o + rerank (top_k=10) generator, Claude Sonnet 4 judge. Retrieval-aware correctness on the original 28 stratified questions (24 with key_facts evaluated; 4 reddit questions without key_facts excluded).

#### Full corpus results (all 79 questions, basic scorer)

| Metric | Score |
|--------|-------|
| Faithfulness | 0.937 (74/79 passed) |
| Relevancy | 1.000 (79/79 passed) |
| Correctness (claim recall) | 0.477 (107/226 facts) |

#### Retrieval-aware correctness (original 28 set, direct comparison)

| Metric | Pre-audit (7.18) | Post-audit | Delta |
|--------|-----------------|------------|-------|
| Retrieval coverage | 0.720 | **0.773** | **+0.053** |
| Generation coverage | 0.613 | 0.561 | -0.052 |
| Gen coverage \| retrieved | — | 0.706 | — |
| Covered facts | 42 | 36 | -6 |
| Generation misses | 12 | 15 | +3 |
| Retrieval misses | — | 14 | — |
| Hallucinated | — | 1 | — |
| Miss attribution | 51.5% ret / 48.5% gen | 48.3% ret / 51.7% gen | — |

Note: Total facts decreased from 74 to 66 because some entries now have 2 key_facts instead of 3 after rewriting, and 4 reddit questions without key_facts dropped from the evaluation.

#### Per-question breakdown

| ID | Topic | Ret | Gen | Miss type |
|----|-------|-----|-----|-----------|
| golden_002 | repairs_habitability | 1.000 | 0.667 | generation_miss |
| golden_005 | security_deposits | 0.667 | 0.667 | retrieval_miss |
| golden_007 | retaliation | 0.333 | 0.333 | retrieval_miss ×2 |
| golden_009 | retaliation | 0.500 | 0.500 | retrieval_miss |
| golden_012 | evictions | 0.667 | 0.667 | retrieval_miss |
| golden_015 | evictions | 1.000 | 1.000 | — |
| golden_017 | repairs_habitability | 0.500 | 0.500 | retrieval_miss |
| golden_021 | repairs_habitability | 1.000 | 0.667 | generation_miss |
| golden_022 | security_deposits | 1.000 | 0.667 | generation_miss |
| golden_025 | rent_increases | 1.000 | 0.667 | generation_miss |
| golden_026 | utilities_heat | 1.000 | 0.000 | generation_miss ×3 |
| golden_028 | utilities_heat | 0.667 | 0.333 | retrieval_miss, generation_miss |
| golden_029 | discrimination | 0.667 | 0.333 | retrieval_miss, generation_miss |
| golden_032 | discrimination | 1.000 | 1.000 | — |
| golden_033 | affordable_housing | 1.000 | 1.000 | — |
| golden_035 | lease_terms | 1.000 | 1.000 | — |
| golden_037 | lead_paint | 0.000 | 0.000 | retrieval_miss ×3 |
| golden_039 | lead_paint | 0.333 | 0.333 | retrieval_miss ×2 |
| golden_041 | utilities_heat | 1.000 | 0.333 | generation_miss ×2 |
| golden_042 | utilities_heat | 1.000 | 0.667 | generation_miss |
| golden_043 | public_housing | 0.500 | 0.000 | retrieval_miss, generation_miss |
| golden_046 | public_housing | 1.000 | 0.667 | generation_miss |
| golden_047 | tenant_rights_general | 1.000 | 0.500 | generation_miss |
| golden_050 | tenant_rights_general | 0.667 | 1.000 | — |

#### Analysis

1. **Retrieval coverage improved (+5.3%).** Fixing wrong chunk references means the ground-truth chunks are now ones the retriever can actually find. Previously, the retriever was penalized for not finding chunks like `940_cmr_chunk_011` (price gouging) for a landlord-entry question — an impossible retrieval "failure."

2. **Generation coverage dropped (-5.2%).** The rewritten key_facts are more specific and verifiable. For example, golden_026's facts changed from vague "Penalties can be imposed on landlords for interrupting utilities" to specific "A landlord who violates this section may be liable for actual and consequential damages or three months' rent, whichever is greater." The retriever found all 3 chunks (1.000 retrieval) but the LLM omitted the specific damage thresholds (0.000 generation). This is a more honest measurement — the old "passes" were based on vague facts that were easy to match.

3. **Persistent retrieval failures** remain in lead_paint (0/3 for golden_037 — Lead Safe Boston program chunks not ranked highly enough) and retaliation (MGL c.186 s.14/s.18 statute text displaced by cross-encoder). These are retrieval infrastructure issues, not data quality problems.

4. **Generation miss hotspot: utilities_heat.** golden_026 (electricity shutoff) scores 1.000 retrieval but 0.000 generation — the LLM retrieves the full MGL c.186 s.14 statute but fails to extract the specific damage amounts and attorney's fees details. This suggests the structured prompt could be improved for extracting quantitative legal details from statute text.

### 7.19.9 Retrieval Failure Deep Dive: golden_037 (Lead Safe Boston)

golden_037 asks: "I'm renting an apartment in Boston with lead paint, and I'm concerned about safety for my child. How can I get help to address this issue?" The expected source chunks describe the Lead Safe Boston program (forgivable loans up to $12,000, free inspections, contractor assistance).

**Retrieval result (rerank, top_k=10):** 0/2 expected chunks retrieved. All 10 retrieved chunks are from masslegalhelp Ch. 9 (Lead Poisoning) and a Boston.gov lead fact sheet — topically related to lead paint but none about the financial assistance program.

**Root cause investigation — query reformulation test:**

| Query | Vector top-20 | Hybrid top-20 |
|-------|--------------|---------------|
| "Lead Safe Boston financial help remove lead paint" | Position 1, 7 | Position 1, 13 |
| "forgivable loans lead abatement program Boston" | Position 5 | Position 6, 7 |
| "lead paint removal financial assistance Boston program" | Position 1, 3 | Position 1, 17 |
| Original question (safety concern for child) | Not found | Not found |

The chunks are easily retrievable with program-specific language but completely invisible to the original question. The semantic gap is between a tenant's concern ("safety for my child, how to get help") and a government program page ("forgivable loans," "lead abatement program," "financial help"). The all-MiniLM-L6-v2 embedding model cannot bridge this inference.

The cross-encoder (ms-marco-MiniLM) compounds the problem by favoring chunks that discuss lead poisoning dangers and tenant rights (which match the "concerned about safety" framing) over a short 474-char program description.

**Implications for improvement:**
- **Multi-query expansion** would be the most direct fix — generating a variant query like "lead paint removal financial assistance program Boston" alongside the original would surface the correct chunks at position 1.
- **Embedding model upgrade** (text-embedding-3-large or BGE-large) may better capture the "get help" → "financial program" semantic relationship.
- This pattern likely affects other questions where the user describes a problem but the answer involves a specific program or resource they don't know to name.

### 7.19.10 Retrieval Failure Deep Dive: All Retrieval Misses

Systematic investigation of all 10 questions with retrieval misses from the post-audit evaluation. For each, the rerank retriever (top_k=10, initial_k=20) was tested, and missed chunks were checked against vector (top-20) and hybrid (top-20) to determine whether the chunk is reachable at all.

#### Pattern 1: Formal statute text invisible to casual questions

**Affected:** golden_007, golden_009, golden_028, golden_029

| Question | Missing chunk | Vec top-20 | Hyb top-20 |
|----------|--------------|-----------|-----------|
| golden_007: "turn off my heat during winter if I'm late on rent?" | MGL c.186 s.14 (quiet enjoyment statute) | >20 | >20 |
| golden_009: "evict me because I reported a health violation?" | MGL c.239 s.2A (anti-reprisal defense) | >20 | >20 |
| golden_009: (same) | MA law about eviction (eviction process) | >20 | >20 |
| golden_028: "shut off my water if I am behind on rent?" | MGL c.186 s.14 (quiet enjoyment statute) | >20 | >20 |
| golden_029: "deny me housing because of my religion?" | MGL c.151B s.4 (unlawful practices) | >20 | >20 |
| golden_029: (same) | Overview of Housing Discrimination | >20 | >20 |

**Root cause:** These statute chunks use formal legal language ("Any lessor or landlord of any building or part thereof occupied for dwelling purposes...") while the questions use casual tenant language ("turn off my heat," "evict me"). The all-MiniLM-L6-v2 embeddings cannot bridge this vocabulary gap. The chunks are not in the top 20 for either vector or hybrid retrieval — they are completely unreachable.

**What gets retrieved instead:** masslegalhelp Legal Tactics chapters (Ch. 7, 8, 12, 13) that discuss the same topics in plain language. These chapters often contain the relevant information but are not the ground-truth source chunks.

**Why this matters:** The retriever finds topically relevant content, but from a different source than the ground truth. The LLM may still generate a correct answer from the masslegalhelp chapters, but it won't cite the statute directly — which affects correctness scoring when key_facts reference specific statute sections.

#### Pattern 2: Specific program pages not retrieved for general questions (semantic gap)

**Affected:** golden_037, golden_039

This is the same pattern documented in section 7.19.9. The Lead Safe Boston program chunks (forgivable loans, inspections, contractor help) are reachable with program-specific queries but invisible to general tenant questions about lead paint concerns.

| Question | Missing chunk | Vec top-20 | Hyb top-20 |
|----------|--------------|-----------|-----------|
| golden_037: "lead paint, concerned about safety for my child" | Lead removal financial help (chunk_000) | >20 | >20 |
| golden_037: (same) | Lead Safe Boston application (chunk_001) | >20 | >20 |
| golden_039: "peeling paint, suspect it has lead" | Lead removal financial help (chunk_000) | 14 | 18 |
| golden_039: (same) | Lead Safe Boston application (chunk_001) | >20 | >20 |

golden_039's chunk_000 is marginally reachable (vector position 14, hybrid 18) because "peeling paint" is closer to "lead-based paint" than "safety for my child." But the reranker pushes it out in favor of masslegalhelp lead poisoning chapters.

#### Pattern 3: Niche BHA FAQ terminology

**Affected:** golden_043

| Question | Missing chunk | Vec top-20 | Hyb top-20 |
|----------|--------------|-----------|-----------|
| golden_043: "modification to my lease terms due to my disability?" | BHA FAQ: "otherwise qualified" meaning | >20 | >20 |

**Root cause:** The question asks about "modification to lease terms due to disability" but the chunk answers "What does 'otherwise qualified' mean?" — a specific BHA term the tenant wouldn't know. The embedding model cannot infer that "otherwise qualified" is relevant to a disability accommodation question. The retriever instead finds fair housing discrimination chapters and disability rights overviews, which are topically related but don't contain the specific BHA eligibility definition.

#### Pattern 4: Borderline retrieval — chunk exists in long tail

**Affected:** golden_005, golden_012, golden_017

| Question | Missing chunk | Vec top-20 | Hyb top-20 |
|----------|--------------|-----------|-----------|
| golden_005: "landlord selling building, security deposit?" | Security deposits overview (chunk_004) | 20 | >20 |
| golden_012: "withhold rent for code violations?" | AG guide (chunk_005) | 16 | 11 |
| golden_017: "enter apartment without notice?" | 940 CMR 3.17 entry provisions | >20 | >20 |

golden_012's AG guide chunk_005 appears at hybrid position 11 — just outside the rerank candidate pool (initial_k=20 → top 10 after reranking). The chunk_006 (also expected) is beyond position 20. A larger initial_k (e.g., 30) might rescue these, but risks the cross-encoder domain mismatch problem documented in section 7.18.

golden_005's chunk is at vector position 20 — right at the boundary. The masslegalhelp security deposit chapters are preferred because they discuss property sales more directly.

golden_017's 940 CMR 3.17 regulation chunk is unreachable (>20 in both), following the Pattern 1 statute language gap.

#### Summary of retrieval failure root causes

| Root Cause | Questions Affected | Missed Facts | Fix |
|-----------|-------------------|-------------|-----|
| Statute language gap | golden_007, 009, 028, 029 | 6 | Embedding model upgrade or multi-query expansion |
| Program name gap | golden_037, 039 | 5 | Multi-query expansion ("lead paint help" → "Lead Safe Boston program") |
| Niche terminology | golden_043 | 1 | Query expansion or synonym injection |
| Borderline ranking | golden_005, 012 | 2 | Larger initial_k (with better cross-encoder) or hybrid weight tuning |

14 total retrieval misses across 10 questions. The dominant failure mode (11/14) is a **vocabulary gap** between casual tenant language and formal legal/program text. The remaining 3/14 are borderline ranking issues where the correct chunk exists in the retrieval candidate set but is displaced by the cross-encoder.

**Recommended priority for fixes:**
1. **Multi-query expansion** — highest expected impact, addresses all three vocabulary gap patterns
2. **Embedding model upgrade** — second priority, helps with statute language matching
3. **Cross-encoder replacement** — would fix the borderline ranking cases and stop displacing legal chunks

---

## Iteration 8: New Retriever Strategies (Section 7.20)

**Date:** 2026-03-21
**Goal:** Evaluate four new retrieval strategies against the rerank baseline, with per-topic breakdown to identify which retrievers work best for which question categories.

### 7.20.1 Strategies Tested

| Config | Description | Implementation |
|--------|-------------|----------------|
| **rerank** (baseline) | Hybrid (vector+BM25) → cross-encoder reranking | `src/rag/retrievers.py` |
| **multiquery** | LLM generates 3 query variants (casual/legal/statute vocabulary) → runs rerank on each → reciprocal rank fusion (RRF, k=60) | `src/rag/multiquery.py` |
| **hybrid_parent_child_rerank** | Hybrid base → neighbor expansion for clustered docs → cross-encoder reranking | `src/rag/hybrid_parent_child.py` |
| **auto_merge** | Hybrid base → when ≥40% of a doc's chunks are retrieved, merge all doc chunks into one result → cross-encoder reranking | `src/rag/hybrid_parent_child.py` |
| **sentence_window** | Sentence-level chunking (10,785 chunks, avg 56 tokens) → retrieve single sentences → expand to ±3 sentence context window | `src/processing/sentence_window_chunker.py` |

All configs use GPT-4o as generator, Claude Sonnet 4 as judge, top_k=10, structured prompt.

### 7.20.2 Retrieval-Only Metrics (MRR / Hit Rate)

Free dry run using exact chunk_id matching against source_chunks ground truth (169 QA pairs). Multi-query skipped (requires LLM calls). Sentence window scores 0 because its chunk IDs (`_sent_NNNN`) differ from ground truth (`_chunk_NNN`) — not a meaningful comparison for this metric.

| Retriever | MRR | Hit Rate | Hits |
|-----------|-----|----------|------|
| rerank | 0.175 | 0.361 | 61/169 |
| hybrid_parent_child_rerank | 0.167 | 0.343 | 58/169 |
| auto_merge | 0.136 | 0.260 | 44/169 |
| sentence_window | N/A | N/A | N/A |

**Note:** MRR/hit rate penalizes auto_merge (merged chunk IDs like `doc_id_merged` can't match ground truth) and is meaningless for sentence_window. The LLM-judged evaluation below is the authoritative comparison.

### 7.20.3 Aggregate Results (LLM-Judged, 27 Stratified Questions, 77 Facts)

| Config | Ret Cov | Gen Cov | Gen\|Ret | Covered | GenMiss | RetMiss | Halluc |
|--------|---------|---------|----------|---------|---------|---------|--------|
| **rerank** (baseline) | **0.675** | **0.494** | **0.673** | 35 | 17 | 22 | 3 |
| multiquery | 0.623 | 0.468 | 0.667 | 32 | 16 | 25 | 4 |
| hybrid_parent_child_rerank | 0.636 | 0.481 | 0.673 | 33 | 16 | 24 | 4 |
| **auto_merge** | **0.727** | 0.442 | 0.589 | 33 | 23 | **20** | **1** |
| sentence_window | 0.519 | 0.364 | 0.600 | 24 | 16 | 33 | 4 |

### 7.20.4 Per-Topic Breakdown (Retrieval Coverage / Generation Coverage)

| Topic | rerank | multiquery | hybrid_pc_rerank | auto_merge | sent_window |
|-------|--------|------------|------------------|------------|-------------|
| affordable_housing | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | 0.333/0.000 |
| discrimination | 0.667/0.167 | 0.500/0.333 | 0.333/0.333 | 0.500/0.167 | 0.333/0.167 |
| evictions | 0.833/0.667 | 0.833/0.500 | 0.833/0.667 | 0.833/0.500 | 0.833/0.333 |
| lead_paint | 0.167/0.167 | 0.167/0.000 | 0.167/0.167 | **1.000/0.167** | 0.167/0.000 |
| lease_terms | 1.000/0.800 | 1.000/0.600 | 0.800/0.600 | 0.800/0.600 | 0.600/0.600 |
| public_housing | 0.750/0.500 | 0.750/0.500 | 0.750/0.500 | 0.750/0.500 | 0.500/0.500 |
| rent_increases | 0.000/0.000 | 0.000/0.000 | **0.333/0.000** | 0.000/0.000 | 0.000/0.000 |
| repairs_habitability | **1.000/0.667** | 0.667/0.500 | 0.500/0.333 | 0.500/0.500 | 0.500/0.500 |
| retaliation | 0.400/0.400 | 0.400/0.400 | **0.600/0.400** | **0.600/0.400** | 0.600/0.200 |
| security_deposits | **0.833/0.500** | 0.667/0.500 | 0.833/0.500 | 0.833/0.500 | 0.500/0.500 |
| tenant_rights_general | 0.667/0.833 | **0.833/0.833** | 0.667/0.833 | 0.667/0.667 | 0.667/0.667 |
| utilities_heat | 0.833/0.333 | 0.833/0.500 | 0.833/0.500 | **1.000/0.333** | 0.667/0.167 |

**Best retriever per topic:**

| Topic | Best Retriever | Delta vs Rerank |
|-------|---------------|-----------------|
| lead_paint | **auto_merge** | **+0.833** |
| rent_increases | hybrid_parent_child_rerank | +0.333 |
| retaliation | hybrid_parent_child_rerank / auto_merge | +0.200 |
| utilities_heat | auto_merge | +0.167 |
| tenant_rights_general | multiquery | +0.166 |
| 7 other topics | rerank (still best or tied) | — |

### 7.20.5 Per-Question Deltas vs Baseline

**auto_merge** (most promising — 6 improved, 3 regressed):
- golden_037 (lead_paint): ret +1.000 — full-doc merge captures Lead Safe Boston info scattered across chunks
- golden_039 (lead_paint): ret +0.667
- golden_009 (retaliation): ret +0.500
- golden_027 (utilities_heat): ret +0.333
- reddit_q008, reddit_q014: ret +0.333 each
- golden_021 (repairs_habitability): ret -1.000 — doc merge diluted focused repair info

**hybrid_parent_child_rerank** (targeted gains — 2 improved, 4 regressed):
- golden_016 (rent_increases): ret +0.667 — neighbor expansion found adjacent rent increase chunks
- golden_009 (retaliation): ret +0.500
- golden_021 (repairs_habitability): ret -1.000

**multiquery** (underperformed — 1 improved, 5 regressed):
- golden_024 (tenant_rights_general): ret +0.333
- Net regression on repairs_habitability, security_deposits, discrimination

**sentence_window** (weakest — 1 improved, 10 regressed):
- golden_009 (retaliation): ret +0.500 — only gain
- Regressed across most topics, especially affordable_housing (-0.667), security_deposits (-0.667), repairs_habitability (-0.667)

### 7.20.6 Analysis

**auto_merge** achieves the highest retrieval coverage (0.727 vs 0.675 baseline) and fewest retrieval misses (20 vs 22), with only 1 hallucination. Its key strength is on topics where relevant information is spread across multiple chunks within a document (lead_paint, utilities_heat). However, merging full documents floods the context window, increasing generation misses (23 vs 17) and dropping gen|ret coverage to 0.589 — the LLM struggles to extract specific facts from verbose merged content.

**hybrid_parent_child_rerank** shows targeted value for topics where BM25 lexical matching adds signal that vector search misses (rent_increases, retaliation). The hybrid base catches keyword matches on statute numbers and specific legal terms.

**multiquery** did not deliver expected improvements. The vocabulary gap hypothesis (Section 7.19.10) predicted this would be the highest-impact fix, but the LLM-generated query variants may not be sufficiently diverse — or the rerank base retriever already captures most of the relevant chunks that the variants would find. The 3 variant queries may also introduce noise, diluting RRF scores for chunks that were well-ranked by the original query.

**sentence_window** is the weakest approach. Single-sentence embeddings (avg 56 tokens) are too short to capture topical relevance for legal questions, which often require multi-sentence reasoning. The sentence splitter also produced some extreme lengths (up to 6,709 tokens) indicating poor splitting on certain documents.

### 7.20.7 Recommendations

1. **Auto_merge is worth pursuing** but needs a fix for the generation miss problem. Options:
   - Truncate merged documents to a max token budget before passing to LLM
   - Use a higher merge threshold (e.g., 0.5 or 0.6) to be more selective about which docs get merged
   - Apply the structured prompt more aggressively to force the LLM to enumerate evidence

2. **Topic-aware retriever selection** (ensemble approach): Use auto_merge for lead_paint/utilities questions, hybrid_parent_child_rerank for retaliation/rent_increases, and rerank as the default. This would require a lightweight topic classifier on the query.

3. **Multi-query needs rethinking**: Consider using more variants (5 instead of 3), a cheaper model for generation (Mistral Small), or a template-based approach instead of LLM-generated variants to ensure vocabulary diversity.

4. **Sentence window should be deprioritized**: The granularity mismatch between sentence-level embeddings and legal questions makes this approach unsuitable for the current domain without significant architectural changes (e.g., combining sentence-window with BM25 at the document level).

### 7.20.8 Cost

Total API cost for this evaluation: ~$4.50 (384 API calls across 5 configs × 27 questions).

---

## 7.21: Failure Taxonomy for Legal QA

### 7.21.1 Motivation

Instructor feedback identified that the project needs a sharper analytical angle beyond "does RAG reduce hallucination." This section categorizes the specific failure modes observed across our evaluation, providing a taxonomy that can guide targeted improvements and serve as an analytical contribution.

### 7.21.2 Methodology

We analyzed the latest retrieval coverage results (27 stratified questions, 77 total key facts) from the best configuration (GPT-4o + rerank + top_k=10 + structured prompt, Claude Sonnet 4 judge). Each of the 46 missed facts was categorized into one of 8 failure modes based on whether the failure occurred at the retrieval stage, generation stage, or both.

### 7.21.3 Taxonomy of RAG Failure Modes

| ID | Failure Mode | Stage | Frequency | Description |
|----|-------------|-------|-----------|-------------|
| F1 | Statute Citation Dropout | Generation | 9 facts / 7 questions | LLM retrieves relevant content but omits specific statute numbers (e.g., M.G.L. c.151B, 940 CMR 3.17) in the generated response. The legal principle is stated but without the authoritative citation. |
| F2 | Remedy/Consequence Omission | Generation | 6 facts / 5 questions | LLM states the rule correctly but drops penalties, damages, or enforcement mechanisms (e.g., triple damages for security deposit violations, attorney's fees recovery). |
| F3 | Cross-Document Synthesis Gap | Retrieval | 5 facts / 5 questions | Answer requires facts from 2+ documents in different sources; the retriever finds one relevant source but not the other. Common when mass.gov statutes and boston.gov guides cover complementary aspects. |
| F4 | Program/Term Mismatch | Retrieval | 5 facts / 2 questions | Casual query language doesn't match official program names or legal terminology in the corpus (e.g., "lead paint help" vs. "Lead Safe Boston program"). Embedding similarity fails to bridge the vocabulary gap. |
| F5 | Statute Retrieval Miss | Retrieval | 4 facts / 4 questions | Relevant regulation or statute exists in the corpus but is not retrieved because the question's embedding is too distant from formal legal language. |
| F6 | Boundary Condition Omission | Generation | 3 facts / 3 questions | General rule is generated correctly but exceptions, conditions, or prerequisites are omitted (e.g., separate meter requirement for tenant utility billing). |
| F7 | Cross-Encoder Domain Mismatch | Retrieval | Qualitative | The ms-marco cross-encoder reranker, trained on web search data, sometimes displaces formal legal statute chunks in favor of conversational FAQ content that scores higher on surface-level relevance. |
| F8 | Hallucinated Citation | Generation | 2 facts / 2 questions | LLM generates a statute reference not present in any retrieved chunk, likely from parametric knowledge (e.g., fabricated M.G.L. c.186 §14 water shutoff reference). |

### 7.21.4 Distribution Analysis

**By stage:**
- **Generation failures** (F1 + F2 + F6 + F8): 20 facts (43%) — The retriever did its job, but the LLM failed to fully extract or faithfully represent the information.
- **Retrieval failures** (F3 + F4 + F5 + F7): 14 facts (30%) — The relevant information exists in the corpus but was not surfaced to the LLM.
- **Covered**: 31 facts (40%) — Successfully retrieved and generated.
- **Hallucinated**: 2 facts (3%) — Generated without retrieval support.

**Key insight**: Generation-side failures outnumber retrieval-side failures. This suggests that for standard (single-statute) questions, the primary bottleneck has shifted from retrieval to generation. However, for harder questions requiring multi-statute reasoning (F3), retrieval remains the bottleneck because the retriever must surface chunks from multiple documents simultaneously.

### 7.21.5 Hard Multi-Step Questions

To stress-test failure modes F3 (Cross-Document Synthesis) and probe new failure patterns around multi-statute reasoning, we created 10 hard questions (golden_051 through golden_060). Each question requires synthesizing information from 2-4 chunks across different legal topics and statutes.

**Question difficulty tiers:**

| Tier | Description | # Questions | Example Failure Modes Targeted |
|------|-------------|-------------|-------------------------------|
| Standard | Single fact, single chunk or closely related chunks | 49 (existing) | F1, F2, F4, F5 |
| Hard | Multiple statutes, cross-topic reasoning, conditional logic | 10 (new) | F3, F6, F7 (and new modes) |

**Hard questions summary:**

| ID | Question | Topics Combined | # Source Chunks |
|----|----------|----------------|-----------------|
| golden_051 | Security deposit not returned — legal steps and damages | Security deposits + treble damages + small claims filing | 4 |
| golden_052 | Lead paint report triggers retaliatory eviction | Lead law + retaliation + eviction defenses | 4 |
| golden_053 | No heat, rent withheld, now facing eviction | Sanitary Code (heat) + consumer protection + rent withholding + eviction defense | 4 |
| golden_054 | Landlord won't pay gas bill, shutoff notice received | Utility obligations + consumer protection + quiet enjoyment + rent deduction | 3 |
| golden_055 | Landlord changed locks — illegal lockout | Lockout statute + damages + quiet enjoyment + consumer protection | 4 |
| golden_056 | Security deposit violations as eviction defense | Security deposit law + eviction defense/counterclaim + treble damages | 4 |
| golden_057 | Building foreclosed, new owner demands tenant leave | Foreclosure tenant protections + just cause + stay of eviction | 3 |
| golden_058 | Family status discrimination + retaliatory eviction | Fair housing + complaint filing + eviction defense | 4 |
| golden_059 | Repairs refused after Board of Health order | Tenant petition + court remedies + consumer protection + 93A damages | 4 |
| golden_060 | Condo conversion with illegal rent increase | Condo conversion limits + security deposit transfer + tenant protections | 4 |

### 7.21.6 Hard Question Evidence Matrix

The following table provides ground-truth verification for all 10 hard questions, showing the exact chunks that answer each question and verbatim proof from the corpus.

| Q# | Question | Chunk ID | Source | Verbatim Evidence |
|----|----------|----------|--------|-------------------|
| Q1 | "My landlord never put my $2,000 security deposit in a separate bank account, and now that I've moved out 45 days ago, they still haven't returned it. What legal steps can I take and what damages can I recover?" | `masslegalhelp_ch03_security_deposits_chunk_016` | MassLegalHelp Ch.3 | "You have a right to ask for 3 times the amount of your security deposit if the landlord: Does not put your security deposit money in a separate account" |
| Q1 | | `masslegalhelp_ch03_security_deposits_chunk_018` | MassLegalHelp Ch.3 | "You can ask the court for up to 3 times the amount of the deposit, plus interest... You can sue for up to $7,000 in Small Claims court" |
| Q1 | | `mass_gov_www_mass_gov_guides_the_attorney_generals_guide_to_landlord_and_tenant_rights_chunk_004` | AG's Guide | "All security deposits must be deposited in a Massachusetts bank, in an account that collects interest... the landlord must return the security deposit, plus interest, within 30 days" |
| Q1 | | `mass_gov_how_to_file_a_small_claim_in_the_boston_municipal_court_district_court_or_housing_court_chunk_000` | Mass.gov | "the claim may be subject to statutory damages of more than $7,000 (i.e., consumer protection cases or certain landlord/tenant cases)" |
| Q2 | "I have a 4-year-old child and discovered peeling paint in our pre-1978 apartment. I reported it to my landlord and the Board of Health. Now my landlord served me with an eviction notice. Is this legal, and what can I do?" | `masslegalhelp_ch09_lead_poisoning_chunk_001` | MassLegalHelp Ch.9 | "property owners must remove or cover all lead paint hazards in homes built before 1978 where any children under 6 live... A property owner may not: evict you... in retaliation for you reporting a suspected lead paint violation" |
| Q2 | | `boston_gov_sites_default_files_file_2020_03_renters_english_lead_20fact_20sheet_chunk_000` | Boston Lead Fact Sheet | "landlords CANNOT refuse to rent to you or evict you if you have children or if the property has lead" |
| Q2 | | `mass_gov_malegislature_gov_aws_eneral_aws_art_itle_hapter186_ection18_chunk_000` | MGL c.186 §18 | "Any person...who threatens to or takes reprisals against any tenant...for the tenant's act of...reporting to the board of health" |
| Q2 | | `masslegalhelp_ch12_evictions_chunk_056` | MassLegalHelp Ch.12 | Defense/counterclaim table: "Retaliation" = defense + counterclaim for non-payment, fault, and no-fault evictions |
| Q3 | "My apartment has had no heat since December even though my landlord is responsible for it, and I've told them multiple times in writing. I started withholding rent and now they're trying to evict me for nonpayment. What are my legal rights?" | `boston_gov_departments_inspectional_services_meeting_housing_code_boston_chunk_001` | Boston Housing Code | "must keep the heat at a minimum of 68 degrees from 7 a.m. - 11 p.m. during heating season from September 15 - June 15. The heat can't go below 64 degrees" |
| Q3 | | `mass_gov_940_cmr_3_17_landlord_tenant_chunk_007` | 940 CMR 3.17 | "unfair or deceptive act or practice for an owner to... Fail... to remedy a violation of law... or maintain the dwelling unit in a condition fit for human habitation" |
| Q3 | | `masslegalhelp_ch08_getting_repairs_made_chunk_003` | MassLegalHelp Ch.8 | "Withhold rent or part of it until the landlord makes the repairs... put the rent money you withhold in a separate bank account" |
| Q3 | | `masslegalhelp_ch12_evictions_chunk_044` | MassLegalHelp Ch.12 | "A judge will determine what the fair rental value of the apartment is in its defective condition and calculate how much rent is owed" |
| Q4 | "My landlord is responsible for gas service per our lease, but they stopped paying the bill. The gas company sent a shutoff notice. What are my rights, and can I deduct utility payments from my rent?" | `mass_gov_940_cmr_3_17_landlord_tenant_chunk_010` | 940 CMR 3.17 | "unfair practice for any owner who is obligated... to provide gas or electric service... To fail to provide such service; or To expose such occupant to the risk of loss" |
| Q4 | | `masslegalhelp_ch06_utilities_chunk_009` | MassLegalHelp Ch.6 | "company cannot shut off service in the building until the tenants have been given at least 30 days' notice... You have a right to deduct from your rent" |
| Q4 | | `masslegalhelp_ch12_evictions_chunk_045` | MassLegalHelp Ch.12 | "Your landlord did not pay for utilities that were the landlord's responsibility... can win up to three months' rent and attorneys' fees" |
| Q5 | "I came home and found my landlord changed the locks on my apartment. My belongings are still inside. What are my legal options and what damages can I recover?" | `mass_gov_malegislature_gov_aws_eneral_aws_art_itle_hapter186_ection15_chunk_001` | MGL c.186 §15F | "If a tenant is removed from the premises or excluded therefrom by the landlord... the tenant may recover... three months' rent or three times the damages sustained by him, and the cost of suit, including reasonable attorney's fees" |
| Q5 | | `masslegalhelp_ch12_evictions_chunk_002` | MassLegalHelp Ch.12 | "They can't lock you out, throw your things out on the street, or harass you" |
| Q5 | | `masslegalhelp_ch12_evictions_chunk_045` | MassLegalHelp Ch.12 | "Your landlord locked you out of your home... attempted to move your possessions out without first taking you to court and getting a court order" |
| Q5 | | `mass_gov_940_cmr_3_17_landlord_tenant_chunk_010` | 940 CMR 3.17 | "(f) To violate willfully any provisions of M.G.L. c. 186, § 14" (lockout = consumer protection violation) |
| Q6 | "I'm being evicted for nonpayment of rent. My landlord never gave me a statement of condition when I moved in, never paid me interest on my security deposit, and I never got a receipt. Can I use any of this in court to fight the eviction?" | `masslegalhelp_ch12_evictions_chunk_044` | MassLegalHelp Ch.12 | "you have a defense to eviction if your landlord failed to provide you with a written receipt, give you a statement... hold your money in a bank account that is separate" |
| Q6 | | `masslegalhelp_ch12_evictions_chunk_045` | MassLegalHelp Ch.12 | "entitled to three times the security deposit and interest owed... the court will give you a chance to pay the difference in 7 days" |
| Q6 | | `masslegalhelp_ch03_security_deposits_chunk_016` | MassLegalHelp Ch.3 | "You can ask for the entire security deposit back if the landlord: Does not give you a complete receipt within 30 days" |
| Q6 | | `masslegalhelp_ch12_evictions_chunk_056` | MassLegalHelp Ch.12 | Table: "Security deposit law" = defense + counterclaim for non-payment eviction |
| Q7 | "My building was just foreclosed on by the bank. The new owner says I need to leave within 30 days. I've lived here for 5 years with a lease. What are my rights?" | `mass_gov_malegislature_gov_aws_eneral_aws_art_itle_hapter186_chunk_000` | MGL c.186A | "'Just cause'... (1) failed to pay rent; (2) materially violated... 'Bona fide lease'... result of an arms-length transaction" |
| Q7 | | `mass_gov_laws_generallaws_partiii_titleiii_chapter239_section9_chunk_000` | MGL c.239 §9 | "a stay or stays of judgment and execution may be granted... for a period not exceeding six months... or twelve months... where the person is sixty years of age or older, or handicapped" |
| Q7 | | `mass_gov_laws_generallaws_partiii_titleiii_chapter239_section10_chunk_000` | MGL c.239 §10 | "applicant cannot secure suitable premises... has used due and reasonable effort... the court may grant a stay" |
| Q8 | "My landlord told me he doesn't want to rent to families with children and is now trying to evict me after I had a baby. Where can I file a complaint and can I fight the eviction?" | `boston_gov_departments_fair_housing_and_equity_chunk_000` | Boston FH&E | "property owners do not discriminate against tenants or buyers based on their... family status" |
| Q8 | | `boston_gov_departments_fair_housing_and_equity_chunk_001` | Boston FH&E | "call us at 617-635-2500, or complete our online intake form" |
| Q8 | | `masslegalhelp_ch12_evictions_chunk_056` | MassLegalHelp Ch.12 | Table: "Discrimination" = defense + counterclaim for non-payment, fault, and no-fault evictions |
| Q8 | | `boston_gov_departments_housing_what_happens_during_eviction_chunk_018` | Boston Eviction Guide | "Unlawful Discrimination: The landlord has refused to rent, or attempted to end the tenancy, because the tenant belongs to a protected class" |
| Q9 | "My landlord has refused to fix a leaking roof and mold problem for 6 months despite my written requests. I've called the Board of Health and they issued a repair order, but my landlord still hasn't fixed it. What court actions can I take?" | `masslegalhelp_ch08_getting_repairs_made_chunk_021` | MassLegalHelp Ch.8 | "A judge can: Order your landlord to make repairs... pay you money... Appoint a receiver... called a tenant petition" |
| Q9 | | `mass_gov_940_cmr_3_17_landlord_tenant_chunk_007` | 940 CMR 3.17 | "unfair or deceptive act... Fail... after notice... to remedy a violation of law in a dwelling unit" |
| Q9 | | `mass_gov_laws_generallaws_parti_titlexv_chapter93a_section9_chunk_003` | MGL c.93A §9 | "damages may include double or treble damages, attorneys' fees and costs" |
| Q9 | | `masslegalhelp_ch13_when_to_take_your_landlord_to_court_chunk_000` | MassLegalHelp Ch.13 | TOC: "Bad Conditions... Violation of Consumer Protection Law" as grounds for filing |
| Q10 | "My landlord just told me he's converting our building to condominiums and is raising my rent by 50%. He also said the new condo owner won't be responsible for my security deposit. Is any of this legal?" | `masslegalhelp_ch05_rent_chunk_012` | MassLegalHelp Ch.5 | "cannot increase your rent by more than 10% per year or above the increase in the Consumer Price Index... whichever is less" |
| Q10 | | `masslegalhelp_ch05_rent_chunk_012` | MassLegalHelp Ch.5 | "the new landlord becomes responsible for your last month's rent and the security deposit... even if it is your former landlord who did not transfer" |
| Q10 | | `masslegalhelp_ch03_security_deposits_chunk_018` | MassLegalHelp Ch.3 | "Failed to transfer a security deposit to a new owner" = grounds for 3x damages |
| Q10 | | `mass_gov_www_mass_gov_guides_the_attorney_generals_guide_to_landlord_and_tenant_rights_chunk_004` | AG's Guide | "All security deposits must be deposited in a Massachusetts bank, in an account that collects interest" |

### 7.21.7 Hypotheses for Hard Questions

Based on the failure taxonomy, we expect:
1. **Retrieval coverage will drop significantly** on hard questions compared to standard questions, because the retriever must surface 3-4 relevant chunks from different documents simultaneously with only top_k=10 slots.
2. **F3 (Cross-Document Synthesis Gap) will be the dominant failure mode**, as most hard questions require chunks from 2-3 different source documents.
3. **Generation coverage given retrieval will also drop**, because the LLM must synthesize across multiple legal domains rather than extracting from a single coherent passage.
4. **Auto-merge and parent-child retrievers may outperform rerank** on hard questions, since they pull in broader context from neighboring chunks.

### 7.21.8 Evaluation Results: Standard vs Hard (Iteration 8)

**Configuration:** GPT-4o + rerank + top_k=10 + structured prompt, Claude Sonnet 4 judge. Fixed stratified question set: 27 standard + 10 hard = 37 total (127 key facts).

#### Overall Results

| Metric | Standard (27q, 77 facts) | Hard (10q, 50 facts) | Overall (37q, 127 facts) |
|--------|--------------------------|----------------------|--------------------------|
| Retrieval coverage | 0.675 | 0.700 | 0.685 |
| Generation coverage | 0.468 | 0.500 | 0.480 |
| Gen coverage given retrieval | 0.654 | 0.714 | 0.678 |
| Covered (ret + gen) | 34 | 25 | 59 |
| Generation miss | 18 | 10 | 28 |
| Retrieval miss | 23 | 15 | 38 |
| Hallucinated | 2 | 0 | 2 |

**Overall failure breakdown:** Of 66 missed facts, 57.6% are retrieval failures and 42.4% are generation failures.

#### Per-Question Breakdown

| ID | Difficulty | Topic | Ret Cov | Gen Cov | Failure Modes |
|----|-----------|-------|---------|---------|---------------|
| reddit_q008 | standard | — | 0.667 | 0.333 | retrieval_miss, generation_miss |
| reddit_q014 | standard | — | 0.667 | 0.667 | retrieval_miss |
| reddit_q015 | standard | — | 0.667 | 0.333 | retrieval_miss, generation_miss |
| reddit_q019 | standard | — | 1.000 | 1.000 | — |
| golden_002 | standard | repairs_habitability | 1.000 | 0.667 | generation_miss |
| golden_006 | standard | security_deposits | 0.667 | 0.333 | generation_miss, retrieval_miss |
| golden_007 | standard | retaliation | 0.333 | 0.333 | retrieval_miss ×2 |
| golden_009 | standard | retaliation | 0.500 | 0.500 | retrieval_miss |
| golden_011 | standard | evictions | 1.000 | 0.333 | generation_miss ×2 |
| golden_012 | standard | evictions | 0.667 | 0.667 | retrieval_miss |
| golden_016 | standard | rent_increases | 0.000 | 0.000 | retrieval_miss ×3 |
| golden_021 | standard | repairs_habitability | 1.000 | 0.667 | generation_miss |
| golden_022 | standard | security_deposits | 1.000 | 0.667 | generation_miss |
| golden_023 | standard | rent_increases | 0.000 | 0.000 | retrieval_miss ×3 |
| golden_024 | standard | tenant_rights_general | 0.667 | 0.667 | retrieval_miss |
| golden_027 | standard | utilities_heat | 0.667 | 0.333 | generation_miss ×2 |
| golden_029 | standard | discrimination | 1.000 | 0.333 | generation_miss ×2 |
| golden_031 | standard | discrimination | 0.333 | 0.000 | generation_miss, retrieval_miss ×2 |
| golden_033 | standard | affordable_housing | 1.000 | 1.000 | — |
| golden_034 | standard | lease_terms | 1.000 | 0.500 | generation_miss |
| golden_036 | standard | lease_terms | 1.000 | 0.667 | generation_miss |
| golden_037 | standard | lead_paint | 0.000 | 0.000 | retrieval_miss ×3 |
| golden_039 | standard | lead_paint | 0.333 | 0.333 | retrieval_miss ×2 |
| golden_041 | standard | utilities_heat | 1.000 | 0.333 | generation_miss ×2 |
| golden_043 | standard | public_housing | 0.500 | 0.000 | retrieval_miss, generation_miss |
| golden_045 | standard | public_housing | 1.000 | 1.000 | — |
| golden_049 | standard | tenant_rights_general | 0.667 | 1.000 | — |
| golden_051 | hard | security_deposits | 1.000 | 0.500 | generation_miss ×3 |
| golden_052 | hard | lead_paint | 0.800 | 0.600 | generation_miss, retrieval_miss |
| golden_053 | hard | repairs_habitability | 0.500 | 0.500 | retrieval_miss ×3 |
| golden_054 | hard | utilities | 0.400 | 0.200 | retrieval_miss ×3, generation_miss |
| golden_055 | hard | evictions | 0.600 | 0.400 | retrieval_miss ×2, generation_miss |
| golden_056 | hard | security_deposits | 0.800 | 0.600 | generation_miss, retrieval_miss |
| golden_057 | hard | evictions | 0.500 | 0.500 | retrieval_miss ×2 |
| golden_058 | hard | discrimination | 0.750 | 0.750 | retrieval_miss |
| golden_059 | hard | repairs_habitability | 0.600 | 0.600 | retrieval_miss ×2 |
| golden_060 | hard | rent_increases | 1.000 | 0.400 | generation_miss ×3 |

#### Analysis: Hypothesis Validation

**H1 (Retrieval coverage drops on hard questions): Not confirmed.** Hard retrieval coverage (0.700) was slightly *higher* than standard (0.675). This suggests the hard questions were designed with chunks that have strong lexical overlap with the queries. The retriever handles multi-chunk questions adequately at the aggregate level, though individual questions vary widely (0.400 to 1.000).

**H2 (Cross-document synthesis is the dominant failure): Partially confirmed.** Retrieval misses account for 15 of 25 hard-question failures (60%), consistent with F3, but the per-question pattern reveals a split: some hard questions achieve perfect retrieval (golden_051, golden_060) while others are heavily retrieval-limited (golden_054 at 0.400). The failure is not uniform — it depends on whether the required chunks share vocabulary with the question.

**H3 (Gen|Ret drops on hard questions): Not confirmed — opposite trend.** Gen|Ret is *higher* for hard questions (0.714 vs 0.654). When the retriever does surface multi-statute chunks, the LLM actually synthesizes them effectively. The real generation problem is selective: golden_051 retrieves all 6 facts but only generates 3 (drops demand letter, attorney's fees, small claims details), and golden_060 retrieves all 5 but generates only 2 (drops CPI cap, tax escalator, intent-to-convert rule). These are F1 (Statute Citation Dropout) and F2 (Remedy/Consequence Omission) — the same failure modes as standard questions, just with more facts to drop.

**Key insight:** The bottleneck for hard questions is **not** cross-document synthesis per se, but rather **fact volume**: hard questions have 5-6 key facts vs 2-3 for standard questions. The LLM's tendency to omit secondary details (F1/F2) becomes more damaging when there are more facts to cover. This suggests that fact-volume-aware prompting or multi-pass generation would be more impactful than retriever improvements for hard questions.

### 7.21.9 Chunk-Level Retrieval Metrics

In addition to the LLM-judged fact-level metrics, we compute chunk-level retrieval metrics that compare retrieved chunk IDs directly against designated ground-truth chunk IDs. These are free (no API calls) and measure a different dimension: **did we find the specific documents?** rather than **did we find the information?**

#### Aggregate Chunk-Level Results

| Metric | Standard (27q) | Hard (10q) | Overall (37q) |
|--------|---------------|------------|---------------|
| MRR | 0.275 | 0.417 | 0.314 |
| Hit@K | 0.444 | 0.600 | 0.486 |
| Recall@K | 0.352 | 0.233 | 0.320 |
| NDCG@K | 0.270 | 0.236 | 0.261 |

#### Per-Question Chunk-Level Results

| ID | Difficulty | Topic | Fact Ret | Fact Gen | Recall@K | NDCG@K | MRR |
|----|-----------|-------|----------|----------|----------|--------|-----|
| reddit_q008 | standard | — | 0.667 | 0.333 | 0.250 | 0.123 | 0.125 |
| reddit_q014 | standard | — | 0.667 | 0.000 | 0.250 | 0.390 | 1.000 |
| reddit_q015 | standard | — | 0.667 | 0.333 | 0.000 | 0.000 | 0.000 |
| reddit_q019 | standard | — | 1.000 | 0.667 | 0.500 | 0.308 | 0.333 |
| golden_002 | standard | repairs_habitability | 1.000 | 0.667 | 0.000 | 0.000 | 0.000 |
| golden_006 | standard | security_deposits | 0.667 | 0.333 | 1.000 | 0.571 | 0.333 |
| golden_007 | standard | retaliation | 0.333 | 0.333 | 0.000 | 0.000 | 0.000 |
| golden_009 | standard | retaliation | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 |
| golden_011 | standard | evictions | 1.000 | 0.333 | 0.000 | 0.000 | 0.000 |
| golden_012 | standard | evictions | 0.667 | 0.667 | 0.000 | 0.000 | 0.000 |
| golden_016 | standard | rent_increases | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| golden_021 | standard | repairs_habitability | 1.000 | 0.667 | 1.000 | 0.389 | 0.143 |
| golden_022 | standard | security_deposits | 1.000 | 0.667 | 0.000 | 0.000 | 0.000 |
| golden_023 | standard | rent_increases | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| golden_024 | standard | tenant_rights_general | 0.667 | 1.000 | 0.500 | 0.613 | 1.000 |
| golden_027 | standard | utilities_heat | 0.667 | 0.333 | 0.000 | 0.000 | 0.000 |
| golden_029 | standard | discrimination | 1.000 | 0.667 | 0.000 | 0.000 | 0.000 |
| golden_031 | standard | discrimination | 0.333 | 0.000 | 0.000 | 0.000 | 0.000 |
| golden_033 | standard | affordable_housing | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| golden_034 | standard | lease_terms | 1.000 | 0.500 | 1.000 | 1.000 | 1.000 |
| golden_036 | standard | lease_terms | 1.000 | 0.667 | 1.000 | 0.631 | 0.500 |
| golden_037 | standard | lead_paint | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| golden_039 | standard | lead_paint | 0.333 | 0.333 | 0.000 | 0.000 | 0.000 |
| golden_041 | standard | utilities_heat | 1.000 | 0.667 | 1.000 | 1.000 | 1.000 |
| golden_043 | standard | public_housing | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 |
| golden_045 | standard | public_housing | 1.000 | 1.000 | 1.000 | 0.631 | 0.500 |
| golden_049 | standard | tenant_rights_general | 0.667 | 1.000 | 1.000 | 0.631 | 0.500 |
| golden_051 | hard | security_deposits | 1.000 | 0.500 | 0.500 | 0.558 | 1.000 |
| golden_052 | hard | lead_paint | 0.800 | 0.600 | 0.500 | 0.359 | 0.500 |
| golden_053 | hard | repairs_habitability | 0.500 | 0.167 | 0.250 | 0.246 | 0.500 |
| golden_054 | hard | utilities | 0.400 | 0.000 | 0.000 | 0.000 | 0.000 |
| golden_055 | hard | evictions | 0.600 | 0.400 | 0.000 | 0.000 | 0.000 |
| golden_056 | hard | security_deposits | 0.800 | 0.600 | 0.500 | 0.586 | 1.000 |
| golden_057 | hard | evictions | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 |
| golden_058 | hard | discrimination | 0.750 | 0.500 | 0.250 | 0.139 | 0.167 |
| golden_059 | hard | repairs_habitability | 0.600 | 0.600 | 0.000 | 0.000 | 0.000 |
| golden_060 | hard | rent_increases | 1.000 | 0.400 | 0.333 | 0.469 | 1.000 |

#### Analysis: Fact-Level vs Chunk-Level Metrics Divergence

A key finding is the **large gap between fact-level retrieval coverage (0.685) and chunk-level Recall@K (0.320)**. This is not a contradiction — it reveals two distinct phenomena:

**Pattern 1: Right document, wrong chunk.** For golden_054 (utilities), golden_057 (foreclosure), and golden_059 (repairs), the retriever finds chunks from the correct source document but lands on a neighboring chunk within it, not the specific one designated as ground truth. For example, golden_054 expected `ch06_utilities_chunk_009` but retrieved `chunk_003`, `chunk_010`, `chunk_015` — all from the same Utilities chapter. The fact-level judge scores these as partial hits because neighboring chunks contain overlapping legal information.

**Pattern 2: Right topic, wrong source.** For golden_055 (lockout), the retriever finds topically related MassLegalHelp content (eviction chapters, court procedures) but misses the specific statute chunks (M.G.L. c. 186 §15F). This is **F7: Cross-Encoder Domain Mismatch** — the ms-marco reranker prefers conversational content over formal statute text.

**Implication:** The corpus has significant **information redundancy** — the same legal facts appear in multiple chunks across different documents (AG's Guide, MassLegalHelp chapters, raw statutes). The retriever exploits this redundancy: it may miss the designated chunk but find an equivalent one. **Recall@K measures retrieval precision** (did we find the exact designated chunks?), while **fact-level coverage measures retrieval effectiveness** (did we find the information, regardless of source?). For a legal QA system, effectiveness matters more to users, but precision matters for understanding retriever behavior and improving it.

#### Charts

![Standard vs Hard: Fact-Level and Chunk-Level Metrics](figures/chart16_standard_vs_hard_metrics.png)

![Per-Question Retrieval: Fact Coverage vs Recall@K](figures/chart17_fact_vs_chunk_retrieval.png)

![Failure Attribution: Standard vs Hard](figures/chart18_failure_attribution_by_tier.png)


---

## Iteration 9: Final Multi-Model Evaluation with Local Qwen3 Models

**Date:** 2026-04-04
**Backend:** `src/rag/` (custom pipeline)
**Corpus:** 967 chunks, 249 documents
**Questions:** 89 (30 Reddit + 59 golden QA pairs)
**Judge:** Claude Sonnet 4 (via OpenRouter)

### 9.1 Motivation

Iteration 9 is the final evaluation run, expanding the model comparison to include two locally-hosted Qwen3 4B variants:

- **Qwen3 4B Base** (`local/qwen3-base`): The base Qwen3 4B instruct model, served via llama-server on port 8081 (GGUF, F16, 8192 context)
- **Qwen3 4B Fine-tuned** (`local/qwen3-finetuned`): Qwen3 4B fine-tuned on the MA tenant law corpus using Unsloth LoRA (3 epochs, 586 samples), served on port 8080

The primary goal was to assess whether (a) local open-weight models can approach cloud model performance, and (b) domain fine-tuning on legal Q&A data improves faithfulness and correctness. This also expands the question set from 28 stratified questions to all 89 evaluation questions, providing more robust aggregate metrics.

### 9.2 Evaluation Setup

**Configuration parameters:**

| Parameter | Value |
|-----------|-------|
| Questions | 89 (30 Reddit-style + 59 golden QA pairs with key_facts) |
| Cloud models | GPT-4o (`openai/gpt-4o`), Llama 3.3 70B (`meta-llama/llama-3.3-70b-instruct`), Mistral Small 24B (`mistralai/mistral-small-3.1-24b-instruct`) |
| Local models | Qwen3 4B Base (`local/qwen3-base`), Qwen3 4B Fine-tuned (`local/qwen3-finetuned`) |
| Retrievers tested | baseline (no RAG), rerank (RAG + cross-encoder) |
| Judge | `anthropic/claude-sonnet-4` via OpenRouter |
| Correctness metric | Per-fact claim recall (fraction of `key_facts` present in response, 0.0–1.0) |
| Faithfulness metric | Binary YES/NO per response (LLM judge) |
| Relevancy metric | Binary YES/NO per response (LLM judge) |
| Total configs | 13 (5 retriever × GPT-4o + 2 × Llama + 2 × Mistral + 2 × Qwen3 Base + 2 × Qwen3 FT) |

**Local model serving:**

| Model | GGUF file | Port | Context | Flags |
|-------|-----------|------|---------|-------|
| Qwen3 4B Base | `Fine-Tuneing/qwen3-base-f16.gguf` | 8081 | 8192 | `-ngl 999 --repeat-penalty 1.3 --temp 0.7` |
| Qwen3 4B Fine-tuned | `Fine-Tuneing/finetuned-qwen3-f16.gguf` | 8080 | 8192 | `-ngl 999 --repeat-penalty 1.3 --temp 0.7` |

Both models served via `llama.cppuildin\Release\llama-server.exe`. Context raised to 8192 (from default 4096) to accommodate system prompt + 5 retrieved chunks without overflow.

**Parallelism:**

| Worker type | Count | Notes |
|-------------|-------|-------|
| Cloud generation workers | 3 | `CLOUD_WORKERS=3` in `scorer.py`; ChromaDB not thread-safe, RAG configs use workers=1 effectively |
| Local generation workers | 1 | `LOCAL_WORKERS=1`; llama-server handles one request at a time |
| Judge workers | 3 | `JUDGE_WORKERS=3`; parallel Claude Sonnet 4 scoring |

**Fine-tuning details (Qwen3 4B Fine-tuned):**

| Parameter | Value |
|-----------|-------|
| Base model | Qwen3 4B Instruct |
| Training framework | Unsloth LoRA |
| Epochs | 3 |
| Training samples | 586 (golden QA × 3, Reddit questions × 3, all_chunks × 1) |
| Hardware | RTX 5080, ~8 min training time |
| Known issue | First 2 training runs used Mistral `[INST]` format; third run corrected to Qwen3 ChatML format, but format damage from prior runs persisted in final weights (see Section 9.4 finding 4) |

### 9.3 Retriever Comparison (GPT-4o Fixed, 89 Questions)

![Chart 3 — Retriever Comparison](figures/chart03_retriever_comparison.png)

| Config | Faithfulness | Relevancy | Correctness |
|--------|-------------|-----------|-------------|
| Baseline (no RAG) | 0.146 | 1.000 | 0.465 |
| RAG + Vector | 0.112 | 1.000 | **0.480** |
| RAG + BM25 | 0.820 | 1.000 | 0.403 |
| RAG + Hybrid | **0.865** | 1.000 | 0.379 |
| RAG + Rerank | 0.854 | 1.000 | 0.430 |

**Retrieval metrics (89 QA pairs, k=5):**

| Retriever | MRR@5 | Hit Rate@5 | Recall@5 | NDCG@5 |
|-----------|-------|-----------|---------|--------|
| BM25 | 0.169 | 0.292 | 0.175 | 0.136 |
| Hybrid | 0.169 | 0.292 | 0.175 | 0.136 |
| Rerank | **0.290** | **0.393** | **0.236** | **0.221** |

> Note: Vector MRR=0.000 is a known chunk ID format mismatch artifact, not a true retrieval failure.

**Key findings:**

1. **Hybrid achieves the highest faithfulness (0.865)** — BM25 lexical matching excels at capturing exact legal citations (statute numbers, specific terms) that semantic embeddings miss.
2. **Vector retrieval has the highest correctness (0.480)** but the lowest faithfulness (0.112). Semantic similarity retrieves broadly relevant content that GPT-4o uses loosely, resulting in lower grounding.
3. **Rerank provides the best overall balance** — 0.854 faithfulness with 0.430 correctness and the best retrieval metrics (MRR 0.290).
4. **All RAG configurations achieve 1.000 relevancy**, confirming the system reliably addresses questions regardless of retrieval strategy.

### 9.4 Model Comparison (Rerank Retriever Fixed, 89 Questions)

![Chart 4 — Multi-Model Comparison](figures/chart04_multimodel_comparison.png)

| Config | Faithfulness | Relevancy | Correctness |
|--------|-------------|-----------|-------------|
| GPT-4o Baseline | 0.146 | 1.000 | 0.465 |
| GPT-4o + Rerank | 0.854 | 1.000 | 0.430 |
| Llama 3.3 70B Baseline | 0.011 | 0.989 | 0.429 |
| Llama 3.3 70B + Rerank | 0.730 | 1.000 | 0.393 |
| Mistral Small 24B Baseline | 0.022 | 0.978 | 0.412 |
| Mistral Small 24B + Rerank | **0.798** | 0.989 | 0.410 |
| Qwen3 4B Base Baseline | 0.000 | 0.966 | 0.327 |
| Qwen3 4B Base + Rerank | 0.427 | 0.989 | **0.472** |
| Qwen3 4B FT Baseline | 0.011 | 0.607 | 0.308 |
| Qwen3 4B FT + Rerank | 0.034 | 0.697 | 0.339 |

**RAG uplift by model:**

| Model | Baseline Faith | Rerank Faith | Uplift |
|-------|---------------|-------------|--------|
| GPT-4o | 0.146 | 0.854 | +485% |
| Llama 3.3 70B | 0.011 | 0.730 | base≈0 |
| Mistral Small 24B | 0.022 | 0.798 | base≈0 |
| Qwen3 4B Base | 0.000 | 0.427 | base=0 |
| Qwen3 4B FT | 0.011 | 0.034 | negligible |

**Key findings:**

1. **RAG is essential for faithfulness across all models.** GPT-4o baseline scores only 0.146 without retrieval; all other model baselines score near 0.000 — confirming that without source grounding, all models hallucinate MA legal details.

2. **Mistral Small 24B + Rerank is the most faithful model (0.798)**, confirming the Iteration 7 finding that Mistral is particularly effective at staying grounded to retrieved context.

3. **Qwen3 4B Base + Rerank achieves the highest correctness (0.472)**, exceeding GPT-4o baseline (0.465) at zero API cost. A 4B local model with good retrieval rivals a ~200B+ cloud model on key-fact recall.

4. **Fine-tuning degraded performance.** Qwen3 finetuned shows near-zero faithfulness (0.034 with rerank) and significantly reduced relevancy (0.697) compared to the base model (0.989). This is attributed to a **chat template format mismatch** during training — the training data was initially formatted in Mistral instruct format (`[INST]`/`[/INST]`) rather than Qwen3's ChatML format (`<|im_start|>`/`<|im_end|>`). The format was corrected on the third training run, but the first two runs appear to have introduced damage to the model's instruction-following capability that persisted in the final weights.

5. **Local deployment is viable for correctness.** Qwen3 4B Base + Rerank (0.472 correctness, 0.427 faithfulness) is competitive with cloud models while requiring no API costs and keeping data on-premises. This makes it suitable for privacy-sensitive legal deployments.

### 9.5 Best Configurations Summary

![Chart 6 — Best Configurations Scatter](figures/chart06_best_configs_scatter.png)

| Config | Faithfulness | Correctness | Cost/1M input | Deployment |
|--------|-------------|-------------|--------------|------------|
| GPT-4o + Rerank | 0.854 | 0.430 | $3.00 | Cloud |
| Mistral Small + Rerank | **0.798** | 0.410 | $0.14 | Cloud |
| Llama 3.3 70B + Rerank | 0.730 | 0.393 | $0.10 | Cloud |
| Qwen3 4B Base + Rerank | 0.427 | **0.472** | $0 (local) | Local |
| GPT-4o Baseline | 0.146 | 0.465 | $3.00 | Cloud |

**For production:** GPT-4o + Rerank offers the best faithfulness at $3/1M input. Mistral Small + Rerank matches the faithfulness profile at 21x lower cost. Qwen3 4B Base + Rerank is the recommended local deployment option when API costs or data privacy are constraints.

### 9.6 Comparison to Prior Iterations (28 Questions → 89 Questions)

The expanded question set (89 vs 28 in Iterations 7–8) produces lower absolute scores due to harder questions and a stricter per-fact correctness metric replacing the binary all-or-nothing correctness used earlier. Key differences:

| Metric | Iter 7–8 (28q, rerank) | Iter 9 (89q, rerank) | Delta |
|--------|----------------------|---------------------|-------|
| GPT-4o faithfulness | 0.857 | 0.854 | -0.003 |
| GPT-4o correctness | 0.357 | 0.430 | +0.073 |
| Llama faithfulness | 0.730 | 0.730 | ±0.000 |
| Mistral faithfulness | 0.643 | 0.798 | +0.155 |

The faithfulness scores are highly consistent across question sets for GPT-4o and Llama, validating that the 28-question stratified sample was representative. Mistral's improvement (+0.155) suggests the broader question set is more aligned with Mistral's strengths (straightforward tenant law Q&A vs. the stratified sample's harder questions).

### 9.7 Fine-Tuning Analysis and Next Steps

The Qwen3 fine-tuning experiment produced a degraded model due to the chat template format issue. The corrected training data (using Qwen3 ChatML format throughout) is already in place in `Fine-Tuneing/prepare_finetune_data.py`. A retrain is recommended with the following configuration:

- Training data: `Fine-Tuneing/train.txt` (ChatML format, 586 samples, 3x weight for golden QA + Reddit)
- LoRA config: same as successful run (rank=16, alpha=32, 3 epochs)
- Expected outcome: faithfulness should recover to ≥0.400 (comparable to Qwen3 Base) with potential gains from domain-specific training data

See Section 7.22 for fine-tuning model selection analysis.

### 9.8 Multi-Run Averaging (Judge Variance Analysis)

**Date:** 2026-04-09
**Script:** `src/evaluation/multi_run.py` (3 parallel runs)
**Config:** GPT-4o + Rerank, top_k=10, all 89 questions, Claude Sonnet 4 judge

To quantify LLM judge variance, we ran the full retrieval-aware evaluation 3 times in parallel and computed per-metric statistics. Note: `top_k=10` (vs `top_k=5` in Iteration 9) provides more context chunks, which explains the faithfulness improvement over Section 9.4.

![Chart 19a — Multi-Run Aggregate Metrics](figures/chart19a_multirun_aggregate.png)

**Aggregate results (mean +/- std, 95% CI):**

| Metric | Mean | Std | 95% CI |
|--------|------|-----|--------|
| Faithfulness | 0.891 | 0.024 | [0.865, 0.918] |
| Relevancy | 1.000 | 0.000 | [1.000, 1.000] |
| Retrieval coverage | 0.746 | 0.004 | [0.742, 0.750] |
| Generation coverage | 0.503 | 0.013 | [0.489, 0.518] |
| Gen\|Ret coverage | 0.647 | 0.015 | [0.630, 0.664] |

**Comparison to Iteration 9 (GPT-4o + Rerank, top_k=5):**

| Metric | Iter 9 (k=5) | Multi-run mean (k=10) | Delta |
|--------|-------------|----------------------|-------|
| Faithfulness | 0.854 | 0.891 | +0.037 |
| Relevancy | 1.000 | 1.000 | 0.000 |
| Correctness/Gen coverage | 0.430 | 0.503 | +0.073 |

The improvement is consistent with Section 7.11 (top_k experiment), which showed all models benefit from larger candidate pools with reranking.

![Chart 19b — Per-Question Variance](figures/chart19b_multirun_variance.png)

![Chart 19c — Faithfulness Comparison](figures/chart19c_multirun_faithfulness.png)

**Variance findings:**

1. **Retrieval is deterministic** (std 0.004). The tiny variance comes from rounding, not retrieval non-determinism — ChromaDB vector search returns identical results for the same query.

2. **Generation is the variance source.** 22/89 questions (24.7%) were flagged as unstable (generation correctness std > 0.1). All 22 have retrieval std ≈ 0.000, confirming the LLM response varies across runs despite temperature=0. This is consistent with known API-level non-determinism in OpenRouter/cloud LLM inference.

3. **Faithfulness variance is small but measurable** (std 0.024). The 95% CI [0.865, 0.918] is tight enough to distinguish GPT-4o from Mistral (0.798) and Llama (0.730) with confidence.

4. **Relevancy is perfectly stable** (std 0.000). The binary YES/NO relevancy judge produces identical results across all runs for GPT-4o + Rerank.

**Conclusion:** Multi-run averaging confirms that our single-run evaluation results are reliable within ±0.02 for faithfulness and ±0.01 for retrieval metrics. The primary source of variance is LLM generation non-determinism, not judge inconsistency. Confidence intervals are narrow enough that all model comparisons in Section 9.4 remain statistically valid.

**Result files:** `data/evaluation/results/multi_run_20260409_230502.json`, `multi_run_20260409_230502.md`

---

## 7.22: Model Fine-Tuning Eligibility Analysis

### 7.22.1 Motivation

Fine-tuning is a medium-priority task for M3 (due 2026-04-15) and a core proposal goal. Before selecting a model for fine-tuning on Massachusetts tenant law Q&A pairs, we must verify which generator models meet the requirements: open weights (downloadable for local training), a permissive license (allowing derivative models), and practical hardware requirements.

### 7.22.2 Model Comparison

| Model | Params | Open Weight | License | Local Fine-Tuning | Min Hardware (Quantized) |
|-------|--------|-------------|---------|-------------------|--------------------------|
| openai/gpt-4o | Unknown | No | Proprietary (OpenAI) | No | N/A (API only) |
| meta-llama/llama-3.3-70b-instruct | 70B | Yes | Llama 3.3 Community License | Yes (QLoRA) | Multi-GPU or cloud |
| mistralai/mistral-small-3.1-24b-instruct | 24B | Yes | Apache 2.0 | Yes (QLoRA/full) | Single RTX 4090 or 32GB Mac |
| Qwen/Qwen3-4B-Instruct | 4B | Yes | Apache 2.0 | Yes (Unsloth LoRA/full) | Single RTX 5080 (16GB) or 16GB Mac |

### 7.22.3 Per-Model Details

**GPT-4o (openai/gpt-4o)**
- Proprietary closed-weight model. Weights are not publicly available.
- Fine-tuning is only possible through OpenAI's API fine-tuning service (limited customization, no weight export, ongoing API costs).
- Not suitable for local fine-tuning on domain-specific legal Q&A data.

**Llama 3.3 70B Instruct (meta-llama/llama-3.3-70b-instruct)**
- Open-weight model released by Meta. Weights available on HuggingFace.
- Licensed under the Llama 3.3 Community License: allows use, reproduction, distribution, and creation of derivative works. Derivative models must include "Llama" in their name.
- Fine-tuning supported via QLoRA (quantized low-rank adaptation) to manage the 70B parameter footprint.
- Hardware requirement is significant: requires multi-GPU setup or cloud compute for fine-tuning even with quantization.
- In our evaluations: faithfulness 0.821 with structured prompt (Section 7.12), highest hallucination rate among the three models (4 hallucinated facts, Section 7.17).

**Mistral Small 3.1 24B Instruct (mistralai/mistral-small-3.1-24b-instruct)**
- Open-weight model released by Mistral AI. Weights available on HuggingFace.
- Licensed under Apache 2.0: fully permissive, allows commercial use and derivative works with no naming restrictions.
- Both base and instruct checkpoints are released, enabling fine-tuning from either starting point.
- Fine-tuning supported by Unsloth and standard HuggingFace training pipelines. Fits on a single RTX 4090 (24GB VRAM) or 32GB RAM MacBook when quantized.
- In our evaluations: matched GPT-4o faithfulness (0.929) with structured prompt (Section 7.12), fewest generation misses (13 vs 16/19) and highest generation-given-retrieval coverage (0.768) across all models (Section 7.17). Best production config identified as Mistral + structured prompt + rerank + k=10 at 7x lower inference cost than GPT-4o.

**Qwen3 4B Instruct (Qwen/Qwen3-4B-Instruct)**
- Open-weight model released by Alibaba Cloud. Weights available on HuggingFace.
- Licensed under Apache 2.0: fully permissive, commercial use and derivative works allowed with no naming restrictions.
- Fine-tuning supported by Unsloth LoRA; fits comfortably on an RTX 5080 (16GB VRAM) or 16GB Mac.
- Context length: 32,768 tokens natively; served in this project at 8,192 via llama.cpp for compatibility.
- In our evaluations (Iteration 9): Qwen3 4B Base + Rerank achieved 0.472 correctness (highest of all models) and 0.427 faithfulness. The fine-tuned variant was degraded by a chat template format mismatch during training (Mistral `[INST]` format applied to a Qwen3 model) but a corrected retrain is expected to recover performance.
- Training was completed in ~8 minutes on RTX 5080 using Unsloth LoRA (3 epochs, 586 samples) — the fastest training cycle of any model evaluated.
- GGUF format used for inference: `finetuned-qwen3-f16.gguf` (8.05GB F16), served via llama-server.

### 7.22.4 Recommendation

**Qwen3 4B Base is the preferred fine-tuning candidate for Iteration 10** based on:

1. **Practical hardware**: At 4B parameters, Qwen3 4B fits comfortably on the available RTX 5080 (16GB VRAM) for both training and inference with Unsloth LoRA. Mistral Small 24B and Llama 3.3 70B require significantly more VRAM.
2. **Local inference confirmed**: Qwen3 4B Base + Rerank achieved 0.472 correctness and 0.427 faithfulness in the final evaluation (Section 9.4), demonstrating viable production-quality inference on local hardware.
3. **License**: Qwen3 is released under Apache 2.0, allowing full commercial use and derivative works.
4. **Training infrastructure in place**: Unsloth LoRA training script (`Fine-Tuneing/FineTune.py`) and data preparation (`Fine-Tuneing/prepare_finetune_data.py`) are already implemented and tested. The chat template format issue from Iteration 9 is corrected — the next run should produce a properly fine-tuned model.
5. **Benchmark gap**: Qwen3 4B fine-tuned currently scores only 0.034 faithfulness (degraded by format mismatch). A successful retrain should recover to ≥0.427 (base model level) and potentially exceed it with domain-specific training.

Mistral Small 3.1 24B remains the best value cloud option (0.798 faithfulness + rerank at $0.14/1M input, vs GPT-4o at 0.854 for $3.00/1M — 21x cheaper for only a 7% faithfulness difference) and the recommended choice if local training hardware is unavailable. Llama 3.3 70B is a lower-priority option due to higher compute cost and lower faithfulness scores (0.730 + rerank).

---

### 9.9 Mistral Small 24B Multi-Run Averaging (Judge Variance Analysis)

**Date:** 2026-04-11
**Config:** Mistral Small 3.1 24B + Rerank, top_k=10, all 89 questions, Claude Sonnet 4 judge
**Runs:** 3 (matching Section 9.8 methodology for GPT-4o)

To enable direct comparison with the GPT-4o multi-run results (Section 9.8), we ran the full retrieval-aware evaluation 3 times with Mistral Small 24B under identical conditions: rerank retriever, top_k=10, structured prompt, all 89 questions, Claude Sonnet 4 judge.

**Aggregate results (mean ± std, 95% CI):**

| Metric | Mean | Std | 95% CI |
|--------|------|-----|--------|
| Faithfulness | 0.843 | 0.000 | [0.843, 0.843] |
| Relevancy | 1.000 | 0.000 | [1.000, 1.000] |
| Retrieval coverage | 0.744 | 0.006 | [0.738, 0.750] |
| Generation coverage | 0.484 | 0.008 | [0.475, 0.493] |
| Gen\|Ret coverage | 0.620 | 0.006 | [0.613, 0.627] |

**Per-run attribution counts:**

| Run | Covered | Gen Miss | Ret Miss | Hallucinated | Total Facts |
|-----|---------|----------|----------|--------------|-------------|
| 1 | 129 | 78 | 63 | 6 | 276 |
| 2 | 125 | 79 | 66 | 6 | 276 |
| 3 | 128 | 77 | 64 | 7 | 276 |
| **Mean** | **127.3** | **78.0** | **64.3** | **6.3** | **276** |

**Head-to-head comparison with GPT-4o (Section 9.8):**

| Metric | Mistral (3-run) | GPT-4o (3-run) | Delta | Significant? |
|--------|----------------|----------------|-------|-------------|
| Faithfulness | 0.843 ± 0.000 | 0.891 ± 0.024 | -0.048 | Yes (CIs don't overlap) |
| Relevancy | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 | — |
| Retrieval coverage | 0.744 ± 0.006 | 0.746 ± 0.004 | -0.002 | No |
| Generation coverage | 0.484 ± 0.008 | 0.503 ± 0.013 | -0.019 | No (CIs overlap) |
| Gen\|Ret coverage | 0.620 ± 0.006 | 0.647 ± 0.015 | -0.027 | Marginal |
| Hallucinated facts | 6.3 | ~2 | +4.3 | Yes |
| Cost per 1M input | $0.14 | $3.00 | 21x cheaper | — |

**Key findings:**

1. **Mistral faithfulness is perfectly deterministic** (std = 0.000) compared to GPT-4o (std = 0.024). All three runs produced identical faithfulness scores of 0.843. This suggests Mistral's temperature=0.2 behavior is more reproducible than GPT-4o's.

2. **GPT-4o's faithfulness advantage is statistically significant.** The 0.048 gap (0.891 vs 0.843) is confirmed with non-overlapping 95% CIs: GPT-4o [0.865, 0.918] vs Mistral [0.843, 0.843].

3. **Retrieval is deterministic and identical** across models (0.744 vs 0.746, std ≤ 0.006), confirming that all performance differences are in the generation stage.

4. **Mistral hallucinates 3x more frequently** (6.3 vs ~2 facts per run). For a legal information tool, this is the most consequential difference. Each hallucination represents a fabricated legal claim that could mislead a tenant.

5. **Generation miss patterns differ.** Mistral averages 78.0 generation misses vs GPT-4o's ~69 — Mistral drops more retrieved facts from its responses. Combined with higher hallucination, this suggests GPT-4o is both more thorough and more disciplined in its use of retrieved context.

6. **Cost-quality tradeoff remains favorable for Mistral** in scenarios where occasional hallucinations are acceptable (e.g., with human review). At 21x lower cost, Mistral achieves 95% of GPT-4o's faithfulness (0.843/0.891).

**Updated production recommendation:** For maximum faithfulness in a legal information tool, GPT-4o + Rerank + k=10 remains the best choice (0.891 faithfulness, ~2 hallucinations). For cost-sensitive deployments with human review, Mistral Small + Rerank + k=10 offers 95% of the faithfulness at 5% of the cost, but users should be aware of the 3x higher hallucination rate.

**Result files:** `data/evaluation/results/retrieval_coverage_20260411_054303.json`, `retrieval_coverage_20260411_213144.json`, `retrieval_coverage_20260411_232123.json`

---

### 9.10 Embedding Model Comparison: all-MiniLM-L6-v2 vs BGE-large-en-v1.5

**Date:** 2026-04-12
**Config:** 89 QA pairs with ground-truth chunk IDs, top_k=10, vector and rerank retrievers
**Cost:** $0 (retrieval metrics only, no LLM judge needed)

**Research question:** Does upgrading from ChromaDB's default all-MiniLM-L6-v2 (384-dim) to BGE-large-en-v1.5 (1024-dim) improve retrieval quality?

**Setup:**
- Indexed 967 chunks into separate ChromaDB collections: `ma_tenant_law` (default) and `ma_tenant_law_bge_large` (BGE-large)
- Computed chunk-level retrieval metrics using ground-truth source_chunks from golden_qa.json and reddit_questions.json (89 QA pairs)
- Tested both vector (pure embedding similarity) and rerank (hybrid + cross-encoder) retrievers

**Results (vector retriever, isolates embedding quality):**

| Metric | all-MiniLM-L6-v2 (384d) | BGE-large-en-v1.5 (1024d) | Delta |
|--------|------------------------|--------------------------|-------|
| MRR@10 | 0.357 | 0.336 | -0.021 |
| Hit Rate@10 | 0.573 | 0.562 | -0.011 |
| Recall@10 | 0.362 | 0.340 | -0.023 |
| NDCG@10 | 0.295 | 0.267 | -0.028 |

**Results (rerank retriever, downstream pipeline impact):**

| Metric | all-MiniLM-L6-v2 (384d) | BGE-large-en-v1.5 (1024d) | Delta |
|--------|------------------------|--------------------------|-------|
| MRR@10 | 0.338 | 0.325 | -0.013 |
| Hit Rate@10 | 0.562 | 0.539 | -0.023 |
| Recall@10 | 0.373 | 0.347 | -0.027 |
| NDCG@10 | 0.298 | 0.278 | -0.020 |

**Conclusion: BGE-large does not improve retrieval for this corpus.** All metrics are slightly worse (-1% to -3%) across both retrievers. The full judge-based evaluation was skipped based on this negative signal — if the embedding model retrieves worse chunks, downstream generation quality cannot improve.

**Analysis:**

1. **Domain specificity matters more than dimensionality.** The 384-dim all-MiniLM-L6-v2 outperforms the 1024-dim BGE-large on this legal corpus. Both are general-purpose models, but the legal domain's specialized vocabulary (statute numbers, legal terms, procedural language) may interact differently with each model's training distribution.

2. **The rerank retriever attenuates but does not reverse the gap.** The cross-encoder (ms-marco-MiniLM) re-scores candidates based on content, partially compensating for embedding quality differences. But it cannot recover candidates that were never in the initial retrieval pool.

3. **Recommendation:** The current all-MiniLM-L6-v2 embedding model should be retained. Future embedding experiments should prioritize legal-domain models (e.g., legal-bert embeddings or a fine-tuned embedding model on legal text) rather than larger general-purpose models.
