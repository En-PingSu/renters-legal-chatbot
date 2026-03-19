# Project TODO -- Renters Legal Assistance Chatbot (CS6180)

> **Last updated:** 2026-03-18 (session 6)
> Claude reads this file at the start of each session and updates it after completing tasks.

---

## Completed

- [x] **Web scraping** -- mass.gov, boston.gov, bostonhousing.org (112 documents)
- [x] **Reddit data collection** -- r/legaladvice, r/renting, r/bostonhousing
- [x] **Document processing & chunking** -- Recursive character splitting, Markdown-aware, FAQ/statute-specific handlers (800 tokens, 200 overlap)
- [x] **ChromaDB vector store** -- 5 collections, all-MiniLM-L6-v2 embeddings (384-dim)
- [x] **RAG pipeline** -- Query -> ChromaDB retrieval (top_k=5) -> prompt assembly -> LLM -> citation check -> response
- [x] **Multiple retrievers** -- Vector, BM25, hybrid (0.6/0.4), cross-encoder rerank
- [x] **Evaluation framework** -- Faithfulness + relevancy + correctness (binary LLM judge), retrieval metrics (MRR, hit rate, precision, recall)
- [x] **Golden QA pairs** -- 20 curated Q&A pairs with key facts and source chunks
- [x] **Reddit question enrichment** -- 30 Reddit-style questions with expected answers
- [x] **Iteration 1: Baseline vs RAG** -- Faithfulness 0.640 -> 0.780, relevancy 1.000
- [x] **Iteration 2: Corpus cleanup** -- Removed 42% junk chunks (1,068 -> 617), faithfulness 0.780 -> 0.900, MRR 0.480 -> 0.568
- [x] **Iteration 3: Separate judge model** -- Switched LLM judge from GPT-4o to Claude Sonnet 4 (eliminates self-evaluation bias)
- [x] **Iteration 3: Multi-model configs** -- Added Llama 3.3-70B and Mistral Small 3.1 generation configs (9 total configurations)
- [x] **Corpus expansion** -- Added masslegalhelp.org Legal Tactics (18 chapters), extended boston.gov (new pages + PDFs), extended BHA FAQ. Enhanced corpus cleaner (non-English, templates, nav-links, PDF fragments, off-topic reports). Final corpus: 146 docs / 871 chunks (was 112 docs / 617 chunks). New coverage: utilities, repairs, court procedures, rooming houses, discrimination, foreclosure.
- [x] **Multi-retriever evaluation (GPT-4o)** -- Baseline vs vector vs BM25 vs hybrid vs rerank. Rerank best: faithfulness 0.762, MRR 0.243, hit rate 0.328. Baseline faithfulness dropped to 0.100 (stricter judge). All RAG configs 100% relevancy.
- [x] **Corpus cleanup round 2** -- Removed nav/sitemap pages, off-topic docs (homelessness, zoning, emergency shelter), and header-only TOC chunks. Added 6 doc_ids to blacklist, improved nav filter with Pattern B (with masslegalhelp exemption), added header-only filter. 871 -> 837 chunks.
- [x] **Corpus expansion round 2** -- Added 41 new documents (249 total). Priority 1: 26 MGL statutes (c.186 s.15A/15D/15E/19-23, c.239 s.9/10/12, c.93A s.2/9/11, c.111 s.127A-127L). Priority 2: Housing Court info, eviction forms, tenants' eviction guide, respond-to-eviction guide, small claims filing (via Chrome browser fallback for 403s). Priority 3: 5 GBLS housing pages. Priority 4: MCAD complaints + housing discrimination overview. Priority 5: MRVP voucher program. Priority 6: Boston ISD housing inspections + constituent services. Final corpus: 249 docs / 967 chunks (was 146 docs / 837 chunks).
- [x] **LlamaIndex migration** -- New `src/rag_llamaindex/` module with 6 files (llm, nodes, index, retrievers, prompts, pipeline). 5 retriever strategies: vector, bm25, hybrid, rerank, parent_child (new small-to-big). Separate ChromaDB at `data/chroma_db_llamaindex/`. Scorer integration via `RAG_BACKEND=llamaindex` env var. All 967 chunks indexed, all retrievers verified.
- [x] **LlamaIndex single-config eval** -- rag_rerank GPT-4o: faithfulness 0.713, relevancy 1.000, correctness 0.412, MRR 0.247. Investigated custom vs LlamaIndex divergence: root cause is metadata embedding (LlamaIndex prepends metadata to text before embedding, mean cosine sim 0.851 vs raw text). Rerank overlap 64%, vector overlap 48%. Findings written to evaluation_report.txt Iteration 6.
- [x] **Fix metadata embedding divergence** -- Excluded all metadata keys from LlamaIndex embedding input (`excluded_embed_metadata_keys`). Re-indexed 967 nodes. Cosine similarity now 1.000 (was 0.851), vector overlap 100% (was 48%), rerank overlap 94% (was 64%). Pipelines at parity.
- [x] **Pipeline decision** -- Custom pipeline chosen for frontend/interface (simpler deployment, identical retrieval quality, better for fine-tuning). LlamaIndex module kept for reference/comparison.
- [x] **Custom parent-child retriever** -- Ported neighbor-expansion logic from LlamaIndex to `src/rag/retrievers.py`. 100% Jaccard overlap with LlamaIndex impl. Hit rate 0.300 (best), MRR 0.155. Added to RETRIEVER_REGISTRY.
- [x] **Multi-model + retriever eval (28q stratified)** -- 7 configs on 28 stratified questions. Best: GPT-4o+rerank (faith 0.857). RAG uplift: GPT-4o +1107%, Mistral +806%, Llama +604%. Mistral 24B outperforms Llama 70B on faithfulness (0.643 vs 0.500). Total cost $4.18. Results in eval report Iter 7 sections 7.8-7.9.
- [x] **Self-evaluation bias experiment** -- Confirmed self-judge inflates faithfulness for all models: Llama +0.429, Mistral +0.214, GPT-4o +0.107. GPT-4o self-inflates correctness by +0.393. Validates use of independent Claude Sonnet 4 judge. Results in eval report section 7.10.
- [x] **top_k=10 experiment** -- All models improve with rerank top_k=10 vs k=5. GPT-4o correctness +30% (0.357→0.464). Llama faith +21% (0.500→0.607). Mistral faith +17% (0.643→0.750). Best config: GPT-4o+rerank+k=10. Results in section 7.11.
- [x] **Structured prompt experiment** -- Evidence-before-answer prompt improves faithfulness: k=5 0.857→0.893, k=10 0.857→0.929 (project best). Slight correctness tradeoff (0.357→0.321). Combined structured+k=10 is best faithfulness config. Results in section 7.12.
- [x] **Analyze multi-model results** -- GPT-4o vs Llama vs Mistral compared on all metrics with RAG uplift per model (section 7.9).
- [x] **Multi-retriever comparison** -- Vector, BM25, hybrid, rerank, parent_child compared on retrieval metrics (section 7.7) and generation quality (section 7.8).

---

## In Progress

(none)

---

## TODO -- High Priority (before M2: 2026-04-01)

- [ ] **Evaluation charts** -- Generate charts for multi-model/retriever results (faithfulness by config, model comparison, retriever comparison, self-judge bias)
- [ ] **M2 progress report** -- Write milestone 2 report (due 2026-04-01) summarizing methodology, results, and next steps
- [ ] **Multi-run averaging** -- Run evaluation 3x and average scores to reduce judge variance (Section 3.4 Item 3 of eval report)

## TODO -- Medium Priority (before M3: 2026-04-15)

- [ ] **Fine-tuning** -- Fine-tune a model on MA tenant law Q&A pairs (core proposal goal); evaluate as additional configuration
- [ ] **Embedding model upgrade** -- Test text-embedding-3-large or BGE-large vs current all-MiniLM-L6-v2; compare retrieval metrics
- [ ] **Question-category analysis** -- Break down evaluation by question type (security deposit, eviction, BHA-specific, etc.) to identify weak areas
- [ ] **Prompt engineering** -- Refine RAG system prompt to balance groundedness with completeness (Section 3.7 Item 4)
- [ ] **Frontend / demo UI** -- Build a simple chat interface for the final presentation

## TODO -- Low Priority / Stretch Goals

- [ ] **Sentence window chunking** -- Implement sentence-level chunking with context expansion at retrieval time
- [ ] **Multi-query expansion** -- Generate query variants to widen retrieval candidate pool
- [ ] **Semantic deduplication** -- Remove near-duplicate chunks from top-k results to increase context diversity
- [ ] **Context window optimization** -- Experiment with larger top_k (10-15) + reranking to top 5-7
- [ ] **Statute-aware chunking** -- Preserve complete legal subsections as atomic chunks

---

## Key Deadlines

| Date | Milestone |
|------|-----------|
| 2026-04-01 | M2 Progress Report |
| 2026-04-15 | M3 Final Submission |
