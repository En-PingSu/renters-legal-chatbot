# Project TODO -- Renters Legal Assistance Chatbot (CS6180)

> **Last updated:** 2026-04-12 (session 18)
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
- [x] **Structured prompt experiment** -- Evidence-before-answer prompt improves faithfulness across all models. GPT-4o: 0.857→0.929. Llama: 0.500→0.821 (+64%). Mistral: 0.643→0.929 (+44%, matches GPT-4o at 7x lower cost). Best production config: Mistral+structured+rerank+k=10. Results in section 7.12.
- [x] **Judge methodology validation** -- Compared our single-call judge vs LlamaIndex-style iterative refine (same prompts, same judge model). Identical aggregate scores (0.964/1.000). 93% per-question agreement on faithfulness. Our method is 5x cheaper. Results in section 7.13.
- [x] **Generator-judge swap experiment** -- Claude S4 as generator judged by GPT-4o/Llama/Mistral. Faithfulness stable (0.893-0.929) across all judges. Correctness varies by judge (GPT-4o=0.750, Mistral=0.464, Llama=0.357, Claude=0.321) — reflects judge leniency not generation quality. Full cross-model table in section 7.14.
- [x] **Analyze multi-model results** -- GPT-4o vs Llama vs Mistral compared on all metrics with RAG uplift per model (section 7.9).
- [x] **Multi-retriever comparison** -- Vector, BM25, hybrid, rerank, parent_child compared on retrieval metrics (section 7.7) and generation quality (section 7.8).
- [x] **Retrieval-aware correctness** -- Decomposed correctness into retrieval coverage (0.757) vs generation coverage (0.554). Of 33 missed facts: 51.5% retrieval failures, 48.5% generation failures. 1 hallucination out of 74 facts. Results in section 7.15.
- [x] **Prompt completeness experiment** -- Tested completeness-focused prompt to reduce generation misses. Result: no net improvement (4 improved, 4 regressed, 20 unchanged). Simple prompt engineering insufficient — generation miss reduction requires structural changes (multi-pass, fine-tuning, or post-gen verification). Prompt reverted. Results in section 7.16.
- [x] **Multi-model retrieval-aware correctness** -- Compared GPT-4o, Mistral, Llama on fact-level attribution. Mistral best: fewest gen misses (13 vs 16/19), highest gen|ret coverage (0.768). Llama worst: most gen misses (19) and hallucinations (4). Reinforces Mistral as best production model. Results in section 7.17.
- [x] **Retrieval bottleneck analysis & fixes** -- Identified 6 root causes of retrieval failures. Applied Fix 1 (rerank candidate pool 2x, tested 3x but cross-encoder domain mismatch pushed out legal chunks) and Fix 2 (golden_qa data quality audit: 8 entries fixed, wrong chunk_011 references replaced, bare statute key_facts replaced with verifiable facts). Net: ret coverage 0.743→0.720, gen coverage 0.541→0.613 (+13%), covered facts 36→42, gen misses 19→12. Cross-encoder domain mismatch (ms-marco on legal content) identified as primary remaining bottleneck. Results in section 7.18.
- [x] **Systematic QA data audit** -- Full audit of all 80 QA entries (golden_qa + reddit_questions) against 967 chunks. Created `src/evaluation/audit_qa_data.py` automated audit script. Phase 1: automated checks found 81 issues (42 low-relevance, 34 low-grounding, 4 duplicate chunks, 1 duplicate question). Phase 2: manual semantic review of all 80 entries identified wrong chunk references (TOC/filler chunks, chunk_011 catch-all), unsupported key_facts, wrong topics, and a duplicate entry. Phase 3: applied 61 fixes — replaced wrong chunks with correct ones, rewrote key_facts to match actual chunk content, fixed topic labels, removed golden_048 duplicate, deduped chunk refs. Phase 4: re-audit shows 12 remaining (all borderline token-overlap false positives). URL spot-check: 10/10 pages live. Final: 79 entries (49 golden + 30 reddit), 0 structural issues.
- [x] **New retriever strategies evaluation** -- Implemented and evaluated 4 new retrievers: multi-query expansion (`src/rag/multiquery.py`), sentence window chunking (`src/processing/sentence_window_chunker.py`), hybrid+parent-child with rerank, and auto-merge (`src/rag/hybrid_parent_child.py`). Eval script: `src/evaluation/eval_new_retrievers.py` with `--retrieval-only` dry run mode. Results (27 stratified questions, per-topic breakdown): auto_merge best retrieval coverage (0.727 vs 0.675 baseline), big gains on lead_paint (+0.833) and retaliation (+0.200), but gen|ret dropped (0.589) due to context flooding. hybrid_parent_child_rerank best for rent_increases (+0.333). multiquery underperformed (1 improved, 5 regressed). sentence_window weakest (1 improved, 10 regressed). Results in eval report section 7.20.

---

- [x] **QA source_chunks audit (session 14)** -- Verified chunk-question alignment for 6 questions. Fixed source_chunks for golden_006 (security deposits: replaced masslegalhelp chunk_004→chunk_015 for deduction rules), golden_021 (asbestos: replaced sanitary_code chunk_005→chunk_006 for cleaner ACM definition), reddit_q008 (rent withholding: replaced AG chunk_003 + tenant_rights chunk_000 with masslegalhelp ch08_015 + sanitary_code chunk_022), reddit_q014 (eviction: replaced 2 weak chunks with 4 covering all key facts). Fixed evaluation report discrepancy: "24 questions" → "27 questions" in Section 7.20. Flagged eviction chunk_005 stale content (page updated after scrape).
- [x] **Failure taxonomy (session 15)** -- Categorized 46 failures across 27 questions into 8 failure modes: F1 Statute Citation Dropout (9 facts), F2 Remedy/Consequence Omission (6), F3 Cross-Document Synthesis Gap (5), F4 Program/Term Mismatch (5), F5 Statute Retrieval Miss (4), F6 Boundary Condition Omission (3), F7 Cross-Encoder Domain Mismatch (qualitative), F8 Hallucinated Citation (2). Generation failures (43%) outnumber retrieval failures (30%). Added as Section 7.21 in evaluation report.
- [x] **Hard multi-step questions (session 15)** -- Created 10 hard questions (golden_051-060) requiring cross-statute reasoning across 2-4 chunks from different legal domains. Topics: security deposit + treble damages + court, lead paint + retaliation + eviction defense, no heat + rent withholding + eviction, utility shutoff + quiet enjoyment, illegal lockout + damages, security deposit as eviction defense, foreclosure + tenant protections, discrimination + fair housing complaint, repair + tenant petition + 93A, condo conversion + rent limits. All chunk references verified against corpus. Evidence matrix with verbatim proof added to Section 7.21.6.
- [x] **M2 progress report** -- Submitted 2026-04-01.
- [x] **Fine-tuning (Qwen3 4B)** -- LoRA fine-tuned Qwen3 4B Instruct using Unsloth (3 epochs, 586 samples from golden QA 5x + Reddit + chunk-derived QA). Scripts: `Fine-Tuneing/FineTune.py` and `Fine-Tuneing/prepare_finetune_data.py`. Served locally via llama-server (GGUF F16). Evaluated in Iteration 9 (89 questions, 13 configs). Qwen3 Base+Rerank achieved highest correctness (0.472) at zero API cost. Fine-tuned model degraded (faith 0.034, rel 0.697) due to chat template format mismatch in early training runs (Mistral `[INST]` format instead of Qwen3 ChatML). Corrected data prep script is in place.
- [x] **Mistral multi-run averaging** -- 3 runs on all 89 questions (Mistral Small 24B + rerank, top_k=10, Claude Sonnet 4 judge). Faithfulness 0.843 ± 0.000 (perfectly deterministic), relevancy 1.000. GPT-4o leads by 0.048 on faithfulness (0.891 vs 0.843, statistically significant). Mistral hallucinates 3x more (6.3 vs ~2 facts). 21x cheaper. Results in eval report Section 9.9.
- [x] **Iteration 9: Full 89-question multi-model eval** -- Expanded from 28 to 89 questions. Added Qwen3 4B Base and Qwen3 4B Fine-tuned (local, llama-server). 13 configs total. Results in local report (`with local report/evaluation_report.md`) Sections 9.1–9.7. Key findings: Qwen3 Base+Rerank best correctness (0.472), fine-tuning degraded by format damage, local models viable for privacy-sensitive deployments.

## In Progress

(none)

---

## TODO -- High Priority (before M3: 2026-04-15)

- [x] **Evaluation charts** -- 11 charts generated covering all iterations: iteration progression, corpus growth, retriever comparison, multi-model, RAG uplift, self-eval bias, top_k experiment, structured prompt, best configs, corpus composition. Report converted to Markdown at `docs/evaluation_report.md`
- [x] **M2 progress report** -- Submitted 2026-04-01.
- [x] **Sync evaluation report** -- Replaced `docs/evaluation_report.md` with `with local report/evaluation_report.md` (Iteration 9 final, 2026-04-04). Adds Qwen3 configs, 89-question results, fine-tuning analysis, updated 7.22 recommendation.
- [x] **Frontend / demo UI** -- Next.js 16 + FastAPI chat interface. SSE streaming, model/retriever/top_k config sidebar, collapsible source cards, markdown rendering. FastAPI at `api/server.py` wraps `pipeline.py` (untouched). Run: `PYTHONPATH=. uvicorn api.server:app --port 8000` + `cd frontend && npm run dev`.
- [ ] **Qwen3 fine-tuning retrain** -- Maxwell's task. Retrain with corrected ChatML format data (already in `Fine-Tuneing/prepare_finetune_data.py`). Expected to recover faithfulness to ≥0.400. See Section 9.7.

## TODO -- Medium Priority (before M3: 2026-04-15)

- [x] **Evaluate hard questions (session 15)** -- Ran retrieval_coverage on fixed set (27 standard + 10 hard = 37 questions). Hard questions did NOT degrade as expected: ret coverage 0.700 vs 0.675 standard, gen|ret 0.714 vs 0.654. Key finding: bottleneck for hard questions is fact volume (5-6 facts vs 2-3), not cross-document synthesis. LLM drops secondary details (F1/F2) more when there are more facts. Created fixed stratified set at `data/evaluation/stratified_questions.json` for reproducible runs. Results in Section 7.21.8.
- [x] **Add NDCG@K and Recall@K metrics (session 15)** -- Fixed scorer.py and eval_new_retrievers.py to use all source_chunks as ground truth (was only using source_chunks[0], causing 8 false misses). Added Recall@K (fraction of relevant chunks found) and NDCG@K (ranking quality with log-discount). Revealed true Recall@10 = 0.312 (retriever finds ~1/3 of relevant chunks). Backward compatible: legacy single-chunk format still supported.
- [x] **Fine-tuning** -- Done (Qwen3 4B, Iteration 9). See completed items above.
- [x] **Multi-run averaging** -- 3 parallel runs on all 89 questions (GPT-4o + rerank, top_k=10, Claude Sonnet 4 judge). Faithfulness 0.891 +/- 0.024, relevancy 1.000 +/- 0.000, retrieval coverage 0.746 +/- 0.004. 22/89 questions unstable on generation correctness (LLM non-determinism, not judge). Retrieval deterministic. All model comparisons from Iter 9 statistically valid. Results in eval report Section 9.8.
- [x] **Embedding model upgrade** -- Indexed 967 chunks with BGE-large-en-v1.5 (1024-dim) and compared chunk-level retrieval metrics against default all-MiniLM-L6-v2 (384-dim) on 89 QA pairs. Result: BGE-large performed slightly worse on all metrics (MRR -0.021, Recall@K -0.023, NDCG@K -0.028). Full judge eval skipped (negative signal). Domain-specific models recommended over larger general-purpose ones. Results in eval report Section 9.10.
- [x] **Question-category analysis** -- Per-topic breakdown across 12 categories (security_deposits, evictions, repairs_habitability, etc.) in new retriever eval (Section 7.20.4). Identified weak areas: lead_paint (0.167 baseline), rent_increases (0.000 baseline), discrimination (0.667 baseline). Best retriever varies by topic.
- [x] **Prompt engineering** -- Tested completeness-focused prompt (Section 7.16); no net improvement. Simple prompt changes insufficient for generation miss reduction. Needs structural approach (multi-pass generation or fine-tuning).

## TODO -- Low Priority / Stretch Goals

- [x] **Sentence window chunking** -- Implemented in `src/processing/sentence_window_chunker.py`. 10,785 sentence chunks (avg 56 tokens), separate ChromaDB collection. Evaluated: weakest approach (ret coverage 0.519 vs 0.675 baseline, 10 questions regressed). Sentence-level embeddings too short for legal question matching. Deprioritized (Section 7.20).
- [x] **Multi-query expansion** -- Implemented in `src/rag/multiquery.py`. LLM generates 3 query variants → RRF merge. Evaluated: underperformed (ret coverage 0.623 vs 0.675 baseline, 5 questions regressed, 1 improved). Variants may not be sufficiently diverse, or rerank already captures relevant chunks. Needs rethinking: more variants, template-based approach, or cheaper model (Section 7.20).
- [x] **Hybrid + parent-child retriever** -- Implemented in `src/rag/hybrid_parent_child.py` (3 variants: hybrid_parent_child, hybrid_parent_child_rerank, auto_merge). Evaluated: auto_merge is most promising (ret coverage 0.727, +0.052 vs baseline), big gains on lead_paint (+0.833) and retaliation (+0.200), but gen|ret dropped to 0.589 due to context flooding (23 gen misses vs 17). hybrid_parent_child_rerank best for rent_increases (+0.333). Needs gen miss fix: truncate merged docs or raise merge threshold (Section 7.20).
- [ ] **Semantic deduplication** -- Remove near-duplicate chunks from top-k results to increase context diversity
- [ ] **Context window optimization** -- Experiment with larger top_k (10-15) + reranking to top 5-7
- [ ] **Statute-aware chunking** -- Preserve complete legal subsections as atomic chunks
- [ ] **Legal-domain cross-encoder** -- Replace ms-marco-MiniLM with a legal-domain cross-encoder to fix displacement of statute chunks by conversational content (Section 7.19.10, Pattern 4)

---

## Key Deadlines

| Date | Milestone |
|------|-----------|
| 2026-04-01 | M2 Progress Report |
| 2026-04-15 | M3 Final Submission |
