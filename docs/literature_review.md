# Literature Review: Retrieval-Augmented Generation (RAG) for Legal Applications

## Context
This literature review supports the CS6180 Final Project -- a RAG-based chatbot for Massachusetts tenant law. The review focuses on recent (2024-2025) scholarly work on RAG systems, with emphasis on legal domain applications, evaluation methodology, and techniques directly applicable to this project (chunking, retrieval strategies, hallucination mitigation, domain-specific embeddings).

---

## Papers Ranked by Relevance to Project

### 1. **Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools**
- **Authors:** Varun Magesh, Faiz Surani, Matthew Dahl, Mirac Suzgun, Christopher D. Manning, Daniel E. Ho (Stanford / Yale)
- **Venue:** *Journal of Empirical Legal Studies*, Vol. 22, pp. 216-242, 2025
- **Link:** [arXiv:2405.20362](https://arxiv.org/abs/2405.20362) | [Wiley](https://onlinelibrary.wiley.com/doi/full/10.1111/jels.12413)
- **Summary:** First preregistered empirical evaluation of RAG-based commercial legal AI tools (Lexis+ AI, Westlaw AI-Assisted Research). Found hallucination rates of 17-33% in production legal RAG systems, compared to higher rates in general-purpose LLMs. Proposes a typology of legal hallucinations (fabricated citations, incorrect holdings, misattributed quotes).
- **Relevance:** **Critical.** Directly motivates our project's emphasis on faithfulness evaluation and citation grounding. The 17-33% hallucination rate even in commercial tools validates our iterative approach to improving faithfulness (0.640 -> 0.900). Their hallucination typology can inform our LLM judge evaluation criteria.

### 2. **HyPA-RAG: A Hybrid Parameter Adaptive Retrieval-Augmented Generation System for AI Legal and Policy Applications**
- **Authors:** Rishi Kalra, Zekun Wu, Ayesha Gulley, Airlie Hilliard, Xin Guan, Adriano Koshiyama, Philip Treleaven
- **Venue:** EMNLP 2024 CustomNLP4U Workshop / NAACL 2025
- **Link:** [arXiv:2409.09046](https://arxiv.org/abs/2409.09046) | [ACL Anthology](https://aclanthology.org/2024.customnlp4u-1.18/)
- **Summary:** Proposes a hybrid retrieval system combining dense, sparse, and knowledge graph methods with a query complexity classifier for adaptive parameter tuning. Tested on NYC Local Law 144 (AI hiring regulations). Demonstrates that different query types benefit from different retrieval strategies.
- **Relevance:** **High.** Our project already implements hybrid retrieval (vector + BM25, 0.6/0.4 weighting) and reranking. HyPA-RAG's adaptive approach -- classifying queries by complexity before selecting retrieval strategy -- is a natural extension. Their focus on a specific local law parallels our focus on MA tenant statutes.

### 3. **Towards Reliable Retrieval in RAG Systems for Large Legal Datasets**
- **Authors:** Markus Reuter, Tobias Lingenberg, Rūta Liepiņa, Francesca Lagioia, Marco Lippi, Giovanni Sartor, Andrea Passerini, Burcu Sayin
- **Venue:** *Natural Legal Language Processing Workshop 2025* (co-located with EMNLP 2025)
- **Link:** [arXiv:2510.06999](https://arxiv.org/abs/2510.06999) | [ACL Anthology](https://aclanthology.org/2025.nllp-1.3/)
- **Summary:** Identifies "Document-Level Retrieval Mismatch" (DRM) -- a failure mode where the retriever pulls chunks from entirely wrong source documents. Proposes Summary-Augmented Chunking (SAC), which prepends a document-level summary to each chunk to inject global context lost during chunking. SAC significantly reduces DRM and improves text-level precision/recall. Notably, generic summaries outperformed domain-expert summaries.
- **Relevance:** **High.** Directly applicable to our chunking strategy (800 tokens, 200 overlap). Our corpus has structurally similar legal documents (multiple MGL chapters, similar statute formats) that are prone to DRM. SAC could be implemented as a metadata enhancement to our existing chunks. This is a concrete, low-cost improvement we could adopt.

### 4. **LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation in the Legal Domain**
- **Authors:** Nicholas Pipitone, Ghita Houir Alami
- **Venue:** arXiv preprint, August 2024
- **Link:** [arXiv:2408.10343](https://arxiv.org/abs/2408.10343) | [GitHub](https://github.com/zeroentropy-ai/legalbenchrag)
- **Summary:** First benchmark specifically designed for evaluating retrieval in legal RAG. Contains 6,858 query-answer pairs over 79M+ character corpus, all human-annotated by legal experts. Emphasizes precise retrieval of minimal relevant text segments rather than whole documents. Includes lightweight LegalBench-RAG-mini for rapid iteration.
- **Relevance:** **High.** Our evaluation framework uses golden QA pairs (20 curated + 30 Reddit-style) with retrieval metrics (MRR, hit rate, precision, recall). LegalBench-RAG's methodology of tracing answers back to source locations validates our approach. Their emphasis on minimal relevant text retrieval aligns with our reranking strategy.

### 5. **CBR-RAG: Case-Based Reasoning for Retrieval Augmented Generation in LLMs for Legal Question Answering**
- **Authors:** Nirmalie Wiratunga, Ramitha Abeyratne, Lasal Jayawardena, Kyle Martin, Stewart Massie, Ikechukwu Nkisi-Orji, Ruvan Weerasinghe, Anne Liret, Bruno Fleisch
- **Venue:** ICCBR 2024 (International Conference on Case-Based Reasoning)
- **Link:** [arXiv:2404.04302](https://arxiv.org/abs/2404.04302) | [Springer](https://link.springer.com/chapter/10.1007/978-3-031-63646-2_29)
- **Summary:** Integrates Case-Based Reasoning (CBR) with RAG for legal QA. Uses CBR's retrieval stage, indexing vocabulary, and similarity knowledge to enrich LLM prompts with contextually relevant cases. Evaluates general vs. domain-specific embeddings (including LegalBERT) and different similarity measures (inter, intra, hybrid). Domain-specific embeddings with hybrid similarity produced best results.
- **Relevance:** **High.** Their finding that domain-specific embeddings outperform general ones directly relates to our planned embedding model upgrade (text-embedding-3-large or BGE-large vs. all-MiniLM-L6-v2). Their hybrid similarity approach parallels our hybrid retrieval strategy.

### 6. **Optimizing Legal Text Summarization Through Dynamic Retrieval-Augmented Generation and Domain-Specific Adaptation**
- **Authors:** S Ajay Mukund, K. S. Easwarakumar
- **Venue:** *Symmetry*, Vol. 17, No. 5, 2025
- **Link:** [MDPI](https://www.mdpi.com/2073-8994/17/5/633)
- **Summary:** Proposes a dynamic RAG framework for legal text summarization. Uses BM25 with top-3 chunk selection. Introduces "dark zone" detection -- identifying unexplained statute-provision pairs or judicial precedents -- and dynamically retrieves authoritative references to fill gaps. Anchors retrieval on structured legal entities rather than lexical similarity.
- **Relevance:** **Medium-High.** The "dark zone" concept (detecting gaps in retrieved context) could improve our pipeline's handling of questions that span multiple statutes. Their entity-anchored retrieval is relevant to our statute-aware chunking stretch goal.

### 7. **RAGAS: Automated Evaluation of Retrieval Augmented Generation**
- **Authors:** Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert
- **Venue:** EACL 2024 (System Demonstrations)
- **Link:** [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) | [ACL Anthology](https://aclanthology.org/2024.eacl-demo.16/)
- **Summary:** Introduces the RAGAS framework for reference-free evaluation of RAG pipelines. Defines metrics: faithfulness (factual consistency with retrieved context), answer relevancy (pertinence to query), and context metrics (recall, precision). Uses LLM-as-judge for automated evaluation without human annotations.
- **Relevance:** **Medium-High.** Our evaluation framework mirrors RAGAS methodology -- binary faithfulness/relevancy/correctness via LLM judge (Claude Sonnet 4). RAGAS provides theoretical grounding for our approach and validates the LLM-as-judge paradigm we adopted from HW4.

### 8. **A Comprehensive Survey of Retrieval-Augmented Generation (RAG): Evolution, Current Landscape and Future Directions**
- **Authors:** Shailja Gupta, Rajesh Ranjan, Surya Narayan Singh
- **Venue:** arXiv preprint, October 2024
- **Link:** [arXiv:2410.12837](https://arxiv.org/abs/2410.12837)
- **Summary:** Comprehensive survey tracing RAG from foundational concepts to state of the art. Covers Naive RAG, Advanced RAG (pre-retrieval query optimization, post-retrieval reranking), and Modular RAG architectures. Discusses retrieval granularity (document, chunk, sentence, token, entity levels), indexing strategies, and evaluation frameworks. Identifies open challenges: scalability, bias, robustness.
- **Relevance:** **Medium.** Provides taxonomic context for our project's position within the RAG landscape. Our system implements Advanced RAG features (hybrid retrieval, cross-encoder reranking, parent-child expansion). The survey's discussion of chunk granularity trade-offs informs our chunking decisions.
- **Note:** Not cited in the M3 report. The M3 report cites Gao et al. (2024) (#11 below) instead, as the original source of the Naive/Advanced/Modular RAG taxonomy.

### 9. **Legal-DC: Benchmarking Retrieval-Augmented Generation for Legal Documents**
- **Authors:** Yaocong Li, Qiang Lan, Leihan Zhang, Le Zhang
- **Venue:** arXiv preprint, March 2025
- **Link:** [arXiv:2603.11772](https://arxiv.org/abs/2603.11772) | [GitHub](https://github.com/legal-dc/Legal-DC)
- **Summary:** Benchmark for Chinese legal RAG with 480 legal documents and 2,475 Q&A pairs annotated with clause-level references. Introduces LegRAG framework with legal adaptive indexing (clause-boundary segmentation) and dual-path self-reflection mechanism. LegRAG outperforms SOTA by 1.3-5.6% across key metrics.
- **Relevance:** **Medium.** While focused on Chinese law, their clause-boundary segmentation approach directly parallels our statute-aware chunking goal. The dual-path self-reflection mechanism (verifying answer against source clauses) could enhance our citation verification step.

### 10. **Enhancing Legal Document Building with Retrieval-Augmented Generation (JusBuild)**
- **Authors:** Matteo Buffa, Alfio Ferrara, Sergio Picascia, Davide Riva, Silvana Castano
- **Venue:** *Computer Law \& Security Review*, Vol. 59, 2025
- **Link:** [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2212473X25001014)
- **Summary:** Presents JusBuild, a document builder for legal practitioners that uses CRF-based segmentation, vector databases for semantic search, and RAG for suggestion retrieval. Designed to assist in drafting new legal documents by retrieving relevant passages from existing ones.
- **Relevance:** **Medium.** While our project focuses on question answering rather than document generation, their CRF-based document segmentation approach is relevant to processing structured legal texts like MA statutes.

### 11. **Retrieval-Augmented Generation for Large Language Models: A Survey**
- **Authors:** Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang
- **Venue:** arXiv preprint (originally Dec 2023, updated March 2024)
- **Link:** [arXiv:2312.10997](https://arxiv.org/abs/2312.10997)
- **Summary:** Earlier foundational RAG survey covering the Naive RAG -> Advanced RAG -> Modular RAG progression. Discusses key challenges: retrieval quality, context window limitations, generation faithfulness. Covers indexing optimization (chunk size, metadata enrichment), query transformation, and iterative retrieval.
- **Relevance:** **Medium.** Provides foundational taxonomy that contextualizes our project's architecture. Their discussion of chunk size optimization (our 800/200 split) and metadata enrichment strategies is directly applicable.

### 12. **Enhancing the Precision and Interpretability of Retrieval-Augmented Generation (RAG) in Legal Technology: A Survey**
- **Authors:** Mahd Hindi, Linda Mohammed, Ommama Maaz, Abdulmalik Alwaraly
- **Venue:** *IEEE Access*, January 2025. DOI: 10.1109/ACCESS.2025.3550145
- **Link:** [ResearchGate](https://www.researchgate.net/publication/389773115_Enhancing_the_Precision_and_Interpretability_of_Retrieval-Augmented_Generation_RAG_in_Legal_Technology_A_Survey)
- **Summary:** Survey specifically focused on RAG in legal technology. Reports hallucination rates of 58-80% for general-purpose LLMs on legal tasks. Discusses precision and interpretability improvements including citation tracing, structured retrieval, and domain adaptation.
- **Relevance:** **Medium.** The 58-80% hallucination baseline for general LLMs on legal tasks contextualizes our baseline faithfulness score (0.640 without RAG for GPT-4o). Their discussion of citation tracing aligns with our citation check step in the pipeline.

---

## Key Themes & Takeaways for This Project

### 1. Hallucination remains the central challenge in legal RAG
Even commercial tools hallucinate 17-33% of the time (Magesh et al., 2025). General LLMs hallucinate 58-80% on legal tasks. Our iterative improvement from 0.640 to 0.900 faithfulness is on the right track but must continue.

### 2. Hybrid retrieval is the emerging standard
Multiple papers (HyPA-RAG, CBR-RAG, comprehensive surveys) converge on combining dense + sparse + reranking retrieval. Our hybrid (0.6/0.4) + cross-encoder rerank approach aligns with current best practices.

### 3. Chunking strategy matters enormously for legal text
Summary-Augmented Chunking (SAC) and clause-boundary segmentation are two promising approaches. Our statute-aware chunking stretch goal is well-supported by the literature. SAC is a low-effort, high-impact improvement we should consider.

### 4. Domain-specific embeddings outperform general ones
CBR-RAG shows LegalBERT > general BERT for legal retrieval. This supports our planned embedding upgrade from all-MiniLM-L6-v2 to a larger/domain-adapted model.

### 5. LLM-as-judge evaluation is validated but imperfect
RAGAS formalizes the approach we use. Our decision to use a separate model family (Claude Sonnet 4) as judge -- avoiding self-evaluation bias -- is a recognized best practice.

### 6. Retrieval precision > recall for legal applications
LegalBench-RAG emphasizes minimal, precise retrieval over broad recall. This supports our reranking approach and suggests we should focus on precision metrics.

---

## Suggested Actions Based on Literature

1. **Implement Summary-Augmented Chunking** (from Paper #3) -- prepend document summaries to chunks to reduce retrieval mismatch
2. **Explore query complexity classification** (from Paper #2) -- route simple vs. complex queries to different retrieval strategies
3. **Test domain-specific embeddings** (from Paper #5) -- compare LegalBERT or BGE-legal against current all-MiniLM-L6-v2
4. **Add hallucination typology to evaluation** (from Paper #1) -- distinguish fabricated citations from incorrect holdings
5. **Cite these papers in M2/M3 reports** to ground our methodology in current literature
