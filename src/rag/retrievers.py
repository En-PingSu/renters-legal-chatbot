"""
Multiple retrieval methods: vector, BM25, hybrid fusion, cross-encoder reranking,
and parent-child (neighbor expansion).
All functions return list[dict] with keys: chunk_id, content, metadata, distance.
"""

import json
import re
from collections import defaultdict

from src.rag.pipeline import retrieve as vector_retrieve
from src.scraping.utils import PROJECT_ROOT

CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "all_chunks.json"

# Module-level caches (built lazily)
_bm25_index = None
_bm25_corpus_ids = None
_bm25_chunks = None
_cross_encoder = None
_all_chunks_by_id = None


def _load_chunks() -> list[dict]:
    """Load all chunks from disk."""
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_all_chunks_by_id() -> dict[str, dict]:
    """Lazy-load all chunks indexed by chunk_id."""
    global _all_chunks_by_id
    if _all_chunks_by_id is not None:
        return _all_chunks_by_id
    chunks = _load_chunks()
    _all_chunks_by_id = {c["chunk_id"]: c for c in chunks}
    return _all_chunks_by_id


def _tokenize(text: str) -> list[str]:
    """Lowercase, extract alphanumeric tokens, stem with Snowball."""
    import Stemmer

    stemmer = Stemmer.Stemmer("english")
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return stemmer.stemWords(tokens)


def _get_bm25():
    """Lazy-build BM25 index from all_chunks.json."""
    global _bm25_index, _bm25_corpus_ids, _bm25_chunks
    if _bm25_index is not None:
        return _bm25_index, _bm25_corpus_ids, _bm25_chunks

    from rank_bm25 import BM25Okapi

    chunks = _load_chunks()
    corpus = [_tokenize(c["content"]) for c in chunks]
    _bm25_index = BM25Okapi(corpus)
    _bm25_corpus_ids = list(range(len(chunks)))
    _bm25_chunks = chunks
    return _bm25_index, _bm25_corpus_ids, _bm25_chunks


def _get_cross_encoder():
    """Lazy-load cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder

    from sentence_transformers import CrossEncoder

    _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def retrieve_vector(query: str, top_k: int = 5) -> list[dict]:
    """Vector retrieval via ChromaDB (thin wrapper)."""
    return vector_retrieve(query, top_k=top_k)


def retrieve_bm25(query: str, top_k: int = 5) -> list[dict]:
    """BM25 lexical retrieval."""
    bm25, corpus_ids, chunks = _get_bm25()
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Get top_k indices by score (descending)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        c = chunks[idx]
        results.append({
            "chunk_id": c["chunk_id"],
            "content": c["content"],
            "metadata": {
                "doc_id": c["doc_id"],
                "source_url": c["source_url"],
                "source_name": c["source_name"],
                "title": c["title"],
                "content_type": c["content_type"],
            },
            "distance": float(scores[idx]),  # BM25 score (higher = better)
        })
    return results


def retrieve_hybrid(
    query: str, top_k: int = 5, vector_weight: float = 0.6, bm25_weight: float = 0.4
) -> list[dict]:
    """Hybrid retrieval: weighted fusion of vector + BM25."""
    pool_k = top_k * 2

    vector_results = retrieve_vector(query, top_k=pool_k)
    bm25_results = retrieve_bm25(query, top_k=pool_k)

    # Guard against empty results
    if not vector_results:
        return bm25_results[:top_k]
    if not bm25_results:
        return vector_results[:top_k]

    # Normalize vector scores: cosine distance (lower=better) -> similarity (higher=better)
    v_distances = [r["distance"] for r in vector_results]
    v_min, v_max = min(v_distances), max(v_distances)
    v_range = v_max - v_min if v_max > v_min else 1.0

    # Normalize BM25 scores (higher=better, keep direction)
    b_scores = [r["distance"] for r in bm25_results]
    b_min, b_max = min(b_scores), max(b_scores)
    b_range = b_max - b_min if b_max > b_min else 1.0

    # Build fused scores by chunk_id
    fused = {}  # chunk_id -> {score, chunk_data}
    for r in vector_results:
        # Invert: lower distance = higher similarity
        norm_score = (v_max - r["distance"]) / v_range
        cid = r["chunk_id"]
        fused[cid] = {"score": vector_weight * norm_score, "data": r}

    for r in bm25_results:
        norm_score = (r["distance"] - b_min) / b_range
        cid = r["chunk_id"]
        if cid in fused:
            fused[cid]["score"] += bm25_weight * norm_score
        else:
            fused[cid] = {"score": bm25_weight * norm_score, "data": r}

    # Sort by fused score (higher = better) and return top_k
    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    results = []
    for item in ranked:
        entry = item["data"].copy()
        entry["distance"] = item["score"]  # fused score
        results.append(entry)
    return results


def retrieve_rerank(query: str, top_k: int = 5, initial_k: int = 10) -> list[dict]:
    """Hybrid retrieval + cross-encoder reranking."""
    # Ensure rerank always has a larger candidate pool than top_k.
    # Without this, top_k=10 with initial_k=10 means the cross-encoder only
    # reorders (no filtering benefit). A 2x pool lets it rescue relevant chunks
    # buried at positions 11-20 in the hybrid results.
    # Note: 3x was tested but the ms-marco cross-encoder's domain mismatch
    # on legal content caused it to push out statute-specific chunks in favor
    # of general conversational content. 2x is a safer compromise.
    initial_k = max(initial_k, top_k * 2)
    candidates = retrieve_hybrid(query, top_k=initial_k)

    try:
        ce = _get_cross_encoder()
    except ImportError:
        print("  [WARN] sentence-transformers not installed, falling back to hybrid")
        return candidates[:top_k]

    # Score each candidate with cross-encoder
    pairs = [(query, c["content"]) for c in candidates]
    ce_scores = ce.predict(pairs)

    # Attach scores and sort
    for c, score in zip(candidates, ce_scores):
        c["distance"] = float(score)  # cross-encoder relevance score

    candidates.sort(key=lambda x: x["distance"], reverse=True)
    return candidates[:top_k]


def retrieve_parent_child(
    query: str, top_k: int = 5, expand_neighbors: int = 1
) -> list[dict]:
    """Vector retrieval + neighbor expansion for clustered documents.

    When multiple retrieved chunks come from the same document, expand
    context by including adjacent chunks from that document.
    """
    # Retrieve 2x candidates for expansion headroom
    initial = retrieve_vector(query, top_k=top_k * 2)

    # Group by doc_id to find clusters
    doc_chunks: dict[str, list[dict]] = defaultdict(list)
    for r in initial:
        doc_id = r["metadata"].get("doc_id", "")
        doc_chunks[doc_id].append(r)

    all_chunks_by_id = _get_all_chunks_by_id()

    expanded_ids: set[str] = set()
    expanded: list[dict] = []

    for r in initial[:top_k]:
        cid = r["chunk_id"]
        if cid in expanded_ids:
            continue
        expanded_ids.add(cid)
        expanded.append(r)

        doc_id = r["metadata"].get("doc_id", "")
        # Only expand if this document has multiple hits in initial results
        if len(doc_chunks[doc_id]) < 2:
            continue

        # Look up chunk_index from the full chunk data
        chunk_data = all_chunks_by_id.get(cid)
        if chunk_data is None:
            continue
        chunk_index = chunk_data["chunk_index"]
        total_chunks = chunk_data["total_chunks"]

        for offset in range(-expand_neighbors, expand_neighbors + 1):
            if offset == 0:
                continue
            neighbor_idx = chunk_index + offset
            if not (0 <= neighbor_idx < total_chunks):
                continue
            neighbor_id = f"{doc_id}_chunk_{neighbor_idx:03d}"
            if neighbor_id in expanded_ids or neighbor_id not in all_chunks_by_id:
                continue
            expanded_ids.add(neighbor_id)
            nc = all_chunks_by_id[neighbor_id]
            expanded.append({
                "chunk_id": nc["chunk_id"],
                "content": nc["content"],
                "metadata": {
                    "doc_id": nc["doc_id"],
                    "source_url": nc["source_url"],
                    "source_name": nc["source_name"],
                    "title": nc["title"],
                    "content_type": nc["content_type"],
                },
                "distance": r["distance"] * 1.2 if r["distance"] else None,  # dampened (higher distance = worse)
            })

    return expanded


RETRIEVER_REGISTRY = {
    "vector": retrieve_vector,
    "bm25": retrieve_bm25,
    "hybrid": retrieve_hybrid,
    "rerank": retrieve_rerank,
    "parent_child": retrieve_parent_child,
}

# Lazy-register new retrievers to avoid import overhead when not needed
def _register_new_retrievers():
    from src.rag.multiquery import retrieve_multiquery
    from src.rag.hybrid_parent_child import (
        retrieve_hybrid_parent_child,
        retrieve_hybrid_parent_child_rerank,
        retrieve_auto_merge,
    )
    RETRIEVER_REGISTRY.update({
        "multiquery": retrieve_multiquery,
        "hybrid_parent_child": retrieve_hybrid_parent_child,
        "hybrid_parent_child_rerank": retrieve_hybrid_parent_child_rerank,
        "auto_merge": retrieve_auto_merge,
    })

_register_new_retrievers()
