"""
Hybrid + parent-child retriever variants:
  - hybrid_parent_child: hybrid (vector+BM25) base with neighbor expansion
  - hybrid_parent_child_rerank: above + cross-encoder reranking
  - auto_merge: hybrid base with automatic parent-level merging
"""

from collections import defaultdict

from src.rag.retrievers import (
    retrieve_hybrid,
    retrieve_rerank,
    _get_all_chunks_by_id,
    _get_cross_encoder,
)


def retrieve_hybrid_parent_child(
    query: str,
    top_k: int = 5,
    expand_neighbors: int = 1,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> list[dict]:
    """Hybrid retrieval (vector+BM25) with neighbor expansion for clustered docs.

    Uses hybrid fusion as the base retriever instead of vector-only,
    then applies the same neighbor-expansion logic as retrieve_parent_child.
    """
    initial = retrieve_hybrid(
        query,
        top_k=top_k * 2,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
    )

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
                "distance": r["distance"] * 0.8,  # dampened (fused score, higher=better)
            })

    return expanded


def retrieve_hybrid_parent_child_rerank(
    query: str,
    top_k: int = 5,
    expand_neighbors: int = 1,
    initial_k: int = 10,
) -> list[dict]:
    """Hybrid parent-child retrieval + cross-encoder reranking.

    Gets a larger candidate pool via hybrid_parent_child, then reranks
    all candidates (including expanded neighbors) with the cross-encoder.
    """
    initial_k = max(initial_k, top_k * 2)
    candidates = retrieve_hybrid_parent_child(
        query, top_k=initial_k, expand_neighbors=expand_neighbors
    )

    try:
        ce = _get_cross_encoder()
    except ImportError:
        print("  [WARN] sentence-transformers not installed, falling back to hybrid_parent_child")
        return candidates[:top_k]

    pairs = [(query, c["content"]) for c in candidates]
    ce_scores = ce.predict(pairs)

    for c, score in zip(candidates, ce_scores):
        c["distance"] = float(score)

    candidates.sort(key=lambda x: x["distance"], reverse=True)
    return candidates[:top_k]


def retrieve_auto_merge(
    query: str,
    top_k: int = 5,
    merge_threshold: float = 0.4,
) -> list[dict]:
    """Auto-merging retriever: merges up to parent level when enough children match.

    When the fraction of retrieved chunks from a document meets or exceeds
    merge_threshold, all chunks from that document are concatenated into a
    single result. Documents below the threshold keep individual chunks.
    Final results are reranked with the cross-encoder.
    """
    # Get a generous initial pool
    initial_k = max(20, top_k * 4)
    candidates = retrieve_hybrid(query, top_k=initial_k)

    all_chunks_by_id = _get_all_chunks_by_id()

    # Group retrieved chunks by doc_id
    doc_hits: dict[str, list[dict]] = defaultdict(list)
    for r in candidates:
        doc_id = r["metadata"].get("doc_id", "")
        doc_hits[doc_id].append(r)

    merged_results: list[dict] = []

    for doc_id, hits in doc_hits.items():
        # Look up total_chunks for this document
        sample_chunk = all_chunks_by_id.get(hits[0]["chunk_id"])
        if sample_chunk is None:
            merged_results.extend(hits)
            continue
        total_chunks = sample_chunk["total_chunks"]

        coverage = len(hits) / total_chunks if total_chunks > 0 else 0

        if coverage >= merge_threshold:
            # Merge: gather ALL chunks from this document in order
            doc_all_chunks = []
            for idx in range(total_chunks):
                cid = f"{doc_id}_chunk_{idx:03d}"
                if cid in all_chunks_by_id:
                    doc_all_chunks.append(all_chunks_by_id[cid])

            doc_all_chunks.sort(key=lambda c: c["chunk_index"])
            merged_content = "\n\n".join(c["content"] for c in doc_all_chunks)

            # Use metadata from the first hit, best fused score from hits
            best_hit = max(hits, key=lambda h: h["distance"])
            merged_results.append({
                "chunk_id": f"{doc_id}_merged",
                "content": merged_content,
                "metadata": best_hit["metadata"],
                "distance": best_hit["distance"],
            })
        else:
            # Keep individual chunks
            merged_results.extend(hits)

    # Rerank with cross-encoder
    try:
        ce = _get_cross_encoder()
    except ImportError:
        print("  [WARN] sentence-transformers not installed, skipping rerank")
        merged_results.sort(key=lambda x: x["distance"], reverse=True)
        return merged_results[:top_k]

    pairs = [(query, c["content"]) for c in merged_results]
    ce_scores = ce.predict(pairs)

    for c, score in zip(merged_results, ce_scores):
        c["distance"] = float(score)

    merged_results.sort(key=lambda x: x["distance"], reverse=True)
    return merged_results[:top_k]


# ---------------------------------------------------------------------------
# Comparison test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_QUERIES = [
        "What are my rights if my landlord doesn't return my security deposit within 30 days?",
        "How does the eviction process work in Massachusetts?",
        "What can I do about bed bugs in my apartment in Boston?",
    ]

    RETRIEVERS = {
        "rerank (baseline)": lambda q, k: retrieve_rerank(q, top_k=k),
        "hybrid_parent_child": lambda q, k: retrieve_hybrid_parent_child(q, top_k=k),
        "hybrid_parent_child_rerank": lambda q, k: retrieve_hybrid_parent_child_rerank(q, top_k=k),
        "auto_merge": lambda q, k: retrieve_auto_merge(q, top_k=k),
    }

    TOP_K = 5
    overlap_totals: dict[str, list[float]] = defaultdict(list)

    for query in TEST_QUERIES:
        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        # Run baseline first
        baseline_results = retrieve_rerank(query, top_k=TOP_K)
        baseline_ids = set(r["chunk_id"] for r in baseline_results)

        print(f"\n  rerank (baseline): {[r['chunk_id'] for r in baseline_results]}")

        for name, fn in RETRIEVERS.items():
            if name == "rerank (baseline)":
                continue
            results = fn(query, TOP_K)
            result_ids = set(r["chunk_id"] for r in results)

            overlap = baseline_ids & result_ids
            unique = result_ids - baseline_ids
            overlap_pct = len(overlap) / len(baseline_ids) * 100 if baseline_ids else 0
            overlap_totals[name].append(overlap_pct)

            print(f"\n  {name}:")
            print(f"    chunks: {[r['chunk_id'] for r in results]}")
            print(f"    overlap with baseline: {len(overlap)}/{len(baseline_ids)} ({overlap_pct:.0f}%)")
            if unique:
                print(f"    unique chunks gained: {sorted(unique)}")

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY: Average overlap with baseline (rerank)")
    print("=" * 80)
    for name, pcts in overlap_totals.items():
        avg = sum(pcts) / len(pcts) if pcts else 0
        print(f"  {name}: {avg:.1f}%")
