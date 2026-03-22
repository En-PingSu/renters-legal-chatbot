"""
Multi-query expansion retrieval.

Generates alternative phrasings of the user query using an LLM to bridge
vocabulary gaps (casual tenant language vs formal legal/statute terminology),
then runs the base retriever on each variant and merges results using
reciprocal rank fusion (RRF).
"""

import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from src.rag.retrievers import RETRIEVER_REGISTRY


def _get_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def generate_query_variants(
    query: str, n: int = 3, model: str = "openai/gpt-4o"
) -> list[str]:
    client = _get_client()

    prompt = (
        f"You are a query expansion assistant for a Massachusetts tenant law knowledge base. "
        f"Given a user question, generate exactly {n} alternative phrasings that would help "
        f"retrieve relevant legal information. Each variant should use different vocabulary:\n"
        f"- One using casual everyday tenant language\n"
        f"- One using formal legal terminology (e.g., 'quiet enjoyment', 'habitability', 'implied warranty')\n"
        f"- One using Massachusetts statute/regulation references (e.g., 'MGL Chapter 186', 'CMR 410')\n\n"
        f"Return ONLY the rephrased queries, one per line, with no numbering or extra text.\n\n"
        f"User question: {query}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()
    variants = [line.strip() for line in raw.splitlines() if line.strip()]
    return variants[:n]


def _reciprocal_rank_fusion(
    ranked_lists: list[list[dict]], k: int = 60
) -> list[dict]:
    """Merge multiple ranked result lists using RRF scoring."""
    scores: dict[str, float] = {}
    best_entry: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, result in enumerate(ranked):
            cid = result["chunk_id"]
            rrf_score = 1.0 / (k + rank + 1)
            scores[cid] = scores.get(cid, 0.0) + rrf_score
            if cid not in best_entry:
                best_entry[cid] = result

    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    merged = []
    for cid in sorted_ids:
        entry = best_entry[cid].copy()
        entry["distance"] = scores[cid]
        merged.append(entry)
    return merged


def retrieve_multiquery(
    query: str,
    top_k: int = 5,
    n_variants: int = 3,
    base_retriever: str = "rerank",
    model: str = "openai/gpt-4o",
) -> list[dict]:
    retriever_fn = RETRIEVER_REGISTRY[base_retriever]

    variants = generate_query_variants(query, n=n_variants, model=model)
    all_queries = [query] + variants

    # Retrieve for each query variant
    ranked_lists = []
    pool_k = top_k * 2  # over-fetch for better fusion
    for q in all_queries:
        results = retriever_fn(q, top_k=pool_k)
        ranked_lists.append(results)

    merged = _reciprocal_rank_fusion(ranked_lists)
    return merged[:top_k]


if __name__ == "__main__":
    load_dotenv()

    test_queries = [
        "Can my landlord keep my security deposit for normal wear and tear?",
        "What do I do if my landlord won't fix the heat?",
        "How much notice does my landlord need to give before entering my apartment?",
    ]

    baseline_retriever = "rerank"
    top_k = 5

    for query in test_queries:
        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        # Generate variants
        variants = generate_query_variants(query)
        print(f"\nGenerated variants:")
        for i, v in enumerate(variants, 1):
            print(f"  {i}. {v}")

        # Baseline
        baseline_results = RETRIEVER_REGISTRY[baseline_retriever](query, top_k=top_k)
        baseline_ids = [r["chunk_id"] for r in baseline_results]

        # Multi-query
        mq_results = retrieve_multiquery(
            query, top_k=top_k, base_retriever=baseline_retriever
        )
        mq_ids = [r["chunk_id"] for r in mq_results]

        print(f"\nBaseline ({baseline_retriever}) chunk_ids:")
        for cid in baseline_ids:
            print(f"  - {cid}")

        print(f"\nMulti-query chunk_ids:")
        for cid in mq_ids:
            marker = " [NEW]" if cid not in baseline_ids else ""
            print(f"  - {cid}{marker}")

        # Overlap stats
        baseline_set = set(baseline_ids)
        mq_set = set(mq_ids)
        overlap = baseline_set & mq_set
        unique_to_mq = mq_set - baseline_set
        pct = len(overlap) / max(len(mq_set), 1) * 100

        print(f"\nOverlap: {len(overlap)}/{len(mq_set)} ({pct:.0f}%)")
        if unique_to_mq:
            print(f"Unique to multi-query: {unique_to_mq}")
        print()
