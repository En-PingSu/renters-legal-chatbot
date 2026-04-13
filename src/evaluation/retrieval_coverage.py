"""
Retrieval-Aware Correctness Scoring (Section 7.15)

Decomposes correctness into:
  - Retrieval coverage: fraction of key facts present in retrieved chunks
  - Generation coverage: fraction of *retrieved* facts the LLM included
  - Overall correctness: fraction of key facts in the LLM response (existing metric)

This separates retrieval failures from generation failures.

Usage:
    venv/bin/python3 -m src.evaluation.retrieval_coverage
"""

import json
import os
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.scorer import (
    EVAL_DIR,
    RESULTS_DIR,
    get_openrouter_client,
    judge_correctness,
    judge_faithfulness,
    judge_relevancy,
    judge_retrieval_coverage,
    load_eval_questions,
)
from src.rag.pipeline import ask, format_context
from src.rag.retrievers import RETRIEVER_REGISTRY


# ── Fixed evaluation set ───────────────────────────────────────────────

FIXED_SET_PATH = EVAL_DIR / "stratified_questions.json"


def load_fixed_questions(questions: list[dict]) -> list[dict]:
    """Load the fixed stratified question set (27 standard + 10 hard).

    Attaches difficulty tags and source_chunks (for chunk-level retrieval metrics).
    Falls back to random stratification if the fixed set file doesn't exist.
    """
    if not FIXED_SET_PATH.exists():
        print("[WARN] Fixed set not found, falling back to random stratification")
        return _stratify_random(questions)

    with open(FIXED_SET_PATH, "r") as f:
        fixed = json.load(f)

    fixed_ids = set(fixed["standard"] + fixed["hard"])
    selected = [q for q in questions if q["id"] in fixed_ids]

    # Tag difficulty for downstream reporting
    hard_ids = set(fixed.get("hard", []))
    for q in selected:
        q["difficulty"] = "hard" if q["id"] in hard_ids else "standard"

    # Attach source_chunks for chunk-level retrieval metrics (MRR, Recall@K, etc.)
    source_chunks_map = {}
    golden_path = EVAL_DIR / "golden_qa.json"
    if golden_path.exists():
        with open(golden_path, "r") as f:
            for item in json.load(f):
                source_chunks_map[item["id"]] = item.get("source_chunks", [])
    reddit_path = EVAL_DIR / "reddit_questions.json"
    if reddit_path.exists():
        with open(reddit_path, "r") as f:
            for item in json.load(f):
                qid = f"reddit_{item['id']}" if not item["id"].startswith("reddit_") else item["id"]
                source_chunks_map[qid] = item.get("source_chunks", [])
    for q in selected:
        q["source_chunk_ids"] = [s["chunk_id"] for s in source_chunks_map.get(q["id"], [])]

    print(f"Loaded fixed set: {len(selected)} questions "
          f"({len(fixed['standard'])} standard + {len(fixed['hard'])} hard)")
    return selected


def _stratify_random(questions: list[dict], per_topic: int = 2,
                     reddit_count: int = 4, seed: int = 42) -> list[dict]:
    """Fallback: select a stratified subset via random sampling."""
    rng = random.Random(seed)

    golden_by_topic: dict[str, list[dict]] = {}
    reddit_qs: list[dict] = []
    for q in questions:
        if q["source"] == "golden":
            golden_by_topic.setdefault(q.get("topic", "unknown"), []).append(q)
        else:
            reddit_qs.append(q)

    selected: list[dict] = []
    for topic in sorted(golden_by_topic):
        pool = golden_by_topic[topic]
        selected.extend(rng.sample(pool, min(per_topic, len(pool))))

    if reddit_qs:
        selected.extend(rng.sample(reddit_qs, min(reddit_count, len(reddit_qs))))

    return selected


def load_eval_questions_with_topics() -> list[dict]:
    """Load eval questions, preserving topic from golden_qa.json."""
    questions = load_eval_questions()

    # Attach topics from golden_qa.json
    golden_path = EVAL_DIR / "golden_qa.json"
    if golden_path.exists():
        with open(golden_path, "r") as f:
            golden = json.load(f)
        topic_map = {item["id"]: item.get("topic", "") for item in golden}
        for q in questions:
            if q["id"] in topic_map:
                q["topic"] = topic_map[q["id"]]

    return questions


# ── Main runner ─────────────────────────────────────────────────────────

def run(
    judge_model: str = "anthropic/claude-sonnet-4",
    gen_model: str = "openai/gpt-4o",
    retriever_name: str = "rerank",
    top_k: int = 10,
    use_all_questions: bool = False,
):
    """Run retrieval-aware correctness scoring.

    Args:
        use_all_questions: If True, use all 89 questions (matching Iteration 9).
                           If False, use the 37 fixed stratified subset.
    """
    print("=" * 60)
    print("Retrieval-Aware Correctness Scoring (Section 7.15)")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = get_openrouter_client()

    all_questions = load_eval_questions_with_topics()
    if use_all_questions:
        questions = all_questions
        # Tag difficulty and attach source_chunks for all questions
        fixed_path = EVAL_DIR / "stratified_questions.json"
        hard_ids = set()
        if fixed_path.exists():
            with open(fixed_path, "r") as f:
                fixed = json.load(f)
            hard_ids = set(fixed.get("hard", []))
        for q in questions:
            q["difficulty"] = "hard" if q["id"] in hard_ids else "standard"

        # Attach source_chunks
        source_chunks_map = {}
        golden_path = EVAL_DIR / "golden_qa.json"
        if golden_path.exists():
            with open(golden_path, "r") as f:
                for item in json.load(f):
                    source_chunks_map[item["id"]] = item.get("source_chunks", [])
        reddit_path = EVAL_DIR / "reddit_questions.json"
        if reddit_path.exists():
            with open(reddit_path, "r") as f:
                for item in json.load(f):
                    qid = f"reddit_{item['id']}" if not item["id"].startswith("reddit_") else item["id"]
                    source_chunks_map[qid] = item.get("source_chunks", [])
        for q in questions:
            q["source_chunk_ids"] = [s["chunk_id"] for s in source_chunks_map.get(q["id"], [])]

        print(f"Using ALL {len(questions)} questions (matching Iteration 9)")
    else:
        # Load fixed question set (27 standard + 10 hard)
        questions = load_fixed_questions(all_questions)

    # Keep only questions with key_facts
    questions = [q for q in questions if q.get("key_facts")]
    print(f"Selected {len(questions)} questions with key_facts")

    retrieve_fn = RETRIEVER_REGISTRY[retriever_name]
    results = []

    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {q['question'][:70]}...")

        # 1. Retrieve chunks
        chunks = retrieve_fn(q["question"], top_k=top_k)
        context_parts = []
        for chunk in chunks:
            meta = chunk["metadata"]
            context_parts.append(
                f"[{meta['title']} ({meta['source_url']})]\n{chunk['content']}"
            )
        retrieved_context = "\n\n---\n\n".join(context_parts)

        # 1b. Chunk-level retrieval metrics (free — no API calls)
        retrieved_ids = [c["chunk_id"] for c in chunks]
        gt_ids = set(q.get("source_chunk_ids", []))
        if gt_ids:
            q_hit = 1 if gt_ids & set(retrieved_ids) else 0
            q_mrr = 0.0
            for rank, rid in enumerate(retrieved_ids, 1):
                if rid in gt_ids:
                    q_mrr = 1.0 / rank
                    break
            q_found = gt_ids & set(retrieved_ids)
            q_recall = len(q_found) / len(gt_ids)
            dcg = sum(1.0 / math.log2(r + 1) for r, rid in enumerate(retrieved_ids, 1) if rid in gt_ids)
            ideal_k = min(len(gt_ids), len(retrieved_ids))
            idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_k + 1))
            q_ndcg = dcg / idcg if idcg > 0 else 0.0
        else:
            q_hit, q_mrr, q_recall, q_ndcg = 0, 0.0, 0.0, 0.0

        # 2. Generate response (structured prompt, matching best config)
        rag_result = ask(
            question=q["question"],
            model=gen_model,
            use_rag=True,
            retriever=retriever_name,
            top_k=top_k,
        )
        response = rag_result["response"]

        # 3. Judge retrieval coverage (facts in chunks)
        print(f"  Judging retrieval coverage...")
        ret_cov = judge_retrieval_coverage(
            q["question"], retrieved_context, q["key_facts"], client, judge_model
        )

        # 4. Judge generation correctness (facts in response)
        print(f"  Judging generation correctness...")
        gen_cor = judge_correctness(
            q["question"], response, q["key_facts"], client, judge_model
        )

        # 4b. Judge faithfulness and relevancy
        print(f"  Judging faithfulness & relevancy...")
        faith = judge_faithfulness(
            q["question"], response, retrieved_context, client, judge_model
        )
        relev = judge_relevancy(
            q["question"], response, client, judge_model
        )

        # 5. Per-fact attribution: retrieval miss vs generation miss
        per_fact_attribution = []
        for fi, fact in enumerate(q["key_facts"]):
            in_retrieval = ret_cov["per_fact"][fi]["present"] if fi < len(ret_cov["per_fact"]) else False
            in_generation = gen_cor["per_fact"][fi]["present"] if fi < len(gen_cor["per_fact"]) else False

            if in_retrieval and in_generation:
                attribution = "covered"  # retrieved AND generated
            elif in_retrieval and not in_generation:
                attribution = "generation_miss"  # retrieved but LLM dropped it
            elif not in_retrieval and in_generation:
                attribution = "hallucinated"  # not retrieved but LLM included it
            else:
                attribution = "retrieval_miss"  # never retrieved

            per_fact_attribution.append({
                "fact": fact,
                "in_retrieval": in_retrieval,
                "in_generation": in_generation,
                "attribution": attribution,
            })

        results.append({
            "question_id": q["id"],
            "question": q["question"],
            "topic": q.get("topic", ""),
            "difficulty": q.get("difficulty", "standard"),
            "retrieval_coverage": ret_cov["score"],
            "retrieval_hits": ret_cov["hits"],
            "retrieval_total": ret_cov["total"],
            "generation_correctness": gen_cor["score"],
            "generation_hits": gen_cor["hits"],
            "generation_total": gen_cor["total"],
            "faithfulness": faith["score"],
            "relevancy": relev["score"],
            "chunk_metrics": {
                "hit_rate": q_hit,
                "mrr": round(q_mrr, 4),
                "recall_at_k": round(q_recall, 4),
                "ndcg_at_k": round(q_ndcg, 4),
                "gt_chunks": len(gt_ids),
                "found_chunks": len(q_found) if gt_ids else 0,
            },
            "per_fact_attribution": per_fact_attribution,
        })

        # Print per-question summary
        print(f"  Retrieval: {ret_cov['hits']}/{ret_cov['total']} "
              f"({ret_cov['score']:.3f})  "
              f"Generation: {gen_cor['hits']}/{gen_cor['total']} "
              f"({gen_cor['score']:.3f})  "
              f"Faith: {faith['score']:.0f}  Rel: {relev['score']:.0f}  "
              f"Recall@K: {q_recall:.3f} ({len(q_found) if gt_ids else 0}/{len(gt_ids)})")

    # ── Aggregate results ───────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    total_facts = sum(r["retrieval_total"] for r in results)
    total_ret_hits = sum(r["retrieval_hits"] for r in results)
    total_gen_hits = sum(r["generation_hits"] for r in results)

    ret_coverage = round(total_ret_hits / total_facts, 3) if total_facts else 0
    gen_coverage = round(total_gen_hits / total_facts, 3) if total_facts else 0

    # Count attributions across all facts
    attr_counts = {"covered": 0, "generation_miss": 0,
                   "retrieval_miss": 0, "hallucinated": 0}
    for r in results:
        for pf in r["per_fact_attribution"]:
            attr_counts[pf["attribution"]] += 1

    # Generation coverage conditioned on retrieval
    retrieved_facts = attr_counts["covered"] + attr_counts["generation_miss"]
    gen_given_ret = (round(attr_counts["covered"] / retrieved_facts, 3)
                     if retrieved_facts else 0)

    # Faithfulness and relevancy aggregates
    valid_faith = [r["faithfulness"] for r in results if r["faithfulness"] >= 0]
    valid_relev = [r["relevancy"] for r in results if r["relevancy"] >= 0]
    avg_faith = round(sum(valid_faith) / len(valid_faith), 3) if valid_faith else 0
    avg_relev = round(sum(valid_relev) / len(valid_relev), 3) if valid_relev else 0

    print(f"\nTotal facts evaluated: {total_facts}")
    print(f"Retrieval coverage:   {total_ret_hits}/{total_facts} = {ret_coverage:.3f}")
    print(f"Generation coverage:  {total_gen_hits}/{total_facts} = {gen_coverage:.3f}")
    print(f"Gen coverage | retrieved: {attr_counts['covered']}/{retrieved_facts} = {gen_given_ret:.3f}")
    print(f"Faithfulness:         {sum(1 for f in valid_faith if f == 1.0)}/{len(valid_faith)} = {avg_faith:.3f}")
    print(f"Relevancy:            {sum(1 for r in valid_relev if r == 1.0)}/{len(valid_relev)} = {avg_relev:.3f}")
    print(f"\nPer-fact attribution:")
    print(f"  Covered (ret+gen):    {attr_counts['covered']}")
    print(f"  Generation miss:      {attr_counts['generation_miss']}")
    print(f"  Retrieval miss:       {attr_counts['retrieval_miss']}")
    print(f"  Hallucinated:         {attr_counts['hallucinated']}")

    missed_total = attr_counts["retrieval_miss"] + attr_counts["generation_miss"]
    if missed_total > 0:
        ret_miss_pct = round(100 * attr_counts["retrieval_miss"] / missed_total, 1)
        gen_miss_pct = round(100 * attr_counts["generation_miss"] / missed_total, 1)
        print(f"\nOf {missed_total} missed facts:")
        print(f"  {ret_miss_pct}% are retrieval failures")
        print(f"  {gen_miss_pct}% are generation failures")

    # ── Chunk-level retrieval metrics (aggregate) ─────────────────────

    results_with_gt = [r for r in results if r["chunk_metrics"]["gt_chunks"] > 0]
    if results_with_gt:
        n = len(results_with_gt)
        avg_mrr = sum(r["chunk_metrics"]["mrr"] for r in results_with_gt) / n
        avg_hit = sum(r["chunk_metrics"]["hit_rate"] for r in results_with_gt) / n
        avg_recall = sum(r["chunk_metrics"]["recall_at_k"] for r in results_with_gt) / n
        avg_ndcg = sum(r["chunk_metrics"]["ndcg_at_k"] for r in results_with_gt) / n
        chunk_metrics_agg = {
            "mrr": round(avg_mrr, 4),
            "hit_rate": round(avg_hit, 4),
            "recall_at_k": round(avg_recall, 4),
            "ndcg_at_k": round(avg_ndcg, 4),
            "num_queries_with_gt": n,
        }
        print(f"\nChunk-level retrieval metrics ({n} questions with ground truth):")
        print(f"  MRR={avg_mrr:.3f}  Hit@K={avg_hit:.3f}  "
              f"Recall@K={avg_recall:.3f}  NDCG@K={avg_ndcg:.3f}")
    else:
        chunk_metrics_agg = {}

    # ── Standard vs Hard breakdown ────────────────────────────────────

    for tier in ("standard", "hard"):
        tier_results = [r for r in results if r.get("difficulty", "standard") == tier]
        if not tier_results:
            continue
        t_facts = sum(r["retrieval_total"] for r in tier_results)
        t_ret = sum(r["retrieval_hits"] for r in tier_results)
        t_gen = sum(r["generation_hits"] for r in tier_results)
        t_ret_cov = round(t_ret / t_facts, 3) if t_facts else 0
        t_gen_cov = round(t_gen / t_facts, 3) if t_facts else 0
        t_attr = {"covered": 0, "generation_miss": 0,
                  "retrieval_miss": 0, "hallucinated": 0}
        for r in tier_results:
            for pf in r["per_fact_attribution"]:
                t_attr[pf["attribution"]] += 1
        t_retrieved = t_attr["covered"] + t_attr["generation_miss"]
        t_gen_given_ret = (round(t_attr["covered"] / t_retrieved, 3)
                           if t_retrieved else 0)
        # Chunk-level metrics per tier
        t_with_gt = [r for r in tier_results if r["chunk_metrics"]["gt_chunks"] > 0]
        t_n = len(t_with_gt)
        t_mrr = sum(r["chunk_metrics"]["mrr"] for r in t_with_gt) / t_n if t_n else 0
        t_hit = sum(r["chunk_metrics"]["hit_rate"] for r in t_with_gt) / t_n if t_n else 0
        t_recall_k = sum(r["chunk_metrics"]["recall_at_k"] for r in t_with_gt) / t_n if t_n else 0
        t_ndcg_k = sum(r["chunk_metrics"]["ndcg_at_k"] for r in t_with_gt) / t_n if t_n else 0
        print(f"\n  [{tier.upper()}] ({len(tier_results)} questions, {t_facts} facts)")
        print(f"    Fact-level:  Ret={t_ret_cov:.3f}  Gen={t_gen_cov:.3f}  Gen|Ret={t_gen_given_ret:.3f}")
        print(f"    Chunk-level: MRR={t_mrr:.3f}  Hit@K={t_hit:.3f}  "
              f"Recall@K={t_recall_k:.3f}  NDCG@K={t_ndcg_k:.3f}")
        print(f"    covered={t_attr['covered']} gen_miss={t_attr['generation_miss']} "
              f"ret_miss={t_attr['retrieval_miss']} halluc={t_attr['hallucinated']}")

    # ── Per-question table ──────────────────────────────────────────────

    print(f"\n{'ID':<14} {'Diff':<5} {'Topic':<20} {'Ret':>5} {'Gen':>5} "
          f"{'Rcl@K':>6} {'NDCG':>6} {'MRR':>5}")
    print("-" * 85)
    for r in results:
        diff = r.get("difficulty", "standard")[:4]
        cm = r["chunk_metrics"]
        print(f"{r['question_id']:<14} {diff:<5} {r['topic']:<20} "
              f"{r['retrieval_coverage']:>5.3f} {r['generation_correctness']:>5.3f} "
              f"{cm['recall_at_k']:>6.3f} {cm['ndcg_at_k']:>6.3f} {cm['mrr']:>5.3f}")

    # ── Save ────────────────────────────────────────────────────────────

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    # Per-tier aggregates for saved output
    tier_aggregates = {}
    for tier in ("standard", "hard"):
        tier_results = [r for r in results if r.get("difficulty", "standard") == tier]
        if not tier_results:
            continue
        t_facts = sum(r["retrieval_total"] for r in tier_results)
        t_ret = sum(r["retrieval_hits"] for r in tier_results)
        t_gen = sum(r["generation_hits"] for r in tier_results)
        t_attr = {"covered": 0, "generation_miss": 0,
                  "retrieval_miss": 0, "hallucinated": 0}
        for r in tier_results:
            for pf in r["per_fact_attribution"]:
                t_attr[pf["attribution"]] += 1
        t_retrieved = t_attr["covered"] + t_attr["generation_miss"]
        # Chunk-level metrics per tier for saved output
        t_with_gt = [r for r in tier_results if r["chunk_metrics"]["gt_chunks"] > 0]
        t_n = len(t_with_gt)
        tier_aggregates[tier] = {
            "num_questions": len(tier_results),
            "total_facts": t_facts,
            "retrieval_coverage": round(t_ret / t_facts, 3) if t_facts else 0,
            "generation_coverage": round(t_gen / t_facts, 3) if t_facts else 0,
            "generation_coverage_given_retrieval": (
                round(t_attr["covered"] / t_retrieved, 3) if t_retrieved else 0),
            "attribution": t_attr,
            "chunk_metrics": {
                "mrr": round(sum(r["chunk_metrics"]["mrr"] for r in t_with_gt) / t_n, 4) if t_n else 0,
                "hit_rate": round(sum(r["chunk_metrics"]["hit_rate"] for r in t_with_gt) / t_n, 4) if t_n else 0,
                "recall_at_k": round(sum(r["chunk_metrics"]["recall_at_k"] for r in t_with_gt) / t_n, 4) if t_n else 0,
                "ndcg_at_k": round(sum(r["chunk_metrics"]["ndcg_at_k"] for r in t_with_gt) / t_n, 4) if t_n else 0,
            },
        }

    output = {
        "config": {
            "embedding_model": os.getenv("EMBEDDING_MODEL", "default"),
            "gen_model": gen_model,
            "judge_model": judge_model,
            "retriever": retriever_name,
            "top_k": top_k,
            "num_questions": len(questions),
        },
        "aggregate": {
            "total_facts": total_facts,
            "retrieval_coverage": ret_coverage,
            "generation_coverage": gen_coverage,
            "generation_coverage_given_retrieval": gen_given_ret,
            "faithfulness": avg_faith,
            "relevancy": avg_relev,
            "attribution": attr_counts,
            "chunk_metrics": chunk_metrics_agg,
        },
        "per_tier": tier_aggregates,
        "results": results,
    }

    out_path = RESULTS_DIR / f"retrieval_coverage_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    return output


if __name__ == "__main__":
    import sys
    use_all = "--all" in sys.argv
    model = "openai/gpt-4o"
    for arg in sys.argv[1:]:
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1]
    run(gen_model=model, use_all_questions=use_all)
