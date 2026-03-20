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
import random
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.scorer import (
    EVAL_DIR,
    RESULTS_DIR,
    get_openrouter_client,
    judge_correctness,
    judge_retrieval_coverage,
    load_eval_questions,
)
from src.rag.pipeline import ask, format_context
from src.rag.retrievers import RETRIEVER_REGISTRY


# ── Stratified sampling (matches Section 7.8 methodology) ──────────────

def stratify_questions(questions: list[dict], per_topic: int = 2,
                       reddit_count: int = 4, seed: int = 42) -> list[dict]:
    """Select a stratified subset: per_topic per golden topic + reddit_count Reddit."""
    rng = random.Random(seed)

    golden_by_topic: dict[str, list[dict]] = {}
    reddit_qs: list[dict] = []
    for q in questions:
        if q["source"] == "golden":
            # look up topic from the original data
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
):
    """Run retrieval-aware correctness scoring on 28 stratified questions."""
    print("=" * 60)
    print("Retrieval-Aware Correctness Scoring (Section 7.15)")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = get_openrouter_client()

    # Load & stratify
    all_questions = load_eval_questions_with_topics()
    questions = stratify_questions(all_questions)
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
            "retrieval_coverage": ret_cov["score"],
            "retrieval_hits": ret_cov["hits"],
            "retrieval_total": ret_cov["total"],
            "generation_correctness": gen_cor["score"],
            "generation_hits": gen_cor["hits"],
            "generation_total": gen_cor["total"],
            "per_fact_attribution": per_fact_attribution,
        })

        # Print per-question summary
        print(f"  Retrieval: {ret_cov['hits']}/{ret_cov['total']} "
              f"({ret_cov['score']:.3f})  "
              f"Generation: {gen_cor['hits']}/{gen_cor['total']} "
              f"({gen_cor['score']:.3f})")

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

    print(f"\nTotal facts evaluated: {total_facts}")
    print(f"Retrieval coverage:   {total_ret_hits}/{total_facts} = {ret_coverage:.3f}")
    print(f"Generation coverage:  {total_gen_hits}/{total_facts} = {gen_coverage:.3f}")
    print(f"Gen coverage | retrieved: {attr_counts['covered']}/{retrieved_facts} = {gen_given_ret:.3f}")
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

    # ── Per-question table ──────────────────────────────────────────────

    print(f"\n{'ID':<14} {'Topic':<22} {'Ret':>5} {'Gen':>5} {'Miss type'}")
    print("-" * 70)
    for r in results:
        misses = [pf["attribution"] for pf in r["per_fact_attribution"]
                  if pf["attribution"] in ("retrieval_miss", "generation_miss")]
        miss_str = ", ".join(misses) if misses else "—"
        print(f"{r['question_id']:<14} {r['topic']:<22} "
              f"{r['retrieval_coverage']:>5.3f} {r['generation_correctness']:>5.3f} "
              f"{miss_str}")

    # ── Save ────────────────────────────────────────────────────────────

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = {
        "config": {
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
            "attribution": attr_counts,
        },
        "results": results,
    }

    out_path = RESULTS_DIR / f"retrieval_coverage_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    return output


if __name__ == "__main__":
    run()
