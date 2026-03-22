"""
Evaluate new retrieval strategies: multi-query expansion, sentence window,
hybrid parent-child (with rerank), and auto-merge.

Two modes:
  --retrieval-only  Free dry run: MRR/hit rate using source_chunks ground truth
  (default)         Full LLM-judged eval: retrieval-aware correctness on 24 stratified questions

Usage:
    venv/bin/python3 -m src.evaluation.eval_new_retrievers --retrieval-only
    venv/bin/python3 -m src.evaluation.eval_new_retrievers
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone

from src.evaluation.scorer import (
    EVAL_DIR,
    RESULTS_DIR,
    get_openrouter_client,
    judge_correctness,
    judge_retrieval_coverage,
    load_eval_questions,
)
from src.evaluation.retrieval_coverage import (
    load_eval_questions_with_topics,
    stratify_questions,
)
from src.rag.pipeline import ask, format_context
from src.rag.retrievers import RETRIEVER_REGISTRY


# ── Config ────────────────────────────────────────────────────────────────

CONFIGS = [
    {"name": "rerank", "retriever": "rerank"},
    {"name": "multiquery", "retriever": "multiquery"},
    {"name": "hybrid_parent_child_rerank", "retriever": "hybrid_parent_child_rerank"},
    {"name": "auto_merge", "retriever": "auto_merge"},
    {"name": "sentence_window", "retriever": "_sentence_window"},  # special-cased
]

GEN_MODEL = "openai/gpt-4o"
JUDGE_MODEL = "anthropic/claude-sonnet-4"
TOP_K = 10


# ── Retrieval-only mode (free) ────────────────────────────────────────────

def _load_qa_with_source_chunks() -> list[dict]:
    """Load all QA pairs that have source_chunks for retrieval metric computation."""
    pairs = []

    golden_path = EVAL_DIR / "golden_qa.json"
    if golden_path.exists():
        with open(golden_path, "r") as f:
            golden = json.load(f)
        for item in golden:
            for sc in item.get("source_chunks", []):
                pairs.append({
                    "question": item["question"],
                    "ground_truth_chunk_id": sc["chunk_id"],
                    "source_title": sc.get("title", ""),
                })

    reddit_path = EVAL_DIR / "reddit_questions.json"
    if reddit_path.exists():
        with open(reddit_path, "r") as f:
            reddit = json.load(f)
        for item in reddit:
            for sc in item.get("source_chunks", []):
                pairs.append({
                    "question": item["question"],
                    "ground_truth_chunk_id": sc["chunk_id"],
                    "source_title": sc.get("title", ""),
                })

    return pairs


def _compute_retrieval_metrics(qa_pairs, retrieve_fn, retriever_name, top_k=10):
    """Compute MRR, Hit Rate for a retriever (no API cost)."""
    print(f"\n--- {retriever_name} ({len(qa_pairs)} QA pairs, top_k={top_k}) ---")

    mrr_total = 0.0
    hits = 0
    total = len(qa_pairs)

    # Cache retrieval results per unique question (many QA pairs share questions)
    cache = {}

    for i, pair in enumerate(qa_pairs):
        q = pair["question"]
        if q not in cache:
            chunks = retrieve_fn(q, top_k=top_k)
            cache[q] = [c["chunk_id"] for c in chunks]

        retrieved_ids = cache[q]
        gt_id = pair["ground_truth_chunk_id"]

        if gt_id in retrieved_ids:
            hits += 1
            rank = retrieved_ids.index(gt_id) + 1
            mrr_total += 1.0 / rank

    metrics = {
        "mrr": round(mrr_total / total, 4) if total else 0,
        "hit_rate": round(hits / total, 4) if total else 0,
        "hits": hits,
        "total": total,
        "top_k": top_k,
    }

    print(f"  MRR={metrics['mrr']:.3f}  Hit Rate={metrics['hit_rate']:.3f}  "
          f"Hits={hits}/{total}")
    return metrics


def run_retrieval_only():
    """Compute retrieval metrics for all retrievers (no API cost)."""
    print("=" * 60)
    print("Retrieval-Only Evaluation (MRR / Hit Rate)")
    print("=" * 60)

    qa_pairs = _load_qa_with_source_chunks()
    print(f"Loaded {len(qa_pairs)} QA pairs with ground truth chunk IDs")

    # Sentence window retriever needs special handling
    sw_available = False
    try:
        from src.processing.sentence_window_chunker import retrieve_sentence_window
        sw_available = True
    except Exception as e:
        print(f"\n[WARN] Sentence window not available: {e}")
        print("  Run: venv/bin/python3 -m src.processing.sentence_window_chunker")

    all_metrics = {}

    for config in CONFIGS:
        name = config["name"]

        if name == "multiquery":
            # Multi-query requires LLM calls for variant generation — not free
            print(f"\n--- {name}: SKIPPED (requires LLM calls, not free) ---")
            continue

        if name == "sentence_window":
            if not sw_available:
                print(f"\n--- {name}: SKIPPED (index not built) ---")
                continue
            retrieve_fn = lambda q, top_k=TOP_K: retrieve_sentence_window(q, top_k=top_k)
        else:
            retriever_name = config["retriever"]
            if retriever_name not in RETRIEVER_REGISTRY:
                print(f"\n--- {name}: SKIPPED (not in registry) ---")
                continue
            retrieve_fn = RETRIEVER_REGISTRY[retriever_name]

        metrics = _compute_retrieval_metrics(qa_pairs, retrieve_fn, name, top_k=TOP_K)
        all_metrics[name] = metrics

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Retriever':<35} {'MRR':>7} {'Hit Rate':>10} {'Hits':>6}")
    print("-" * 60)
    for name, m in all_metrics.items():
        print(f"{name:<35} {m['mrr']:>7.3f} {m['hit_rate']:>10.3f} "
              f"{m['hits']:>3}/{m['total']}")
    print("=" * 60)

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"new_retrievers_retrieval_only_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"top_k": TOP_K, "metrics": all_metrics}, f, indent=2)
    print(f"\nSaved to {out_path}")

    return all_metrics


# ── Full LLM-judged evaluation ────────────────────────────────────────────

def _retrieve_for_config(config_name, question, top_k):
    """Retrieve chunks for a given config, handling sentence_window specially."""
    if config_name == "sentence_window":
        from src.processing.sentence_window_chunker import retrieve_sentence_window
        results = retrieve_sentence_window(question, top_k=top_k)
        # Normalize: use expanded_content as content for generation/judging
        normalized = []
        for r in results:
            normalized.append({
                "chunk_id": r["chunk_id"],
                "content": r.get("expanded_content", r["content"]),
                "metadata": r["metadata"],
                "distance": r.get("distance"),
            })
        return normalized
    else:
        retriever = config_name if config_name != "rerank" else "rerank"
        # Find the actual retriever name from CONFIGS
        for c in CONFIGS:
            if c["name"] == config_name:
                retriever = c["retriever"]
                break
        retrieve_fn = RETRIEVER_REGISTRY[retriever]
        return retrieve_fn(question, top_k=top_k)


def _build_context(chunks):
    """Build context string from retrieved chunks."""
    parts = []
    for chunk in chunks:
        meta = chunk["metadata"]
        parts.append(f"[{meta['title']} ({meta['source_url']})]\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


def run_full_eval():
    """Run full LLM-judged retrieval-aware correctness evaluation."""
    print("=" * 60)
    print("New Retrievers — Full LLM-Judged Evaluation")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client = get_openrouter_client()

    # Load & stratify
    all_questions = load_eval_questions_with_topics()
    questions = stratify_questions(all_questions)
    questions = [q for q in questions if q.get("key_facts")]
    print(f"Selected {len(questions)} questions with key_facts")

    # Check sentence window availability
    sw_available = False
    try:
        from src.processing.sentence_window_chunker import retrieve_sentence_window
        sw_available = True
    except Exception:
        print("[WARN] Sentence window index not built. Skipping that config.")
        print("  Run: venv/bin/python3 -m src.processing.sentence_window_chunker")

    configs_to_run = [c for c in CONFIGS
                      if c["name"] != "sentence_window" or sw_available]

    all_results = {}  # config_name -> list of per-question results

    for config in configs_to_run:
        config_name = config["name"]
        print(f"\n{'='*60}")
        print(f"CONFIG: {config_name}")
        print(f"{'='*60}")

        results = []

        for i, q in enumerate(questions):
            print(f"\n  [{i+1}/{len(questions)}] {q['question'][:65]}...")

            # 1. Retrieve
            chunks = _retrieve_for_config(config_name, q["question"], TOP_K)
            context = _build_context(chunks)

            # 2. Generate response (using pipeline for structured prompt)
            if config_name == "sentence_window":
                # Can't use ask() since it uses standard retriever
                # Generate directly with context
                from src.rag.pipeline import generate_response, verify_citations, SYSTEM_PROMPT
                system_msg = SYSTEM_PROMPT.format(context=context, question=q["question"])
                from src.rag.pipeline import OpenAI as _OAI
                import os
                gen_client = _OAI(
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                )
                gen_response = gen_client.chat.completions.create(
                    model=GEN_MODEL,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": q["question"]},
                    ],
                    temperature=0.2,
                    max_tokens=1500,
                )
                response = gen_response.choices[0].message.content
            else:
                rag_result = ask(
                    question=q["question"],
                    model=GEN_MODEL,
                    use_rag=True,
                    retriever=config["retriever"],
                    top_k=TOP_K,
                )
                response = rag_result["response"]

            # 3. Judge retrieval coverage
            print(f"    Judging retrieval coverage...")
            ret_cov = judge_retrieval_coverage(
                q["question"], context, q["key_facts"], client, JUDGE_MODEL
            )

            # 4. Judge generation correctness
            print(f"    Judging generation correctness...")
            gen_cor = judge_correctness(
                q["question"], response, q["key_facts"], client, JUDGE_MODEL
            )

            # 5. Per-fact attribution
            per_fact_attribution = []
            for fi, fact in enumerate(q["key_facts"]):
                in_ret = (ret_cov["per_fact"][fi]["present"]
                          if fi < len(ret_cov["per_fact"]) else False)
                in_gen = (gen_cor["per_fact"][fi]["present"]
                          if fi < len(gen_cor["per_fact"]) else False)

                if in_ret and in_gen:
                    attr = "covered"
                elif in_ret and not in_gen:
                    attr = "generation_miss"
                elif not in_ret and in_gen:
                    attr = "hallucinated"
                else:
                    attr = "retrieval_miss"

                per_fact_attribution.append({
                    "fact": fact,
                    "in_retrieval": in_ret,
                    "in_generation": in_gen,
                    "attribution": attr,
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
                "retrieved_chunk_ids": [c["chunk_id"] for c in chunks],
            })

            print(f"    Ret: {ret_cov['hits']}/{ret_cov['total']} ({ret_cov['score']:.3f})  "
                  f"Gen: {gen_cor['hits']}/{gen_cor['total']} ({gen_cor['score']:.3f})")

        all_results[config_name] = results

    # ── Aggregate & compare ─────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("AGGREGATE COMPARISON")
    print("=" * 70)

    header = (f"{'Config':<32} {'Ret Cov':>8} {'Gen Cov':>8} {'Gen|Ret':>8} "
              f"{'Covered':>8} {'GenMiss':>8} {'RetMiss':>8} {'Halluc':>7}")
    print(header)
    print("-" * len(header))

    config_aggregates = {}

    for config_name, results in all_results.items():
        total_facts = sum(r["retrieval_total"] for r in results)
        total_ret_hits = sum(r["retrieval_hits"] for r in results)
        total_gen_hits = sum(r["generation_hits"] for r in results)

        ret_cov = round(total_ret_hits / total_facts, 3) if total_facts else 0
        gen_cov = round(total_gen_hits / total_facts, 3) if total_facts else 0

        attr = {"covered": 0, "generation_miss": 0,
                "retrieval_miss": 0, "hallucinated": 0}
        for r in results:
            for pf in r["per_fact_attribution"]:
                attr[pf["attribution"]] += 1

        retrieved_facts = attr["covered"] + attr["generation_miss"]
        gen_given_ret = (round(attr["covered"] / retrieved_facts, 3)
                         if retrieved_facts else 0)

        config_aggregates[config_name] = {
            "total_facts": total_facts,
            "retrieval_coverage": ret_cov,
            "generation_coverage": gen_cov,
            "generation_coverage_given_retrieval": gen_given_ret,
            "attribution": attr,
        }

        print(f"{config_name:<32} {ret_cov:>8.3f} {gen_cov:>8.3f} {gen_given_ret:>8.3f} "
              f"{attr['covered']:>8} {attr['generation_miss']:>8} "
              f"{attr['retrieval_miss']:>8} {attr['hallucinated']:>7}")

    # ── Per-question deltas vs baseline ────────────────────────────────

    if "rerank" in all_results:
        baseline_results = {r["question_id"]: r for r in all_results["rerank"]}

        for config_name, results in all_results.items():
            if config_name == "rerank":
                continue

            print(f"\n--- {config_name} vs rerank (per-question deltas) ---")
            print(f"{'ID':<14} {'Topic':<22} {'dRet':>6} {'dGen':>6} {'Note'}")
            print("-" * 65)

            improved = 0
            regressed = 0

            for r in results:
                baseline = baseline_results.get(r["question_id"])
                if not baseline:
                    continue
                d_ret = r["retrieval_coverage"] - baseline["retrieval_coverage"]
                d_gen = r["generation_correctness"] - baseline["generation_correctness"]

                note = ""
                if d_ret > 0.001:
                    note += " ret+"
                    improved += 1
                elif d_ret < -0.001:
                    note += " ret-"
                    regressed += 1
                if d_gen > 0.001:
                    note += " gen+"
                elif d_gen < -0.001:
                    note += " gen-"

                if abs(d_ret) > 0.001 or abs(d_gen) > 0.001:
                    print(f"{r['question_id']:<14} {r['topic']:<22} "
                          f"{d_ret:>+6.3f} {d_gen:>+6.3f} {note}")

            print(f"  Improved: {improved}  Regressed: {regressed}")

    # ── Per-topic breakdown ───────────────────────────────────────────

    print("\n" + "=" * 70)
    print("PER-TOPIC BREAKDOWN (retrieval coverage by question category)")
    print("=" * 70)

    # Collect all topics across questions
    topics = sorted({r["topic"] for results in all_results.values()
                     for r in results if r["topic"]})

    config_names = list(all_results.keys())

    # Header
    topic_header = f"{'Topic':<25}"
    for cn in config_names:
        short = cn[:12]
        topic_header += f" {short:>13}"
    print(topic_header)
    print("-" * len(topic_header))

    # Per-topic aggregates stored for JSON output
    topic_aggregates = {}

    for topic in topics:
        row = f"{topic:<25}"
        topic_data = {}

        for cn in config_names:
            topic_results = [r for r in all_results[cn] if r["topic"] == topic]
            if not topic_results:
                row += f" {'—':>13}"
                continue

            t_facts = sum(r["retrieval_total"] for r in topic_results)
            t_ret = sum(r["retrieval_hits"] for r in topic_results)
            t_gen = sum(r["generation_hits"] for r in topic_results)

            ret_cov = round(t_ret / t_facts, 3) if t_facts else 0
            gen_cov = round(t_gen / t_facts, 3) if t_facts else 0

            # Count attributions for this topic
            t_attr = {"covered": 0, "generation_miss": 0,
                      "retrieval_miss": 0, "hallucinated": 0}
            for r in topic_results:
                for pf in r["per_fact_attribution"]:
                    t_attr[pf["attribution"]] += 1

            topic_data[cn] = {
                "retrieval_coverage": ret_cov,
                "generation_coverage": gen_cov,
                "facts": t_facts,
                "attribution": t_attr,
            }

            row += f" {ret_cov:>5.3f}/{gen_cov:.3f}"

        topic_aggregates[topic] = topic_data
        print(row)

    print(f"\n(Format: ret_coverage/gen_coverage per cell)")

    # Identify best retriever per topic
    print(f"\n--- Best retriever per topic (by retrieval coverage) ---")
    for topic in topics:
        best_config = None
        best_ret = -1
        for cn in config_names:
            td = topic_aggregates.get(topic, {}).get(cn)
            if td and td["retrieval_coverage"] > best_ret:
                best_ret = td["retrieval_coverage"]
                best_config = cn
        if best_config:
            baseline_td = topic_aggregates.get(topic, {}).get("rerank")
            baseline_ret = baseline_td["retrieval_coverage"] if baseline_td else 0
            delta = best_ret - baseline_ret
            marker = f" (+{delta:.3f} vs rerank)" if delta > 0.001 else " (= rerank)"
            print(f"  {topic:<25} {best_config}{marker}")

    # ── Save ──────────────────────────────────────────────────────────

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = {
        "config": {
            "gen_model": GEN_MODEL,
            "judge_model": JUDGE_MODEL,
            "top_k": TOP_K,
            "num_questions": len(questions),
            "configs_evaluated": [c["name"] for c in configs_to_run],
        },
        "aggregates": config_aggregates,
        "topic_aggregates": topic_aggregates,
        "results": all_results,
    }

    out_path = RESULTS_DIR / f"new_retrievers_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")

    return output


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate new retriever strategies")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Retrieval metrics only (MRR/hit rate, no API cost)")
    args = parser.parse_args()

    if args.retrieval_only:
        run_retrieval_only()
    else:
        run_full_eval()
