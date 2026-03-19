"""
Evaluation scorer for the RAG chatbot.
Runs all evaluation questions through baseline and RAG configurations,
then scores on Faithfulness and Relevancy (matching HW4 methodology).
Also computes retrieval metrics (MRR, Hit Rate, Precision, Recall) for RAG.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

def _get_ask_fn():
    backend = os.getenv("RAG_BACKEND", "custom")
    if backend == "llamaindex":
        from src.rag_llamaindex.pipeline import ask
    else:
        from src.rag.pipeline import ask
    return ask


def _get_retriever_registry():
    backend = os.getenv("RAG_BACKEND", "custom")
    if backend == "llamaindex":
        from src.rag_llamaindex.retrievers import RETRIEVER_REGISTRY
    else:
        from src.rag.retrievers import RETRIEVER_REGISTRY
    return RETRIEVER_REGISTRY


ask = _get_ask_fn()
RETRIEVER_REGISTRY = _get_retriever_registry()
from src.scraping.utils import PROJECT_ROOT

load_dotenv()

EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "all_chunks.json"

# --- LLM Judge Prompts (binary pass/fail, matching LlamaIndex evaluators) ---

FAITHFULNESS_PROMPT = """You are evaluating whether a response is faithful to the provided source context.

A response is FAITHFUL if every claim it makes is supported by the source context. It should not contain information that contradicts or goes beyond what the sources provide. If no source context is provided (baseline mode), evaluate whether the response avoids fabricating specific statutes, case numbers, or URLs that don't exist.

QUESTION: {question}

SOURCE CONTEXT:
{context}

RESPONSE: {response}

Is this response faithful to the source context? Answer with ONLY "YES" or "NO", followed by a brief explanation.
Format: YES/NO: <explanation>"""

RELEVANCY_PROMPT = """You are evaluating whether a response is relevant to the question asked.

A response is RELEVANT if it directly addresses the question and provides useful information toward answering it. A response that declines to answer but appropriately explains why and suggests resources is still relevant. A response is NOT relevant if it discusses unrelated topics or fails to address the core question.

QUESTION: {question}

RESPONSE: {response}

Is this response relevant to the question? Answer with ONLY "YES" or "NO", followed by a brief explanation.
Format: YES/NO: <explanation>"""

CORRECTNESS_PROMPT = """You are evaluating whether a response contains the correct key facts.

QUESTION: {question}

EXPECTED KEY FACTS:
{key_facts}

RESPONSE: {response}

Does the response contain the key facts listed above? Answer YES or NO, then explain which facts are present/missing.
Format: YES/NO: <explanation>"""


def get_openrouter_client() -> OpenAI:
    """Get OpenRouter client."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set. Add it to .env file.")
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


def load_eval_questions() -> list[dict]:
    """Load evaluation questions from Reddit + Golden QA test sets."""
    questions = []

    # Reddit questions (may include enriched golden answers)
    reddit_path = EVAL_DIR / "reddit_questions.json"
    if reddit_path.exists():
        with open(reddit_path, "r") as f:
            reddit = json.load(f)
        for item in reddit:
            questions.append({
                "id": f"reddit_{item['id']}",
                "source": "reddit",
                "question": item["question"],
                "context": item.get("context", ""),
                "expected_answer": item.get("expected_answer", ""),
                "key_facts": item.get("key_facts", []),
            })

    # Golden QA pairs (with expected answers and key facts)
    golden_path = EVAL_DIR / "golden_qa.json"
    if golden_path.exists():
        with open(golden_path, "r") as f:
            golden = json.load(f)
        for item in golden:
            questions.append({
                "id": item["id"],
                "source": "golden",
                "question": item["question"],
                "context": "",
                "expected_answer": item.get("expected_answer", ""),
                "key_facts": item.get("key_facts", []),
            })

    return questions


def generate_all_responses(
    questions: list[dict],
    configurations: list[dict] | None = None,
) -> list[dict]:
    """Generate responses for all questions under each configuration."""
    if configurations is None:
        configurations = [
            # --- Retriever comparison (GPT-4o fixed) ---
            {"name": "baseline",         "use_rag": False, "model": "openai/gpt-4o",                            "retriever": "none"},
            {"name": "rag_vector",       "use_rag": True,  "model": "openai/gpt-4o",                            "retriever": "vector"},
            {"name": "rag_bm25",         "use_rag": True,  "model": "openai/gpt-4o",                            "retriever": "bm25"},
            {"name": "rag_hybrid",       "use_rag": True,  "model": "openai/gpt-4o",                            "retriever": "hybrid"},
            {"name": "rag_rerank",       "use_rag": True,  "model": "openai/gpt-4o",                            "retriever": "rerank"},
            # --- Model comparison (rerank retriever fixed) ---
            {"name": "llama_baseline",   "use_rag": False, "model": "meta-llama/llama-3.3-70b-instruct",        "retriever": "none"},
            {"name": "llama_rerank",     "use_rag": True,  "model": "meta-llama/llama-3.3-70b-instruct",        "retriever": "rerank"},
            {"name": "mistral_baseline", "use_rag": False, "model": "mistralai/mistral-small-3.1-24b-instruct", "retriever": "none"},
            {"name": "mistral_rerank",   "use_rag": True,  "model": "mistralai/mistral-small-3.1-24b-instruct", "retriever": "rerank"},
        ]

    results = []
    for config in configurations:
        print(f"\n--- Generating {config['name']} responses ---")
        for i, q in enumerate(questions):
            print(f"  [{i+1}/{len(questions)}] {q['question'][:60]}...")
            result = ask(
                question=q["question"],
                model=config["model"],
                use_rag=config["use_rag"],
                retriever=config.get("retriever", "vector"),
            )
            # Build context string from retrieved chunks for faithfulness eval
            context_str = ""
            if result.get("retrieved_chunks"):
                context_parts = []
                for chunk in result["retrieved_chunks"]:
                    meta = chunk["metadata"]
                    context_parts.append(
                        f"[{meta['title']} ({meta['source_url']})]\n{chunk['content']}"
                    )
                context_str = "\n\n---\n\n".join(context_parts)

            results.append({
                "question_id": q["id"],
                "question": q["question"],
                "configuration": config["name"],
                "model": config["model"],
                "use_rag": config["use_rag"],
                "response": result["response"],
                "num_chunks_retrieved": len(result.get("retrieved_chunks", [])),
                "retrieved_context": context_str,
            })
    return results


def judge_faithfulness(question: str, response: str, context: str,
                       client: OpenAI, judge_model: str) -> dict:
    """Binary faithfulness evaluation (1.0 = pass, 0.0 = fail)."""
    if not context:
        context = "(No source context provided — baseline mode)"

    prompt = FAITHFULNESS_PROMPT.format(
        question=question, response=response, context=context
    )
    try:
        result = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        text = result.choices[0].message.content.strip()
        passing = text.upper().startswith("YES")
        return {"score": 1.0 if passing else 0.0, "reasoning": text}
    except Exception as e:
        print(f"  [ERROR] Faithfulness judge failed: {e}")
        return {"score": -1.0, "reasoning": str(e)}


def judge_relevancy(question: str, response: str,
                    client: OpenAI, judge_model: str) -> dict:
    """Binary relevancy evaluation (1.0 = pass, 0.0 = fail)."""
    prompt = RELEVANCY_PROMPT.format(question=question, response=response)
    try:
        result = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        text = result.choices[0].message.content.strip()
        passing = text.upper().startswith("YES")
        return {"score": 1.0 if passing else 0.0, "reasoning": text}
    except Exception as e:
        print(f"  [ERROR] Relevancy judge failed: {e}")
        return {"score": -1.0, "reasoning": str(e)}


def judge_correctness(question: str, response: str, key_facts: list[str],
                      client: OpenAI, judge_model: str) -> dict:
    """Binary correctness evaluation against expected key facts (1.0 = pass, 0.0 = fail)."""
    if not key_facts:
        return {"score": -1.0, "reasoning": "No key facts to evaluate against"}

    facts_str = "\n".join(f"- {f}" for f in key_facts)
    prompt = CORRECTNESS_PROMPT.format(
        question=question, response=response, key_facts=facts_str
    )
    try:
        result = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        text = result.choices[0].message.content.strip()
        passing = text.upper().startswith("YES")
        return {"score": 1.0 if passing else 0.0, "reasoning": text}
    except Exception as e:
        print(f"  [ERROR] Correctness judge failed: {e}")
        return {"score": -1.0, "reasoning": str(e)}


def score_all(results: list[dict], questions: list[dict] | None = None,
              judge_model: str = "anthropic/claude-sonnet-4") -> list[dict]:
    """Score all responses on Faithfulness, Relevancy, and Correctness (when available)."""
    client = get_openrouter_client()
    scored = []

    # Build lookup for key_facts by question_id
    key_facts_map: dict[str, list[str]] = {}
    if questions:
        for q in questions:
            if q.get("key_facts"):
                key_facts_map[q["id"]] = q["key_facts"]

    for i, r in enumerate(results):
        print(f"  Scoring [{i+1}/{len(results)}] {r['configuration']}: {r['question'][:40]}...")

        faith = judge_faithfulness(
            r["question"], r["response"], r["retrieved_context"],
            client, judge_model
        )
        relev = judge_relevancy(
            r["question"], r["response"],
            client, judge_model
        )

        scores = {
            "faithfulness": faith["score"],
            "relevancy": relev["score"],
            "faithfulness_reasoning": faith["reasoning"],
            "relevancy_reasoning": relev["reasoning"],
        }

        # Correctness scoring for golden QA pairs with key_facts
        key_facts = key_facts_map.get(r["question_id"], [])
        if key_facts:
            correct = judge_correctness(
                r["question"], r["response"], key_facts,
                client, judge_model
            )
            scores["correctness"] = correct["score"]
            scores["correctness_reasoning"] = correct["reasoning"]

        scored.append({**r, "scores": scores})
    return scored


# --- Retrieval Metrics ---

def generate_qa_pairs_from_chunks(
    num_pairs: int = 50,
    model: str = "openai/gpt-4o",
) -> list[dict]:
    """Auto-generate QA pairs from chunks for retrieval evaluation.

    Each pair has a question and the chunk_id of the source chunk,
    enabling MRR/Hit Rate computation.
    """
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    client = get_openrouter_client()
    qa_pairs = []

    # Sample chunks evenly
    step = max(1, len(chunks) // num_pairs)
    sampled = chunks[::step][:num_pairs]

    print(f"\n--- Generating {len(sampled)} QA pairs from chunks ---")
    for i, chunk in enumerate(sampled):
        print(f"  [{i+1}/{len(sampled)}] Generating question from: {chunk['title'][:40]}...")
        try:
            result = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": (
                    "Based on the following text about Massachusetts tenant law, "
                    "generate ONE specific question that can be answered using this text. "
                    "Return ONLY the question, nothing else.\n\n"
                    f"TEXT: {chunk['content'][:1000]}"
                )}],
                temperature=0.5,
                max_tokens=100,
            )
            question = result.choices[0].message.content.strip()
            qa_pairs.append({
                "question": question,
                "ground_truth_chunk_id": chunk["chunk_id"],
                "source_title": chunk["title"],
            })
        except Exception as e:
            print(f"  [ERROR] QA generation failed: {e}")

    return qa_pairs


def compute_retrieval_metrics(
    qa_pairs: list[dict], top_k: int = 5, retriever_name: str = "vector"
) -> dict:
    """Compute MRR, Hit Rate, Precision, Recall for a given retriever."""
    retrieve_fn = RETRIEVER_REGISTRY.get(retriever_name)
    if retrieve_fn is None:
        raise ValueError(f"Unknown retriever: {retriever_name}")

    print(f"\n--- Computing retrieval metrics [{retriever_name}] "
          f"({len(qa_pairs)} QA pairs, top_k={top_k}) ---")

    mrr_total = 0.0
    hits = 0
    total = len(qa_pairs)

    for i, pair in enumerate(qa_pairs):
        print(f"  [{i+1}/{total}] Retrieving for: {pair['question'][:50]}...")
        chunks = retrieve_fn(pair["question"], top_k=top_k)
        retrieved_ids = [c["chunk_id"] for c in chunks]
        gt_id = pair["ground_truth_chunk_id"]

        if gt_id in retrieved_ids:
            hits += 1
            rank = retrieved_ids.index(gt_id) + 1
            mrr_total += 1.0 / rank

    metrics = {
        "mrr": round(mrr_total / total, 4) if total > 0 else 0,
        "hit_rate": round(hits / total, 4) if total > 0 else 0,
        "precision": round(hits / (total * top_k), 4) if total > 0 else 0,
        "recall": round(hits / total, 4) if total > 0 else 0,
        "num_queries": total,
        "top_k": top_k,
    }

    print(f"\n  MRR={metrics['mrr']:.3f}  Hit Rate={metrics['hit_rate']:.3f}  "
          f"Precision={metrics['precision']:.3f}  Recall={metrics['recall']:.3f}")
    return metrics


def compute_summary(scored_results: list[dict]) -> dict:
    """Compute aggregate statistics for Faithfulness, Relevancy, and Correctness."""
    summary = {}
    configs = set(r["configuration"] for r in scored_results)

    for config in sorted(configs):
        config_results = [r for r in scored_results if r["configuration"] == config]
        valid = [r for r in config_results if r["scores"]["faithfulness"] >= 0]

        if not valid:
            summary[config] = {"count": len(config_results), "note": "no valid scores"}
            continue

        stats = {
            "count": len(valid),
            "faithfulness_mean": round(
                sum(r["scores"]["faithfulness"] for r in valid) / len(valid), 3
            ),
            "relevancy_mean": round(
                sum(r["scores"]["relevancy"] for r in valid) / len(valid), 3
            ),
        }

        # Correctness (only for golden QA pairs that have key_facts)
        correctness_results = [
            r for r in valid if r["scores"].get("correctness", -1) >= 0
        ]
        if correctness_results:
            stats["correctness_mean"] = round(
                sum(r["scores"]["correctness"] for r in correctness_results)
                / len(correctness_results), 3
            )
            stats["correctness_count"] = len(correctness_results)

        summary[config] = stats

    return summary


def run(judge_model: str = "anthropic/claude-sonnet-4", run_retrieval_metrics: bool = True):
    """Full evaluation pipeline."""
    print("=" * 60)
    print("Evaluation Pipeline (Faithfulness + Relevancy + Correctness)")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_eval_questions()
    print(f"Loaded {len(questions)} evaluation questions")
    golden_count = sum(1 for q in questions if q.get("source") == "golden")
    if golden_count:
        print(f"  ({golden_count} golden QA pairs with expected answers)")

    if not questions:
        print("No evaluation questions found. Run generate_golden_qa.py first.")
        return

    # Generate responses
    results = generate_all_responses(questions)

    # Score with LLM judge
    print("\n--- Scoring responses (Faithfulness + Relevancy + Correctness) ---")
    scored = score_all(results, questions=questions, judge_model=judge_model)

    # Summary
    summary = compute_summary(scored)
    print("\n--- Results Summary ---")
    for config, stats in summary.items():
        print(f"\n{config}:")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")

    # Retrieval metrics (all retrievers)
    retrieval_metrics = None
    if run_retrieval_metrics:
        # Combine golden + enriched reddit QA pairs for retrieval metrics
        qa_pairs = []
        golden_path = EVAL_DIR / "golden_qa.json"
        if golden_path.exists():
            with open(golden_path, "r", encoding="utf-8") as f:
                golden = json.load(f)
            qa_pairs.extend([
                {
                    "question": g["question"],
                    "ground_truth_chunk_id": g["source_chunks"][0]["chunk_id"],
                    "source_title": g["source_chunks"][0]["title"],
                }
                for g in golden if g.get("source_chunks")
            ])
        reddit_path = EVAL_DIR / "reddit_questions.json"
        if reddit_path.exists():
            with open(reddit_path, "r", encoding="utf-8") as f:
                reddit = json.load(f)
            qa_pairs.extend([
                {
                    "question": r["question"],
                    "ground_truth_chunk_id": r["source_chunks"][0]["chunk_id"],
                    "source_title": r["source_chunks"][0]["title"],
                }
                for r in reddit if r.get("source_chunks")
            ])
        if qa_pairs:
            print(f"\n  Using {len(qa_pairs)} QA pairs for retrieval metrics")
        else:
            # Fallback to auto-generated QA pairs
            qa_path = RESULTS_DIR / "retrieval_qa_pairs.json"
            if qa_path.exists():
                with open(qa_path, "r", encoding="utf-8") as f:
                    qa_pairs = json.load(f)
                print(f"\n  Reusing {len(qa_pairs)} saved QA pairs from {qa_path}")
            else:
                qa_pairs = generate_qa_pairs_from_chunks(num_pairs=50, model="openai/gpt-4o")
                with open(qa_path, "w", encoding="utf-8") as f:
                    json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
                print(f"  Saved QA pairs to {qa_path}")

        retrieval_metrics = {}
        for retriever_name in ["vector", "bm25", "hybrid", "rerank"]:
            try:
                metrics = compute_retrieval_metrics(
                    qa_pairs, top_k=5, retriever_name=retriever_name
                )
                retrieval_metrics[retriever_name] = metrics
            except ImportError as e:
                print(f"  [SKIP] {retriever_name}: {e}")
            except Exception as e:
                print(f"  [ERROR] {retriever_name}: {e}")
                retrieval_metrics[retriever_name] = {"error": str(e)}

    # Save results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"eval_{timestamp}.json"
    output = {
        "summary": summary,
        "retrieval_metrics": retrieval_metrics,
        "judge_model": judge_model,
        "results": scored,
    }
    # Remove retrieved_context from saved results to reduce file size
    for r in output["results"]:
        r.pop("retrieved_context", None)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {results_path}")

    return summary, retrieval_metrics


if __name__ == "__main__":
    run()
