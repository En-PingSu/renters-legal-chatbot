"""
Multi-Run Averaging for Retrieval-Aware Correctness Scoring

Launches N parallel evaluation processes, collects result files,
and computes per-question and aggregate statistics (mean, std, 95% CI).

Usage:
    venv/bin/python3 -m src.evaluation.multi_run --n-runs 3
"""

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

from src.evaluation.scorer import RESULTS_DIR

PROJECT_ROOT = Path(__file__).parent.parent.parent
VENV_PYTHON = str(PROJECT_ROOT / "venv" / "bin" / "python3")


# ── Phase A: Parallel execution ──────────────────────────────────────────

def launch_parallel_runs(
    n_runs: int,
    model: str | None = None,
    use_all: bool = False,
) -> list[Path]:
    """Launch N retrieval_coverage processes in parallel, return result file paths."""
    # Snapshot existing result files before launching
    existing = set(RESULTS_DIR.glob("retrieval_coverage_*.json"))

    # Build command
    cmd = [VENV_PYTHON, "-m", "src.evaluation.retrieval_coverage"]
    if use_all:
        cmd.append("--all")
    if model:
        cmd.append(f"--model={model}")

    label = model or "openai/gpt-4o"
    q_label = "89 (all)" if use_all else "37 (stratified)"
    print(f"Launching {n_runs} parallel runs: model={label}, questions={q_label}")
    procs = []
    for i in range(n_runs):
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        procs.append(proc)
        print(f"  Run {i+1}/{n_runs} started (PID {proc.pid})")

    # Wait for all to complete
    for i, proc in enumerate(procs):
        retcode = proc.wait()
        status = "OK" if retcode == 0 else f"FAILED (exit {retcode})"
        print(f"  Run {i+1}/{n_runs} finished: {status}")
        if retcode != 0:
            stdout = proc.stdout.read().decode() if proc.stdout else ""
            print(f"    Output: {stdout[-500:]}")

    # Identify new result files
    all_files = set(RESULTS_DIR.glob("retrieval_coverage_*.json"))
    new_files = sorted(all_files - existing)

    if len(new_files) < n_runs:
        print(f"  WARNING: Expected {n_runs} new result files, found {len(new_files)}")

    print(f"  Found {len(new_files)} new result files")
    for f in new_files:
        print(f"    {f.name}")

    return new_files


# ── Phase B: Per-question aggregation ────────────────────────────────────

def _metric_stats(values: list[float]) -> dict:
    """Compute mean, std, 95% CI for a list of values."""
    n = len(values)
    m = mean(values)
    s = stdev(values) if n > 1 else 0.0
    ci = 1.96 * s / sqrt(n) if n > 1 else 0.0
    return {
        "mean": round(m, 4),
        "std": round(s, 4),
        "ci_95_low": round(m - ci, 4),
        "ci_95_high": round(m + ci, 4),
        "values": [round(v, 4) for v in values],
    }


def aggregate_results(run_data: list[dict]) -> dict:
    """Aggregate N run results into per-question and overall statistics."""
    n_runs = len(run_data)

    # Group per-question results by question_id
    by_question: dict[str, list[dict]] = defaultdict(list)
    for run in run_data:
        for r in run["results"]:
            by_question[r["question_id"]].append(r)

    # Per-question statistics
    per_question = []
    for qid, runs in sorted(by_question.items()):
        q_info = runs[0]  # metadata from first run

        ret_covs = [r["retrieval_coverage"] for r in runs]
        gen_cors = [r["generation_correctness"] for r in runs]
        faiths = [r.get("faithfulness", -1) for r in runs]
        relevs = [r.get("relevancy", -1) for r in runs]

        # Chunk-level metrics
        chunk_metric_names = ["mrr", "hit_rate", "recall_at_k", "ndcg_at_k"]
        chunk_stats = {}
        for cm in chunk_metric_names:
            vals = [r["chunk_metrics"][cm] for r in runs]
            chunk_stats[cm] = _metric_stats(vals)

        # Per-fact attribution: majority vote across runs
        n_facts = len(runs[0].get("per_fact_attribution", []))
        fact_attributions = []
        for fi in range(n_facts):
            votes = []
            for r in runs:
                if fi < len(r.get("per_fact_attribution", [])):
                    votes.append(r["per_fact_attribution"][fi]["attribution"])
            majority = Counter(votes).most_common(1)[0][0] if votes else "unknown"
            fact_text = runs[0]["per_fact_attribution"][fi]["fact"] if fi < n_facts else ""
            fact_attributions.append({
                "fact": fact_text,
                "majority_vote": majority,
                "votes": votes,
            })

        ret_stats = _metric_stats(ret_covs)
        gen_stats = _metric_stats(gen_cors)
        faith_valid = [f for f in faiths if f >= 0]
        relev_valid = [r for r in relevs if r >= 0]
        faith_stats = _metric_stats(faith_valid) if faith_valid else None
        relev_stats = _metric_stats(relev_valid) if relev_valid else None

        # Flag unstable questions
        unstable = ret_stats["std"] > 0.1 or gen_stats["std"] > 0.1

        q_entry = {
            "question_id": qid,
            "question": q_info["question"],
            "topic": q_info.get("topic", ""),
            "difficulty": q_info.get("difficulty", "standard"),
            "retrieval_coverage": ret_stats,
            "generation_correctness": gen_stats,
            "chunk_metrics": chunk_stats,
            "per_fact_attribution": fact_attributions,
            "unstable": unstable,
        }
        if faith_stats:
            q_entry["faithfulness"] = faith_stats
        if relev_stats:
            q_entry["relevancy"] = relev_stats

        per_question.append(q_entry)

    # Aggregate-level statistics (across runs, not across questions)
    agg_metrics = {}
    for metric_key in ["retrieval_coverage", "generation_coverage",
                       "generation_coverage_given_retrieval",
                       "faithfulness", "relevancy"]:
        vals = [run["aggregate"][metric_key] for run in run_data]
        agg_metrics[metric_key] = _metric_stats(vals)

    for cm in ["mrr", "hit_rate", "recall_at_k", "ndcg_at_k"]:
        vals = [run["aggregate"]["chunk_metrics"][cm] for run in run_data
                if cm in run["aggregate"].get("chunk_metrics", {})]
        if vals:
            agg_metrics[cm] = _metric_stats(vals)

    # Attribution counts across runs
    attr_keys = ["covered", "generation_miss", "retrieval_miss", "hallucinated"]
    attr_stats = {}
    for ak in attr_keys:
        vals = [float(run["aggregate"]["attribution"][ak]) for run in run_data]
        attr_stats[ak] = _metric_stats(vals)
    agg_metrics["attribution"] = attr_stats

    return {
        "aggregate": agg_metrics,
        "per_question": per_question,
        "n_runs": n_runs,
    }


# ── Phase D: Output ──────────────────────────────────────────────────────

def format_markdown(stats: dict, run_files: list[str]) -> str:
    """Format statistics as markdown tables."""
    lines = [
        "# Multi-Run Averaging Results",
        "",
        f"**Runs:** {stats['n_runs']}  ",
        f"**Result files:** {', '.join(run_files)}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Mean | Std | 95% CI |",
        "|--------|------|-----|--------|",
    ]

    agg = stats["aggregate"]
    display_order = [
        "retrieval_coverage", "generation_coverage",
        "generation_coverage_given_retrieval",
        "mrr", "hit_rate", "recall_at_k", "ndcg_at_k",
    ]
    for key in display_order:
        if key in agg:
            m = agg[key]
            lines.append(
                f"| {key} | {m['mean']:.4f} | {m['std']:.4f} | "
                f"[{m['ci_95_low']:.4f}, {m['ci_95_high']:.4f}] |"
            )

    # Attribution
    lines.extend(["", "### Attribution Counts (mean across runs)", ""])
    lines.append("| Attribution | Mean | Std |")
    lines.append("|-------------|------|-----|")
    for ak in ["covered", "generation_miss", "retrieval_miss", "hallucinated"]:
        m = agg["attribution"][ak]
        lines.append(f"| {ak} | {m['mean']:.1f} | {m['std']:.1f} |")

    # Per-question table (sorted by max std descending)
    lines.extend(["", "## Per-Question Variance", ""])
    lines.append(
        "| Question ID | Topic | Diff | Ret Cov (mean/std) | Gen Cor (mean/std) | Unstable |"
    )
    lines.append(
        "|-------------|-------|------|--------------------|--------------------|----------|"
    )

    pq_sorted = sorted(
        stats["per_question"],
        key=lambda q: max(
            q["retrieval_coverage"]["std"],
            q["generation_correctness"]["std"],
        ),
        reverse=True,
    )
    for q in pq_sorted:
        rc = q["retrieval_coverage"]
        gc = q["generation_correctness"]
        flag = "YES" if q["unstable"] else ""
        lines.append(
            f"| {q['question_id']} | {q['topic'][:18]} | "
            f"{q['difficulty'][:4]} | "
            f"{rc['mean']:.3f} / {rc['std']:.3f} | "
            f"{gc['mean']:.3f} / {gc['std']:.3f} | {flag} |"
        )

    lines.append("")
    return "\n".join(lines)


def save_outputs(stats: dict, run_files: list[Path]) -> tuple[Path, Path]:
    """Save JSON and markdown output files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    file_names = [f.name for f in run_files]

    # JSON
    output = {
        "config": {
            "n_runs": stats["n_runs"],
            "eval_type": "retrieval_coverage",
        },
        "run_files": file_names,
        **stats,
    }
    json_path = RESULTS_DIR / f"multi_run_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Markdown
    md_content = format_markdown(stats, file_names)
    md_path = RESULTS_DIR / f"multi_run_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return json_path, md_path


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run retrieval_coverage evaluation N times and compute statistics"
    )
    parser.add_argument(
        "--n-runs", type=int, default=3,
        help="Number of parallel evaluation runs (default: 3)"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Generator model (default: openai/gpt-4o)"
    )
    parser.add_argument(
        "--all", action="store_true", dest="use_all",
        help="Use all 89 questions instead of 37 stratified subset"
    )
    args = parser.parse_args()

    n_runs = args.n_runs
    print(f"\n{'='*60}")
    print(f"Multi-Run Averaging ({n_runs} runs)")
    print(f"{'='*60}\n")

    # Phase A: Launch parallel runs
    run_files = launch_parallel_runs(n_runs, model=args.model, use_all=args.use_all)

    if not run_files:
        print("ERROR: No result files produced. Exiting.")
        sys.exit(1)

    # Load results
    print(f"\nLoading {len(run_files)} result files...")
    run_data = []
    for f in run_files:
        with open(f, "r", encoding="utf-8") as fh:
            run_data.append(json.load(fh))

    # Phase B+C: Aggregate
    print("Computing statistics...")
    stats = aggregate_results(run_data)

    # Phase D: Save
    json_path, md_path = save_outputs(stats, run_files)

    # Print summary
    print(f"\n{'='*60}")
    print("MULTI-RUN SUMMARY")
    print(f"{'='*60}")
    agg = stats["aggregate"]
    for key in ["retrieval_coverage", "generation_coverage",
                "generation_coverage_given_retrieval",
                "faithfulness", "relevancy"]:
        if key in agg:
            m = agg[key]
            print(f"  {key}: {m['mean']:.4f} +/- {m['std']:.4f} "
                  f"(95% CI: [{m['ci_95_low']:.4f}, {m['ci_95_high']:.4f}])")

    unstable = [q for q in stats["per_question"] if q["unstable"]]
    print(f"\n  Unstable questions (std > 0.1): {len(unstable)}/{len(stats['per_question'])}")
    for q in unstable:
        print(f"    {q['question_id']}: ret={q['retrieval_coverage']['std']:.3f} "
              f"gen={q['generation_correctness']['std']:.3f}")

    print(f"\nSaved to:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()
