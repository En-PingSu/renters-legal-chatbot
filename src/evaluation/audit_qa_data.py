"""
Audit QA data files against all_chunks.json.

Checks:
1. Chunk ID existence — every source_chunks[].chunk_id exists in all_chunks.json
2. Metadata match — title and source_url in QA references match the actual chunk
3. Duplicate detection — no duplicate chunk_ids within a single QA entry, no duplicate questions
4. Chunk-question relevance — token containment ratio between question and chunk content
5. Key fact grounding — token containment ratio between key_facts and chunk content

Run: venv/bin/python3 -m src.evaluation.audit_qa_data
"""

import json
import re
from pathlib import Path
from collections import Counter
from datetime import datetime


DATA_DIR = Path("data")
CHUNKS_PATH = DATA_DIR / "chunks" / "all_chunks.json"
GOLDEN_QA_PATH = DATA_DIR / "evaluation" / "golden_qa.json"
REDDIT_QA_PATH = DATA_DIR / "evaluation" / "reddit_questions.json"
OUTPUT_PATH = DATA_DIR / "evaluation" / "results" / "audit_report.json"

# Thresholds
CHUNK_RELEVANCE_THRESHOLD = 0.10
KEY_FACT_GROUNDING_THRESHOLD = 0.15


def tokenize(text: str) -> set[str]:
    """Simple whitespace + punctuation tokenizer, lowercased, stopwords removed."""
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "just", "or", "and", "but", "if", "it",
        "its", "i", "me", "my", "we", "our", "you", "your", "he", "him",
        "his", "she", "her", "they", "them", "their", "what", "which", "who",
        "whom", "this", "that", "these", "those", "am", "about", "up",
    }
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return tokens - stopwords


def token_containment(query_text: str, reference_text: str) -> float:
    """Fraction of query tokens found in reference text."""
    query_tokens = tokenize(query_text)
    if not query_tokens:
        return 0.0
    ref_tokens = tokenize(reference_text)
    overlap = query_tokens & ref_tokens
    return len(overlap) / len(query_tokens)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def build_chunk_index(chunks: list[dict]) -> dict[str, dict]:
    return {c["chunk_id"]: c for c in chunks}


def audit_entry(entry: dict, chunk_index: dict, source_file: str) -> dict:
    """Audit a single QA entry. Returns dict of issues found."""
    entry_id = entry.get("id", "unknown")
    issues = []

    source_chunks = entry.get("source_chunks", [])
    question = entry.get("question", "")
    key_facts = entry.get("key_facts", [])

    # --- Check 1: Chunk ID existence ---
    for sc in source_chunks:
        cid = sc.get("chunk_id", "")
        if cid not in chunk_index:
            issues.append({
                "check": "chunk_id_existence",
                "severity": "error",
                "chunk_id": cid,
                "message": f"Chunk ID '{cid}' not found in all_chunks.json",
            })

    # --- Check 2: Metadata match ---
    for sc in source_chunks:
        cid = sc.get("chunk_id", "")
        if cid not in chunk_index:
            continue
        actual = chunk_index[cid]
        qa_title = sc.get("title", "")
        qa_url = sc.get("source_url", "")

        if qa_title and qa_title != actual["title"]:
            issues.append({
                "check": "metadata_title_mismatch",
                "severity": "warning",
                "chunk_id": cid,
                "message": f"Title mismatch: QA='{qa_title}' vs chunk='{actual['title']}'",
            })
        if qa_url and qa_url != actual["source_url"]:
            issues.append({
                "check": "metadata_url_mismatch",
                "severity": "warning",
                "chunk_id": cid,
                "message": f"URL mismatch: QA='{qa_url}' vs chunk='{actual['source_url']}'",
            })

    # --- Check 3: Duplicate chunk_ids within entry ---
    chunk_ids = [sc.get("chunk_id", "") for sc in source_chunks]
    id_counts = Counter(chunk_ids)
    for cid, count in id_counts.items():
        if count > 1:
            issues.append({
                "check": "duplicate_chunk_in_entry",
                "severity": "warning",
                "chunk_id": cid,
                "message": f"Chunk ID '{cid}' appears {count} times in this entry",
            })

    # --- Check 4: Chunk-question relevance ---
    for sc in source_chunks:
        cid = sc.get("chunk_id", "")
        if cid not in chunk_index:
            continue
        content = chunk_index[cid]["content"]
        ratio = token_containment(question, content)
        if ratio < CHUNK_RELEVANCE_THRESHOLD:
            issues.append({
                "check": "low_chunk_relevance",
                "severity": "warning",
                "chunk_id": cid,
                "ratio": round(ratio, 3),
                "message": f"Low question-chunk overlap ({ratio:.3f} < {CHUNK_RELEVANCE_THRESHOLD})",
            })

    # --- Check 5: Key fact grounding ---
    # Combine all referenced chunk content
    combined_content = " ".join(
        chunk_index[sc["chunk_id"]]["content"]
        for sc in source_chunks
        if sc.get("chunk_id", "") in chunk_index
    )
    for i, fact in enumerate(key_facts):
        # Strip source references like "(Source 1)" for cleaner matching
        clean_fact = re.sub(r"\(Source\s*\d+[^)]*\)", "", fact).strip()
        ratio = token_containment(clean_fact, combined_content)
        if ratio < KEY_FACT_GROUNDING_THRESHOLD:
            issues.append({
                "check": "low_fact_grounding",
                "severity": "warning",
                "fact_index": i,
                "fact": fact,
                "ratio": round(ratio, 3),
                "message": f"Key fact poorly grounded ({ratio:.3f} < {KEY_FACT_GROUNDING_THRESHOLD})",
            })

    return {
        "entry_id": entry_id,
        "source_file": source_file,
        "num_issues": len(issues),
        "issues": issues,
    }


def check_duplicate_questions(golden: list[dict], reddit: list[dict]) -> list[dict]:
    """Check for duplicate questions across all entries."""
    issues = []
    seen = {}
    for entry in golden + reddit:
        q = entry.get("question", "").strip().lower()
        eid = entry.get("id", "unknown")
        if q in seen:
            issues.append({
                "check": "duplicate_question",
                "severity": "warning",
                "entries": [seen[q], eid],
                "message": f"Duplicate question found in {seen[q]} and {eid}",
            })
        else:
            seen[q] = eid
    return issues


def run():
    print("Loading data files...")
    chunks = load_json(CHUNKS_PATH)
    golden = load_json(GOLDEN_QA_PATH)
    reddit = load_json(REDDIT_QA_PATH)

    chunk_index = build_chunk_index(chunks)
    print(f"  Chunks: {len(chunks)}, Golden QA: {len(golden)}, Reddit Q: {len(reddit)}")

    # Audit each entry
    entry_results = []
    for entry in golden:
        entry_results.append(audit_entry(entry, chunk_index, "golden_qa.json"))
    for entry in reddit:
        entry_results.append(audit_entry(entry, chunk_index, "reddit_questions.json"))

    # Cross-file duplicate check
    dup_issues = check_duplicate_questions(golden, reddit)

    # Summary
    total_issues = sum(r["num_issues"] for r in entry_results) + len(dup_issues)
    entries_with_issues = sum(1 for r in entry_results if r["num_issues"] > 0)

    # Count by check type
    check_counts = Counter()
    for r in entry_results:
        for issue in r["issues"]:
            check_counts[issue["check"]] += 1
    for issue in dup_issues:
        check_counts[issue["check"]] += 1

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_entries": len(golden) + len(reddit),
            "entries_with_issues": entries_with_issues,
            "total_issues": total_issues,
            "issues_by_check": dict(check_counts),
        },
        "duplicate_questions": dup_issues,
        "entry_results": entry_results,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f"Total entries audited: {len(golden) + len(reddit)}")
    print(f"Entries with issues:   {entries_with_issues}")
    print(f"Total issues found:    {total_issues}")
    if check_counts:
        print(f"\nIssues by type:")
        for check, count in sorted(check_counts.items()):
            print(f"  {check}: {count}")

    # Print details for entries with issues
    if entries_with_issues > 0:
        print(f"\n{'='*60}")
        print("DETAILS")
        print(f"{'='*60}")
        for r in entry_results:
            if r["num_issues"] > 0:
                print(f"\n[{r['entry_id']}] ({r['source_file']}) — {r['num_issues']} issue(s):")
                for issue in r["issues"]:
                    severity = issue["severity"].upper()
                    print(f"  [{severity}] {issue['message']}")

    if dup_issues:
        print(f"\nDuplicate questions:")
        for issue in dup_issues:
            print(f"  {issue['message']}")

    # Save report
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {OUTPUT_PATH}")

    return report


if __name__ == "__main__":
    run()
