"""
Generate golden QA pairs from the corpus for evaluation.

Reads all chunks, groups by topic via keyword matching,
then uses GPT-4o to generate question-answer pairs with source citations.

Usage:
    python -m src.evaluation.generate_golden_qa
"""

import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.scraping.utils import PROJECT_ROOT

load_dotenv()

CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "all_chunks.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "golden_qa.json"

# --- Topic definitions: keywords + quota ---

TOPIC_QUOTAS = {
    "security_deposits":     6,
    "retaliation":           3,   # before evictions so anti-retaliation chunks match here
    "evictions":             6,
    "repairs_habitability":  6,
    "rent_increases":        4,
    "landlord_entry":        3,
    "discrimination":        4,
    "lease_terms":           4,
    "lead_paint":            3,
    "utilities_heat":        3,
    "public_housing":        4,
    "tenant_rights_general": 4,
}

TOPIC_KEYWORDS = {
    "security_deposits": [
        "security deposit", "15B", "last month", "s.15B",
        "tenants' security deposits",
    ],
    "retaliation": [
        "retaliation", "retaliatory", "anti-retaliation",
        "anti-reprisal", "reprisal", "186 s.18",
    ],
    "evictions": [
        "eviction", "notice to quit", "summary process", "c.239",
        "evict", "s.1A", "possession of land",
    ],
    "repairs_habitability": [
        "sanitary code", "105 CMR 410", "habitability", "repair",
        "housing code", "code violation", "fitness for human",
        "inspectional services",
    ],
    "rent_increases": [
        "rent increase", "rent control", "late fee", "rent stabilization",
        "c.40P", "940 CMR 3.17",
    ],
    "landlord_entry": [
        "quiet enjoyment", "landlord entry", "s.14", "privacy",
        "wrongful acts of landlord", "forcible entry",
    ],
    "discrimination": [
        "discrimination", "fair housing", "c.151B", "s.4",
        "protected class", "familial status",
    ],
    "lease_terms": [
        "lease", "tenancy at will", "s.12", "estate at will",
        "month-to-month", "breaking lease", "s.11A", "nonpayment",
    ],
    "lead_paint": [
        "lead paint", "lead poisoning", "deleading", "lead removal",
        "lead from your home",
    ],
    "utilities_heat": [
        "heat", "heating", "utility", "hot water", "excessive insufficient",
        "shutoff",
    ],
    "public_housing": [
        "BHA", "section 8", "public housing", "waiting list",
        "c.121B", "housing assistance", "screening",
    ],
    "tenant_rights_general": [
        "tenant rights", "landlord and tenant", "attorney general",
        "top ten things", "know your rights",
    ],
}

QA_GENERATION_PROMPT = """Based on the following Massachusetts tenant law source text(s), generate ONE specific question that a real Boston renter might ask, and a clear factual answer.

RULES:
1. The question should sound natural (like a Reddit post or legal aid inquiry)
2. The answer must ONLY use information from the provided sources
3. The answer should cite specific statutes or regulations when available
4. Keep the answer to 2-4 sentences

{sources_text}

Return ONLY valid JSON (no markdown fences):
{{"question": "...", "answer": "...", "key_facts": ["fact1", "fact2", "fact3"]}}"""


def get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set.")
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


def load_chunks() -> list[dict]:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_chunks_by_topic(chunks: list[dict]) -> dict[str, list[dict]]:
    """Assign chunks to topics via keyword matching."""
    topic_chunks: dict[str, list[dict]] = {t: [] for t in TOPIC_QUOTAS}
    for chunk in chunks:
        text = (chunk.get("content", "") + " " + chunk.get("title", "")).lower()
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(kw.lower() in text for kw in keywords):
                topic_chunks[topic].append(chunk)
                break  # each chunk goes to first matching topic only
    return topic_chunks


def generate_qa_for_chunks(
    client: OpenAI,
    selected_chunks: list[dict],
    model: str = "openai/gpt-4o",
) -> dict | None:
    """Generate one QA pair from a group of 1-3 chunks."""
    sources_parts = []
    for i, c in enumerate(selected_chunks, 1):
        sources_parts.append(
            f"[Source {i}: {c['title']} ({c['source_url']})]\n{c['content'][:1500]}"
        )
    sources_text = "\n\n---\n\n".join(sources_parts)
    prompt = QA_GENERATION_PROMPT.format(sources_text=sources_text)

    try:
        result = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        text = result.choices[0].message.content.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        return json.loads(text)
    except Exception as e:
        print(f"  [ERROR] QA generation failed: {e}")
        return None


def main():
    print("Loading chunks...")
    chunks = load_chunks()
    print(f"  {len(chunks)} chunks loaded")

    print("\nClassifying chunks by topic...")
    topic_chunks = classify_chunks_by_topic(chunks)
    for topic, tc in topic_chunks.items():
        print(f"  {topic}: {len(tc)} chunks")

    client = get_client()
    golden_qa = []
    qa_id = 1

    for topic, quota in TOPIC_QUOTAS.items():
        available = topic_chunks[topic]
        if not available:
            print(f"\n[WARN] No chunks for topic '{topic}', skipping")
            continue

        print(f"\n--- Generating {quota} QA pairs for '{topic}' "
              f"({len(available)} chunks available) ---")

        for i in range(quota):
            # Sample 1-3 chunks per question
            n_chunks = min(random.randint(1, 3), len(available))
            selected = random.sample(available, n_chunks)

            print(f"  [{i+1}/{quota}] From: {selected[0]['title'][:50]}...")
            result = generate_qa_for_chunks(client, selected)
            if result is None:
                continue

            golden_qa.append({
                "id": f"golden_{qa_id:03d}",
                "topic": topic,
                "question": result["question"],
                "expected_answer": result["answer"],
                "key_facts": result.get("key_facts", []),
                "source_chunks": [
                    {
                        "chunk_id": c["chunk_id"],
                        "title": c["title"],
                        "source_url": c["source_url"],
                    }
                    for c in selected
                ],
            })
            qa_id += 1

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(golden_qa, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(golden_qa)} golden QA pairs")
    print(f"Saved to {OUTPUT_PATH}")

    # Topic distribution summary
    from collections import Counter
    dist = Counter(q["topic"] for q in golden_qa)
    print("\nTopic distribution:")
    for topic, count in sorted(dist.items()):
        print(f"  {topic}: {count}")


if __name__ == "__main__":
    main()
