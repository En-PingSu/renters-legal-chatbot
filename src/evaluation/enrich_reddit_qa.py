"""
Enrich reddit evaluation questions with golden answers and key facts.

For each reddit question, retrieves relevant chunks from the corpus,
then uses GPT-4o to generate a reference answer + key_facts grounded
in those chunks.

Usage:
    python -m src.evaluation.enrich_reddit_qa
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.rag.retrievers import RETRIEVER_REGISTRY
from src.scraping.utils import PROJECT_ROOT

load_dotenv()

REDDIT_PATH = PROJECT_ROOT / "data" / "evaluation" / "reddit_questions.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "reddit_questions.json"

ANSWER_PROMPT = """Based on the following Massachusetts tenant law source texts, answer the renter's question with a clear, factual response.

RULES:
1. The answer must ONLY use information from the provided sources
2. Cite specific statutes or regulations when available
3. Keep the answer to 2-4 sentences

QUESTION: {question}
CONTEXT: {context}

SOURCE TEXTS:
{sources_text}

Return ONLY valid JSON (no markdown fences):
{{"answer": "...", "key_facts": ["fact1", "fact2", "fact3"]}}"""


def get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set.")
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


def main():
    with open(REDDIT_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    retrieve = RETRIEVER_REGISTRY["rerank"]
    client = get_client()

    print(f"Enriching {len(questions)} reddit questions with golden answers...\n")

    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q['question'][:60]}...")

        # Retrieve relevant chunks
        chunks = retrieve(q["question"], top_k=5)

        # Build source text
        sources_parts = []
        source_chunks = []
        for j, c in enumerate(chunks, 1):
            meta = c["metadata"]
            sources_parts.append(
                f"[Source {j}: {meta['title']} ({meta['source_url']})]\n{c['content'][:1500]}"
            )
            source_chunks.append({
                "chunk_id": c["chunk_id"],
                "title": meta["title"],
                "source_url": meta["source_url"],
            })

        sources_text = "\n\n---\n\n".join(sources_parts)
        prompt = ANSWER_PROMPT.format(
            question=q["question"],
            context=q.get("context", ""),
            sources_text=sources_text,
        )

        try:
            result = client.chat.completions.create(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            text = result.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            parsed = json.loads(text)

            q["expected_answer"] = parsed["answer"]
            q["key_facts"] = parsed.get("key_facts", [])
            q["source_chunks"] = source_chunks
            print(f"  -> {len(q['key_facts'])} key facts, {len(source_chunks)} sources")

        except Exception as e:
            print(f"  [ERROR] {e}")
            q["expected_answer"] = ""
            q["key_facts"] = []
            q["source_chunks"] = source_chunks

    # Save back
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    enriched = sum(1 for q in questions if q.get("expected_answer"))
    print(f"\nEnriched {enriched}/{len(questions)} questions")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
