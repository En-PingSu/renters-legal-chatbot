"""
Fine-Tuneing/prepare_finetune_data.py

Prepares training data for local Qwen3 4B LoRA finetuning.
Pulls from three sources in priority order:
  1. golden_qa.json        — human-curated QA pairs (5x weight, highest signal)
  2. reddit_questions.json — enriched real-user questions (1x weight)
  3. all_chunks.json       — clean 967-chunk corpus → synthetic QA (1x weight)

Output: Fine-Tuneing/train.txt and Fine-Tuneing/val.txt
        Formatted as Qwen3 ChatML tokens, one sample per blank line.
"""

import json
import random
import re
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent
GOLDEN_QA      = PROJECT_ROOT / "data" / "evaluation" / "golden_qa.json"
REDDIT_QA      = PROJECT_ROOT / "data" / "evaluation" / "reddit_questions.json"
CHUNKS_PATH    = PROJECT_ROOT / "data" / "chunks" / "all_chunks.json"
OUT_DIR        = PROJECT_ROOT / "Fine-Tuneing"
TRAIN_OUT      = OUT_DIR / "train.txt"
VAL_OUT        = OUT_DIR / "val.txt"

# ── Config ────────────────────────────────────────────────────────────────────
GOLDEN_WEIGHT    = 5     # repeat golden QA pairs N times (highest signal)
REDDIT_WEIGHT    = 1
CHUNK_WEIGHT     = 1

VAL_SPLIT        = 0.10  # 10% held out for validation
RANDOM_SEED      = 42

# Clean corpus (967 chunks) has avg 464 chars (~116 tokens), range ~32-3900 chars.
# MIN_CHUNK_LEN lowered to 50 chars — short chunks in the CLEAN corpus are high
#   signal (statute limits, notice periods, penalty amounts) not nav fragments.
#   The corpus_cleaner already removed those. MIN_WORD_COUNT catches true stubs.
# MAX_CHUNK_LEN: skip merged blocks that would produce rambling answers.
MIN_CHUNK_LEN    = 50    # chars — corpus_cleaner already filtered real junk
MIN_WORD_COUNT   = 8     # words — catches number-only or single-phrase stubs
MAX_CHUNK_LEN    = 3200  # chars — let truncate_answer() handle long chunks
MAX_ANSWER_CHARS = 800   # truncate chunk answers at last paragraph/sentence boundary

SYSTEM_CONTEXT = (
    "You are a knowledgeable Massachusetts tenant law assistant. "
    "Provide accurate legal information grounded in Massachusetts statutes, "
    "regulations, and housing resources. Always recommend consulting an attorney "
    "for specific legal situations."
)


# ── Answer truncation ─────────────────────────────────────────────────────────
def truncate_answer(text: str, max_chars: int = MAX_ANSWER_CHARS) -> str:
    """Truncate chunk answer at last paragraph or sentence boundary before max_chars.

    Prefers paragraph breaks (\n\n) over sentence breaks to keep legal provisions
    intact. Only applied to chunk QA — golden and Reddit answers kept in full.
    """
    if len(text) <= max_chars:
        return text

    window = text[:max_chars]

    # Prefer paragraph boundary
    last_para = window.rfind("\n\n")
    if last_para > max_chars // 2:
        return text[:last_para].strip()

    # Fall back to sentence boundary
    last_sent = max(
        window.rfind(". "),
        window.rfind(".\n"),
        window.rfind("! "),
        window.rfind("? "),
    )
    if last_sent > max_chars // 2:
        return text[:last_sent + 1].strip()

    # Hard truncate as last resort
    return window.strip()


# ── Qwen3 ChatML format ───────────────────────────────────────────────────────
def format_sample(question: str, answer: str) -> str:
    """
    Qwen3 ChatML format:
      <|im_start|>system\n{system}<|im_end|>
      <|im_start|>user\n{question}<|im_end|>
      <|im_start|>assistant\n{answer}<|im_end|>
    """
    question = question.strip()
    answer   = answer.strip()
    return (
        f"<|im_start|>system\n{SYSTEM_CONTEXT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n{answer}<|im_end|>"
    )


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_golden_qa() -> list[dict]:
    if not GOLDEN_QA.exists():
        print(f"  [WARN] {GOLDEN_QA} not found, skipping golden QA")
        return []

    with open(GOLDEN_QA, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    for item in data:
        question = item.get("question", "").strip()
        if not question:
            continue

        # golden_qa.json uses 'expected_answer' as the primary field
        answer = item.get("expected_answer", "").strip()
        if not answer and "key_facts" in item:
            facts = item["key_facts"]
            if isinstance(facts, list):
                answer = " ".join(str(f) for f in facts).strip()

        if answer:
            pairs.append({"question": question, "answer": answer, "source": "golden"})

    print(f"  Loaded {len(pairs)} golden QA pairs")
    return pairs


def load_reddit_qa() -> list[dict]:
    if not REDDIT_QA.exists():
        print(f"  [WARN] {REDDIT_QA} not found, skipping Reddit QA")
        return []

    with open(REDDIT_QA, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    for item in data:
        question = item.get("question", "").strip()
        if not question:
            continue

        # reddit_questions.json uses 'expected_answer' as primary field
        answer = (
            item.get("expected_answer")
            or item.get("enriched_answer")
            or item.get("ideal_answer")
            or item.get("answer")
            or ""
        ).strip()

        if answer:
            pairs.append({"question": question, "answer": answer, "source": "reddit"})

    print(f"  Loaded {len(pairs)} Reddit QA pairs")
    return pairs


def chunk_to_qa(chunk: dict) -> dict | None:
    """Convert a single clean chunk to a QA training pair.

    Uses content_type to generate a realistic question. Skips chunks that are
    too short (fragments) or too long (merged statute blocks). Truncates answers
    at paragraph/sentence boundary to keep training targets focused.
    """
    content = chunk.get("content", "").strip()
    if len(content) < MIN_CHUNK_LEN or len(content) > MAX_CHUNK_LEN:
        return None
    if len(content.split()) < MIN_WORD_COUNT:
        return None  # stub: too few words to form a useful QA pair

    title        = chunk.get("title", "").strip()
    source_name  = chunk.get("source_name", "").strip()
    content_type = chunk.get("content_type", "").strip()

    if content_type == "faq":
        lines = content.splitlines()
        first_line = lines[0].strip() if lines else ""
        question = (
            first_line if first_line.endswith("?")
            else f"What does {source_name} say about {title.lower()}?"
        )
    elif content_type in ("statute", "regulation"):
        statute_match = re.search(r"(MGL\s+c\.\s*\d+[A-Z]?|CMR\s+\d+)", content)
        if statute_match:
            question = f"What does {statute_match.group(1)} say about tenant rights in Massachusetts?"
        else:
            question = f"What are the legal requirements under {title}?"
    elif content_type == "guide":
        question = f"What guidance does {source_name} provide about {title.lower()}?"
    else:
        question = f"What does Massachusetts law say about {title.lower()}?"

    return {
        "question": question,
        "answer":   truncate_answer(content),
        "source":   "chunk",
    }


def load_chunk_qa() -> list[dict]:
    if not CHUNKS_PATH.exists():
        print(f"  [WARN] {CHUNKS_PATH} not found, skipping chunk QA")
        return []

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    pairs   = []
    skipped = 0
    for chunk in chunks:
        qa = chunk_to_qa(chunk)
        if qa:
            pairs.append(qa)
        else:
            skipped += 1

    # Deduplicate by question — many chunks from the same doc produce identical
    # template questions (e.g. dozens of "What guidance does masslegalhelp provide
    # about evictions?"). Keep only the first occurrence of each question so the
    # model doesn't learn that one question has many contradictory answers.
    before_dedup = len(pairs)
    # Keep longest answer per unique question — many chunks from the same doc
    # produce identical template questions; the longest answer has the most signal.
    seen_questions = {}
    for qa in pairs:
        q_norm = re.sub(r"\s+", " ", qa["question"].strip().lower())
        if q_norm not in seen_questions or len(qa["answer"]) > len(seen_questions[q_norm]["answer"]):
            seen_questions[q_norm] = qa
    pairs = list(seen_questions.values())

    print(f"  Converted {before_dedup} chunks to QA pairs ({skipped} skipped — too short/long)")
    print(f"  Deduplicated chunk questions: {before_dedup} -> {len(pairs)} (-{before_dedup - len(pairs)} duplicate questions)")
    return pairs


# ── Assembly ──────────────────────────────────────────────────────────────────
def build_dataset() -> list[str]:
    print("\nLoading training data sources...")
    golden = load_golden_qa()
    reddit = load_reddit_qa()
    chunks = load_chunk_qa()

    weighted = (
        golden * GOLDEN_WEIGHT +
        reddit * REDDIT_WEIGHT +
        chunks * CHUNK_WEIGHT
    )

    print(f"\nDataset composition:")
    print(f"  Golden QA:   {len(golden)} x {GOLDEN_WEIGHT} = {len(golden) * GOLDEN_WEIGHT} samples (full expected_answer)")
    print(f"  Reddit QA:   {len(reddit)} x {REDDIT_WEIGHT} = {len(reddit) * REDDIT_WEIGHT} samples")
    print(f"  Chunk QA:    {len(chunks)} x {CHUNK_WEIGHT} = {len(chunks) * CHUNK_WEIGHT} samples (>={MIN_CHUNK_LEN} chars, >={MIN_WORD_COUNT} words, truncated <={MAX_ANSWER_CHARS} chars)")
    print(f"  Total:       {len(weighted)} samples")

    random.seed(RANDOM_SEED)
    random.shuffle(weighted)

    formatted = [format_sample(p["question"], p["answer"]) for p in weighted]
    return formatted


def write_splits(samples: list[str]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_val   = max(1, int(len(samples) * VAL_SPLIT))
    n_train = len(samples) - n_val

    train_samples = samples[:n_train]
    val_samples   = samples[n_train:]

    TRAIN_OUT.write_text("\n\n".join(train_samples), encoding="utf-8")
    VAL_OUT.write_text(  "\n\n".join(val_samples),   encoding="utf-8")

    print(f"\nWrote training data:")
    print(f"  Train: {len(train_samples)} samples -> {TRAIN_OUT}")
    print(f"  Val:   {len(val_samples)} samples   -> {VAL_OUT}")
    print(f"\n  Train file size: {TRAIN_OUT.stat().st_size / 1024:.1f} KB")
    print(f"  Val file size:   {VAL_OUT.stat().st_size / 1024:.1f} KB")


def preview(samples: list[str], n: int = 3):
    print(f"\n{'=' * 70}")
    print(f"Sample preview ({n} samples):")
    print(f"{'=' * 70}")
    for i, s in enumerate(samples[:n], 1):
        print(f"\n--- Sample {i} ---")
        print(s[:500] + "..." if len(s) > 500 else s)


if __name__ == "__main__":
    import sys

    print("MA Tenant Law -- Finetune Data Preparation (Qwen3 ChatML format)")
    print("=" * 50)

    samples = build_dataset()

    if "--preview" in sys.argv or "-p" in sys.argv:
        preview(samples)

    write_splits(samples)

    print("\nDone. Next step:")
    print("  python Fine-Tuneing/FineTune.py")
