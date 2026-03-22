"""
Sentence-level chunking with context window expansion at retrieval time.

Strategy: Each chunk stores a single sentence for precise embedding matching.
At retrieval time, the surrounding window (N sentences on each side) is expanded
to provide the LLM with richer context.

FAQ documents are kept as single chunks (not sentence-split).
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import chromadb
import tiktoken

from src.scraping.utils import PROJECT_ROOT, load_processed_docs

CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
SENTENCE_COLLECTION = "ma_tenant_law_sentences"

enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken."""
    return len(enc.encode(text))


# ---------------------------------------------------------------------------
# 1. Sentence splitting
# ---------------------------------------------------------------------------

# Abbreviations that should NOT trigger a sentence break when followed by a period
_ABBREVS = {
    "e.g", "i.e", "vs", "etc", "approx", "inc", "corp", "ltd",
    "jr", "sr", "dr", "mr", "mrs", "ms", "prof", "dept", "est",
    "govt", "no", "vol", "ch", "sec", "art", "div", "pt", "ex",
    "gen", "mass", "app", "ct", "mgl", "st",
}


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex.

    Handles abbreviations like e.g., i.e., MGL c., s., section references.
    Filters out empty/whitespace-only sentences.
    """
    # Normalize whitespace but preserve paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)

    # Strategy: find candidate split points (sentence-ending punctuation
    # followed by whitespace + uppercase letter), then filter out false positives.
    # We split on: [.!?] followed by space(s) + uppercase letter or quote.
    _split_re = re.compile(
        r'([.!?])'
        r'(\s+)'
        r'(?=[A-Z(\u201c"\u2018])',
    )

    parts = _split_re.split(text)

    # Reassemble: parts alternate between text, punctuation, whitespace
    sentences = []
    current = ""
    i = 0
    while i < len(parts):
        current += parts[i]
        if i + 2 < len(parts) and len(parts[i + 1]) == 1 and parts[i + 1] in ".!?":
            punct = parts[i + 1]
            ws = parts[i + 2]
            # Check if the word before the period is an abbreviation
            # or if it looks like a legal citation (e.g. "c.", "s.", digit before period)
            word_before = re.search(r'(\S+)$', current)
            is_abbrev = False
            if word_before:
                wb = word_before.group(1).rstrip(".")
                if wb.lower() in _ABBREVS:
                    is_abbrev = True
                # Single letter followed by period (e.g. "c.", "s.")
                if len(wb) == 1 and wb.isalpha():
                    is_abbrev = True
                # Digit before period (e.g. "15B.", section numbers)
                if re.search(r'\d$', wb):
                    is_abbrev = True

            if is_abbrev:
                # Don't split here
                current += punct + ws
                i += 3
            else:
                current += punct
                sentences.append(current.strip())
                current = ""
                i += 3
        else:
            i += 1

    if current.strip():
        sentences.append(current.strip())

    # Filter empties
    sentences = [s for s in sentences if s.strip()]

    # If regex produced no splits, fall back to simple newline splitting
    if len(sentences) <= 1 and len(text) > 200:
        fallback = [s.strip() for s in text.split("\n") if s.strip()]
        if len(fallback) > 1:
            return fallback

    return sentences


# ---------------------------------------------------------------------------
# 2. Create sentence-level chunks
# ---------------------------------------------------------------------------

def create_sentence_chunks(docs: list[dict], window_size: int = 3) -> list[dict]:
    """Create sentence-level chunks from documents.

    Each chunk contains a single sentence for embedding precision.
    Metadata includes window indices for expansion at retrieval time.

    FAQ documents are kept as single chunks (not sentence-split).

    Args:
        docs: List of processed document dicts.
        window_size: Number of sentences on each side for the context window.

    Returns:
        List of chunk dicts with sentence-level granularity.
    """
    all_chunks = []

    for doc in docs:
        doc_id = doc["doc_id"]
        content_type = doc.get("content_type", "guide")

        # Skip duplicate full-page FAQ
        if "faq_full_page" in doc_id:
            continue

        # FAQ docs: keep as single chunk
        if content_type == "faq":
            chunk = {
                "chunk_id": f"{doc_id}_sent_0000",
                "doc_id": doc_id,
                "source_url": doc["source_url"],
                "source_name": doc["source_name"],
                "title": doc["title"],
                "content": doc["content"],
                "content_type": content_type,
                "legal_citations": doc.get("legal_citations", []),
                "chunk_index": 0,
                "total_chunks": 1,
                "sentence_index": 0,
                "total_sentences": 1,
                "window_start": 0,
                "window_end": 1,
            }
            all_chunks.append(chunk)
            continue

        # Split into sentences
        sentences = split_into_sentences(doc["content"])
        if not sentences:
            continue

        total_sentences = len(sentences)

        for i, sentence in enumerate(sentences):
            w_start = max(0, i - window_size)
            w_end = min(total_sentences, i + window_size + 1)

            chunk = {
                "chunk_id": f"{doc_id}_sent_{i:04d}",
                "doc_id": doc_id,
                "source_url": doc["source_url"],
                "source_name": doc["source_name"],
                "title": doc["title"],
                "content": sentence,
                "content_type": content_type,
                "legal_citations": doc.get("legal_citations", []),
                "chunk_index": i,
                "total_chunks": total_sentences,
                "sentence_index": i,
                "total_sentences": total_sentences,
                "window_start": w_start,
                "window_end": w_end,
            }
            all_chunks.append(chunk)

    return all_chunks


# ---------------------------------------------------------------------------
# 3. Window expansion
# ---------------------------------------------------------------------------

def _build_doc_sentence_index(chunks: list[dict]) -> dict[str, list[dict]]:
    """Build a mapping from doc_id to sorted list of sentence chunks."""
    by_doc = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk["doc_id"]].append(chunk)
    # Sort each doc's chunks by sentence_index
    for doc_id in by_doc:
        by_doc[doc_id].sort(key=lambda c: c["sentence_index"])
    return dict(by_doc)


def expand_window(chunk: dict, all_chunks_by_doc: dict[str, list[dict]]) -> str:
    """Expand a retrieved sentence chunk to include its surrounding window.

    Args:
        chunk: A retrieved sentence chunk dict.
        all_chunks_by_doc: Mapping from doc_id to sorted list of sentence chunks.

    Returns:
        Expanded text (concatenation of window sentences).
    """
    doc_chunks = all_chunks_by_doc.get(chunk["doc_id"], [])
    if not doc_chunks:
        return chunk["content"]

    w_start = chunk.get("window_start", chunk["sentence_index"])
    w_end = chunk.get("window_end", chunk["sentence_index"] + 1)

    window_sentences = []
    for c in doc_chunks:
        if w_start <= c["sentence_index"] < w_end:
            window_sentences.append(c["content"])

    return " ".join(window_sentences) if window_sentences else chunk["content"]


# ---------------------------------------------------------------------------
# 4. Sentence-window retrieval
# ---------------------------------------------------------------------------

def _get_sentence_collection() -> chromadb.Collection:
    """Get the sentence-level ChromaDB collection."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=SENTENCE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def retrieve_sentence_window(
    query: str,
    top_k: int = 5,
    window_size: int = 3,
    all_sentence_chunks: list[dict] | None = None,
) -> list[dict]:
    """Retrieve sentence chunks and expand them to context windows.

    Args:
        query: User query.
        top_k: Number of results to return.
        window_size: Context window size (used only if all_sentence_chunks
                     need to be loaded from disk).
        all_sentence_chunks: Pre-loaded sentence chunks. If None, loaded from disk.

    Returns:
        List of result dicts with expanded content.
    """
    collection = _get_sentence_collection()

    # Retrieve more candidates than needed so we can deduplicate overlaps
    n_candidates = top_k * 2
    results = collection.query(
        query_texts=[query],
        n_results=min(n_candidates, collection.count()),
    )

    if not results["ids"][0]:
        return []

    # Load sentence chunks for window expansion
    if all_sentence_chunks is None:
        chunks_path = CHUNKS_DIR / "sentence_chunks.json"
        with open(chunks_path, "r", encoding="utf-8") as f:
            all_sentence_chunks = json.load(f)

    all_chunks_by_doc = _build_doc_sentence_index(all_sentence_chunks)

    # Build retrieved chunk objects
    retrieved = []
    for i in range(len(results["ids"][0])):
        chunk_id = results["ids"][0][i]
        # Find the full chunk data
        matching = [c for c in all_sentence_chunks if c["chunk_id"] == chunk_id]
        if not matching:
            continue
        chunk = matching[0]
        chunk_with_meta = {
            **chunk,
            "distance": results["distances"][0][i] if results.get("distances") else None,
        }
        retrieved.append(chunk_with_meta)

    # Expand windows and deduplicate overlapping windows from the same doc
    expanded_results = []
    seen_ranges = {}  # doc_id -> list of (start, end) ranges already included

    for chunk in retrieved:
        doc_id = chunk["doc_id"]
        w_start = chunk.get("window_start", chunk["sentence_index"])
        w_end = chunk.get("window_end", chunk["sentence_index"] + 1)

        # Check for overlap with already-included ranges from same doc
        if doc_id in seen_ranges:
            merged = False
            for j, (s, e) in enumerate(seen_ranges[doc_id]):
                # If ranges overlap, merge them
                if w_start < e and w_end > s:
                    new_start = min(s, w_start)
                    new_end = max(e, w_end)
                    seen_ranges[doc_id][j] = (new_start, new_end)
                    # Update the expanded content for this merged range
                    for r in expanded_results:
                        if r["doc_id"] == doc_id and r["window_start"] == s:
                            r["window_start"] = new_start
                            r["window_end"] = new_end
                            # Re-expand with merged range
                            doc_chunks = all_chunks_by_doc.get(doc_id, [])
                            window_sentences = [
                                c["content"] for c in doc_chunks
                                if new_start <= c["sentence_index"] < new_end
                            ]
                            r["expanded_content"] = " ".join(window_sentences)
                            break
                    merged = True
                    break
            if merged:
                continue

        # New range
        if doc_id not in seen_ranges:
            seen_ranges[doc_id] = []
        seen_ranges[doc_id].append((w_start, w_end))

        expanded_text = expand_window(chunk, all_chunks_by_doc)
        result = {
            "chunk_id": chunk["chunk_id"],
            "doc_id": doc_id,
            "content": chunk["content"],
            "expanded_content": expanded_text,
            "metadata": {
                "doc_id": chunk["doc_id"],
                "source_url": chunk["source_url"],
                "source_name": chunk["source_name"],
                "title": chunk["title"],
                "content_type": chunk["content_type"],
            },
            "distance": chunk.get("distance"),
            "window_start": w_start,
            "window_end": w_end,
        }
        expanded_results.append(result)

        if len(expanded_results) >= top_k:
            break

    return expanded_results[:top_k]


# ---------------------------------------------------------------------------
# 5. Indexing
# ---------------------------------------------------------------------------

def index_sentence_chunks(chunks: list[dict]):
    """Index sentence chunks into ChromaDB collection.

    Uses the same batched approach as the main pipeline indexer.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Clear and recreate
    try:
        client.delete_collection(SENTENCE_COLLECTION)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=SENTENCE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["content"] for c in batch],
            metadatas=[
                {
                    "doc_id": c["doc_id"],
                    "source_url": c["source_url"],
                    "source_name": c["source_name"],
                    "title": c["title"],
                    "content_type": c["content_type"],
                }
                for c in batch
            ],
        )
        print(f"  Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} sentence chunks")

    print(f"Done! {collection.count()} sentence chunks in collection '{SENTENCE_COLLECTION}'")


# ---------------------------------------------------------------------------
# 6. Run / comparison
# ---------------------------------------------------------------------------

def _print_comparison(query: str, standard_chunks: list[dict], sw_results: list[dict]):
    """Print a side-by-side comparison of standard vs sentence-window retrieval."""
    print(f"\n{'=' * 70}")
    print(f"Query: {query}")
    print(f"{'=' * 70}")

    print(f"\n--- Standard Retrieval ({len(standard_chunks)} chunks) ---")
    standard_ids = set()
    for i, chunk in enumerate(standard_chunks, 1):
        meta = chunk["metadata"]
        dist = chunk.get("distance", "?")
        print(f"  {i}. [{dist:.3f}] {meta['title']}")
        print(f"     {chunk['content'][:120]}...")
        standard_ids.add(chunk["chunk_id"])

    print(f"\n--- Sentence Window Retrieval ({len(sw_results)} expanded windows) ---")
    sw_doc_ids = set()
    for i, result in enumerate(sw_results, 1):
        meta = result["metadata"]
        dist = result.get("distance", "?")
        expanded = result.get("expanded_content", result["content"])
        print(f"  {i}. [{dist:.3f}] {meta['title']} (sentences {result['window_start']}-{result['window_end'] - 1})")
        print(f"     Sentence: {result['content'][:100]}...")
        print(f"     Expanded ({count_tokens(expanded)} tokens): {expanded[:150]}...")
        sw_doc_ids.add(result["doc_id"])

    # Unique analysis
    std_docs = {c["metadata"]["doc_id"] for c in standard_chunks}
    print(f"\n  Unique docs in standard: {len(std_docs)}")
    print(f"  Unique docs in sentence-window: {len(sw_doc_ids)}")
    only_std = std_docs - sw_doc_ids
    only_sw = sw_doc_ids - std_docs
    if only_std:
        print(f"  Only in standard: {only_std}")
    if only_sw:
        print(f"  Only in sentence-window: {only_sw}")


def run():
    """Build sentence chunks, index them, and compare with standard retrieval."""
    from src.rag.pipeline import retrieve

    print("=" * 70)
    print("Sentence Window Chunker")
    print("=" * 70)

    # Load docs and create sentence chunks
    docs = load_processed_docs()
    print(f"Loaded {len(docs)} processed documents")

    chunks = create_sentence_chunks(docs, window_size=3)
    print(f"Created {len(chunks)} sentence chunks")

    # Stats
    token_counts = [count_tokens(c["content"]) for c in chunks]
    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
        print(f"Avg sentence length: {avg_tokens:.1f} tokens")
        print(f"Token range: {min(token_counts)}-{max(token_counts)}")

    # Count by content type
    by_type = defaultdict(int)
    for c in chunks:
        by_type[c["content_type"]] += 1
    for ct, count in sorted(by_type.items()):
        print(f"  {ct}: {count} chunks")

    # Save
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    chunks_path = CHUNKS_DIR / "sentence_chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {chunks_path}")

    # Index into ChromaDB
    print("\nIndexing into ChromaDB...")
    index_sentence_chunks(chunks)

    # Compare with standard retrieval
    test_queries = [
        "What are the rules about security deposit interest in Massachusetts?",
        "Can my landlord evict me without going to court?",
        "What is the penalty for landlord retaliation?",
    ]

    print("\n" + "=" * 70)
    print("Comparison: Standard Chunks vs Sentence Window")
    print("=" * 70)

    for query in test_queries:
        # Standard retrieval
        std_chunks = retrieve(query, top_k=5)

        # Sentence window retrieval
        sw_results = retrieve_sentence_window(
            query, top_k=5, all_sentence_chunks=chunks
        )

        _print_comparison(query, std_chunks, sw_results)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    run()
