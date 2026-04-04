"""
Chunking pipeline for scraped documents.
- Recursive character splitting with Markdown-aware boundaries
- Special handling for statutes (section boundaries) and FAQs (never split)
- 800 token chunks with 200 token overlap
"""

import json
import re
from pathlib import Path

import tiktoken

from src.scraping.utils import PROJECT_ROOT, load_processed_docs

CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"

# Chunking parameters
CHUNK_SIZE = 800  # tokens
CHUNK_OVERLAP = 200  # tokens
CHARS_PER_TOKEN = 4  # rough estimate

# Markdown-aware separators (highest to lowest priority)
SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]

# Statute subsection pattern
STATUTE_SECTION_RE = re.compile(r"\n\s*\([a-z]\)\s+|\n\s*Section\s+\d+", re.IGNORECASE)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens using tiktoken."""
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def split_by_separator(text: str, separator: str) -> list[str]:
    """Split text by separator, keeping the separator at the start of each chunk."""
    if separator not in text:
        return [text]
    parts = text.split(separator)
    result = [parts[0]]
    for part in parts[1:]:
        result.append(separator + part)
    return [p for p in result if p.strip()]


def recursive_split(text: str, separators: list[str], max_tokens: int) -> list[str]:
    """Recursively split text using hierarchical separators."""
    if count_tokens(text) <= max_tokens:
        return [text]

    if not separators:
        # Last resort: hard split by characters
        chunks = []
        max_chars = max_tokens * CHARS_PER_TOKEN
        for i in range(0, len(text), max_chars):
            chunks.append(text[i : i + max_chars])
        return chunks

    sep = separators[0]
    remaining_seps = separators[1:]
    parts = split_by_separator(text, sep)

    chunks = []
    for part in parts:
        if count_tokens(part) <= max_tokens:
            chunks.append(part)
        else:
            chunks.extend(recursive_split(part, remaining_seps, max_tokens))

    return chunks


def merge_with_overlap(chunks: list[str], overlap_tokens: int) -> list[str]:
    """Add overlap between consecutive chunks, always starting at a paragraph boundary.

    Takes the last `overlap_tokens` tokens of the previous chunk, then walks
    forward to the first paragraph break (double newline) so the overlap prefix
    begins at a clean paragraph start. For legal text this is almost always a
    complete self-contained thought — better than sentence boundaries which can
    split multi-sentence legal provisions mid-rule.

    Falls back to sentence boundary if no paragraph break is found, then to
    the raw overlap if neither is found.
    """
    if len(chunks) <= 1:
        return chunks

    enc = tiktoken.get_encoding("cl100k_base")
    PARA_BREAK = re.compile(r"\n\n+")
    SENT_END   = re.compile(r"(?<=[.!?])\s+")
    merged = []

    for i, chunk in enumerate(chunks):
        if i == 0:
            merged.append(chunk)
            continue

        prev_tokens = enc.encode(chunks[i - 1])
        if len(prev_tokens) > overlap_tokens:
            raw_overlap = enc.decode(prev_tokens[-overlap_tokens:])
            # Prefer paragraph boundary, fall back to sentence, then raw
            m = PARA_BREAK.search(raw_overlap)
            if m:
                overlap_text = raw_overlap[m.end():]
            else:
                m = SENT_END.search(raw_overlap)
                overlap_text = raw_overlap[m.end():] if m else raw_overlap
            chunk_with_overlap = overlap_text + chunk
        else:
            chunk_with_overlap = chunk

        merged.append(chunk_with_overlap)

    return merged


def chunk_faq(doc: dict) -> list[dict]:
    """FAQ documents: each Q&A = one chunk, never split."""
    return [{
        "chunk_id": f"{doc['doc_id']}_chunk_000",
        "doc_id": doc["doc_id"],
        "source_url": doc["source_url"],
        "source_name": doc["source_name"],
        "title": doc["title"],
        "content": doc["content"],
        "content_type": doc["content_type"],
        "legal_citations": doc.get("legal_citations", []),
        "chunk_index": 0,
        "total_chunks": 1,
    }]


def chunk_statute(doc: dict) -> list[dict]:
    """Statutes: split at section/subsection boundaries first."""
    text = doc["content"]
    # Split at section boundaries
    sections = STATUTE_SECTION_RE.split(text)
    separators_for_sections = STATUTE_SECTION_RE.findall(text)

    # Reconstruct sections with their markers
    parts = [sections[0]]
    for sep, section in zip(separators_for_sections, sections[1:]):
        parts.append(sep + section)

    # Further split any large sections
    all_splits = []
    for part in parts:
        if count_tokens(part) > CHUNK_SIZE:
            all_splits.extend(recursive_split(part, SEPARATORS, CHUNK_SIZE))
        else:
            all_splits.append(part)

    all_splits = merge_with_overlap(all_splits, CHUNK_OVERLAP)
    all_splits = [s for s in all_splits if len(s.strip()) >= 50]

    chunks = []
    for i, text in enumerate(all_splits):
        chunks.append({
            "chunk_id": f"{doc['doc_id']}_chunk_{i:03d}",
            "doc_id": doc["doc_id"],
            "source_url": doc["source_url"],
            "source_name": doc["source_name"],
            "title": doc["title"],
            "content": text.strip(),
            "content_type": doc["content_type"],
            "legal_citations": doc.get("legal_citations", []),
            "chunk_index": i,
            "total_chunks": len(all_splits),
        })
    return chunks


def chunk_generic(doc: dict) -> list[dict]:
    """Generic documents: recursive character splitting."""
    text = doc["content"]

    if count_tokens(text) <= CHUNK_SIZE:
        return [{
            "chunk_id": f"{doc['doc_id']}_chunk_000",
            "doc_id": doc["doc_id"],
            "source_url": doc["source_url"],
            "source_name": doc["source_name"],
            "title": doc["title"],
            "content": text,
            "content_type": doc["content_type"],
            "legal_citations": doc.get("legal_citations", []),
            "chunk_index": 0,
            "total_chunks": 1,
        }]

    splits = recursive_split(text, SEPARATORS, CHUNK_SIZE)
    splits = merge_with_overlap(splits, CHUNK_OVERLAP)
    splits = [s for s in splits if len(s.strip()) >= 50]

    chunks = []
    for i, chunk_text in enumerate(splits):
        chunks.append({
            "chunk_id": f"{doc['doc_id']}_chunk_{i:03d}",
            "doc_id": doc["doc_id"],
            "source_url": doc["source_url"],
            "source_name": doc["source_name"],
            "title": doc["title"],
            "content": chunk_text.strip(),
            "content_type": doc["content_type"],
            "legal_citations": doc.get("legal_citations", []),
            "chunk_index": i,
            "total_chunks": len(splits),
        })
    return chunks


def chunk_document(doc: dict) -> list[dict]:
    """Route document to appropriate chunking strategy."""
    # Skip duplicate full-page FAQ (individual Q&A docs are better)
    if "faq_full_page" in doc.get("doc_id", ""):
        return []

    content_type = doc.get("content_type", "guide")

    if content_type == "faq" and count_tokens(doc["content"]) <= CHUNK_SIZE * 2:
        return chunk_faq(doc)
    elif content_type == "statute":
        return chunk_statute(doc)
    else:
        return chunk_generic(doc)


def run():
    """Process all documents into chunks."""
    print("=" * 60)
    print("Chunking Pipeline")
    print("=" * 60)

    docs = load_processed_docs()
    print(f"Loaded {len(docs)} processed documents")

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []

    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['doc_id']}: {len(chunks)} chunks")

    # Save all chunks
    chunks_path = CHUNKS_DIR / "all_chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Print stats
    token_counts = [count_tokens(c["content"]) for c in all_chunks]
    print(f"\n{'=' * 60}")
    print(f"Total chunks: {len(all_chunks)}")
    if token_counts:
        print(f"Token range: {min(token_counts)}-{max(token_counts)} (avg {sum(token_counts)//len(token_counts)})")
    print(f"Saved to: {chunks_path}")
    print(f"{'=' * 60}")

    return all_chunks


if __name__ == "__main__":
    run()
