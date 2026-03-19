"""Chunk-to-TextNode conversion with parent-child document relationships."""

import json
from pathlib import Path

from llama_index.core.schema import Document, NodeRelationship, RelatedNodeInfo, TextNode

from src.scraping.utils import PROJECT_ROOT

CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "all_chunks.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_chunks() -> list[dict]:
    """Load all chunks from disk."""
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_parent_documents() -> dict[str, Document]:
    """Load processed documents as LlamaIndex Documents, keyed by doc_id."""
    docs = {}
    for path in PROCESSED_DIR.glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        doc_id = data["doc_id"]
        docs[doc_id] = Document(
            doc_id=doc_id,
            text=data["content"],
            metadata={
                "doc_id": doc_id,
                "source_url": data["source_url"],
                "source_name": data["source_name"],
                "title": data["title"],
                "content_type": data["content_type"],
            },
        )
    return docs


def chunks_to_nodes(chunks: list[dict] | None = None) -> list[TextNode]:
    """Convert chunk dicts to LlamaIndex TextNodes with stable IDs and parent refs."""
    if chunks is None:
        chunks = load_chunks()

    nodes = []
    for chunk in chunks:
        node = TextNode(
            id_=chunk["chunk_id"],
            text=chunk["content"],
            metadata={
                "doc_id": chunk["doc_id"],
                "source_url": chunk["source_url"],
                "source_name": chunk["source_name"],
                "title": chunk["title"],
                "content_type": chunk["content_type"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
            },
            excluded_embed_metadata_keys=[
                "doc_id", "source_url", "source_name", "title",
                "content_type", "chunk_index", "total_chunks",
            ],
            excluded_llm_metadata_keys=["doc_id", "chunk_index", "total_chunks"],
        )
        # Link to parent document
        node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
            node_id=chunk["doc_id"],
        )
        nodes.append(node)

    return nodes
