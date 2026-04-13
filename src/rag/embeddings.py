"""
Embedding model configuration for ChromaDB.

Supports multiple embedding models via the EMBEDDING_MODEL env var:
  - "default"               -> ChromaDB's built-in all-MiniLM-L6-v2 (384-dim)
  - "bge-large"             -> BAAI/bge-large-en-v1.5 (1024-dim, local/free)
  - "text-embedding-3-large" -> OpenAI text-embedding-3-large (3072-dim, paid API)

Each model gets its own ChromaDB collection to allow A/B comparison.
"""

import os

COLLECTION_BASE = "ma_tenant_law"

# Map model aliases to collection suffixes and model identifiers
MODEL_REGISTRY = {
    "default": {
        "collection_suffix": "",
        "description": "ChromaDB default (all-MiniLM-L6-v2, 384-dim)",
    },
    "bge-large": {
        "collection_suffix": "_bge_large",
        "model_name": "BAAI/bge-large-en-v1.5",
        "description": "BGE-large-en-v1.5 (1024-dim, local/free)",
    },
    "text-embedding-3-large": {
        "collection_suffix": "_oai_3_large",
        "model_name": "text-embedding-3-large",
        "description": "OpenAI text-embedding-3-large (3072-dim, paid API)",
    },
}


def _resolve_model_name(model_name: str | None = None) -> str:
    """Resolve model name from argument or EMBEDDING_MODEL env var."""
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL", "default")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown embedding model: {model_name!r}. "
            f"Valid options: {list(MODEL_REGISTRY.keys())}"
        )
    return model_name


def get_embedding_function(model_name: str | None = None):
    """Return a ChromaDB-compatible embedding function.

    Returns None for "default" (ChromaDB uses its built-in all-MiniLM-L6-v2).
    """
    model_name = _resolve_model_name(model_name)

    if model_name == "default":
        return None

    info = MODEL_REGISTRY[model_name]

    if model_name == "bge-large":
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )
        return SentenceTransformerEmbeddingFunction(
            model_name=info["model_name"],
        )

    if model_name == "text-embedding-3-large":
        from chromadb.utils.embedding_functions import (
            OpenAIEmbeddingFunction,
        )
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY env var required for text-embedding-3-large"
            )
        return OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=info["model_name"],
        )

    return None


def get_collection_name(model_name: str | None = None) -> str:
    """Return the ChromaDB collection name for the given embedding model.

    Examples:
        "default"                -> "ma_tenant_law"
        "bge-large"              -> "ma_tenant_law_bge_large"
        "text-embedding-3-large" -> "ma_tenant_law_oai_3_large"
    """
    model_name = _resolve_model_name(model_name)
    suffix = MODEL_REGISTRY[model_name]["collection_suffix"]
    return f"{COLLECTION_BASE}{suffix}"
