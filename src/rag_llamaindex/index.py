"""ChromaDB-backed VectorStoreIndex for LlamaIndex."""

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.rag_llamaindex.llm import get_embed_model
from src.scraping.utils import PROJECT_ROOT

CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db_llamaindex"
COLLECTION_NAME = "ma_tenant_law"

# Module-level cache
_index = None


def _get_chroma_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_nodes(nodes: list[TextNode]) -> VectorStoreIndex:
    """Build a VectorStoreIndex from TextNodes, storing in ChromaDB."""
    global _index

    # Clear existing collection and recreate
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except (ValueError, Exception):
        pass  # Collection may not exist yet
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    embed_model = get_embed_model()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    _index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    print(f"Indexed {len(nodes)} nodes into ChromaDB at {CHROMA_DIR}")
    return _index


def get_or_create_index() -> VectorStoreIndex:
    """Get existing index or create from persisted ChromaDB store."""
    global _index
    if _index is not None:
        return _index

    collection = _get_chroma_collection()
    embed_model = get_embed_model()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    _index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return _index
