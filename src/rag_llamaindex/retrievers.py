"""Five retrieval strategies: vector, bm25, hybrid, rerank, parent_child.

All retrievers subclass LlamaIndex's BaseRetriever and return list[NodeWithScore].
"""

import json
import re
from collections import defaultdict
from typing import Any

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.rag_llamaindex.index import get_or_create_index
from src.rag_llamaindex.nodes import CHUNKS_PATH, load_chunks

# Module-level caches
_bm25_index = None
_bm25_chunks = None
_all_chunks_by_id = None


def _tokenize(text: str) -> list[str]:
    """Lowercase, extract alphanumeric tokens, stem with Snowball."""
    import Stemmer

    stemmer = Stemmer.Stemmer("english")
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return stemmer.stemWords(tokens)


def _get_bm25():
    """Lazy-build BM25 index from all_chunks.json."""
    global _bm25_index, _bm25_chunks
    if _bm25_index is not None:
        return _bm25_index, _bm25_chunks

    from rank_bm25 import BM25Okapi

    chunks = load_chunks()
    corpus = [_tokenize(c["content"]) for c in chunks]
    _bm25_index = BM25Okapi(corpus)
    _bm25_chunks = chunks
    return _bm25_index, _bm25_chunks


def _get_all_chunks_by_id() -> dict[str, dict]:
    """Lazy-load all chunks indexed by chunk_id."""
    global _all_chunks_by_id
    if _all_chunks_by_id is not None:
        return _all_chunks_by_id
    chunks = load_chunks()
    _all_chunks_by_id = {c["chunk_id"]: c for c in chunks}
    return _all_chunks_by_id


def _chunk_to_node_with_score(chunk: dict, score: float) -> NodeWithScore:
    """Convert a chunk dict to a NodeWithScore."""
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
    )
    return NodeWithScore(node=node, score=score)


class VectorRetriever(BaseRetriever):
    """Vector retrieval via LlamaIndex VectorStoreIndex."""

    def __init__(self, top_k: int = 5, **kwargs: Any):
        super().__init__(**kwargs)
        self._top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        index = get_or_create_index()
        retriever = index.as_retriever(similarity_top_k=self._top_k)
        results = retriever.retrieve(query_bundle)
        # ChromaDB returns distance (lower=better); LlamaIndex converts to
        # similarity score (higher=better) automatically via ChromaVectorStore.
        return results


class BM25Retriever(BaseRetriever):
    """BM25 lexical retrieval."""

    def __init__(self, top_k: int = 5, **kwargs: Any):
        super().__init__(**kwargs)
        self._top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        bm25, chunks = _get_bm25()
        query_tokens = _tokenize(query_bundle.query_str)
        scores = bm25.get_scores(query_tokens)

        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[: self._top_k]

        results = []
        for idx in top_indices:
            results.append(_chunk_to_node_with_score(chunks[idx], float(scores[idx])))
        return results


class HybridRetriever(BaseRetriever):
    """Weighted fusion of vector + BM25 retrieval (0.6/0.4)."""

    def __init__(
        self,
        top_k: int = 5,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._top_k = top_k
        self._vector_weight = vector_weight
        self._bm25_weight = bm25_weight

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        pool_k = self._top_k * 2

        vector_ret = VectorRetriever(top_k=pool_k)
        bm25_ret = BM25Retriever(top_k=pool_k)

        vector_results = vector_ret.retrieve(query_bundle)
        bm25_results = bm25_ret.retrieve(query_bundle)

        # Normalize vector scores (higher=better already from LlamaIndex)
        v_scores = [r.score for r in vector_results]
        v_min, v_max = min(v_scores), max(v_scores)
        v_range = v_max - v_min if v_max > v_min else 1.0

        # Normalize BM25 scores (higher=better)
        b_scores = [r.score for r in bm25_results]
        b_min, b_max = min(b_scores), max(b_scores)
        b_range = b_max - b_min if b_max > b_min else 1.0

        # Fuse scores by node ID
        fused: dict[str, dict] = {}
        for r in vector_results:
            norm = (r.score - v_min) / v_range
            nid = r.node.id_
            fused[nid] = {"score": self._vector_weight * norm, "node_with_score": r}

        for r in bm25_results:
            norm = (r.score - b_min) / b_range
            nid = r.node.id_
            if nid in fused:
                fused[nid]["score"] += self._bm25_weight * norm
            else:
                fused[nid] = {"score": self._bm25_weight * norm, "node_with_score": r}

        ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[
            : self._top_k
        ]
        return [
            NodeWithScore(node=item["node_with_score"].node, score=item["score"])
            for item in ranked
        ]


class RerankRetriever(BaseRetriever):
    """Hybrid retrieval + cross-encoder reranking."""

    def __init__(self, top_k: int = 5, initial_k: int = 10, **kwargs: Any):
        super().__init__(**kwargs)
        self._top_k = top_k
        self._initial_k = initial_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        from llama_index.core.postprocessor import SentenceTransformerRerank

        hybrid = HybridRetriever(top_k=self._initial_k)
        candidates = hybrid.retrieve(query_bundle)

        reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=self._top_k,
        )
        reranked = reranker.postprocess_nodes(candidates, query_bundle)
        return reranked


class ParentChildRetriever(BaseRetriever):
    """Retrieve on chunks, expand context with neighboring chunks from same document.

    When multiple retrieved chunks cluster in one document, include their
    neighboring chunks to provide broader context.
    """

    def __init__(self, top_k: int = 5, expand_neighbors: int = 1, **kwargs: Any):
        super().__init__(**kwargs)
        self._top_k = top_k
        self._expand_neighbors = expand_neighbors

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        # Initial retrieval with more candidates
        vector_ret = VectorRetriever(top_k=self._top_k * 2)
        initial_results = vector_ret.retrieve(query_bundle)

        # Group by doc_id to find clusters
        doc_chunks: dict[str, list[NodeWithScore]] = defaultdict(list)
        for r in initial_results:
            doc_id = r.node.metadata.get("doc_id", "")
            doc_chunks[doc_id].append(r)

        all_chunks_by_id = _get_all_chunks_by_id()

        # Expand context for clustered docs
        expanded_ids: set[str] = set()
        expanded_results: list[NodeWithScore] = []

        for r in initial_results[: self._top_k]:
            nid = r.node.id_
            if nid in expanded_ids:
                continue
            expanded_ids.add(nid)
            expanded_results.append(r)

            doc_id = r.node.metadata.get("doc_id", "")
            chunk_index = r.node.metadata.get("chunk_index", 0)
            total_chunks = r.node.metadata.get("total_chunks", 1)

            # Only expand if this doc has multiple hits
            if len(doc_chunks[doc_id]) >= 2:
                for offset in range(-self._expand_neighbors, self._expand_neighbors + 1):
                    if offset == 0:
                        continue
                    neighbor_idx = chunk_index + offset
                    if 0 <= neighbor_idx < total_chunks:
                        # Build expected neighbor chunk_id pattern
                        # chunk_id format: {doc_id}_chunk_{index:03d}
                        neighbor_id = f"{doc_id}_chunk_{neighbor_idx:03d}"
                        if neighbor_id in all_chunks_by_id and neighbor_id not in expanded_ids:
                            expanded_ids.add(neighbor_id)
                            neighbor_chunk = all_chunks_by_id[neighbor_id]
                            expanded_results.append(
                                _chunk_to_node_with_score(
                                    neighbor_chunk, r.score * 0.8
                                )
                            )

        return expanded_results


def get_retriever(name: str, top_k: int = 5) -> BaseRetriever:
    """Factory function to get a retriever by name."""
    retrievers = {
        "vector": lambda: VectorRetriever(top_k=top_k),
        "bm25": lambda: BM25Retriever(top_k=top_k),
        "hybrid": lambda: HybridRetriever(top_k=top_k),
        "rerank": lambda: RerankRetriever(top_k=top_k),
        "parent_child": lambda: ParentChildRetriever(top_k=top_k),
    }
    factory = retrievers.get(name)
    if factory is None:
        raise ValueError(f"Unknown retriever: {name}. Options: {list(retrievers.keys())}")
    return factory()


# Registry matching the interface of src/rag/retrievers.py
# Each entry is a callable(query, top_k) -> list[dict]
def _make_retriever_fn(name: str):
    """Create a retriever function matching the old interface: (query, top_k) -> list[dict]."""
    def fn(query: str, top_k: int = 5) -> list[dict]:
        retriever = get_retriever(name, top_k=top_k)
        results = retriever.retrieve(query)
        return [
            {
                "chunk_id": r.node.id_,
                "content": r.node.text,
                "metadata": {
                    k: v
                    for k, v in r.node.metadata.items()
                    if k in ("doc_id", "source_url", "source_name", "title", "content_type")
                },
                "distance": r.score,
            }
            for r in results
        ]
    return fn


RETRIEVER_REGISTRY = {
    name: _make_retriever_fn(name)
    for name in ["vector", "bm25", "hybrid", "rerank", "parent_child"]
}
