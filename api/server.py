"""
FastAPI server wrapping the RAG pipeline with SSE streaming.

Run from project root:
    PYTHONPATH=. venv/bin/python3 -m uvicorn api.server:app --reload --port 8000
"""

import json
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Ensure project root is on sys.path so src.rag imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from src.rag.pipeline import (
    SYSTEM_PROMPT,
    format_context,
    retrieve,
    verify_citations,
)
from src.rag.retrievers import RETRIEVER_REGISTRY

app = FastAPI(title="MA Tenant Law RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = [
    {"id": "mistralai/mistral-small-3.1-24b-instruct", "label": "Mistral Small 24B"},
    {"id": "openai/gpt-4o", "label": "GPT-4o"},
    {"id": "meta-llama/llama-3.3-70b-instruct", "label": "Llama 3.3 70B"},
]

DEFAULT_MODEL = "mistralai/mistral-small-3.1-24b-instruct"
DEFAULT_RETRIEVER = "rerank"
DEFAULT_TOP_K = 5

SCORE_TYPE_MAP = {
    "vector": "cosine_distance",
    "bm25": "bm25",
    "hybrid": "fusion",
    "rerank": "cross_encoder",
    "parent_child": "cosine_distance",
    "multiquery": "rrf",
    "hybrid_parent_child": "fusion",
    "hybrid_parent_child_rerank": "cross_encoder",
    "auto_merge": "cross_encoder",
}


class ChatRequest(BaseModel):
    question: str
    model: str = DEFAULT_MODEL
    retriever: str = DEFAULT_RETRIEVER
    top_k: int = DEFAULT_TOP_K
    use_rag: bool = True


def _serialize_chunks(chunks: list[dict], score_type: str = "unknown") -> list[dict]:
    return [
        {
            "chunk_id": c["chunk_id"],
            "content": c["content"][:500],
            "metadata": c["metadata"],
            "distance": c.get("distance"),
            "score_type": score_type,
        }
        for c in chunks
    ]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    return {
        "models": MODELS,
        "retrievers": sorted(RETRIEVER_REGISTRY.keys()),
        "defaults": {
            "model": DEFAULT_MODEL,
            "retriever": DEFAULT_RETRIEVER,
            "top_k": DEFAULT_TOP_K,
        },
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    async def event_generator():
        # 1. Retrieve chunks
        chunks: list[dict] = []
        context: str | None = None
        if req.use_rag:
            retrieve_fn = RETRIEVER_REGISTRY.get(req.retriever, retrieve)
            chunks = retrieve_fn(req.question, top_k=req.top_k)
            context = format_context(chunks)

        # 2. Send sources immediately
        score_type = SCORE_TYPE_MAP.get(req.retriever, "unknown")
        yield {"event": "sources", "data": json.dumps(_serialize_chunks(chunks, score_type))}

        # 3. Build prompt (same logic as pipeline.generate_response)
        if context:
            system_msg = SYSTEM_PROMPT.format(context=context, question=req.question)
        else:
            system_msg = (
                "You are a legal information assistant. Answer the following question "
                "about Massachusetts tenant law to the best of your knowledge. "
                "Always recommend consulting an attorney for specific situations.\n\n"
                f"QUESTION: {req.question}"
            )

        # 4. Stream LLM response
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            yield {"event": "error", "data": "OPENROUTER_API_KEY not set"}
            return

        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        stream = client.chat.completions.create(
            model=req.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": req.question},
            ],
            temperature=0.2,
            max_tokens=1500,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield {"event": "token", "data": delta.content}

        # 5. Send done signal
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())
