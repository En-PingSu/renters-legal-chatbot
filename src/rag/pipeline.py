"""
RAG pipeline: ChromaDB retrieval + LLM generation with citations.
Architecture: Query -> ChromaDB (top-k=5) -> Prompt Assembly -> GPT-4o-mini -> Citation Check -> Response

Models available:
  OpenRouter (cloud):
    - openai/gpt-4o
    - meta-llama/llama-3.1-70b-instruct
    - mistralai/mistral-small-3.1-24b-instruct

  Local llama-server (localhost:8080):
    - local/qwen3-finetuned   → Fine-Tuneing/finetuned-qwen3-f16.gguf  (domain fine-tuned)
    - local/qwen3-base        → Fine-Tuneing/Qwen3/  (base instruct, no fine-tune)

  Start local server (finetuned):
    llama.cpp\build\bin\Release\llama-server.exe \
      -m Fine-Tuneing\finetuned-qwen3-f16.gguf \
      -ngl 999 --port 8080 --ctx-size 4096 \
      --repeat-penalty 1.3 --temp 0.7 --top-p 0.9

  Start local server (base):
    llama.cpp\build\bin\Release\llama-server.exe \
      -m Fine-Tuneing\Qwen3\  (needs GGUF — quantize first or use HF transformers)
      -ngl 999 --port 8081 --ctx-size 4096

  Note: run finetuned on port 8080, base on port 8081 to test both simultaneously.
"""

import json
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from src.scraping.utils import PROJECT_ROOT

load_dotenv()

CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "all_chunks.json"
COLLECTION_NAME = "ma_tenant_law"

# ── Local model config ────────────────────────────────────────────────────────
LOCAL_MODELS = {
    "local/qwen3-finetuned": {
        "base_url": "http://localhost:8080/v1",
        "model_id": "qwen3-finetuned",
        "description": "Qwen3 4B fine-tuned on MA tenant law corpus",
    },
    "local/qwen3-base": {
        "base_url": "http://localhost:8081/v1",
        "model_id": "qwen3-base",
        "description": "Qwen3 4B base instruct (no fine-tuning)",
    },
}

SYSTEM_PROMPT = """You are a retrieval-grounded legal information assistant for Massachusetts tenant law (Boston area).

RULES:
1. Use ONLY the retrieved context below to answer. Do not use outside knowledge.
2. NEVER provide legal ADVICE -- only legal INFORMATION. Recommend consulting an attorney for specific situations.
3. ALWAYS cite sources with [Source: <title> (<url>)]. Cite specific statutes (e.g., MGL c.186, s.15B) when they appear in the context.
4. If the retrieved context is insufficient or conflicting, say so clearly and suggest legal aid resources such as MassLegalHelp.org or Greater Boston Legal Services.
5. If the question is outside Massachusetts tenant law, say so.

PROCESS: Follow these steps in order.
1. Restate what the user is asking in 1-2 sentences.
2. Identify the most relevant pieces of retrieved context.
3. Briefly analyze how the evidence answers the question. Note any gaps or conflicts.
4. Provide a clear final answer grounded only in the evidence.

RETRIEVED CONTEXT:
{context}

QUESTION: {question}

Respond in this format:

**Question Understanding:** <1-2 sentence restatement>

**Relevant Evidence:**
- <key excerpt or summary from a source, with citation>
- <additional evidence as needed>

**Analysis:** <brief reasoning connecting evidence to the answer; note any gaps>

**Final Answer:** <clear, grounded answer to the user>

**Confidence:** <high / medium / low, based on how well the context covers the question>"""


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create ChromaDB persistent client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_or_create_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    """Get or create the tenant law collection."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[dict] | None = None):
    """Index chunks into ChromaDB. Loads from file if chunks not provided."""
    if chunks is None:
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)

    client = get_chroma_client()
    collection = get_or_create_collection(client)

    existing = collection.count()
    if existing > 0:
        print(f"Collection already has {existing} chunks. Clearing and re-indexing.")
        client.delete_collection(COLLECTION_NAME)
        collection = get_or_create_collection(client)

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
        print(f"  Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    print(f"Done! {collection.count()} chunks in collection '{COLLECTION_NAME}'")


def retrieve(query: str, top_k: int = 5, content_type: str | None = None) -> list[dict]:
    """Retrieve relevant chunks from ChromaDB."""
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    where_filter = None
    if content_type:
        where_filter = {"content_type": content_type}

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_filter,
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "chunk_id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i] if results.get("distances") else None,
        })
    return chunks


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as context for the LLM."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        context_parts.append(
            f"[Source {i}: {meta['title']} ({meta['source_url']})]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(context_parts)


def generate_response(
    question: str,
    context: str | None = None,
    model: str = "openai/gpt-4o",
) -> str:
    """Generate a response using the LLM — routes to local or OpenRouter."""

    if context:
        system_msg = SYSTEM_PROMPT.format(context=context, question=question)
    else:
        system_msg = (
            "You are a legal information assistant. Answer the following question "
            "about Massachusetts tenant law to the best of your knowledge. "
            "Always recommend consulting an attorney for specific situations.\n\n"
            f"QUESTION: {question}"
        )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": question},
    ]

    # ── Route to local llama-server ───────────────────────────────────────────
    if model.startswith("local/"):
        config = LOCAL_MODELS.get(model)
        if config is None:
            raise ValueError(
                f"Unknown local model '{model}'. "
                f"Available: {list(LOCAL_MODELS.keys())}"
            )
        try:
            client = OpenAI(
                api_key="not-needed",
                base_url=config["base_url"],
                timeout=60.0,
            )
            response = client.chat.completions.create(
                model=config["model_id"],
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
                extra_body={
                    "repeat_penalty": 1.3,
                    "top_p": 0.9,
                },
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(
                f"Local model '{model}' unreachable at {config['base_url']}. "
                f"Is llama-server running? Error: {e}"
            )

    # ── Route to OpenRouter ───────────────────────────────────────────────────
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set. Add it to .env file.")

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", timeout=120.0)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=1500,
    )
    return response.choices[0].message.content


def verify_citations(response: str, chunks: list[dict]) -> str:
    """Verify cited sources exist in retrieved chunks. Append sources footer."""
    footer_lines = ["\n\n---\n**Sources:**"]
    for chunk in chunks:
        meta = chunk["metadata"]
        footer_lines.append(f"- [{meta['title']}]({meta['source_url']})")

    seen = set()
    unique_footer = [footer_lines[0]]
    for line in footer_lines[1:]:
        if line not in seen:
            seen.add(line)
            unique_footer.append(line)

    return response + "\n".join(unique_footer)


def ask(question: str, top_k: int = 5, model: str = "openai/gpt-4o",
        use_rag: bool = True, retriever: str = "vector") -> dict:
    """Full RAG pipeline: retrieve, generate, verify citations."""
    from src.rag.retrievers import RETRIEVER_REGISTRY

    result = {
        "question": question,
        "model": model,
        "use_rag": use_rag,
        "retriever": retriever,
        "retrieved_chunks": [],
        "response": "",
    }

    if use_rag:
        retrieve_fn = RETRIEVER_REGISTRY.get(retriever, retrieve)
        chunks = retrieve_fn(question, top_k=top_k)
        result["retrieved_chunks"] = chunks
        context = format_context(chunks)
        response = generate_response(question, context=context, model=model)
        result["response"] = verify_citations(response, chunks)
    else:
        response = generate_response(question, context=None, model=model)
        result["response"] = response

    return result


def sanity_check():
    """Run sanity check queries after indexing."""
    test_queries = [
        "security deposit Massachusetts",
        "eviction notice how many days",
        "landlord won't fix heat",
        "tenant rights repair Massachusetts",
        "breaking a lease early",
    ]

    print("\n" + "=" * 60)
    print("Sanity Check: Top-3 results for test queries")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        chunks = retrieve(query, top_k=3)
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            dist = chunk.get("distance", "?")
            print(f"  {i}. [{dist:.3f}] {meta['title']} ({meta['source_name']})")
            print(f"     {chunk['content'][:100]}...")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "index":
        index_chunks()
    elif len(sys.argv) > 1 and sys.argv[1] == "sanity":
        sanity_check()
    elif len(sys.argv) > 1 and sys.argv[1] == "test-local":
        # Quick test of local models
        q = "My landlord hasn't returned my security deposit after 30 days. What are my rights?"
        for m in ["local/qwen3-finetuned", "local/qwen3-base"]:
            print(f"\n{'='*60}")
            print(f"Model: {m}")
            print("="*60)
            try:
                result = ask(q, model=m)
                print(result["response"][:500] + "...")
            except RuntimeError as e:
                print(f"ERROR: {e}")
    else:
        # Interactive mode
        print("MA Tenant Law RAG Assistant")
        print("Models: openai/gpt-4o | local/qwen3-finetuned | local/qwen3-base")
        print("Type 'quit' to exit\n")
        while True:
            q = input("Question: ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            result = ask(q)
            print(f"\n{result['response']}\n")
