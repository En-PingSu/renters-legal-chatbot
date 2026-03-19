"""LlamaIndex RAG pipeline: ask() matching the exact interface of src/rag/pipeline.ask()."""

import os

from dotenv import load_dotenv
from openai import OpenAI

from src.rag_llamaindex.nodes import chunks_to_nodes, load_chunks
from src.rag_llamaindex.index import index_nodes
from src.rag_llamaindex.prompts import BASELINE_PROMPT, SYSTEM_PROMPT, format_context, verify_citations
from src.rag_llamaindex.retrievers import RETRIEVER_REGISTRY

load_dotenv()


def _generate_response(
    question: str,
    context: str | None = None,
    model: str = "openai/gpt-4o",
) -> str:
    """Generate a response using the LLM via OpenRouter (direct call for exact prompt control)."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set. Add it to .env file.")

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    if context:
        system_msg = SYSTEM_PROMPT.format(context=context, question=question)
    else:
        system_msg = BASELINE_PROMPT.format(question=question)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
        ],
        temperature=0.2,
        max_tokens=1500,
    )
    return response.choices[0].message.content


def ask(
    question: str,
    top_k: int = 5,
    model: str = "openai/gpt-4o",
    use_rag: bool = True,
    retriever: str = "vector",
) -> dict:
    """Full RAG pipeline: retrieve, generate, verify citations.

    Returns dict with keys: question, model, use_rag, retriever,
    retrieved_chunks (list of {chunk_id, content, metadata, distance}),
    response (str).
    """
    result = {
        "question": question,
        "model": model,
        "use_rag": use_rag,
        "retriever": retriever,
        "retrieved_chunks": [],
        "response": "",
    }

    if use_rag:
        retrieve_fn = RETRIEVER_REGISTRY.get(retriever)
        if retrieve_fn is None:
            raise ValueError(
                f"Unknown retriever: {retriever}. "
                f"Options: {list(RETRIEVER_REGISTRY.keys())}"
            )
        chunks = retrieve_fn(question, top_k=top_k)
        result["retrieved_chunks"] = chunks
        context = format_context(chunks)
        response = _generate_response(question, context=context, model=model)
        result["response"] = verify_citations(response, chunks)
    else:
        response = _generate_response(question, context=None, model=model)
        result["response"] = response

    return result


def index_chunks(chunks: list[dict] | None = None):
    """Build the LlamaIndex vector store from chunks."""
    if chunks is None:
        chunks = load_chunks()
    nodes = chunks_to_nodes(chunks)
    index_nodes(nodes)
    print(f"Indexed {len(nodes)} chunks into LlamaIndex vector store")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "index":
        index_chunks()
    else:
        print("MA Tenant Law RAG Assistant (LlamaIndex)")
        print("Type 'quit' to exit\n")
        while True:
            q = input("Question: ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            result = ask(q)
            print(f"\n{result['response']}\n")
