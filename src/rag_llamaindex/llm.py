"""OpenRouter LLM and HuggingFace embedding model setup for LlamaIndex."""

import os

from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI

load_dotenv()


def get_llm(model: str = "openai/gpt-4o") -> LlamaOpenAI:
    """Get LlamaIndex LLM configured for OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set. Add it to .env file.")
    return LlamaOpenAI(
        model=model,
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0.2,
        max_tokens=1500,
    )


def get_embed_model() -> HuggingFaceEmbedding:
    """Get HuggingFace embedding model matching ChromaDB's default (all-MiniLM-L6-v2)."""
    return HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
