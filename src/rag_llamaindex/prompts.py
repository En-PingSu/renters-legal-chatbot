"""Prompt templates and utility functions for the LlamaIndex RAG pipeline."""

SYSTEM_PROMPT = """You are a legal information assistant for Massachusetts tenant law (Boston area).

RULES:
1. ONLY answer from provided source documents. If insufficient, say so and suggest legal aid resources such as MassLegalHelp.org or Greater Boston Legal Services.
2. NEVER provide legal ADVICE -- only legal INFORMATION. Recommend consulting an attorney for specific situations.
3. ALWAYS cite sources: [Source: <title> (<url>)].
4. Cite specific statutes (e.g., MGL c.186, s.15B) when relevant.
5. Synthesize multiple sources when relevant.
6. If the question is outside Massachusetts tenant law, say so.

CONTEXT:
{context}

QUESTION: {question}"""

BASELINE_PROMPT = (
    "You are a legal information assistant. Answer the following question "
    "about Massachusetts tenant law to the best of your knowledge. "
    "Always recommend consulting an attorney for specific situations.\n\n"
    "QUESTION: {question}"
)


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks as context for the LLM."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        context_parts.append(
            f"[Source {i}: {meta['title']} ({meta['source_url']})]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(context_parts)


def verify_citations(response: str, chunks: list[dict]) -> str:
    """Verify cited sources exist in retrieved chunks. Append sources footer."""
    footer_lines = ["\n\n---\n**Sources:**"]
    seen = set()
    for chunk in chunks:
        meta = chunk["metadata"]
        line = f"- [{meta['title']}]({meta['source_url']})"
        if line not in seen:
            seen.add(line)
            footer_lines.append(line)
    return response + "\n".join(footer_lines)
