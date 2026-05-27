"""Prompt construction for RAG-augmented generation.

Exists as a separate module so the prompt template can be versioned,
A/B tested, and swapped without touching retrieval or generation logic.
"""

from prompts.templates import RAG_PROMPT_TEMPLATE


def build_prompt(query: str, context_chunks: list[str]) -> str:
    """Assemble the final LLM prompt by injecting context and the user query.

    Each chunk is separated by a delimiter so the model can clearly identify
    chunk boundaries. The template instructs the model to answer ONLY from
    context, which directly supports groundedness evaluation.

    Args:
        query: The user's question.
        context_chunks: Retrieved text chunks from the vector store.

    Returns:
        Fully assembled prompt string ready for the LLM.

    Example:
        >>> prompt = build_prompt("What is RAG?", ["RAG stands for Retrieval-Augmented Generation."])
        >>> "RAG stands for" in prompt
        True
    """
    delimiter = "\n---\n"
    context = delimiter.join(chunk.strip() for chunk in context_chunks)
    return RAG_PROMPT_TEMPLATE.format(context=context, question=query)
