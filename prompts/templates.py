"""Prompt templates for RAG-augmented generation.

Centralizes all prompt template strings so they can be versioned,
tested, and swapped independently of the pipeline logic.
"""

RAG_PROMPT_TEMPLATE = """\
You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the answer cannot be found in the context, say: "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
