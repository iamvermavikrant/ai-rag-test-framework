"""RAG pipeline package — document loading, embedding, retrieval, and generation.

Exports RAGPipeline as the single entry point for all test fixtures.
The individual sub-modules (loader, embedder, vector_store, retriever,
prompt_builder, generator) remain importable for unit testing each layer
in isolation.
"""

import os

from rag.embedder import Embedder
from rag.generator import Generator
from rag.loader import chunk_text, load_documents
from rag.prompt_builder import build_prompt
from rag.retriever import Retriever
from rag.vector_store import VectorStore
from observability.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """Facade over all RAG pipeline layers.

    Composes Embedder, VectorStore, Retriever, PromptBuilder, and Generator
    into a single object so test fixtures only need to hold one reference.
    Mirrors the "Page Object" pattern — tests call pipeline.query(), not
    raw embedder/retriever/generator methods.

    Example:
        >>> pipeline = RAGPipeline()
        >>> pipeline.ingest("datasets/documents")
        >>> result = pipeline.query("How do I reset my password?")
        >>> isinstance(result["answer"], str)
        True
        >>> isinstance(result["chunks"], list)
        True
    """

    def __init__(self) -> None:
        self._embedder = Embedder()
        self._vector_store = VectorStore()
        self._retriever = Retriever(self._embedder, self._vector_store)
        self._generator = Generator()

    def ingest(self, documents_dir: str) -> None:
        """Load, chunk, embed, and store all documents from a directory.

        Skips ingestion if the collection already has documents — safe to
        call multiple times (idempotent for the session-scoped fixture).

        Args:
            documents_dir: Path to the folder containing .txt source files.
        """
        if self._vector_store.collection_exists():
            logger.info("Collection already populated — skipping ingestion")
            return

        documents = load_documents(documents_dir)
        all_chunks: list[str] = []
        for doc in documents:
            all_chunks.extend(chunk_text(doc))

        embeddings = self._embedder.embed(all_chunks)
        self._vector_store.add_documents(all_chunks, embeddings)
        logger.info("Ingested %d chunks from %s", len(all_chunks), documents_dir)

    def query(self, question: str, n_results: int = 3) -> dict[str, object]:
        """Run the full RAG pipeline for a single question.

        Args:
            question: The user question to answer.
            n_results: Number of context chunks to retrieve.

        Returns:
            Dict with keys:
                "question" (str)  — the original question
                "chunks"   (list[str]) — retrieved context chunks
                "prompt"   (str)  — assembled prompt sent to the LLM
                "answer"   (str)  — generated answer from the LLM
        """
        chunks: list[str] = self._retriever.retrieve(question, n_results=n_results)
        prompt: str = build_prompt(question, chunks)
        answer: str = self._generator.generate(prompt)

        logger.info("RAG query complete | question: %r | chunks: %d", question[:80], len(chunks))

        return {
            "question": question,
            "chunks": chunks,
            "prompt": prompt,
            "answer": answer,
        }