"""Chunk retrieval logic for the RAG pipeline.

Exists as a separate module to compose Embedder and VectorStore cleanly,
keeping query-time orchestration independent of both storage and generation.
"""

from rag.embedder import Embedder
from rag.vector_store import VectorStore


# INTERVIEW NOTE: Why sentence-transformers over OpenAI embeddings?
# sentence-transformers runs locally (free, no API call).
# OpenAI embeddings cost ~$0.0001/1K tokens — acceptable in production
# but unnecessary for an MVP demo. The retrieval quality is comparable
# for small document sets.
class Retriever:
    """Orchestrates embedding a query and fetching the top-k relevant chunks.

    Composes Embedder and VectorStore so callers only need to pass a plain
    query string and receive ready-to-use context chunks.

    Example:
        >>> from rag.embedder import Embedder
        >>> from rag.vector_store import VectorStore
        >>> retriever = Retriever(Embedder(), VectorStore())
        >>> chunks = retriever.retrieve("What is RAG?", n_results=3)
        >>> isinstance(chunks, list)
        True
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(self, query: str, n_results: int = 3) -> list[str]:
        """Embed the query and return the most relevant stored chunks.

        Args:
            query: User question string.
            n_results: Number of chunks to retrieve.

        Returns:
            List of text chunks ranked by similarity to the query.
        """
        query_embedding: list[float] = self._embedder.embed([query])[0]
        return self._vector_store.query(query_embedding, n_results=n_results)
