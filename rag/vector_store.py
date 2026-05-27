"""ChromaDB vector store operations.

Exists as a separate module so the persistence layer can be tested,
reset, or swapped without touching the embedding or retrieval logic.
"""

import os
import uuid

import chromadb

from observability.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """Manages a persistent ChromaDB collection for chunk storage and similarity search.

    Example:
        >>> store = VectorStore()
        >>> store.collection_exists()
        False
        >>> store.add_documents(["Hello world"], [[0.1, 0.2, 0.3]])
        >>> store.collection_exists()
        True
    """

    def __init__(self) -> None:
        chroma_path: str = os.getenv("CHROMA_PATH", "./chroma_db")
        self._collection_name: str = os.getenv("CHROMA_COLLECTION", "rag_collection")
        self._client = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name
        )
        logger.info("VectorStore initialized: collection=%s", self._collection_name)

    def collection_exists(self) -> bool:
        """Return True if the collection already contains documents."""
        return self._collection.count() > 0

    def add_documents(
        self, chunks: list[str], embeddings: list[list[float]]
    ) -> None:
        """Store text chunks and their pre-computed embeddings.

        Args:
            chunks: Raw text chunks to store.
            embeddings: Corresponding embedding vectors (same length as chunks).
        """
        ids = [str(uuid.uuid4()) for _ in chunks]
        self._collection.add(documents=chunks, embeddings=embeddings, ids=ids)
        logger.info("Added %d chunks to collection", len(chunks))

    def query(self, embedding: list[float], n_results: int = 3) -> list[str]:
        """Retrieve the top-k most similar chunks for a query embedding.

        Args:
            embedding: Query vector produced by the Embedder.
            n_results: Number of chunks to return.

        Returns:
            List of matching document strings.
        """
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, self._collection.count()),
        )
        documents: list[list[str]] = results.get("documents", [[]])
        return documents[0] if documents else []
