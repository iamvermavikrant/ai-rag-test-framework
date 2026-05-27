"""Embedding generation using sentence-transformers.

Exists as a separate module to keep embedding model concerns isolated from
retrieval logic, and to allow swapping models without touching other layers.
"""

import os
from typing import TYPE_CHECKING

from observability.logger import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)


class Embedder:
    """Generates dense vector embeddings using a local sentence-transformers model.

    The model is lazy-loaded on the first call to embed() to avoid
    slowing down imports and fixture setup.

    Example:
        >>> embedder = Embedder()
        >>> vectors = embedder.embed(["Hello world"])
        >>> len(vectors[0]) > 0
        True
    """

    def __init__(self) -> None:
        self._model_name: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self._model: "SentenceTransformer | None" = None

    def _load_model(self) -> "SentenceTransformer":
        """Lazy-load the sentence-transformers model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a list of text strings.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors, one per input string.
        """
        model = self._load_model()
        vectors = model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vectors]
