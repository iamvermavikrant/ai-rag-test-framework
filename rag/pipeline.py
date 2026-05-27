"""Public interface for the RAG pipeline.

Exists as the single entry point for all tests and external callers,
composing the six pipeline modules behind a clean, stable API.
"""

from observability.logger import get_logger
from rag.embedder import Embedder
from rag.generator import Generator
from rag.loader import chunk_text, load_documents
from rag.prompt_builder import build_prompt
from rag.retriever import Retriever
from rag.vector_store import VectorStore
from session.manager import SessionManager

logger = get_logger(__name__)


class RAGPipeline:
    """Composes loader, embedder, vector store, retriever, prompt builder, and generator.

    This is the ONLY class tests import from the rag module.

    Example:
        >>> pipeline = RAGPipeline()
        >>> pipeline.ingest("datasets/documents")
        >>> result = pipeline.query("What is RAG?")
        >>> set(result.keys()) == {"question", "answer", "chunks"}
        True
    """

    def __init__(self) -> None:
        self._embedder = Embedder()
        self._vector_store = VectorStore()
        self._retriever = Retriever(self._embedder, self._vector_store)
        self._generator = Generator()
        self._session_manager = SessionManager()

    def ingest(self, directory: str) -> None:
        """Load documents from disk, chunk them, embed, and store in ChromaDB.

        Skips ingestion if the collection already contains documents to avoid
        duplicate entries on repeated test runs.

        Args:
            directory: Path to the folder containing .txt source documents.
        """
        if self._vector_store.collection_exists():
            logger.info("Collection already populated — skipping ingestion")
            return

        documents = load_documents(directory)
        all_chunks: list[str] = []
        for doc in documents:
            all_chunks.extend(chunk_text(doc))

        logger.info("Ingesting %d chunks from %d documents", len(all_chunks), len(documents))
        embeddings = self._embedder.embed(all_chunks)
        self._vector_store.add_documents(all_chunks, embeddings)

    def query(self, question: str, n_results: int = 3) -> dict[str, object]:
        """Run a question through the full RAG pipeline and return a structured result.

        Args:
            question: The user's question string.
            n_results: Number of context chunks to retrieve.

        Returns:
            Dict with keys:
                - "question": the original question string
                - "answer": the LLM-generated answer string
                - "chunks": list of retrieved context chunk strings
        """
        chunks = self._retriever.retrieve(question, n_results=n_results)
        prompt = build_prompt(question, chunks)
        answer = self._generator.generate(prompt)

        logger.info("Query complete | question=%r | chunks=%d", question, len(chunks))
        return {"question": question, "answer": answer, "chunks": chunks}

    def query_with_session(self, question: str, session_id: str, n_results: int = 3) -> dict[str, object]:
        """Query using session history as additional context for multi-turn conversations.

        Builds a context-enriched prompt from prior turns before retrieval so the
        LLM can resolve references like "it" or "the previous answer" correctly.

        Args:
            question: The user's current question.
            session_id: Active session ID managed by the caller.
            n_results: Number of context chunks to retrieve.

        Returns:
            Dict with keys:
                - "question": the original question string
                - "answer": the LLM-generated answer string
                - "chunks": list of retrieved context chunk strings
        """
        # Build context-aware prompt from session history before retrieval.
        context_prompt = self._session_manager.build_context_prompt(session_id, question)

        chunks = self._retriever.retrieve(context_prompt, n_results=n_results)
        prompt = build_prompt(context_prompt, chunks)
        answer = self._generator.generate(prompt)

        self._session_manager.add_message(session_id, "user", question)
        self._session_manager.add_message(session_id, "assistant", answer)

        logger.info("Session query complete | session=%s | question=%r | chunks=%d", session_id, question, len(chunks))
        return {"question": question, "answer": answer, "chunks": chunks}
