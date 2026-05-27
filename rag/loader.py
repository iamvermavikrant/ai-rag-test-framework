"""Document loading and chunking for RAG ingestion.

Exists as a separate module so chunking strategy can be tuned and tested
independently without touching the embedding or storage layers.
"""

from pathlib import Path


def load_documents(directory: str) -> list[str]:
    """Read all .txt files from a directory and return their contents.

    Args:
        directory: Path to the folder containing source documents.

    Returns:
        List of raw text strings, one per file.

    Example:
        >>> docs = load_documents("datasets/documents")
        >>> isinstance(docs[0], str)
        True
    """
    texts: list[str] = []
    doc_dir = Path(directory)

    if not doc_dir.exists():
        raise FileNotFoundError(f"Document directory not found: {directory!r}")

    for filepath in sorted(doc_dir.glob("*.txt")):
        try:
            texts.append(filepath.read_text(encoding="utf-8"))
        except OSError as exc:
            raise FileNotFoundError(f"Could not read document {filepath}: {exc}") from exc

    return texts


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping fixed-size character chunks.

    Overlap ensures that sentences split across a boundary are still
    represented fully in at least one chunk, improving retrieval recall.

    Args:
        text: Raw document text.
        chunk_size: Maximum characters per chunk.
        overlap: Characters shared between consecutive chunks.

    Returns:
        List of text chunks.

    Example:
        >>> chunks = chunk_text("Hello world " * 100, chunk_size=100, overlap=10)
        >>> all(len(c) <= 100 for c in chunks)
        True
    """
    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        if end == text_length:
            break
        start += chunk_size - overlap

    return chunks
