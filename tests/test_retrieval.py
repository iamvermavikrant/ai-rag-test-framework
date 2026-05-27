"""Tests for vector relevance and chunk retrieval.

Validates that the retriever returns contextually relevant chunks
for a given query, in the correct quantity, and within acceptable latency.

Fixture dependency: `rag_pipeline` (session-scoped, from conftest.py).
All tests call rag_pipeline.query() — no direct class instantiation.
"""

import time

import pytest

from rag import RAGPipeline


@pytest.mark.retrieval
def test_relevant_chunks_returned_for_known_question(rag_pipeline: RAGPipeline) -> None:
    """Verify top-3 retrieved chunks contain keywords from the question topic.

    WHY: The retriever's primary job is topical relevance. If chunks for a
    password-reset question contain none of the expected domain words, the
    embedding + similarity search is broken regardless of generation quality.

    Fixture: rag_pipeline — shared pipeline with documents already ingested.
    """
    result = rag_pipeline.query("How do I reset my password?", n_results=3)
    chunks: list[str] = result["chunks"]  # type: ignore[assignment]

    combined_text = " ".join(chunks).lower()
    topic_keywords = ["password", "reset", "login", "email", "forgot"]

    assert any(kw in combined_text for kw in topic_keywords), (
        f"None of {topic_keywords} found in retrieved chunks: {chunks}"
    )


@pytest.mark.retrieval
def test_chunk_count_matches_requested_n_results(rag_pipeline: RAGPipeline) -> None:
    """Verify retrieval returns exactly n_results chunks when requested.

    WHY: Callers rely on a predictable number of context chunks when building
    prompts. If the retriever silently returns fewer chunks, prompt quality
    degrades and downstream evaluation scores become unreliable.

    Fixture: rag_pipeline — session-scoped pipeline.
    """
    requested = 3
    result = rag_pipeline.query("What is the return policy?", n_results=requested)
    chunks: list[str] = result["chunks"]  # type: ignore[assignment]

    assert len(chunks) == requested


@pytest.mark.retrieval
def test_empty_query_returns_empty_or_raises(rag_pipeline: RAGPipeline) -> None:
    """Edge case: empty string query should not crash the pipeline.

    WHY: An empty query is an invalid but realistic user mistake. The pipeline
    must either return an empty chunk list or raise a clear exception — it must
    NOT raise an unhandled internal error that would hide the real cause.

    Fixture: rag_pipeline — session-scoped pipeline.
    """
    try:
        result = rag_pipeline.query("", n_results=3)
        chunks: list[str] = result["chunks"]  # type: ignore[assignment]
        # Acceptable outcome: returned an empty list
        assert isinstance(chunks, list)
    except (ValueError, RuntimeError):
        # Acceptable outcome: raised a meaningful exception
        pass


@pytest.mark.retrieval
def test_retrieval_is_faster_than_threshold(rag_pipeline: RAGPipeline) -> None:
    """Performance: retrieval should complete in under 2 seconds.

    WHY: ChromaDB similarity search on a local collection should be sub-second.
    A latency regression (e.g. from an index rebuild or embedding re-run) that
    crosses the 2-second boundary would degrade interactive use cases.

    Fixture: rag_pipeline — session-scoped pipeline (embedder already warm).
    """
    threshold_seconds = 2.0
    start = time.perf_counter()
    rag_pipeline.query("What file formats are supported?", n_results=3)
    elapsed = time.perf_counter() - start

    assert elapsed < threshold_seconds, (
        f"Retrieval took {elapsed:.3f}s — exceeds {threshold_seconds}s threshold"
    )


@pytest.mark.retrieval
def test_chunks_are_non_empty_strings(rag_pipeline: RAGPipeline) -> None:
    """Verify all returned chunks are non-empty strings.

    WHY: Downstream components (prompt builder, evaluator) iterate over chunks
    and call string methods on them. An empty string or non-string element
    would cause silent prompt degradation or an AttributeError in generation.

    Fixture: rag_pipeline — session-scoped pipeline.
    """
    result = rag_pipeline.query("How is employee data protected?", n_results=3)
    chunks: list[str] = result["chunks"]  # type: ignore[assignment]

    assert len(chunks) > 0, "Expected at least one chunk to be returned"
    for chunk in chunks:
        assert isinstance(chunk, str), f"Chunk is not a string: {type(chunk)}"
        assert len(chunk.strip()) > 0, f"Chunk is empty or whitespace-only: {chunk!r}"
