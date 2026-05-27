"""Tests for AI evaluation metrics: hallucination, relevancy, and faithfulness.

Runs the full RAG pipeline and evaluates generated answers using DeepEval.
Each test targets a single metric threshold defined in CLAUDE.md.

Fixture dependencies:
    - `rag_pipeline` (session-scoped) — shared, pre-ingested pipeline.
    - `evaluator` (session-scoped) — shared Evaluator with all metrics configured.
    - `test_dataset` (session-scoped) — list of Q&A pairs from qa_pairs.json.
    - `mock_generator` (function-scoped) — patches Generator.generate; no OpenAI calls.
    - `mock_metrics` (function-scoped) — patches Evaluator._run_metric; no DeepEval calls.
"""

import pytest

from evaluation.evaluator import EvaluationResult, Evaluator
from evaluation.metrics import (
    FAITHFULNESS_THRESHOLD,
    HALLUCINATION_THRESHOLD,
    RELEVANCY_THRESHOLD,
)
from rag import RAGPipeline


@pytest.mark.evaluation
def test_hallucination_score_within_threshold(
    rag_pipeline: RAGPipeline,
    evaluator: Evaluator,
    test_dataset: list[dict[str, object]],
    mock_generator: None,
    mock_metrics: None,
) -> None:
    """Run first Q&A pair through full RAG + evaluation. Assert hallucination < 0.3.

    WHY: Hallucination is the most critical failure mode for a RAG system.
    Using a known factual question from the dataset gives a reproducible
    signal: if the answer drifts from the retrieved context, the score rises.

    Fixtures: rag_pipeline (pipeline), evaluator (metrics), test_dataset (Q&A pairs).
    """
    pair = test_dataset[0]
    result = rag_pipeline.query(pair["question"])

    eval_result: EvaluationResult = evaluator.evaluate(
        question=result["question"],  # type: ignore[arg-type]
        answer=result["answer"],  # type: ignore[arg-type]
        chunks=result["chunks"],  # type: ignore[arg-type]
    )

    assert eval_result.hallucination_score < HALLUCINATION_THRESHOLD, (
        f"Hallucination {eval_result.hallucination_score:.3f} >= threshold {HALLUCINATION_THRESHOLD}. "
        f"Reasons: {eval_result.failure_reasons}"
    )


@pytest.mark.evaluation
def test_answer_relevancy_above_threshold(
    rag_pipeline: RAGPipeline,
    evaluator: Evaluator,
    test_dataset: list[dict[str, object]],
    mock_generator: None,
    mock_metrics: None,
) -> None:
    """Verify answer relevancy score > 0.7 for a known factual question.

    WHY: A high relevancy score confirms the answer directly addresses the
    question. Using dataset entry q002 (file formats) gives a concrete,
    domain-specific question where an off-topic answer would be obvious.

    Fixtures: rag_pipeline (pipeline), evaluator (metrics), test_dataset (Q&A pairs).
    """
    pair = test_dataset[1]  # q002: file format question
    result = rag_pipeline.query(pair["question"])

    eval_result: EvaluationResult = evaluator.evaluate(
        question=result["question"],  # type: ignore[arg-type]
        answer=result["answer"],  # type: ignore[arg-type]
        chunks=result["chunks"],  # type: ignore[arg-type]
    )

    assert eval_result.relevancy_score >= RELEVANCY_THRESHOLD, (
        f"Relevancy {eval_result.relevancy_score:.3f} < threshold {RELEVANCY_THRESHOLD}. "
        f"Reasons: {eval_result.failure_reasons}"
    )


@pytest.mark.evaluation
def test_faithfulness_score_above_threshold(
    rag_pipeline: RAGPipeline,
    evaluator: Evaluator,
    test_dataset: list[dict[str, object]],
    mock_generator: None,
    mock_metrics: None,
) -> None:
    """Verify answer is grounded in retrieved context (faithfulness > 0.6).

    WHY: Faithfulness checks that every factual claim in the answer can be
    traced back to the retrieved chunks. A score below 0.6 means the LLM is
    introducing facts not in the context — the definition of hallucination
    at the claim level.

    Fixtures: rag_pipeline (pipeline), evaluator (metrics), test_dataset (Q&A pairs).
    """
    pair = test_dataset[3]  # q004: data protection question — verifiable claims
    result = rag_pipeline.query(pair["question"])

    eval_result: EvaluationResult = evaluator.evaluate(
        question=result["question"],  # type: ignore[arg-type]
        answer=result["answer"],  # type: ignore[arg-type]
        chunks=result["chunks"],  # type: ignore[arg-type]
    )

    assert eval_result.faithfulness_score >= FAITHFULNESS_THRESHOLD, (
        f"Faithfulness {eval_result.faithfulness_score:.3f} < threshold {FAITHFULNESS_THRESHOLD}. "
        f"Reasons: {eval_result.failure_reasons}"
    )


@pytest.mark.evaluation
def test_all_metrics_pass_for_clean_question(
    rag_pipeline: RAGPipeline,
    evaluator: Evaluator,
    mock_generator: None,
    mock_metrics: None,
) -> None:
    """Integration: a well-formed question should pass ALL 3 thresholds.

    WHY: Individual metric tests verify thresholds in isolation. This test
    confirms the pipeline passes holistically — a regression in any one
    metric surfaces here even if the individual tests haven't caught it yet.

    Fixtures: rag_pipeline (pipeline), evaluator (metrics).
    """
    result = rag_pipeline.query("What is the support response SLA for Enterprise plan customers?")

    eval_result: EvaluationResult = evaluator.evaluate(
        question=result["question"],  # type: ignore[arg-type]
        answer=result["answer"],  # type: ignore[arg-type]
        chunks=result["chunks"],  # type: ignore[arg-type]
    )

    assert eval_result.passed, (
        f"Expected all metrics to pass. Failures: {eval_result.failure_reasons}"
    )


@pytest.mark.evaluation
def test_evaluation_result_contains_all_fields(
    rag_pipeline: RAGPipeline,
    evaluator: Evaluator,
    mock_generator: None,
    mock_metrics: None,
) -> None:
    """Verify EvaluationResult dataclass has all required fields populated.

    WHY: Downstream reporting and assertion logic (other tests, HTML reports)
    depends on every EvaluationResult field being present and typed correctly.
    This test catches regressions if the evaluator ever returns a partial result.

    Fixtures: rag_pipeline (pipeline), evaluator (metrics).
    """
    result = rag_pipeline.query("What happens to my data after I cancel my subscription?")

    eval_result: EvaluationResult = evaluator.evaluate(
        question=result["question"],  # type: ignore[arg-type]
        answer=result["answer"],  # type: ignore[arg-type]
        chunks=result["chunks"],  # type: ignore[arg-type]
    )

    assert isinstance(eval_result.question, str) and len(eval_result.question) > 0
    assert isinstance(eval_result.answer, str) and len(eval_result.answer) > 0
    assert isinstance(eval_result.chunks, list) and len(eval_result.chunks) > 0
    assert isinstance(eval_result.hallucination_score, float)
    assert isinstance(eval_result.relevancy_score, float)
    assert isinstance(eval_result.faithfulness_score, float)
    assert isinstance(eval_result.passed, bool)
    assert isinstance(eval_result.failure_reasons, list)
