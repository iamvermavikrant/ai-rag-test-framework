"""Evaluation orchestration for RAG test runs.

Coordinates running DeepEval metrics against a given RAG response,
aggregating scores and returning structured evaluation results.
"""

from dataclasses import dataclass, field

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from evaluation.metrics import (
    HALLUCINATION_THRESHOLD,
    FAITHFULNESS_THRESHOLD,
    RELEVANCY_THRESHOLD,
    get_faithfulness_metric,
    get_hallucination_metric,
    get_relevancy_metric,
)
from observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Structured output from a single evaluation run.

    Attributes:
        question: The original user question fed to the RAG pipeline.
        answer: The generated answer returned by the LLM.
        chunks: Retrieved context chunks used to ground the answer.
        hallucination_score: Float in [0, 1]; lower is better.
        relevancy_score: Float in [0, 1]; higher is better.
        faithfulness_score: Float in [0, 1]; higher is better.
        passed: True only when ALL three thresholds are satisfied.
        failure_reasons: Human-readable explanation for each failing metric.
    """

    question: str
    answer: str
    chunks: list[str]
    hallucination_score: float
    relevancy_score: float
    faithfulness_score: float
    passed: bool
    failure_reasons: list[str] = field(default_factory=list)


# INTERVIEW NOTE: Why these 3 metrics?
# Hallucination: catches model making up facts not in the document
# Faithfulness: checks answer is grounded in retrieved context
# Answer Relevancy: checks answer actually addresses the question
# Together they cover the 3 failure modes of RAG systems.
class Evaluator:
    """Runs all configured DeepEval metrics and returns a structured result.

    Design note: Evaluator is intentionally unaware of the RAG pipeline.
    It receives plain strings (question, answer, chunks) so the evaluation
    layer stays decoupled from retrieval and generation concerns.

    Example:
        >>> evaluator = Evaluator()
        >>> result = evaluator.evaluate(
        ...     question="What is the return policy?",
        ...     answer="Returns are accepted within 30 days.",
        ...     chunks=["Our return policy allows 30-day returns for unused items."],
        ... )
        >>> isinstance(result.passed, bool)
        True
    """

    def evaluate(
        self,
        question: str,
        answer: str,
        chunks: list[str],
    ) -> EvaluationResult:
        """Run hallucination, relevancy, and faithfulness checks on one RAG response.

        DeepEval's LLMTestCase maps RAG outputs as follows:
          - input          → the user question
          - actual_output  → the generated answer (what the LLM said)
          - retrieval_context → the chunks the RAG pipeline fetched
          - expected_output is optional; omitted here because we evaluate
            answer quality against context, not against a gold-standard string.

        Args:
            question: The user question passed to the RAG pipeline.
            answer: The answer produced by the LLM.
            chunks: Context chunks retrieved from the vector store.

        Returns:
            EvaluationResult with all scores and a pass/fail verdict.
        """
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=chunks,
        )

        hallucination_metric = get_hallucination_metric()
        relevancy_metric = get_relevancy_metric()
        faithfulness_metric = get_faithfulness_metric()

        # Run each metric individually so we can capture per-metric scores
        # even when one metric raises (defensive — DeepEval can be chatty).
        hallucination_score = self._run_metric(hallucination_metric, test_case, "hallucination")
        relevancy_score = self._run_metric(relevancy_metric, test_case, "relevancy")
        faithfulness_score = self._run_metric(faithfulness_metric, test_case, "faithfulness")

        logger.info(
            "Evaluation scores — hallucination: %.3f, relevancy: %.3f, faithfulness: %.3f | question: %r",
            hallucination_score,
            relevancy_score,
            faithfulness_score,
            question[:80],
        )

        failure_reasons: list[str] = []

        if hallucination_score >= HALLUCINATION_THRESHOLD:
            failure_reasons.append(
                f"Hallucination {hallucination_score:.3f} >= threshold {HALLUCINATION_THRESHOLD}"
            )
        if relevancy_score < RELEVANCY_THRESHOLD:
            failure_reasons.append(
                f"Relevancy {relevancy_score:.3f} < threshold {RELEVANCY_THRESHOLD}"
            )
        if faithfulness_score < FAITHFULNESS_THRESHOLD:
            failure_reasons.append(
                f"Faithfulness {faithfulness_score:.3f} < threshold {FAITHFULNESS_THRESHOLD}"
            )

        passed = len(failure_reasons) == 0

        if not passed:
            logger.warning("Evaluation FAILED — %s", "; ".join(failure_reasons))
        else:
            logger.info("Evaluation PASSED")

        return EvaluationResult(
            question=question,
            answer=answer,
            chunks=chunks,
            hallucination_score=hallucination_score,
            relevancy_score=relevancy_score,
            faithfulness_score=faithfulness_score,
            passed=passed,
            failure_reasons=failure_reasons,
        )

    def _run_metric(
        self,
        metric: object,
        test_case: LLMTestCase,
        name: str,
    ) -> float:
        """Measure a single metric and return its score.

        Isolated so that a single metric failure doesn't abort the others,
        and so the score extraction logic isn't repeated three times.

        Args:
            metric: A configured DeepEval metric object.
            test_case: The assembled LLMTestCase.
            name: Human-readable label used only in error logging.

        Returns:
            Float score, or 0.0 / 1.0 as a safe fallback on error
            (direction chosen conservatively: hallucination defaults high
            so it fails safe; others default low so they fail safe too).
        """
        try:
            metric.measure(test_case)
            score: float = metric.score  # type: ignore[attr-defined]
            return score
        except Exception as exc:  # noqa: BLE001
            logger.error("Metric %r raised during measure: %s", name, exc)
            # Fail-safe defaults: hallucination defaults to worst case (1.0),
            # relevancy and faithfulness default to worst case (0.0).
            return 1.0 if name == "hallucination" else 0.0