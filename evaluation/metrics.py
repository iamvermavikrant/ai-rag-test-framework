"""DeepEval metric definitions for RAG evaluation.

Defines hallucination, answer relevance, and faithfulness metrics
with configured thresholds used across all evaluation test cases.
"""

import os

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric
from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI

# Thresholds are fixed in CLAUDE.md — do not change them here.
HALLUCINATION_THRESHOLD: float = 0.3   # lower is better; fail if score >= threshold
RELEVANCY_THRESHOLD: float = 0.7       # higher is better; fail if score < threshold
FAITHFULNESS_THRESHOLD: float = 0.6    # higher is better; fail if score < threshold


class _GPT4oMiniJudge(DeepEvalBaseLLM):
    """Thin wrapper that points DeepEval's judge at gpt-4o-mini.

    DeepEval's default judge resolves via its own API key flow.
    This wrapper injects the project's OPENAI_API_KEY and model
    so evaluation stays on a single key with no extra setup.
    """

    def __init__(self) -> None:
        self._model_name: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_model_name(self) -> str:
        """Return the model identifier string."""
        return self._model_name

    def load_model(self) -> "OpenAI":
        """Return the underlying OpenAI client (satisfies DeepEval interface)."""
        return self._client

    def generate(self, prompt: str) -> str:
        """Run a single prompt through the judge model and return the text.

        Args:
            prompt: Evaluation prompt assembled by DeepEval internally.

        Returns:
            Raw string response from the model.
        """
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    async def a_generate(self, prompt: str) -> str:
        """Async variant — delegates to synchronous generate for simplicity."""
        return self.generate(prompt)


def _judge() -> _GPT4oMiniJudge:
    """Return a shared judge instance (constructed fresh each call is fine for tests)."""
    return _GPT4oMiniJudge()


def get_hallucination_metric() -> HallucinationMetric:
    """Return a configured HallucinationMetric.

    Hallucination measures whether the generated answer introduces facts
    that are NOT grounded in the retrieved context chunks.  A score of 0
    means no hallucination; a score of 1 means the answer is entirely
    fabricated.  We fail if the score is >= 0.3.

    Returns:
        HallucinationMetric with threshold and judge pre-configured.
    """
    return HallucinationMetric(
        threshold=HALLUCINATION_THRESHOLD,
        model=_judge(),
        include_reason=True,
    )


def get_relevancy_metric() -> AnswerRelevancyMetric:
    """Return a configured AnswerRelevancyMetric.

    Answer relevancy measures how directly the generated answer addresses
    the user's question, independent of the context.  A score of 1 means
    the answer is perfectly on-topic; 0 means it is entirely off-topic.
    We require a score >= 0.7.

    Returns:
        AnswerRelevancyMetric with threshold and judge pre-configured.
    """
    return AnswerRelevancyMetric(
        threshold=RELEVANCY_THRESHOLD,
        model=_judge(),
        include_reason=True,
    )


def get_faithfulness_metric() -> FaithfulnessMetric:
    """Return a configured FaithfulnessMetric.

    Faithfulness (groundedness) measures what fraction of the claims in
    the generated answer can be directly supported by the retrieved context
    chunks.  A score of 1 means every claim is grounded; 0 means none are.
    We require a score >= 0.6.

    Returns:
        FaithfulnessMetric with threshold and judge pre-configured.
    """
    return FaithfulnessMetric(
        threshold=FAITHFULNESS_THRESHOLD,
        model=_judge(),
        include_reason=True,
    )