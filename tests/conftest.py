"""Test-level conftest — function-scoped fixtures for individual tests.

Provides a fresh conversation session and pre-fetched RAG results for
each test function, ensuring full state isolation between tests.

Every fixture here uses scope="function" (the default). This matches
beforeEach/afterEach semantics: setup runs before the test body, and
teardown (code after yield) runs after it — even if the test fails.

# Why we mock in evaluation tests
# ============================================================
# The evaluation suite validates pipeline *wiring* — that scores flow
# through EvaluationResult correctly and that thresholds are asserted in
# the right direction.  It does NOT validate model quality (that requires
# a real API key and is done in manual / CI integration runs).
#
# Two external calls are mocked:
#   1. Generator.generate  — replaces the OpenAI chat completion with a
#      fixed answer string.  Keeps retrieval tests fast and free.
#   2. Evaluator._run_metric — replaces the DeepEval metric.measure() +
#      metric.score read with fixed floats chosen to sit safely on the
#      passing side of every threshold.  Patching _run_metric rather than
#      the individual DeepEval metric classes means the mock is isolated
#      to our own code and is immune to DeepEval API changes.
#
# A third concern is construction-time: openai.OpenAI() validates the API
# key in __init__, which runs inside the session-scoped rag_pipeline fixture
# before any function-scoped mock can activate.  A session-scoped autouse
# fixture sets a dummy OPENAI_API_KEY env var so the client constructs
# without error.  The key is never used for a real call because
# Generator.generate is patched before any test body runs.
#
# The fixtures are function-scoped so they are applied fresh per test and
# automatically undone by monkeypatch teardown — no manual cleanup needed.
# ============================================================
"""

from typing import Generator

import pytest

from observability.logger import get_logger
from rag import RAGPipeline
from session.manager import ConversationSession, SessionManager

logger = get_logger(__name__)


@pytest.fixture(scope="function")
def fresh_session(session_manager: SessionManager) -> Generator[ConversationSession, None, None]:
    """Provide a new isolated conversation session for each test.

    Scope: function — each test must start with an empty message history.
    Sharing sessions would let one test's queries pollute another's context,
    breaking multi-turn isolation tests entirely.

    The yield splits setup (before) from teardown (after), mirroring
    beforeEach/afterEach without needing a class or explicit finalizer.

    Usage in a test:
        def test_session_remembers_first_turn(
            fresh_session: ConversationSession,
        ) -> None:
            ...

    Yields:
        A new ConversationSession with an empty message history.

    Teardown:
        Closes the session and clears its message history after the test.
    """
    session = session_manager.create_session()
    logger.debug("fresh_session created: %s", session.id)
    yield session
    # Teardown — equivalent to afterEach: clean up regardless of pass/fail.
    session_manager.close_session(session.id)
    logger.debug("fresh_session closed: %s", session.id)


@pytest.fixture(scope="function")
def rag_result(rag_pipeline: RAGPipeline, request: pytest.FixtureRequest) -> dict[str, object]:
    """Fetch a pre-built RAG result for the question supplied by the test.

    Scope: function — each test may ask a different question and needs its
    own retrieved chunks + generated answer. Caching at session scope would
    return stale results for parametrized tests with different questions.

    The question is supplied via pytest.mark.parametrize or request.param:

        @pytest.mark.parametrize("rag_result", ["How do I reset my password?"], indirect=True)
        def test_answer_not_empty(rag_result: dict) -> None:
            assert rag_result["answer"]

    Args:
        rag_pipeline: Session-scoped pipeline (injected by pytest).
        request: Pytest FixtureRequest — carries the indirect parameter.

    Returns:
        Dict with keys: question (str), chunks (list[str]),
        prompt (str), answer (str).

    Raises:
        pytest.UsageError: If no question parameter is provided via indirect.
    """
    question: str = getattr(request, "param", None)  # type: ignore[assignment]
    if not question:
        raise pytest.UsageError(
            "rag_result requires a question via indirect parametrize:\n"
            '  @pytest.mark.parametrize("rag_result", ["your question"], indirect=True)'
        )
    logger.debug("rag_result fixture querying pipeline: %r", question)
    result: dict[str, object] = rag_pipeline.query(question)
    return result


# ---------------------------------------------------------------------------
# Environment guard — ensures openai.OpenAI() constructs without a real key
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _ensure_openai_env(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Set a dummy OPENAI_API_KEY so the OpenAI client constructs without error.

    Scope: session, autouse — runs once before any fixture or test in this
    process, including the session-scoped rag_pipeline fixture.

    Why needed: openai.OpenAI.__init__ raises OpenAIError immediately when
    OPENAI_API_KEY is absent.  That happens during RAGPipeline construction,
    which is before any function-scoped mock can activate.  Setting a dummy
    value satisfies the SDK's credential check; the key is never sent over the
    wire because Generator.generate is patched before every evaluation test.

    The env var is set for the lifetime of the test session and left in place
    (os.environ changes in session scope are not auto-reverted by monkeypatch,
    but the process exits after the session, so there is no leak).
    """
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key-for-local-mocked-runs"


# ---------------------------------------------------------------------------
# Mock fixtures — evaluation suite (no OpenAI calls)
# ---------------------------------------------------------------------------

_FIXED_ANSWER = (
    "Based on the provided documentation, the answer to your question is "
    "fully supported by the retrieved context."
)

# Scores are chosen to pass every threshold defined in CLAUDE.md:
#   hallucination < 0.3  →  0.05 passes
#   relevancy     > 0.7  →  0.90 passes
#   faithfulness  > 0.6  →  0.85 passes
_METRIC_SCORES: dict[str, float] = {
    "hallucination": 0.05,
    "relevancy": 0.90,
    "faithfulness": 0.85,
}


@pytest.fixture()
def mock_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch Generator.generate to return a fixed string without hitting OpenAI.

    Scope: function — monkeypatch automatically undoes the patch after each test.

    Why this target: Generator.generate is the single method responsible for
    all OpenAI chat completion calls in the RAG pipeline.  Patching it at the
    class level intercepts every call regardless of how the Generator instance
    was constructed (session-scoped rag_pipeline included).
    """
    monkeypatch.setattr(
        "rag.generator.Generator.generate",
        lambda self, prompt: _FIXED_ANSWER,
    )


@pytest.fixture()
def mock_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch Evaluator._run_metric to return fixed floats without calling DeepEval.

    Scope: function — monkeypatch automatically undoes the patch after each test.

    Why this target: _run_metric is the single choke-point that calls
    metric.measure() and reads metric.score.  Patching it means we never
    import or instantiate DeepEval metric classes during evaluation tests,
    so the mock is immune to changes in DeepEval's internal API.

    The name argument passed by Evaluator._run_metric is used to select the
    correct fixed score from _METRIC_SCORES.
    """
    monkeypatch.setattr(
        "evaluation.evaluator.Evaluator._run_metric",
        lambda self, metric, test_case, name: _METRIC_SCORES[name],
    )