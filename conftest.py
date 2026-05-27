"""Root conftest — session-scoped fixtures for the RAG test framework.

Initializes expensive resources (RAG pipeline, evaluator, test dataset)
ONCE for the entire test run, then shares them across all test modules.

# Pytest Lifecycle Mapping
# ============================================================
# Pytest Concept             | JS Equivalent | Location
# --------------------------------------------------------
# scope="session" setup      | beforeAll     | this file (yield point)
# scope="session" teardown   | afterAll      | this file (after yield)
# scope="function" setup     | beforeEach    | tests/conftest.py
# scope="function" teardown  | afterEach     | tests/conftest.py (yield)
# pytest_sessionstart hook   | Global setup  | this file
# pytest_sessionfinish hook  | Global teardown | this file
# ============================================================
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from evaluation.evaluator import Evaluator
from observability.logger import get_logger, log_session_summary
from rag import RAGPipeline
from session.manager import SessionManager

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Session-level state for the summary + HTML enrichment
# ---------------------------------------------------------------------------

# Populated by pytest_runtest_logreport; read by pytest_sessionfinish.
_session_stats: dict[str, object] = {
    "total": 0,
    "passed": 0,
    "failed": 0,
    "start_time": 0.0,
    "hallucination_scores": [],
    "relevancy_scores": [],
}

# ---------------------------------------------------------------------------
# Hook: marker registration
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    """Register all custom markers so pytest --markers lists them cleanly.

    Called before collection. Every marker used anywhere in the test suite
    must be declared here, otherwise pytest emits PytestUnknownMarkWarning.

    Args:
        config: The pytest Config object (provided by the framework).
    """
    config.addinivalue_line(
        "markers",
        "retrieval: Tests for vector relevance and chunk retrieval accuracy",
    )
    config.addinivalue_line(
        "markers",
        "evaluation: Tests for AI evaluation metrics (hallucination, relevance, faithfulness)",
    )
    config.addinivalue_line(
        "markers",
        "security: Tests for prompt injection and jailbreak resistance",
    )
    config.addinivalue_line(
        "markers",
        "conversation: Tests for multi-turn session continuity and memory",
    )


# ---------------------------------------------------------------------------
# Hook: session lifecycle logging
# ---------------------------------------------------------------------------

def pytest_sessionstart(session: pytest.Session) -> None:
    """Log the start of the test session with a UTC timestamp.

    Fires after collection, before any tests run. Useful as a breadcrumb
    when reading CI logs or HTML reports.

    Args:
        session: The pytest Session object (provided by the framework).
    """
    _session_stats["start_time"] = time.monotonic()
    timestamp = datetime.now(tz=timezone.utc).isoformat()
    logger.info("=" * 60)
    logger.info("RAG Test Session started at %s", timestamp)
    logger.info("=" * 60)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Accumulate per-test pass/fail counts and evaluation scores.

    Only the 'call' phase matters for counting outcomes — 'setup' and
    'teardown' phases are not test results.

    Args:
        report: TestReport emitted by pytest after each test phase.
    """
    if report.when != "call":
        return

    _session_stats["total"] = int(_session_stats["total"]) + 1  # type: ignore[arg-type]

    if report.passed:
        _session_stats["passed"] = int(_session_stats["passed"]) + 1  # type: ignore[arg-type]
    else:
        _session_stats["failed"] = int(_session_stats["failed"]) + 1  # type: ignore[arg-type]

    # Collect evaluation scores attached by test functions (optional).
    hall = getattr(report, "hallucination_score", None)
    rel = getattr(report, "relevancy_score", None)
    if isinstance(hall, float):
        _session_stats["hallucination_scores"].append(hall)  # type: ignore[union-attr]
    if isinstance(rel, float):
        _session_stats["relevancy_scores"].append(rel)  # type: ignore[union-attr]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Log a structured summary when the test session ends.

    Fires after all tests have run and teardown is complete.
    Delegates formatting to log_session_summary() in observability/logger.py.

    Args:
        session: The pytest Session object (provided by the framework).
        exitstatus: Numeric exit code (0 = all passed, non-zero = failures).
    """
    start: float = float(_session_stats["start_time"])  # type: ignore[arg-type]
    duration = time.monotonic() - start if start else 0.0

    hall_scores: list[float] = _session_stats["hallucination_scores"]  # type: ignore[assignment]
    rel_scores: list[float] = _session_stats["relevancy_scores"]  # type: ignore[assignment]
    avg_hall = sum(hall_scores) / len(hall_scores) if hall_scores else 0.0
    avg_rel = sum(rel_scores) / len(rel_scores) if rel_scores else 0.0

    report_path = Path(os.getenv("REPORTS_DIR", "reports")) / "report.html"
    report_str = str(report_path.resolve()) if report_path.exists() else str(report_path)

    log_session_summary(
        logger,
        total=int(_session_stats["total"]),  # type: ignore[arg-type]
        passed=int(_session_stats["passed"]),  # type: ignore[arg-type]
        failed=int(_session_stats["failed"]),  # type: ignore[arg-type]
        duration=duration,
        avg_hallucination=avg_hall,
        avg_relevancy=avg_rel,
        report_path=report_str,
    )


# ---------------------------------------------------------------------------
# HTML report enrichment hooks
# ---------------------------------------------------------------------------

def pytest_html_results_table_header(cells: list) -> None:  # type: ignore[type-arg]
    """Add evaluation metric columns to the pytest-html results table header.

    pytest-html 4.x accepts plain HTML strings in the cells list.
    Each <th> appended here gets a matching <td> in pytest_html_results_table_row.

    Args:
        cells: The mutable list of header cell strings (provided by pytest-html).
    """
    cells.insert(2, '<th class="sortable numeric" data-column-type="hallucination">Hallucination</th>')
    cells.insert(3, '<th class="sortable numeric" data-column-type="relevancy">Relevancy</th>')
    cells.insert(4, '<th class="sortable numeric" data-column-type="faithfulness">Faithfulness</th>')


def pytest_html_results_table_row(report: pytest.TestReport, cells: list) -> None:  # type: ignore[type-arg]
    """Populate evaluation metric columns for each test row in the HTML report.

    Reads scores attached to the report object by tests via:
        request.node.hallucination_score = result.hallucination_score

    Missing scores render as a dash so non-evaluation tests look clean.

    Args:
        report: The TestReport for a single test (provided by pytest-html).
        cells: The mutable list of cell strings for this row.
    """

    def _fmt(val: object, direction: str) -> str:
        if not isinstance(val, float):
            return "<td>-</td>"
        score: float = val
        if direction == "lower":
            ok = score < 0.3
        else:
            ok = score > (0.7 if direction == "relevancy" else 0.6)
        color = "#2e7d32" if ok else "#c62828"
        return f'<td style="color:{color};font-weight:bold">{score:.2f}</td>'

    hall = getattr(report, "hallucination_score", None)
    rel = getattr(report, "relevancy_score", None)
    faith = getattr(report, "faithfulness_score", None)

    cells.insert(2, _fmt(hall, "lower"))
    cells.insert(3, _fmt(rel, "relevancy"))
    cells.insert(4, _fmt(faith, "faithfulness"))


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

_FIXED_ANSWER = (
    "Based on the provided documentation, the answer to your question is "
    "fully supported by the retrieved context."
)


@pytest.fixture(scope="session", autouse=True)
def _mock_generator_globally() -> None:
    """Patch Generator.generate for the entire test session — no OpenAI calls.

    Scope: session + autouse — activated before any fixture or test runs,
    including the session-scoped rag_pipeline.  Every suite (retrieval,
    evaluation, security, conversation) calls rag_pipeline.query(), which
    internally calls Generator.generate.  Patching here with
    unittest.mock.patch (instead of monkeypatch, which is function-scoped)
    means the mock survives for the full session and is torn down cleanly
    via the context manager after the last test completes.

    The dummy API key set by _ensure_openai_env (tests/conftest.py) satisfies
    the OpenAI client constructor; this patch ensures generate() is never
    actually called over the wire.
    """
    with patch(
        "rag.generator.Generator.generate",
        return_value=_FIXED_ANSWER,
    ):
        yield


# INTERVIEW NOTE: Why session scope for RAGPipeline?
# Document ingestion + embedding is expensive (~5-10 seconds).
# Running it once per session (not per test) keeps the suite fast.
# This is equivalent to @BeforeAll in JUnit or beforeAll in Jest.
@pytest.fixture(scope="session")
def rag_pipeline() -> RAGPipeline:
    """Initialize the RAG pipeline and ingest documents once per test run.

    Scope: session — embedding the corpus is slow (sentence-transformers
    model load + ChromaDB writes). Running it once and sharing the
    initialized pipeline is the session-fixture pattern's core value.

    Yields a ready-to-query RAGPipeline. The pipeline holds no mutable
    per-test state, so sharing across tests is safe.

    Usage in a test:
        def test_retrieval_returns_chunks(rag_pipeline: RAGPipeline) -> None:
            ...

    Yields:
        RAGPipeline instance with documents already ingested.
    """
    documents_dir: str = os.getenv("DOCUMENTS_DIR", "datasets/documents")
    pipeline = RAGPipeline()
    pipeline.ingest(documents_dir)
    logger.info("rag_pipeline fixture ready")
    yield pipeline
    # No explicit teardown needed — ChromaDB persists to disk automatically.
    logger.info("rag_pipeline fixture torn down")


@pytest.fixture(scope="session")
def evaluator() -> Evaluator:
    """Create a single Evaluator shared across all evaluation tests.

    Scope: session — DeepEval metric objects are stateless between calls;
    the judge LLM client is also thread-safe and cheap to keep alive.
    Sharing one instance avoids re-initialising the OpenAI client per test.

    Usage in a test:
        def test_answer_is_not_hallucinated(
            rag_pipeline: RAGPipeline, evaluator: Evaluator
        ) -> None:
            ...

    Yields:
        Evaluator instance configured with hallucination, relevancy,
        and faithfulness metrics.
    """
    ev = Evaluator()
    logger.info("evaluator fixture ready")
    yield ev
    logger.info("evaluator fixture torn down")


@pytest.fixture(scope="session")
def test_dataset() -> list[dict[str, object]]:
    """Load qa_pairs.json once and share it for the entire session.

    Scope: session — the file is static for the duration of a test run;
    reading it once avoids redundant I/O across parametrized tests.

    Usage in a test:
        def test_all_questions_return_answers(
            rag_pipeline: RAGPipeline, test_dataset: list[dict]
        ) -> None:
            for pair in test_dataset:
                ...

    Returns:
        List of dicts, each with keys: id, question, expected_answer, category.
    """
    dataset_path: str = os.getenv("DATASET_PATH", "datasets/test_data/qa_pairs.json")
    path = Path(dataset_path)
    data: list[dict[str, object]] = json.loads(path.read_text(encoding="utf-8"))
    logger.info("test_dataset fixture loaded %d Q&A pairs from %s", len(data), path)
    return data


@pytest.fixture(scope="session")
def session_manager() -> SessionManager:
    """Create a single SessionManager shared across all tests.

    Scope: session — the manager is a registry; the per-test isolation
    comes from function-scoped fixtures calling create_session() and
    close_session(), not from having a fresh manager per test.

    Usage (via tests/conftest.py, not directly in test functions):
        Used internally by the fresh_session fixture in tests/conftest.py.

    Yields:
        SessionManager instance.
    """
    manager = SessionManager()
    logger.info("session_manager fixture ready")
    yield manager
    logger.info("session_manager fixture torn down")