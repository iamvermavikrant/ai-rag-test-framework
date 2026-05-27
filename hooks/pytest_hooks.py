"""Custom Pytest hooks for the RAG test framework.

Registered as a plugin via conftest.py imports. Enriches the HTML report
with evaluation context and logs per-test outcomes for CI readability.

These hooks use the pytest plugin API — they are called by the framework
automatically when the hook name matches a registered hook spec.
"""

from observability.logger import get_logger

logger = get_logger(__name__)


def pytest_runtest_logreport(report: object) -> None:
    """Log PASSED or FAILED for each test with its name and duration.

    Called three times per test: for the setup, call, and teardown phases.
    We only log the "call" phase — that is the actual test body outcome.
    Setup/teardown failures are still captured by pytest's own output.

    Args:
        report: pytest.TestReport object (typed as object to avoid a
                hard import — pytest injects this at runtime).
    """
    # report is a pytest.TestReport; use getattr for type-checker safety
    phase: str = getattr(report, "when", "")
    if phase != "call":
        return

    nodeid: str = getattr(report, "nodeid", "unknown")
    duration: float = getattr(report, "duration", 0.0)
    passed: bool = getattr(report, "passed", False)
    failed: bool = getattr(report, "failed", False)

    if passed:
        logger.info("PASSED  [%.3fs]  %s", duration, nodeid)
    elif failed:
        longrepr = getattr(report, "longreprtext", "") or str(getattr(report, "longrepr", ""))
        # Only log the first line — full traceback is in the HTML report.
        first_line = longrepr.splitlines()[0] if longrepr else "no details"
        logger.warning("FAILED  [%.3fs]  %s  —  %s", duration, nodeid, first_line)


def pytest_html_report_title(report: object) -> None:  # type: ignore[override]
    """Set a descriptive title on the generated HTML report.

    Called by pytest-html after the report object is created but before
    the HTML is rendered.

    Args:
        report: pytest_html report object with a `title` attribute.
    """
    report.title = "AI RAG Test Framework — Evaluation Report"  # type: ignore[attr-defined]


def pytest_html_results_summary(
    prefix: list,  # list of html tags prepended to the summary section
    summary: list,  # list of html tags for the main summary section
    postfix: list,  # list of html tags appended after the summary section
) -> None:
    """Append evaluation metric context to the HTML report summary section.

    Called by pytest-html when building the results summary block.
    The prefix/summary/postfix lists accept raw HTML strings that
    pytest-html injects directly into the page.

    Args:
        prefix: HTML prepended before the default summary table.
        summary: HTML for the main summary block.
        postfix: HTML appended after the default summary table.
    """
    postfix.append(
        "<p><strong>Evaluation Thresholds:</strong> "
        "Hallucination &lt; 0.3 &nbsp;|&nbsp; "
        "Answer Relevance &gt; 0.7 &nbsp;|&nbsp; "
        "Faithfulness &gt; 0.6"
        "</p>"
    )
    postfix.append(
        "<p><em>LLM judge: gpt-4o-mini &nbsp;|&nbsp; "
        "Vector store: ChromaDB (local) &nbsp;|&nbsp; "
        "Embeddings: sentence-transformers</em></p>"
    )