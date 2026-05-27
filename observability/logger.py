"""Structured logging for the RAG test framework.

All modules import get_logger(__name__) from here — no print() anywhere.
Log output goes to both stdout (with optional color) and reports/test_run.log.
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluation.evaluator import EvaluationResult

# ---------------------------------------------------------------------------
# ANSI color codes — used only when the handler writes to a real terminal
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_COLORS: dict[int, str] = {
    logging.DEBUG: "\033[36m",     # cyan
    logging.INFO: "\033[32m",      # green
    logging.WARNING: "\033[33m",   # yellow
    logging.ERROR: "\033[31m",     # red
    logging.CRITICAL: "\033[35m",  # magenta
}


class _ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI color to the level name when writing to a TTY."""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelno, "")
        record.levelname = f"{color}{_BOLD}{record.levelname:<8}{_RESET}"
        return super().format(record)


_LOG_FMT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# Module-level cache so each name only gets one logger configured
_configured: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with structured format.

    Format: [TIMESTAMP] [LEVEL] [MODULE] message

    Writes to:
      - stdout (colored when attached to a TTY)
      - reports/test_run.log (plain text, path from REPORTS_DIR env var)

    Log level is controlled by the LOG_LEVEL env var (default: INFO).

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if name in _configured:
        return logger

    _configured.add(name)

    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logger.setLevel(log_level)

    # --- Console handler (colored when stdout is a TTY) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    if console_handler.stream.isatty():  # type: ignore[union-attr]
        console_handler.setFormatter(_ColorFormatter(_LOG_FMT, datefmt=_DATE_FMT))
    else:
        console_handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
    logger.addHandler(console_handler)

    # --- File handler (plain text, always) ---
    reports_dir = Path(os.getenv("REPORTS_DIR", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_file = reports_dir / "test_run.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT))
    logger.addHandler(file_handler)

    # Prevent propagation to the root logger to avoid duplicate lines
    logger.propagate = False

    return logger


# ---------------------------------------------------------------------------
# Structured helper: RAG query cycle
# ---------------------------------------------------------------------------

def log_rag_query(
    logger: logging.Logger,
    question: str,
    chunks: list[str],
    answer: str,
) -> None:
    """Structured log for a single RAG query-response cycle.

    Args:
        logger: The caller's logger instance.
        question: The user question sent to the RAG pipeline.
        chunks: Context chunks retrieved from the vector store.
        answer: The generated answer from the LLM.
    """
    logger.info(
        "RAG Query completed\n"
        "  Question      : %s\n"
        "  Chunks fetched: %d\n"
        "  Answer length : %d chars",
        question,
        len(chunks),
        len(answer),
    )


# ---------------------------------------------------------------------------
# Structured helper: evaluation result
# ---------------------------------------------------------------------------

def log_evaluation_result(logger: logging.Logger, result: "EvaluationResult") -> None:
    """Structured log for evaluation scores with PASS/FAIL indicators.

    Uses check (✓) and cross (✗) symbols so a human scanning logs can spot
    failing metrics without reading the numbers.

    Args:
        logger: The caller's logger instance.
        result: The EvaluationResult produced by Evaluator.evaluate().
    """
    hall_sym = "✓" if result.hallucination_score < 0.3 else "✗"
    rel_sym = "✓" if result.relevancy_score > 0.7 else "✗"
    faith_sym = "✓" if result.faithfulness_score > 0.6 else "✗"
    status = "PASSED" if result.passed else "FAILED"

    log_fn = logger.info if result.passed else logger.warning
    log_fn(
        "Evaluation result\n"
        "  Question     : %s\n"
        "  Hallucination: %.2f %s | Relevancy: %.2f %s | Faithfulness: %.2f %s\n"
        "  Status       : %s",
        result.question,
        result.hallucination_score,
        hall_sym,
        result.relevancy_score,
        rel_sym,
        result.faithfulness_score,
        faith_sym,
        status,
    )


# ---------------------------------------------------------------------------
# Session summary (called from pytest_sessionfinish in conftest.py)
# ---------------------------------------------------------------------------

def log_session_summary(
    logger: logging.Logger,
    total: int,
    passed: int,
    failed: int,
    duration: float,
    avg_hallucination: float,
    avg_relevancy: float,
    report_path: str,
) -> None:
    """Print a fixed-width session summary block at the end of the run.

    Args:
        logger: The caller's logger instance.
        total: Total number of tests collected.
        passed: Number of tests that passed.
        failed: Number of tests that failed.
        duration: Total wall-clock duration in seconds.
        avg_hallucination: Mean hallucination score across evaluation tests (0 if none ran).
        avg_relevancy: Mean relevancy score across evaluation tests (0 if none ran).
        report_path: Absolute or relative path to the HTML report.
    """
    bar = "=" * 44
    logger.info(
        "\n%s\n"
        "TEST SESSION SUMMARY\n"
        "%s\n"
        "Total Tests      : %d\n"
        "Passed           : %d\n"
        "Failed           : %d\n"
        "Duration         : %.1fs\n"
        "Avg Hallucination: %.2f\n"
        "Avg Relevancy    : %.2f\n"
        "Report           : %s\n"
        "%s",
        bar,
        bar,
        total,
        passed,
        failed,
        duration,
        avg_hallucination,
        avg_relevancy,
        report_path,
        bar,
    )
