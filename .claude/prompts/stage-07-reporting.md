# Stage 07 — Reporting & Observability

> Refer to CLAUDE.md for conventions and structure.
> Stages 01–06 must be complete before this stage.

---

## Session Goal
Implement structured logging and HTML report customization.
Reports must be readable by a non-technical interviewer.

---

## Module 1: `observability/logger.py` — Structured Logger

Implement:
```python
def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with structured format.
    Format: [TIMESTAMP] [LEVEL] [MODULE] message
    """
```

Logger must output:
- To console (stdout) with color support if possible
- To `reports/test_run.log` file
- Log level from `LOG_LEVEL` env var (default: INFO)

Include helper functions:
```python
def log_rag_query(logger, question: str, chunks: list[str], answer: str) -> None:
    """Structured log for a single RAG query-response cycle."""

def log_evaluation_result(logger, result: EvaluationResult) -> None:
    """Structured log for evaluation scores with PASS/FAIL indicators."""
```

Log format example:
```
[2024-01-15 10:23:45] [INFO] [rag.generator] RAG Query completed
  Question: What is the return policy?
  Chunks retrieved: 3
  Answer length: 142 chars
  Hallucination: 0.12 ✓ | Relevancy: 0.85 ✓ | Faithfulness: 0.78 ✓
  Status: PASSED
```

---

## Module 2: Custom `conftest.py` Hook — HTML Report Enrichment

Update root `conftest.py` to add evaluation scores to HTML report:

```python
def pytest_html_results_table_header(cells):
    """Add Hallucination / Relevancy / Faithfulness columns to HTML table."""

def pytest_html_results_table_row(report, cells):
    """Populate evaluation score columns per test row."""
```

Store scores on the report object during test execution:
```python
# Inside test (show how to attach metadata):
request.node.hallucination_score = result.hallucination_score
```

---

## Module 3: `observability/logger.py` — Test Summary
Add a session-end summary log:
```
========================================
TEST SESSION SUMMARY
========================================
Total Tests      : 18
Passed           : 15
Failed           : 3
Duration         : 42.3s
Avg Hallucination: 0.18
Avg Relevancy    : 0.79
Report           : reports/report.html
========================================
```
Log this inside `pytest_sessionfinish` hook.

---

## Output Format
1. Full `observability/logger.py`
2. Updated `conftest.py` hooks for HTML enrichment
3. Sample output showing what the HTML report looks like (ASCII mockup is fine)
4. Sample log file output (5–8 lines showing the format)

---

## Validation Checklist
- [ ] Logger writes to both console AND `reports/test_run.log`
- [ ] Log level controlled by env var
- [ ] HTML report includes evaluation score columns (not just PASS/FAIL)
- [ ] Session summary prints to console at the end of the run
- [ ] Logger is imported from `observability/logger.py` — no raw `print()` anywhere
- [ ] Log file path is from `REPORTS_DIR` env var

---

## DO NOT in This Stage
- Do not use external logging services (no Datadog, no Splunk)
- Do not implement LangSmith or Phoenix tracing
- Do not make logging async
