# Stage 04 — Pytest Framework Architecture

> Refer to CLAUDE.md for conftest scope, conventions, and structure.
> Stages 01–03 must be complete before this stage.

---

## Session Goal
Build the Pytest infrastructure — conftest files, fixtures, markers, hooks.
NO test cases yet. Think of this as building the test harness that test
cases will plug into.

---

## Files to Implement

### `conftest.py` (root) — Session-Scoped Fixtures
Scope: `session` — initialized ONCE for the entire test run.

Implement these fixtures:
```python
@pytest.fixture(scope="session")
def rag_pipeline() -> RAGPipeline:
    """Initialize and ingest documents once for all tests."""
    # Load from datasets/documents/
    # Returns ready-to-query RAGPipeline instance

@pytest.fixture(scope="session")
def evaluator() -> Evaluator:
    """Single Evaluator instance shared across all evaluation tests."""

@pytest.fixture(scope="session")
def test_dataset() -> list[dict]:
    """Load qa_pairs.json once for the whole session."""
```

Also implement:
- `pytest_configure(config)` — register all custom markers with descriptions
- `pytest_sessionstart(session)` — log test session start with timestamp
- `pytest_sessionfinish(session, exitstatus)` — log summary and report path

### `tests/conftest.py` — Function-Scoped Fixtures
Scope: `function` — fresh instance per test.

Implement:
```python
@pytest.fixture(scope="function")
def fresh_session(session_manager) -> ConversationSession:
    """New isolated conversation session per test.
    Equivalent to beforeEach/afterEach."""
    session = session_manager.create_session()
    yield session
    session_manager.close_session(session.id)  # cleanup = afterEach

@pytest.fixture(scope="function")
def rag_result(rag_pipeline, request) -> dict:
    """Pre-fetched RAG result for a question.
    Question provided via pytest.mark.parametrize or request.param."""
```

### `hooks/pytest_hooks.py` — Custom Hooks
Implement:
```python
def pytest_runtest_logreport(report):
    """Log PASSED/FAILED per test with test name and duration."""

def pytest_html_report_title(report):
    """Custom HTML report title."""

def pytest_html_results_summary(prefix, summary, postfix):
    """Add evaluation score summary to HTML report."""
```

### `pytest.ini` — Configuration
```ini
[pytest]
testpaths = tests
markers =
    retrieval: Tests for vector retrieval accuracy
    evaluation: Tests for AI evaluation metrics (hallucination, relevance)
    security: Tests for prompt injection and jailbreak resistance
    conversation: Tests for multi-turn session continuity
log_cli = true
log_cli_level = INFO
addopts = --html=reports/report.html --self-contained-html
```

---

## Lifecycle Mapping (For Interview Explanation)
| Pytest Concept | Equivalent | Where |
|---|---|---|
| `scope="session"` fixture setup | `beforeAll` | root `conftest.py` |
| `scope="session"` fixture teardown | `afterAll` | root `conftest.py` (yield) |
| `scope="function"` fixture setup | `beforeEach` | `tests/conftest.py` |
| `scope="function"` fixture teardown | `afterEach` | `tests/conftest.py` (yield) |
| `pytest_sessionstart` | Global setup hook | root `conftest.py` |
| `pytest_sessionfinish` | Global teardown hook | root `conftest.py` |

Include this table as a comment block in `conftest.py` — interviewers love it.

---

## Output Format
1. Full code for each file
2. Comment on EVERY fixture explaining its scope and why that scope was chosen
3. Show how a test function would USE these fixtures (signature only, no body)

---

## Validation Checklist
- [ ] Root conftest uses `scope="session"` for expensive resources
- [ ] Tests conftest uses `scope="function"` for isolation
- [ ] All markers registered in `pytest_configure`
- [ ] `yield` used in fixtures (not `return`) so teardown runs
- [ ] No test logic in conftest files
- [ ] `conftest.py` imports work without circular dependencies

---

## DO NOT in This Stage
- Do not write actual test functions yet (Stage 05)
- Do not implement session manager yet (Stage 06) — stub it if needed
- Do not add DeepEval assertions inside conftest
