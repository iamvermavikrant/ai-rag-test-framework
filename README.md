# ai-rag-test-framework

**An AI-powered RAG (Retrieval-Augmented Generation) test framework demonstrating enterprise-grade AI evaluation architecture.**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pytest](https://img.shields.io/badge/Pytest-8.x-green)
![DeepEval](https://img.shields.io/badge/DeepEval-1.x-orange)

> Built for: SDET / AI Test Architect interview demonstration

---

## Architecture Overview

The framework is organized into three independent, vertically separated layers. Each layer has a single responsibility and can be evolved without touching the others.

- **RAG Layer** (`rag/`) — Handles document ingestion, embedding, vector storage, retrieval, prompt construction, and LLM generation. This is the system under test.
- **Evaluation Layer** (`evaluation/`) — Wraps DeepEval metrics (hallucination, answer relevance, faithfulness) and produces scored `EvaluationResult` objects that tests can assert against.
- **Test Layer** (`tests/`) — Pytest test suites that use fixtures to drive the RAG pipeline and evaluate outputs. No pipeline logic lives here.

```
        [Test Suites]
              ↓
   [Pytest Fixtures + Hooks]
              ↓
   [RAG Pipeline] ←→ [Evaluation Layer]
        ↓                    ↓
   [ChromaDB]          [DeepEval Metrics]
   [OpenAI API]
```

The Pytest fixture system provides the same lifecycle guarantees as `beforeAll` / `afterAll` / `beforeEach` / `afterEach` in other frameworks — but with explicit scoping rather than hidden global state. The RAG pipeline is initialized **once per session**; conversation sessions are created **fresh per test**.

---

## Project Structure

```
ai-rag-test-framework/
├── .env.example              # Required environment variable keys (no secrets)
├── .env                      # Actual secrets — gitignored, never committed
├── requirements.txt          # Pinned dependencies
├── pytest.ini                # Pytest settings: markers, HTML report path, log level
├── conftest.py               # Root conftest — session-scoped fixtures and hooks
├── rag/                      # RAG pipeline: load → chunk → embed → store → retrieve → generate
│   ├── loader.py             # Reads .txt files and splits them into overlapping chunks
│   ├── embedder.py           # Converts text chunks to vector embeddings (sentence-transformers)
│   ├── vector_store.py       # ChromaDB read/write operations
│   ├── retriever.py          # Similarity search — returns top-N chunks for a query
│   ├── prompt_builder.py     # Assembles retrieved chunks + query into an LLM prompt
│   └── generator.py          # Calls OpenAI and returns the generated answer
├── evaluation/               # AI evaluation layer — wraps DeepEval metrics
│   ├── metrics.py            # Metric definitions and threshold constants
│   └── evaluator.py          # Orchestrates multi-metric evaluation per test case
├── session/
│   └── manager.py            # Per-test conversation memory and session isolation
├── security/
│   └── payloads.py           # Prompt injection and jailbreak test inputs
├── datasets/
│   ├── documents/            # Source .txt files ingested into ChromaDB
│   └── test_data/
│       └── qa_pairs.json     # Question-answer pairs used in evaluation tests
├── prompts/
│   └── templates.py          # Prompt templates for RAG generation
├── fixtures/
│   └── rag_fixtures.py       # Reusable Pytest fixtures (additional helpers)
├── hooks/
│   └── pytest_hooks.py       # Custom Pytest hooks (HTML report enrichment)
├── observability/
│   └── logger.py             # Structured JSON logger — no print() in production code
├── tests/                    # All test suites
│   ├── conftest.py           # Function-scoped fixtures: fresh session per test
│   ├── test_retrieval.py     # Vector relevance, chunk count, latency tests
│   ├── test_evaluation.py    # Hallucination, answer relevance, faithfulness tests
│   ├── test_security.py      # Prompt injection and jailbreak resistance tests
│   └── test_conversation.py  # Multi-turn memory and session isolation tests
└── reports/                  # Generated HTML reports — gitignored
```

---

## Tech Stack & Why Each Was Chosen

| Tool | Version | Why |
|---|---|---|
| Python | 3.11 | Stable LTS release; type hints and `match` statements are fully supported |
| Pytest | 8.x | Industry standard; fixture scoping model maps cleanly to `beforeAll`/`afterAll`/`beforeEach` |
| OpenAI API | `openai>=1.30` | `gpt-4o-mini` is capable enough for RAG generation at low cost — no over-engineering |
| DeepEval | 1.x | Purpose-built for AI evaluation — provides hallucination, faithfulness, and relevance out of the box |
| ChromaDB | 0.5.x | Zero-config local vector DB; persistent on disk with no Docker or network dependency |
| sentence-transformers | 2.x | Free local embeddings via `all-MiniLM-L6-v2`; no OpenAI cost for the indexing step |
| pytest-html | 4.x | Generates a self-contained HTML report with custom columns for evaluation scores |
| python-dotenv | 1.x | Loads `.env` into `os.environ` at startup — keeps secrets out of code |

---

## Local Setup

```bash
git clone https://github.com/vikrantverma080/ai-rag-test-framework.git
cd ai-rag-test-framework

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Open .env and set OPENAI_API_KEY to your actual key
```

**.env values explained:**

| Variable | Example | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | `sk-...` | Used by generator and DeepEval judge |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM for RAG generation |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model |
| `CHROMA_PATH` | `./chroma_db` | Persistent ChromaDB directory |
| `CHROMA_COLLECTION` | `rag_collection` | Collection name inside ChromaDB |
| `REPORTS_DIR` | `./reports` | Output directory for HTML report |

---

## RAG Pipeline Flow

The pipeline follows a standard retrieve-then-generate pattern. All steps are implemented as focused single-responsibility classes in `rag/`.

1. **Load** — `loader.py` reads every `.txt` file from `datasets/documents/`.
2. **Chunk** — Each document is split into 500-character chunks with 50-character overlap. The overlap prevents sentences from being cut in half across chunk boundaries, which would hurt retrieval recall.
3. **Embed** — `embedder.py` converts each chunk to a dense vector using `all-MiniLM-L6-v2` (runs locally, no API call).
4. **Store** — Vectors and their source text are written to a persistent ChromaDB collection. This survives process restarts.
5. **Retrieve** — At query time, the question is embedded and ChromaDB returns the top-3 most similar chunks by cosine distance.
6. **Build prompt** — `prompt_builder.py` assembles a structured prompt: system instructions + retrieved chunks as context + the user question.
7. **Generate** — `generator.py` sends the prompt to `gpt-4o-mini` and returns the grounded answer.

---

## Pytest Lifecycle (Fixture Scopes)

Pytest fixtures replace `setUp`/`tearDown` with explicit, composable scopes. This framework uses two scopes:

| Scope | File | Fixture | Equivalent |
|---|---|---|---|
| `session` | `conftest.py` (root) | `rag_pipeline`, `evaluator`, `test_dataset`, `session_manager` | `beforeAll` / `afterAll` |
| `function` | `tests/conftest.py` | `fresh_session`, `test_metadata` | `beforeEach` / `afterEach` |

**Execution order for a single test:**

```
pytest_sessionstart                  ← global hook, fires once
  └─ rag_pipeline setup              ← session fixture: ingest docs, warm embedder
  └─ evaluator setup                 ← session fixture: build metric objects
       └─ fresh_session setup        ← function fixture: new conversation session
            └─ test_example runs
       └─ fresh_session teardown     ← session closed, memory cleared
  └─ rag_pipeline teardown           ← (no-op; ChromaDB persists to disk)
pytest_sessionfinish                 ← logs summary, report path
```

The session-scoped `rag_pipeline` fixture is the key performance win: loading the `sentence-transformers` model and writing embeddings to ChromaDB happens **once**, not once per test.

---

## DeepEval Integration Flow

DeepEval evaluates LLM outputs by running a second LLM (the "judge") that scores the answer against the question and retrieved context. All three metrics use `gpt-4o-mini` as the judge via a thin wrapper (`_GPT4oMiniJudge`) that injects the project's `OPENAI_API_KEY`.

### Metrics, Thresholds, and Reasoning

| Metric | What it measures | Threshold | Direction | Why this threshold |
|---|---|---|---|---|
| **Hallucination** | Fraction of claims in the answer that are NOT in the retrieved context | `< 0.3` | Lower is better | A score of 0.3 means 30% of claims are unsupported — anything above that is an unacceptable fabrication rate |
| **Answer Relevance** | How directly the answer addresses the question, regardless of context | `> 0.7` | Higher is better | Scores below 0.7 indicate the model is drifting off-topic or hedging excessively |
| **Faithfulness** | Fraction of claims in the answer that are directly supported by the context | `> 0.6` | Higher is better | Slightly more lenient than relevance because short answers can lose a few grounded points while still being correct |

### How test pass/fail is determined

```python
result = evaluator.evaluate(question, answer, context_chunks)
assert result.passed  # True only if ALL metrics meet their thresholds
```

`EvaluationResult.passed` is `True` only when every metric clears its threshold. A single metric failure fails the test. Scores are also attached to the Pytest report object so they appear as colored columns in the HTML report.

---

## Running Tests

```bash
# Run the full test suite
pytest

# Run by category
pytest -m retrieval
pytest -m evaluation
pytest -m security
pytest -m conversation

# Verbose output (shows each test name and PASSED/FAILED)
pytest -v

# Open the generated HTML report
open reports/report.html          # macOS/Linux
start reports\report.html         # Windows
```

> The test suite patches `Generator.generate` for the entire session so no live OpenAI calls are made during retrieval, security, and conversation tests. Only evaluation tests use a real judge model (configurable via `DEEPEVAL_MODEL` in `.env`).

---

## Test Suite Overview

| Suite | File | Tests | What it Validates |
|---|---|---|---|
| Retrieval | `test_retrieval.py` | 5 | Chunk relevance to query, correct chunk count, empty query edge case, latency < 2s, non-empty string type check |
| Evaluation | `test_evaluation.py` | 5 | Hallucination score, answer relevance score, faithfulness score, multi-metric combined pass, dataset-driven evaluation |
| Security | `test_security.py` | 6 | Prompt injection resistance, jailbreak attempt handling, instruction override attempts, output boundary enforcement |
| Conversation | `test_conversation.py` | 4 | Multi-turn context retention, session isolation between tests, memory reset on new session, conversation history accuracy |

---

## Future Roadmap (Interview Discussion Points)

These are deliberate omissions at MVP scope — each represents a real architectural decision:

- **LangSmith observability** — trace every RAG call (prompt, retrieved chunks, generation, latency) in a queryable UI; currently handled by the local structured logger
- **Playwright UI test layer** — if the RAG system has a chat frontend, Playwright E2E tests would sit above this framework as a 4th layer
- **CI/CD with GitHub Actions** — run `pytest -m retrieval -m security` on every PR; gate merges on evaluation score regressions
- **Docker containerization** — package ChromaDB + the test runner so any environment gets an identical baseline without Python version conflicts
- **LangGraph agent workflow testing** — if the RAG evolves into a multi-step agent (tool use, routing), the evaluation layer would need to track intermediate steps, not just final answers
- **Async test execution** — `pytest-asyncio` + async DeepEval calls to run evaluation tests in parallel; currently blocked on DeepEval's sync judge API
- **Parametrized evaluation dataset** — replace the 5 static `test_evaluation.py` tests with `@pytest.mark.parametrize` over `qa_pairs.json` for full dataset coverage in one test run
