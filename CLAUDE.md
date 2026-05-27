# ai-rag-test-framework

## What This Project Is

An AI-powered RAG (Retrieval-Augmented Generation) test framework MVP.
Built to demonstrate enterprise AI testing architecture and automation maturity for an SDET/EPAM interview.

**Goal:** Show architectural clarity + AI evaluation understanding + automation maturity.
**NOT a goal:** Build a production SaaS platform.

---

## Tech Stack (Pinned Versions)

| Tool                  | Version      | Purpose                       |
| --------------------- | ------------ | ----------------------------- |
| Python                | 3.11         | Core language                 |
| Pytest                | 8.x          | Test runner and framework     |
| OpenAI API            | openai>=1.30 | LLM — use `gpt-4o-mini` model |
| DeepEval              | 1.x          | AI evaluation metrics         |
| ChromaDB              | 0.5.x        | Local persistent vector DB    |
| sentence-transformers | 2.x          | Embeddings (local, free)      |
| pytest-html           | 4.x          | HTML reporting                |
| python-dotenv         | 1.x          | Env var management            |

---

## Project Structure

```
ai-rag-test-framework/
├── CLAUDE.md                        # Claude instruction file (this file)
├── .claude/
│   └── prompts/                     # Stage-by-stage session prompts
├── .env.example                     # Required env keys (no secrets)
├── .env                             # Actual secrets (gitignored)
├── requirements.txt                 # Pinned dependencies
├── pytest.ini                       # Pytest configuration
├── conftest.py                      # Root conftest — session-scoped fixtures
├── README.md                        # Project documentation
├── rag/
│   ├── __init__.py
│   ├── loader.py                    # Document loading and chunking
│   ├── embedder.py                  # Embedding generation
│   ├── vector_store.py              # ChromaDB operations
│   ├── retriever.py                 # Chunk retrieval logic
│   ├── prompt_builder.py            # Prompt construction
│   └── generator.py                 # OpenAI response generation
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                   # DeepEval metric definitions
│   └── evaluator.py                 # Evaluation orchestration
├── session/
│   ├── __init__.py
│   └── manager.py                   # Conversation memory and session isolation
├── security/
│   ├── __init__.py
│   └── payloads.py                  # Prompt injection and jailbreak test inputs
├── datasets/
│   ├── documents/                   # Source documents for RAG ingestion
│   │   └── sample.txt
│   └── test_data/
│       └── qa_pairs.json            # Question-answer pairs for evaluation
├── prompts/
│   ├── __init__.py
│   └── templates.py                 # Prompt templates
├── fixtures/
│   ├── __init__.py
│   └── rag_fixtures.py              # Reusable Pytest fixtures
├── hooks/
│   └── pytest_hooks.py              # Custom Pytest hooks
├── tests/
│   ├── conftest.py                  # Test-level conftest — function-scoped fixtures
│   ├── test_retrieval.py            # Vector relevance and chunk retrieval tests
│   ├── test_evaluation.py           # Hallucination, groundedness, relevance tests
│   ├── test_security.py             # Prompt injection and jailbreak tests
│   └── test_conversation.py         # Multi-turn and session continuity tests
├── observability/
│   ├── __init__.py
│   └── logger.py                    # Structured logging
└── reports/                         # Generated HTML reports (gitignored)
```

---

## Architecture Decisions

- **Strict separation of concerns** — RAG pipeline, evaluation, and test layers are independent modules
- **Page Object equivalent** — RAG pipeline is wrapped in clean classes, not called raw in tests
- **Local-first** — everything runs offline except OpenAI API call (ChromaDB is persistent local)
- **Session isolation** — each test function gets a fresh session; no shared state between tests
- **Fixtures over setup/teardown** — follow Pytest best practices throughout

---

## conftest.py Scope (Important)

| File                 | Scope      | Contains                                           |
| -------------------- | ---------- | -------------------------------------------------- |
| `conftest.py` (root) | `session`  | RAG pipeline init, ChromaDB client, OpenAI client  |
| `tests/conftest.py`  | `function` | Fresh conversation session per test, test metadata |

---

## DeepEval Evaluation Thresholds

| Metric           | Threshold | Assert Direction |
| ---------------- | --------- | ---------------- |
| Hallucination    | < 0.3     | Lower is better  |
| Answer Relevance | > 0.7     | Higher is better |
| Groundedness     | > 0.6     | Higher is better |

---

## Coding Conventions

- Python 3.11 type hints on all functions (`def fn(x: str) -> dict:`)
- Docstrings on all classes and public methods (Google style)
- All secrets via `.env` — never hardcoded
- `BASE_URL`, `OPENAI_API_KEY`, `CHROMA_PATH` always from environment
- No `print()` in production code — use the logger from `observability/logger.py`
- Prefer explicit variable names over one-liners
- Each test function tests exactly ONE thing

---

## Current Sprint

> ⬇️ Update this section at the start of each work session

- [Done] Stage 1 — Project structure and scaffolding
- [Done] Stage 2 — RAG pipeline implementation
- [Done] Stage 3 — DeepEval integration
- [Done] Stage 4 — Pytest framework setup
- [Done] Stage 5 — Test suites
- [Done] Stage 6 — Session management
- [Done] Stage 7 — Reporting
- [Done] Stage 8 — README
- [Done] Stage 9 — Interview polish
- [ ] Stage 10 — Final review

---

## How Claude Should Help

- Write clean, readable, explainable code — an interviewer will read this
- Follow existing patterns in the codebase before introducing new ones
- Ask ONE clarifying question if requirement is ambiguous — then proceed
- Always check if a utility/fixture already exists before creating a new one
- Explain WHY a design decision was made, not just WHAT the code does
- Keep implementations minimal — avoid over-engineering

---

## DO NOT

- Use `test.only` or `test.skip` without a comment
- Hardcode any URLs, API keys, or file paths
- Mix RAG pipeline logic inside test files
- Use `Any` type annotation — be explicit
- Generate code without checking existing utils/fixtures first
- Add dependencies not listed in the tech stack without asking
- Use LangChain, LangSmith, LangGraph, or any agent framework
- Use Docker, Kubernetes, or distributed components
