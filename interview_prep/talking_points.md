# Interview Talking Points — AI RAG Test Framework

---

## "Walk me through your framework architecture"
*(~3 minutes)*

The framework is organized into three distinct layers that mirror a classic test pyramid, adapted for AI systems.

**Layer 1 — RAG Pipeline** (`rag/`)
This is the system under test. It's a six-stage document pipeline: a Loader reads and chunks source documents, an Embedder converts chunks to vectors using a local sentence-transformers model, a VectorStore persists those vectors in ChromaDB, a Retriever queries the store for semantically similar chunks at query time, a PromptBuilder assembles the retrieved context into a structured prompt, and a Generator calls the OpenAI API to produce a final answer. Each stage is a clean class — no business logic leaks into tests.

**Layer 2 — Evaluation** (`evaluation/`)
This layer wraps DeepEval and defines the assertion layer for AI outputs. Because AI responses are non-deterministic, we can't assert exact strings. Instead, `metrics.py` defines score-based metrics — hallucination, answer relevance, and faithfulness — and `evaluator.py` orchestrates running them against a (question, context, answer) triple.

**Layer 3 — Tests** (`tests/`)
The Pytest layer consumes both layers above purely through fixtures. Tests never import RAG classes directly — they receive a fully-initialized `rag_pipeline` fixture from `conftest.py`. This is the Page Object pattern applied to AI: the test only cares about behavior, not internals.

**How they connect:** The root `conftest.py` initializes the RAG pipeline once at session scope, which is expensive (embedding model load + ChromaDB connection). The `tests/conftest.py` creates a fresh conversation session per test function, ensuring isolation. No test shares mutable state.

---

## "Why DeepEval over writing your own assertions?"
*(~60 seconds)*

Simple string matching fails for AI for two reasons: the answer is paraphrased differently each run, and partial correctness isn't binary — "somewhat relevant" is a meaningful signal.

DeepEval provides three things I'd otherwise have to build myself:

1. **Hallucination score** — measures how much of the answer is unsupported by the retrieved context. A score of 0.0 means every claim is grounded. We fail above 0.3.
2. **Answer Relevance** — measures whether the answer actually addresses the question asked, not just whether it's factually correct. We require above 0.7.
3. **Faithfulness** — measures whether the answer stays within the retrieved context versus drawing on outside knowledge. We require above 0.6.

The thresholds were set conservatively for a factual document corpus. A hallucination threshold of 0.3 means the model can lightly paraphrase without failing — but if it's fabricating 30% of its answer, that's a test failure.

---

## "How is this different from a regular Pytest framework?"
*(~90 seconds)*

Three things make AI testing structurally different from API or UI testing.

**First, assertions are score-based, not binary.** In a regular test you assert `response.status_code == 200` or `"error" not in body`. For an AI response you assert `hallucination_score < 0.3`. The test is checking a distribution of quality, not a discrete outcome. DeepEval gives us that numeric layer.

**Second, non-determinism is a first-class concern.** The same question asked twice will get slightly different answers. Writing `assert answer == expected_string` would produce flaky tests. We handle this by evaluating against stable, factual source documents, and by using score thresholds with tolerance bands wide enough to absorb phrasing variation while still catching real regressions.

**Third, AI tests have stateful sessions.** A conversational AI maintains a history of prior turns — question 3's answer depends on questions 1 and 2. `test_conversation.py` tests this explicitly: it sends two turns and asserts the second response references context from the first. The `SessionManager` wraps that state, and the `function`-scoped fixture in `tests/conftest.py` ensures each test starts with a clean session so tests can't contaminate each other.

---

## "What would you add in a production version?"
*(~2 minutes)*

Four areas in priority order:

**1. Observability with LangSmith.**
Right now we log structured JSON but don't trace individual RAG steps. LangSmith would give us a flamegraph-style trace of each pipeline stage — chunk retrieval latency, token counts, which chunks were actually used in the final answer. That's essential for debugging regressions in production.

**2. Async execution.**
The current pipeline is fully synchronous. For a production load test or parallel evaluation run, you'd wrap the OpenAI call in `asyncio` and run evaluations concurrently. This is an architectural change to `generator.py` and `evaluator.py` but the rest of the stack is already stateless and would benefit immediately.

**3. CI/CD integration.**
The framework already runs cleanly as `pytest -v` with an HTML report. The next step is a GitHub Actions workflow that runs the retrieval tests on every PR (cheap, no API cost) and the full evaluation suite on merge to main. The `pytest-xdist` plugin is already installed for parallelism.

**4. Docker deployment.**
Containerizing the framework with a pinned Python image and a mounted ChromaDB volume makes the environment fully reproducible — no "works on my machine" issues with sentence-transformers model downloads or ChromaDB path conflicts. A `docker-compose.yml` with a test runner service and a volume for the DB would be the MVP.

---

## "How do you handle non-determinism in AI tests?"
*(~90 seconds)*

Non-determinism in AI testing comes from three sources: the LLM generating slightly different text each run, the embedding model having version drift over time, and the retrieved chunks changing if the document corpus is updated. Here's how each is handled:

**Score-based thresholds instead of exact match.** No test in this framework asserts a specific string. Every AI output assertion is a numeric threshold — hallucination below 0.3, relevance above 0.7. This gives a tolerance band that absorbs natural variation without hiding genuine failures.

**Retrieval tests are deterministic.** Vector similarity scoring is stable for the same model version and the same documents. `test_retrieval.py` asserts structural properties — the right number of chunks returned, non-empty strings, sub-100ms latency — not semantic content. These tests can run in CI with zero flakiness risk.

**Evaluation tests run against controlled, stable documents.** The `datasets/documents/sample.txt` corpus is static and factual. We're not testing the LLM's general knowledge — we're testing whether it stays grounded in the provided context. Because the source material doesn't change between runs, the evaluation scores are stable within a few percentage points.

**Session isolation prevents cross-test contamination.** Each test function gets a fresh `SessionManager` instance via a `function`-scoped fixture. If a prior test produced an unexpected response, it cannot bleed into the next test's conversation history.
