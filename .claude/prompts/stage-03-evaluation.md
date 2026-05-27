# Stage 03 — AI Evaluation with DeepEval

> Refer to CLAUDE.md for thresholds, conventions, and structure.
> Stage 02 (RAG Pipeline) must be complete before this stage.
> Note: Use OpenAI gpt-4o-mini as DeepEval judge. Single OPENAI_API_KEY.

---

## Session Goal

Implement the evaluation layer using DeepEval. This layer sits BETWEEN
the RAG pipeline and the test assertions. Tests call the evaluator —
they do not call DeepEval directly.

---

## Modules to Implement

### `evaluation/metrics.py` — Metric Definitions

Define DeepEval metric objects with thresholds from CLAUDE.md:

```python
# Use these exact thresholds (from CLAUDE.md):
# Hallucination  < 0.3
# AnswerRelevancy > 0.7
# Faithfulness   > 0.6
```

Implement:

- `get_hallucination_metric() -> HallucinationMetric`
- `get_relevancy_metric() -> AnswerRelevancyMetric`
- `get_faithfulness_metric() -> FaithfulnessMetric`
- Each function returns a configured metric object
- Add docstring explaining what each metric measures in plain English

### `evaluation/evaluator.py` — Evaluation Orchestration

Implement:

- `EvaluationResult` dataclass:
  ```python
  @dataclass
  class EvaluationResult:
      question: str
      answer: str
      chunks: list[str]
      hallucination_score: float
      relevancy_score: float
      faithfulness_score: float
      passed: bool
      failure_reasons: list[str]
  ```
- `Evaluator` class:
  - `evaluate(question: str, answer: str, chunks: list[str]) -> EvaluationResult`
  - Runs all 3 metrics
  - Sets `passed = True` only if ALL thresholds are met
  - Populates `failure_reasons` for any failing metric
  - Logs scores using `observability/logger.py`

---

## Test Data Integration

Update `datasets/test_data/qa_pairs.json` format:

```json
[
  {
    "id": "q001",
    "question": "What is the return policy?",
    "expected_answer": "...",
    "category": "retrieval"
  }
]
```

---

## Output Format

For each file:

1. Full implementation
2. Explain how DeepEval LLMTestCase maps to RAG outputs
3. Show a standalone usage example (no pytest yet):
   ```python
   # Example: how evaluator.py is used
   result = evaluator.evaluate(question, answer, chunks)
   print(result.passed, result.hallucination_score)
   ```

---

## Validation Checklist

- [ ] Thresholds match CLAUDE.md exactly
- [ ] `EvaluationResult.passed` is False if ANY metric fails
- [ ] `failure_reasons` lists which metric failed and by how much
- [ ] Evaluator does NOT import from `tests/` — one-way dependency only
- [ ] All scores are logged with structured logger
- [ ] `evaluation/` has no dependency on `session/` or `security/`

---

## DO NOT in This Stage

- Do not write pytest test cases yet (that is Stage 05)
- Do not call `RAGPipeline` from inside `evaluator.py`
- Do not use LangSmith or any external observability platform
- Do not change thresholds — they are fixed in CLAUDE.md
