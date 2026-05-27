# Stage 10 — Final Review & Interview Dry Run

> This is the final stage. All previous stages must be complete.
> Goal: Verify the framework runs end-to-end and prepare interview talking points.

---

## Session Goal
Run the full framework, fix any runtime errors, and generate
the interview talking points document.

---

## Step 1 — End-to-End Smoke Test
Run in this exact order and confirm each step passes:

```bash
# 1. Verify environment
python --version          # Must be 3.11.x
pip list | grep deepeval  # Must show version

# 2. Verify imports work
python -c "from rag.pipeline import RAGPipeline; print('RAG OK')"
python -c "from evaluation.evaluator import Evaluator; print('Eval OK')"
python -c "from session.manager import SessionManager; print('Session OK')"

# 3. Run retrieval tests only (fastest, no OpenAI cost)
pytest -m retrieval -v

# 4. Run full suite
pytest -v

# 5. Verify report generated
ls reports/report.html
```

For each step, report:
- ✅ PASS
- ❌ FAIL + error message + fix applied

---

## Step 2 — Common Issues to Check

| Issue | Where to Look | Fix |
|---|---|---|
| `ModuleNotFoundError` | Missing `__init__.py` | Add empty `__init__.py` |
| `ChromaDB collection already exists` | `vector_store.py` | Check `collection_exists()` before ingest |
| `openai.AuthenticationError` | `.env` not loaded | Verify `load_dotenv()` called at entry |
| `DeepEval API key required` | `metrics.py` | DeepEval 1.x needs `OPENAI_API_KEY` for evaluation |
| `sentence-transformers slow first run` | `embedder.py` | Model downloads on first use — expected |

---

## Step 3 — Interview Talking Points Document

Generate `interview_prep/talking_points.md` with:

### "Walk me through your framework architecture"
Write a 3-minute verbal explanation covering:
- The 3 layers (RAG / Evaluation / Test)
- Why they're separated
- How they connect at the fixture level

### "Why DeepEval over writing your own assertions?"
Prepare a 60-second answer covering:
- What DeepEval provides that simple string matching doesn't
- The 3 metrics and what failure looks like
- How thresholds were chosen

### "How is this different from a regular Pytest framework?"
Prepare a 90-second answer covering:
- AI evaluation as a test assertion (not just HTTP status codes)
- Non-determinism handling (why you don't assert exact strings)
- Session management for stateful AI testing

### "What would you add in a production version?"
Prepare a 2-minute answer covering the roadmap from README:
- LangSmith observability
- Async execution
- CI/CD integration
- Docker deployment

### "How do you handle non-determinism in AI tests?"
Answer should explain:
- Score-based thresholds instead of exact match assertions
- Running tests against stable, factual documents
- Why retrieval tests are deterministic (vector similarity is stable)
- Why evaluation tests have tolerance bands

---

## Step 4 — Final File Checklist
Confirm these files exist and are non-empty:

```
CLAUDE.md                         ✓
.env.example                      ✓
requirements.txt                  ✓
pytest.ini                        ✓
conftest.py                       ✓
rag/pipeline.py                   ✓
evaluation/evaluator.py           ✓
session/manager.py                ✓
security/payloads.py              ✓
tests/test_retrieval.py           ✓
tests/test_evaluation.py          ✓
tests/test_security.py            ✓
tests/test_conversation.py        ✓
observability/logger.py           ✓
datasets/documents/sample.txt     ✓
datasets/test_data/qa_pairs.json  ✓
README.md                         ✓
interview_prep/talking_points.md  ✓
```

---

## Output Format
1. Step-by-step smoke test results
2. Any fixes applied
3. Full `interview_prep/talking_points.md`
4. Final confirmation: "Ready for interview demo: YES / NO + reason"

---

## Validation Checklist
- [ ] `pytest` runs without import errors
- [ ] At least 1 test passes end-to-end (retrieval test is cheapest to verify)
- [ ] `reports/report.html` is generated
- [ ] Talking points cover all 5 common interview questions
- [ ] No `TODO` or `pass` statements in production code files

---

## Congratulations
If all checks pass, your MVP is interview-ready.
The framework demonstrates:
- ✅ AI evaluation understanding (DeepEval + 3 metrics)
- ✅ Pytest architecture maturity (fixtures, hooks, markers, scopes)
- ✅ RAG pipeline design thinking (loader → embedder → store → retriever → generator)
- ✅ Security awareness (prompt injection testing)
- ✅ Conversational AI testing (multi-turn session management)
