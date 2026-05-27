# Stage 09 — Interview Polish & Code Review

> Refer to CLAUDE.md for conventions.
> All stages 01–08 must be complete before this stage.

---

## Session Goal
Review the entire codebase for interview readiness.
An interviewer will walk through this code live — it must be explainable,
consistent, and clean. This is a code audit pass, not new feature work.

---

## Review Checklist (Go Through Each)

### 1. Docstring Completeness
Verify every file has:
- Module-level docstring (what this file does, 2–3 lines)
- Class-level docstring (what this class models)
- Method-level docstring on all public methods

Flag any missing docstrings and generate them.

### 2. Type Hint Consistency
- Every function signature must have argument types and return type
- No bare `dict` — use `dict[str, str]` or define a TypedDict
- No `Any` allowed

Audit all files in: `rag/`, `evaluation/`, `session/`, `tests/`

### 3. Error Handling
Check every external call:
- `openai.ChatCompletion` → wrapped in try/except with clear error message
- `chromadb` operations → handle `CollectionNotFoundError`
- File I/O in `loader.py` → handle `FileNotFoundError`
- All errors logged before raising

### 4. Naming Consistency
Verify:
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Test functions: start with `test_` + describe what is tested in plain English
- Fixture names: descriptive nouns (`rag_pipeline`, not `rp` or `pipeline`)

### 5. Import Organization
Every file should follow:
```python
# 1. Standard library
import os
import uuid

# 2. Third-party
import openai
import chromadb

# 3. Internal modules
from rag.pipeline import RAGPipeline
```

Fix any files that mix import order.

### 6. Test Quality
For each test function, verify:
- Single responsibility (one thing tested)
- Meaningful assertion message: `assert score < 0.3, f"Hallucination too high: {score}"`
- No `time.sleep()` or hardcoded waits
- Fixture used correctly (not bypassed)

### 7. Interview "Explain This" Prep
Add a comment block to these key files that an interviewer might ask about:

In `rag/retriever.py`:
```python
# INTERVIEW NOTE: Why sentence-transformers over OpenAI embeddings?
# sentence-transformers runs locally (free, no API call).
# OpenAI embeddings cost ~$0.0001/1K tokens — acceptable in production
# but unnecessary for an MVP demo. The retrieval quality is comparable
# for small document sets.
```

In `evaluation/evaluator.py`:
```python
# INTERVIEW NOTE: Why these 3 metrics?
# Hallucination: catches model making up facts not in the document
# Faithfulness: checks answer is grounded in retrieved context
# Answer Relevancy: checks answer actually addresses the question
# Together they cover the 3 failure modes of RAG systems.
```

In `conftest.py` (root):
```python
# INTERVIEW NOTE: Why session scope for RAGPipeline?
# Document ingestion + embedding is expensive (~5-10 seconds).
# Running it once per session (not per test) keeps the suite fast.
# This is equivalent to @BeforeAll in JUnit or beforeAll in Jest.
```

---

## Output Format
1. List every issue found (file + line + issue type)
2. Corrected code for each issue
3. Final "interview readiness score" out of 10 with reasoning

---

## Validation Checklist
- [ ] Zero functions without docstrings
- [ ] Zero bare `except:` clauses
- [ ] Zero hardcoded values (check with grep for any quoted URLs or numbers)
- [ ] All 3 INTERVIEW NOTE comments added to key files
- [ ] Test assertion messages are descriptive (`assert x, "explanation"`)
- [ ] `requirements.txt` matches what is actually imported

---

## DO NOT in This Stage
- Do not add new features or new test cases
- Do not refactor working logic — only fix quality issues
- Do not add comments that explain WHAT the code does (the code should be self-explanatory)
- Only add comments that explain WHY a decision was made
