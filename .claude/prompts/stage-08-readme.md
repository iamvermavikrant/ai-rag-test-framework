# Stage 08 — README Documentation

> Refer to CLAUDE.md for the full architecture picture.
> All implementation stages (01–07) must be complete before writing the README.

---

## Session Goal
Generate a production-grade `README.md` that helps an interviewer understand
your architectural thinking in under 10 minutes of reading.

---

## README Structure (Required Sections in Order)

### 1. Header
- Project name + one-line description
- Badges: Python version, Pytest, DeepEval
- "Built for: SDET/AI Test Architect interview demonstration"

### 2. Architecture Overview
- 1 paragraph explaining the 3-layer architecture:
  - **RAG Layer** → `rag/`
  - **Evaluation Layer** → `evaluation/`
  - **Test Layer** → `tests/`
- ASCII architecture diagram:
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

### 3. Project Structure
- Full folder tree (copy from CLAUDE.md but add one-line description per folder)

### 4. Tech Stack & Why Each Was Chosen
Table format:
| Tool | Version | Why |
|---|---|---|
| Pytest | 8.x | Industry-standard, fixture model maps to beforeAll/afterAll |
| DeepEval | 1.x | Purpose-built AI evaluation — hallucination, faithfulness, relevance |
| ChromaDB | 0.5.x | Zero-config local vector DB, no Docker needed |
| sentence-transformers | 2.x | Free local embeddings, no OpenAI cost for indexing |

### 5. Local Setup
Step-by-step:
```bash
git clone ...
cd ai-rag-test-framework
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add OPENAI_API_KEY
```

### 6. RAG Pipeline Flow
Numbered walkthrough:
1. Documents loaded from `datasets/documents/`
2. Text chunked (500 chars, 50 overlap)
3. Chunks embedded with sentence-transformers
4. Embeddings stored in ChromaDB (persistent)
5. At query time: question embedded → top-3 chunks retrieved
6. Prompt built with chunks as context
7. OpenAI generates grounded answer

### 7. Pytest Lifecycle (Fixture Scopes)
Explain session vs function scope with the table from Stage 04.
Show the execution order for a single test run.

### 8. DeepEval Integration Flow
Explain:
- What each metric measures (plain English)
- Threshold values and why they were chosen
- How `EvaluationResult.passed` gates test pass/fail

### 9. Running Tests
```bash
# Run all tests
pytest

# Run by marker
pytest -m retrieval
pytest -m evaluation
pytest -m security
pytest -m conversation

# Run with verbose output
pytest -v

# Open report
open reports/report.html
```

### 10. Test Suite Overview
Table:
| Suite | File | Tests | What it Validates |
|---|---|---|---|
| Retrieval | test_retrieval.py | 5 | Chunk relevance, speed, count |
| Evaluation | test_evaluation.py | 5 | Hallucination, relevancy, faithfulness |
| Security | test_security.py | 6 | Prompt injection, jailbreak |
| Conversation | test_conversation.py | 4 | Multi-turn, session isolation |

### 11. Future Roadmap (Interview Discussion Points)
List as bullet points:
- LangSmith for production observability
- Playwright UI test layer
- CI/CD with GitHub Actions
- Docker containerization
- LangGraph for agent workflow testing
- Async test execution for speed

---

## Writing Style for README
- Write for an **interviewer who is technical but unfamiliar with this codebase**
- Every section should answer: "Why does this exist?"
- Avoid jargon without explanation
- No marketing language — be honest about what's MVP scope

---

## Output Format
Full `README.md` content ready to paste.

---

## Validation Checklist
- [ ] Setup steps are copy-paste executable (tested mentally)
- [ ] Architecture diagram is ASCII (no image dependencies)
- [ ] Evaluation thresholds are documented with reasoning
- [ ] Future roadmap section shows you know what's missing
- [ ] Every `pytest -m marker` command actually works
