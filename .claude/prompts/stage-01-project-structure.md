# Stage 01 — Project Structure & Scaffolding

> Refer to CLAUDE.md for full stack, conventions, and folder structure.

---

## Session Goal

Generate the complete project scaffold — folder structure, empty files with docstrings,
configuration files, and dependency list. NO implementation code yet.

---

## Task Breakdown

### 1. Generate Folder + File Tree

Create the exact structure defined in CLAUDE.md > Project Structure.

- Add `__init__.py` wherever needed for Python imports
- Add placeholder `.py` files with:
  - Module-level docstring (2–3 lines explaining purpose)
  - No implementation yet

### 2. Generate `requirements.txt`

Use pinned versions from CLAUDE.md > Tech Stack.
Include:

```
openai>=1.30.0
deepeval>=1.0.0
chromadb>=0.5.0
sentence-transformers>=2.0.0
pytest>=8.0.0
pytest-html>=4.0.0
python-dotenv>=1.0.0
```

### 3. Generate `.env.example`

```
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-4o-mini
DEEPEVAL_MODEL=gpt-4o-mini
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PATH=./chroma_db
CHROMA_COLLECTION=rag_collection
LOG_LEVEL=INFO
REPORTS_DIR=./reports
```

### 4. Generate `pytest.ini`

Include:

- testpaths = tests
- markers: retrieval, evaluation, security, conversation
- log_cli = true
- html report default path

### 5. Generate Sample Dataset File

Create `datasets/documents/sample.txt` with 10–15 lines of realistic
domain content (e.g., a product FAQ or policy document) that will be used
for RAG ingestion in Stage 2.

Create `datasets/test_data/qa_pairs.json` with 3–5 Q&A pairs based on
the sample document.

---

## Output Format

1. Full folder tree with all files listed
2. Each file's content (placeholder files just need docstring + imports)
3. `requirements.txt`
4. `.env.example`
5. `pytest.ini`
6. Sample dataset files

---

## Validation Checklist

Before finishing, confirm:

- [ ] All folders have `__init__.py`
- [ ] No implementation code exists yet — only docstrings
- [ ] `.env` is in `.gitignore`
- [ ] `reports/` and `chroma_db/` are in `.gitignore`
- [ ] requirements.txt has pinned versions

---

## DO NOT in This Stage

- Do not implement any RAG logic
- Do not write test cases
- Do not import modules that don't exist yet
