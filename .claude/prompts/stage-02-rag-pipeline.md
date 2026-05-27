# Stage 02 — Local RAG Pipeline Implementation

> Refer to CLAUDE.md for full stack, conventions, and folder structure.
> Stage 01 must be complete before starting this stage.

---

## Session Goal
Implement the full local RAG pipeline across 6 modules inside `rag/`.
Each module has ONE responsibility. Keep each file under 80 lines.

---

## Modules to Implement

### `rag/loader.py` — Document Loading & Chunking
Implement:
- `load_documents(directory: str) -> list[str]` — reads all `.txt` files from a folder
- `chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]` — splits text into overlapping chunks
- Use only Python stdlib (no LangChain)
- Add docstring explaining chunking strategy

### `rag/embedder.py` — Embedding Generation
Implement:
- `Embedder` class
- Use `sentence-transformers` with model from `EMBEDDING_MODEL` env var
- `embed(texts: list[str]) -> list[list[float]]` — returns embedding vectors
- Lazy-load the model on first call (not at import time)
- Add docstring explaining embedding choice

### `rag/vector_store.py` — ChromaDB Operations
Implement:
- `VectorStore` class
- Initialize ChromaDB persistent client from `CHROMA_PATH` env var
- `add_documents(chunks: list[str], embeddings: list[list[float]]) -> None`
- `query(embedding: list[float], n_results: int = 3) -> list[str]`
- `collection_exists() -> bool` — check before re-ingesting
- Use collection name from `CHROMA_COLLECTION` env var

### `rag/retriever.py` — Retrieval Orchestration
Implement:
- `Retriever` class (composes `Embedder` + `VectorStore`)
- `retrieve(query: str, n_results: int = 3) -> list[str]` — end-to-end retrieval
- Returns list of relevant chunks

### `rag/prompt_builder.py` — Prompt Construction
Implement:
- `build_prompt(query: str, context_chunks: list[str]) -> str`
- Uses template from `prompts/templates.py`
- Context chunks are clearly delimited in the prompt
- Includes instruction to answer ONLY from context (groundedness)

### `rag/generator.py` — OpenAI Response Generation
Implement:
- `Generator` class
- Initialize OpenAI client with key from env
- `generate(prompt: str) -> str` — calls `gpt-4o-mini`, returns answer string
- Handle `openai.APIError` with a clear error message
- Log token usage to logger

---

## Integration Script
Create `rag/pipeline.py`:
- `RAGPipeline` class that composes all 6 modules
- `ingest(directory: str) -> None` — load, chunk, embed, store
- `query(question: str) -> dict` — retrieve, build prompt, generate
  - Returns: `{"question": str, "answer": str, "chunks": list[str]}`
- This is the ONLY class tests will import from the `rag` module

---

## Output Format
For each file:
1. Full implementation code
2. 2-line comment explaining WHY this module exists separately
3. Example usage (in a docstring `Example:` block)

---

## Validation Checklist
- [ ] No RAG logic leaks into test files
- [ ] All env vars loaded via `os.getenv()` with defaults where safe
- [ ] `RAGPipeline.query()` returns a dict with `question`, `answer`, `chunks`
- [ ] ChromaDB uses persistent storage (not in-memory)
- [ ] Embedder uses sentence-transformers, NOT OpenAI embeddings (cost saving)
- [ ] `pipeline.py` is the single public interface for all tests

---

## DO NOT in This Stage
- Do not write test cases
- Do not use LangChain for any part of the pipeline
- Do not call OpenAI embeddings — use sentence-transformers only
- Do not hardcode file paths or model names
