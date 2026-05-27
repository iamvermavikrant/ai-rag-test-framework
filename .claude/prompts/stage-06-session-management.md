# Stage 06 — Conversation Session Management

> Refer to CLAUDE.md for conventions and structure.
> Stages 01–05 must be complete before this stage.

---

## Session Goal
Implement lightweight in-memory conversational session management.
This must be simple enough to explain in 2 minutes in an interview.

---

## Module to Implement: `session/manager.py`

### Data Structures
```python
@dataclass
class Message:
    role: str        # "user" or "assistant"
    content: str
    timestamp: str   # ISO format

@dataclass
class ConversationSession:
    id: str                    # UUID
    history: list[Message]     # Full conversation history
    created_at: str            # ISO format
    metadata: dict             # Optional tags (e.g., test_name)
```

### `SessionManager` Class
Implement:
```python
class SessionManager:
    def __init__(self):
        self._sessions: dict[str, ConversationSession] = {}

    def create_session(self, metadata: dict = None) -> ConversationSession:
        """Create a new isolated session with unique UUID."""

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session history."""

    def get_history(self, session_id: str) -> list[Message]:
        """Return full conversation history for a session."""

    def build_context_prompt(self, session_id: str, new_question: str) -> str:
        """Build a prompt that includes prior turns as context.
        Format:
        Prior conversation:
        User: ...
        Assistant: ...

        New question: ...
        """

    def close_session(self, session_id: str) -> None:
        """Remove session from memory. Used in fixture teardown."""

    def get_active_session_count(self) -> int:
        """Return number of currently active sessions."""
```

---

## Integration with RAGPipeline
Update `rag/pipeline.py` to add a multi-turn method:
```python
def query_with_session(self, question: str, session_id: str) -> dict:
    """Query using session context for multi-turn conversations.
    Returns same dict as query() but builds prompt with history."""
```

---

## Key Design Points to Explain in Interview
Add these as comments in the code:
1. **Why in-memory?** — For MVP, no persistence needed. State lives for the test run duration.
2. **Why UUID?** — Ensures session isolation even with parallel test execution.
3. **Why `close_session` in teardown?** — Prevents memory leaks in long test runs.
4. **Why separate `build_context_prompt`?** — Keeps session logic out of RAG pipeline.

---

## Output Format
1. Full `session/manager.py` implementation
2. Updated `rag/pipeline.py` with `query_with_session` method
3. A standalone usage demo (not pytest, just show the flow):
   ```python
   manager = SessionManager()
   session = manager.create_session()
   manager.add_message(session.id, "user", "What is the return policy?")
   manager.add_message(session.id, "assistant", "Returns are accepted within 30 days.")
   prompt = manager.build_context_prompt(session.id, "Can I return after 45 days?")
   print(prompt)  # Shows full context prompt
   ```

---

## Validation Checklist
- [ ] Each session has a unique UUID
- [ ] `close_session` removes session from `_sessions` dict
- [ ] `build_context_prompt` includes ALL prior turns, not just the last one
- [ ] `get_history` raises `KeyError` with clear message if session not found
- [ ] No file I/O — sessions are purely in-memory
- [ ] Logger logs session create and close events

---

## DO NOT in This Stage
- Do not persist sessions to disk or database
- Do not implement authentication or user management
- Do not make session management async
