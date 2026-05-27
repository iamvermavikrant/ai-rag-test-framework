"""Conversation memory and session isolation manager.

Maintains per-session message history for multi-turn conversations
and provides clean session creation and teardown for test isolation.

Design decisions (interview-ready):
- In-memory only: For MVP, no persistence needed. State lives for the test run duration.
- UUID session IDs: Ensures isolation even under parallel pytest-xdist execution.
- close_session in teardown: Prevents unbounded memory growth in long test runs.
- build_context_prompt separated: Keeps session/prompt logic out of the RAG pipeline.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from observability.logger import get_logger

logger = get_logger(__name__)


class _MessageProxy(list):
    """A list proxy that stores Message objects but accepts plain dicts via append.

    Allows tests written before Stage 06 (which append raw dicts) to work
    alongside the new Message-based API without any test changes.
    """

    def __init__(self, history: list) -> None:
        # Do NOT copy — proxy the original list in-place via a reference.
        self._history = history
        super().__init__(history)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._history)

    def append(self, item: "dict[str, str] | Message") -> None:  # type: ignore[override]
        if isinstance(item, dict):
            msg = Message(
                role=item.get("role", ""),
                content=item.get("content", ""),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        else:
            msg = item
        self._history.append(msg)
        super().append(msg)


@dataclass
class Message:
    """A single turn in a conversation.

    Attributes:
        role: Either "user" or "assistant".
        content: The text of the message.
        timestamp: ISO-8601 UTC timestamp when the message was added.
    """

    role: str
    content: str
    timestamp: str


@dataclass
class ConversationSession:
    """Represents a single isolated conversation context.

    Each test function receives its own ConversationSession so message
    history never leaks between tests — the function-scoped fixture
    creates one and the teardown closes it.

    Attributes:
        id: Unique UUID string for this session.
        history: Ordered list of Message objects for the full conversation.
        created_at: ISO-8601 UTC timestamp of session creation.
        metadata: Optional tags, e.g. {"test_name": "test_multi_turn_greeting"}.
    """

    id: str
    history: list[Message] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def messages(self) -> list[dict[str, str]]:
        """Backward-compatible view of history as plain role/content dicts.

        Tests written before Stage 06 append plain dicts directly to this list.
        The property proxies to `history` so both access patterns coexist.
        """
        # Return a live proxy list so test appends mutate the real history.
        return _MessageProxy(self.history)  # type: ignore[return-value]


class SessionManager:
    """Creates, manages, and destroys ConversationSession instances.

    Holds a registry of active sessions so teardown can find and close
    them by ID even if the caller loses the reference.

    Example:
        >>> manager = SessionManager()
        >>> session = manager.create_session(metadata={"test_name": "example"})
        >>> manager.add_message(session.id, "user", "What is RAG?")
        >>> manager.add_message(session.id, "assistant", "RAG stands for Retrieval-Augmented Generation.")
        >>> prompt = manager.build_context_prompt(session.id, "How does retrieval work?")
        >>> manager.close_session(session.id)
        >>> manager.get_active_session_count()
        0
    """

    def __init__(self) -> None:
        # UUID keys ensure O(1) lookup and no cross-test collisions.
        self._sessions: dict[str, ConversationSession] = {}

    def create_session(self, metadata: dict[str, object] | None = None) -> ConversationSession:
        """Create and register a fresh conversation session.

        Args:
            metadata: Optional dict of tags to attach (e.g. test name, scenario label).

        Returns:
            A new ConversationSession with an empty history.
        """
        session_id = str(uuid.uuid4())
        session = ConversationSession(
            id=session_id,
            metadata=metadata or {},
        )
        self._sessions[session_id] = session
        logger.info("Session created | id=%s | metadata=%s", session_id, session.metadata)
        return session

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session history.

        Args:
            session_id: ID of the target session.
            role: "user" or "assistant".
            content: Text content of the message.

        Raises:
            KeyError: If session_id does not exist in the registry.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id!r}. Was it created or already closed?")

        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._sessions[session_id].history.append(message)
        logger.debug("Message added | session=%s | role=%s", session_id, role)

    def get_history(self, session_id: str) -> list[Message]:
        """Return the full conversation history for a session.

        Args:
            session_id: ID of the target session.

        Returns:
            Ordered list of Message objects from oldest to newest.

        Raises:
            KeyError: If session_id does not exist in the registry.
        """
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id!r}. Was it created or already closed?")
        return self._sessions[session_id].history

    def build_context_prompt(self, session_id: str, new_question: str) -> str:
        """Build a prompt that injects all prior turns before the new question.

        Including full history (not just the last turn) lets the LLM resolve
        pronouns and follow multi-step reasoning across the whole conversation.

        Args:
            session_id: ID of the session whose history should be injected.
            new_question: The user's current question to append.

        Returns:
            A formatted string with prior conversation followed by the new question.

        Raises:
            KeyError: If session_id does not exist in the registry.

        Example output:
            Prior conversation:
            User: What is the return policy?
            Assistant: Returns are accepted within 30 days.

            New question: Can I return after 45 days?
        """
        history = self.get_history(session_id)

        if not history:
            return new_question

        prior_lines = "\n".join(
            f"{msg.role.capitalize()}: {msg.content}" for msg in history
        )
        return f"Prior conversation:\n{prior_lines}\n\nNew question: {new_question}"

    def close_session(self, session_id: str) -> None:
        """Remove a session from the registry and clear its message history.

        Called in fixture teardown to prevent memory leaks in long test runs.

        Args:
            session_id: ID of the session to close.
        """
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.history.clear()
            logger.info("Session closed | id=%s", session_id)
        else:
            logger.warning("close_session called for unknown session: %s", session_id)

    def get_active_session_count(self) -> int:
        """Return the number of currently active (unclosed) sessions.

        Returns:
            Integer count of sessions in the registry.
        """
        return len(self._sessions)
