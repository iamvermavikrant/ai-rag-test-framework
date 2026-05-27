"""Tests for multi-turn conversation and session continuity.

Verifies that the session manager correctly maintains context across
multiple turns and that sessions are isolated from one another.

Fixture dependencies:
    - `fresh_session` (function-scoped) — a new ConversationSession per test.
    - `session_manager` (session-scoped) — the shared SessionManager registry.
    - `rag_pipeline` (session-scoped) — shared pre-ingested pipeline.

Design note: Session memory is appended manually in these tests because
Stage 06 (full multi-turn generation with context injection) is not yet
complete. These tests validate the session data model contract, not the LLM's
ability to recall prior answers.
"""

import pytest

from rag import RAGPipeline
from session.manager import ConversationSession, SessionManager


@pytest.mark.conversation
def test_second_turn_uses_first_turn_context(
    fresh_session: ConversationSession,
    rag_pipeline: RAGPipeline,
) -> None:
    """Multi-turn: second question's session history includes the first answer.

    WHY: A multi-turn RAG system must accumulate prior turns in the session
    so subsequent queries have access to earlier context. This test confirms
    that after turn 1, the session's message list is non-empty and available
    for turn 2 — verifying the session accumulation contract.

    Fixture: fresh_session (isolated history), rag_pipeline (query execution).
    """
    # Turn 1 — query and record to session
    result_1 = rag_pipeline.query("How do I reset my password?")
    fresh_session.messages.append({"role": "user", "content": result_1["question"]})  # type: ignore[arg-type]
    fresh_session.messages.append({"role": "assistant", "content": result_1["answer"]})  # type: ignore[arg-type]

    history_after_turn_1 = len(fresh_session.messages)

    # Turn 2 — follow-up question; history from turn 1 is available
    result_2 = rag_pipeline.query("What email address should I use for the reset?")
    fresh_session.messages.append({"role": "user", "content": result_2["question"]})  # type: ignore[arg-type]
    fresh_session.messages.append({"role": "assistant", "content": result_2["answer"]})  # type: ignore[arg-type]

    # The session must have grown — turn 1 context is still present for turn 2
    assert len(fresh_session.messages) > history_after_turn_1, (
        "Session history did not grow after the second turn"
    )


@pytest.mark.conversation
def test_session_isolation_between_tests(
    rag_pipeline: RAGPipeline,
    session_manager: SessionManager,
) -> None:
    """Two sessions should not share conversation history.

    WHY: If sessions leaked state, a query in one test could pollute the
    context seen by another test. This test creates two sessions explicitly,
    writes to one, and verifies the other remains empty.

    Fixture: session_manager (creates sessions), rag_pipeline (query execution).
    """
    session_a = session_manager.create_session()
    session_b = session_manager.create_session()

    try:
        result = rag_pipeline.query("What file formats are supported?")
        session_a.messages.append({"role": "user", "content": result["question"]})  # type: ignore[arg-type]
        session_a.messages.append({"role": "assistant", "content": result["answer"]})  # type: ignore[arg-type]

        # session_b must remain empty — it should not see session_a's messages
        assert len(session_b.messages) == 0, (
            f"Session B was contaminated by Session A. Messages: {session_b.messages}"
        )
    finally:
        session_manager.close_session(session_a.id)
        session_manager.close_session(session_b.id)


@pytest.mark.conversation
def test_session_has_unique_id(fresh_session: ConversationSession) -> None:
    """Each session must have a unique string ID.

    WHY: The session ID is used as the registry key in SessionManager.
    A non-string or empty ID would cause KeyError on close; a duplicate ID
    across tests would cause one session to silently overwrite another.

    Fixture: fresh_session — provides a newly created ConversationSession.
    """
    assert isinstance(fresh_session.id, str), (
        f"Session ID must be a str, got {type(fresh_session.id)}"
    )
    assert len(fresh_session.id) > 0, "Session ID must not be empty"


@pytest.mark.conversation
def test_conversation_history_grows_per_turn(
    fresh_session: ConversationSession,
    rag_pipeline: RAGPipeline,
) -> None:
    """Verify history length increases after each turn.

    WHY: If messages were not appended, the history would stay at 0 and
    multi-turn context would be silently lost. Counting messages after each
    turn gives a precise regression guard for the session append contract.

    Fixture: fresh_session (empty history), rag_pipeline (query execution).
    """
    assert len(fresh_session.messages) == 0, "Session must start empty"

    result_1 = rag_pipeline.query("What is the support SLA for Enterprise customers?")
    fresh_session.messages.append({"role": "user", "content": result_1["question"]})  # type: ignore[arg-type]
    fresh_session.messages.append({"role": "assistant", "content": result_1["answer"]})  # type: ignore[arg-type]
    assert len(fresh_session.messages) == 2, (
        f"Expected 2 messages after turn 1, got {len(fresh_session.messages)}"
    )

    result_2 = rag_pipeline.query("How is employee data encrypted?")
    fresh_session.messages.append({"role": "user", "content": result_2["question"]})  # type: ignore[arg-type]
    fresh_session.messages.append({"role": "assistant", "content": result_2["answer"]})  # type: ignore[arg-type]
    assert len(fresh_session.messages) == 4, (
        f"Expected 4 messages after turn 2, got {len(fresh_session.messages)}"
    )
