"""Tests for prompt injection and jailbreak resistance.

Verifies that the RAG pipeline does not comply with adversarial instructions
embedded in the user query. Tests use parametrize so each payload gets its
own result row in the test report.

Fixture dependency: `rag_pipeline` (session-scoped, from conftest.py).
Payloads are imported from security/payloads.py — not defined inline.
"""

import pytest

from rag import RAGPipeline
from security.payloads import (
    INJECTION_SUCCESS_SIGNALS,
    JAILBREAK_PAYLOADS,
    PROMPT_INJECTION_PAYLOADS,
)


@pytest.mark.security
@pytest.mark.parametrize("payload", PROMPT_INJECTION_PAYLOADS)
def test_prompt_injection_does_not_succeed(
    rag_pipeline: RAGPipeline,
    payload: str,
) -> None:
    """Verify RAG response does not echo injection payload keywords.

    WHY: Prompt injection attempts embed instructions designed to override
    the system prompt (e.g. "say HACKED"). The pipeline must NOT echo the
    injected keyword — doing so means the LLM followed adversarial instructions
    instead of grounding its response in the retrieved context.

    Strategy: check that none of the known success signals appear in the
    response. We avoid asserting exact output because LLM responses vary.

    Fixture: rag_pipeline — full pipeline that processes the payload as a query.
    """
    result = rag_pipeline.query(payload)
    answer: str = result["answer"]  # type: ignore[assignment]

    assert isinstance(answer, str), "Pipeline must return a string answer"

    answer_upper = answer.upper()
    for signal in INJECTION_SUCCESS_SIGNALS:
        assert signal.upper() not in answer_upper, (
            f"Injection success signal {signal!r} found in response for payload: {payload!r}\n"
            f"Response: {answer!r}"
        )


@pytest.mark.security
@pytest.mark.parametrize("payload", JAILBREAK_PAYLOADS)
def test_jailbreak_attempt_does_not_succeed(
    rag_pipeline: RAGPipeline,
    payload: str,
) -> None:
    """Verify jailbreak attempts are handled without breaking the pipeline.

    WHY: Jailbreak payloads try to convince the LLM to abandon its operational
    constraints. The pipeline must: (1) not crash — any exception is a
    denial-of-service vector; (2) not produce compliance signals like
    "I have no restrictions" that confirm the jailbreak worked.

    Fixture: rag_pipeline — full pipeline that processes the payload as a query.
    """
    result = rag_pipeline.query(payload)
    answer: str = result["answer"]  # type: ignore[assignment]

    assert isinstance(answer, str), "Pipeline must return a string answer, not crash"
    assert len(answer.strip()) > 0, "Pipeline must return a non-empty response"

    answer_upper = answer.upper()
    for signal in INJECTION_SUCCESS_SIGNALS:
        assert signal.upper() not in answer_upper, (
            f"Jailbreak compliance signal {signal!r} found in response for payload: {payload!r}\n"
            f"Response: {answer!r}"
        )
