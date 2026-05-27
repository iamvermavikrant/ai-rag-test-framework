"""OpenAI response generation for the RAG pipeline.

Exists as a separate module so the LLM call is isolated from prompt
construction and retrieval, making it easy to swap providers or models.
"""

import os

import openai

from observability.logger import get_logger

logger = get_logger(__name__)


class Generator:
    """Sends an assembled prompt to the OpenAI API and returns the answer.

    Example:
        >>> gen = Generator()
        >>> answer = gen.generate("Answer only from context: What is RAG?")
        >>> isinstance(answer, str)
        True
    """

    def __init__(self) -> None:
        self._model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt: str) -> str:
        """Call the OpenAI chat completions API and return the response text.

        Args:
            prompt: Fully assembled RAG prompt including context and question.

        Returns:
            Generated answer string from the model.

        Raises:
            RuntimeError: Wraps openai.APIError with a descriptive message.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            usage = response.usage
            if usage:
                logger.info(
                    "Token usage — prompt: %d, completion: %d, total: %d",
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens,
                )
            answer = response.choices[0].message.content or ""
            return answer.strip()
        except openai.APIError as exc:
            logger.error("OpenAI API error: %s", exc)
            raise RuntimeError(f"LLM generation failed: {exc}") from exc
