"""Direct sequence-to-sequence correction strategy."""

from __future__ import annotations

from llm_grammar_bench.backends.base import BaseBackend
from llm_grammar_bench.strategies.base import BaseStrategy

_DEFAULT_SYSTEM_PROMPT = (
    "You are a grammatical error correction system. "
    "Correct all grammatical, lexical, and orthographical errors in the input text. "
    "Return only the corrected text, with no additional commentary."
)

_DEFAULT_TASK_PREFIX = "grammar: "


class Seq2SeqStrategy(BaseStrategy):
    """Direct correction: send the erroneous text and get back the corrected version.

    For seq2seq models (T5, BART), prepends a task prefix.
    For chat API backends, wraps the source in a system+user prompt via **kwargs.
    """

    def __init__(
        self,
        task_prefix: str = _DEFAULT_TASK_PREFIX,
        system_prompt: str | None = None,
    ) -> None:
        self._task_prefix = task_prefix
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

    def correct(self, backend: BaseBackend, source: str) -> str:
        """Correct a single sentence.

        Args:
            backend: The model backend to use for inference.
            source: The erroneous source sentence.

        Returns:
            The corrected sentence.
        """
        prompt = f"{self._task_prefix}{source}"
        raw_output = backend.correct(prompt, system_prompt=self._system_prompt)

        # Strip task prefix if the model echoed it back (case-insensitive)
        corrected = raw_output
        if self._task_prefix and corrected.lower().startswith(self._task_prefix.lower()):
            corrected = corrected[len(self._task_prefix) :]

        return corrected.strip()

    @property
    def task_prefix(self) -> str:
        return self._task_prefix
