"""Edit-based correction strategy where the model outputs edit spans as JSON."""

from __future__ import annotations

import json
import logging
import re

from llm_grammar_bench.backends.base import BaseBackend
from llm_grammar_bench.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

_DEFAULT_EDIT_SYSTEM_PROMPT = (
    "You are a grammatical error correction system. "
    "For the given sentence, identify all grammatical, lexical, and orthographical "
    "errors. Return a JSON array of edit operations. Each edit must have: "
    "'start' (integer character index), 'end' (integer character index, exclusive), "
    "and 'text' (the corrected replacement string for that span). "
    "Return ONLY valid JSON, with no additional commentary or markdown."
)


def _extract_json(text: str) -> list[dict]:
    """Extract a JSON array from model output that may contain extra text."""
    # Direct parse attempt
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Find JSON array in text (greedy)
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try code-block extraction
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON edits from: {text[:200]}")


class EditBasedStrategy(BaseStrategy):
    """Model outputs edits as a JSON array of {start, end, text} objects,
    which are then applied to the source sentence to produce the corrected text."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self._system_prompt = system_prompt or _DEFAULT_EDIT_SYSTEM_PROMPT

    def correct(self, backend: BaseBackend, source: str) -> str:
        """Correct a single sentence via edit-based JSON output.

        Args:
            backend: The model backend to use for inference.
            source: The erroneous source sentence.

        Returns:
            The corrected sentence, or the original if parsing fails.
        """
        prompt = f"Correct this sentence. Return only a JSON array of edits.\n\n{source}"
        raw = backend.correct(prompt, system_prompt=self._system_prompt)

        try:
            edits = _extract_json(raw)
        except ValueError as exc:
            logger.warning("Failed to parse edit JSON: %s. Returning source unchanged.", exc)
            return source

        # Apply edits in reverse order to preserve character indices
        chars = list(source)
        for edit in sorted(edits, key=lambda e: int(e["start"]), reverse=True):
            start = int(edit["start"])
            end = int(edit["end"])
            text = str(edit["text"])
            if 0 <= start <= end <= len(chars):
                chars[start:end] = list(text)

        return "".join(chars)
