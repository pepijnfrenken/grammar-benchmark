"""Apply bert-score compatibility patch for transformers >= 5.x."""

from __future__ import annotations

from typing import Any


def _apply_patch() -> None:
    """Monkeypatch bert_score.utils.sent_encode for transformers 5.x compatibility.

    In transformers >= 5.x, PreTrainedTokenizer.build_inputs_with_special_tokens
    was removed. bert-score calls it for empty strings. This patch uses
    tokenizer.encode("", ...) as a drop-in replacement.
    """
    import bert_score.utils

    _original = bert_score.utils.sent_encode

    def patched(tokenizer: Any, sent: str) -> Any:
        sent = sent.strip()
        if sent == "":
            if hasattr(tokenizer, "build_inputs_with_special_tokens"):
                return tokenizer.build_inputs_with_special_tokens([])
            return tokenizer.encode("", add_special_tokens=True)
        return _original(tokenizer, sent)

    # Using __dict__ assignment to avoid type checker noise
    bert_score.utils.__dict__["sent_encode"] = patched

    # Also patch submodules that imported sent_encode by reference
    import bert_score.score  # type: ignore[import-untyped]
    import bert_score.scorer  # type: ignore[import-untyped]

    bert_score.score.__dict__["sent_encode"] = patched
    bert_score.scorer.__dict__["sent_encode"] = patched


_apply_patch()
