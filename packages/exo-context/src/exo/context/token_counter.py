"""TiktokenCounter — accurate token counting via tiktoken with char-based fallback.

Uses tiktoken for exact token counts when available; falls back to a
character-based heuristic (~4 chars per token) when tiktoken is not installed.
"""

from __future__ import annotations

from typing import Any

# Model → encoding mapping.  o200k_base covers GPT-4o family;
# cl100k_base covers GPT-4/3.5-turbo.
_MODEL_ENCODINGS: dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o1-preview": "o200k_base",
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-turbo-preview": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
}

# Default encoding when model is not in the map.
_DEFAULT_ENCODING = "o200k_base"

# Per-message token overhead for chat-format messages (role + delimiters).
_MESSAGE_OVERHEAD = 3

# Extra tokens appended after all messages (assistant reply priming).
_REPLY_OVERHEAD = 3

# Chars-per-token ratio for the fallback estimator.
_CHARS_PER_TOKEN = 4.0


class TiktokenCounter:
    """Count tokens for text and chat messages.

    Uses tiktoken for exact counts when installed; otherwise falls back
    to a character-length heuristic.

    Parameters
    ----------
    model:
        Model name used to select the tiktoken encoding.
        Defaults to ``"gpt-4o"`` (o200k_base).

    Example::

        counter = TiktokenCounter()
        n = counter.count("Hello, world!")
        total = counter.count_messages([
            {"role": "user", "content": "Hi"},
        ])
    """

    __slots__ = ("_encoding", "_has_tiktoken", "_model")

    def __init__(self, model: str = "gpt-4o") -> None:
        self._model = model
        self._encoding: Any = None
        self._has_tiktoken = False

        encoding_name = _MODEL_ENCODINGS.get(model, _DEFAULT_ENCODING)

        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding(encoding_name)
            self._has_tiktoken = True
        except ImportError:
            pass

    # ── public API ────────────────────────────────────────────────────

    @property
    def model(self) -> str:
        """The model name this counter was created for."""
        return self._model

    @property
    def has_tiktoken(self) -> bool:
        """Whether tiktoken is available for exact counting."""
        return self._has_tiktoken

    def count(self, text: str) -> int:
        """Count tokens in a plain text string.

        Returns exact count when tiktoken is installed, otherwise a
        character-based estimate.
        """
        if self._has_tiktoken:
            return len(self._encoding.encode(text))
        return max(1, round(len(text) / _CHARS_PER_TOKEN))

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        """Count tokens for a list of chat messages.

        Each message is expected to have ``role`` and ``content`` keys.
        Adds per-message overhead (3 tokens) and reply priming overhead
        (3 tokens) following the OpenAI token counting convention.

        Returns exact count when tiktoken is installed, otherwise a
        character-based estimate with the same overhead constants.
        """
        total = 0
        for msg in messages:
            total += _MESSAGE_OVERHEAD
            for value in msg.values():
                if isinstance(value, str):
                    total += self.count(value)
        total += _REPLY_OVERHEAD
        return total

    def __repr__(self) -> str:
        backend = "tiktoken" if self._has_tiktoken else "fallback"
        return f"TiktokenCounter(model={self._model!r}, backend={backend!r})"
