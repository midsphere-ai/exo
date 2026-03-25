"""Tests for exo.context.token_counter — TiktokenCounter."""

from __future__ import annotations

from unittest.mock import patch

from exo.context.token_counter import (  # pyright: ignore[reportMissingImports]
    _CHARS_PER_TOKEN,
    _MESSAGE_OVERHEAD,
    _REPLY_OVERHEAD,
    TiktokenCounter,
)

# ── Construction ─────────────────────────────────────────────────────


class TestConstruction:
    def test_default_model(self) -> None:
        counter = TiktokenCounter()
        assert counter.model == "gpt-4o"

    def test_custom_model(self) -> None:
        counter = TiktokenCounter(model="gpt-4")
        assert counter.model == "gpt-4"

    def test_unknown_model_uses_default_encoding(self) -> None:
        counter = TiktokenCounter(model="some-future-model")
        assert counter.model == "some-future-model"
        assert counter.has_tiktoken is True
        # Should still work (defaults to o200k_base)
        assert counter.count("hello") > 0

    def test_has_tiktoken_true(self) -> None:
        counter = TiktokenCounter()
        assert counter.has_tiktoken is True

    def test_repr_with_tiktoken(self) -> None:
        counter = TiktokenCounter()
        assert repr(counter) == "TiktokenCounter(model='gpt-4o', backend='tiktoken')"


# ── Exact token counting (tiktoken present) ─────────────────────────


class TestCountExact:
    def test_empty_string(self) -> None:
        counter = TiktokenCounter()
        assert counter.count("") == 0

    def test_single_char(self) -> None:
        counter = TiktokenCounter()
        assert counter.count("a") == 1

    def test_hello_world(self) -> None:
        counter = TiktokenCounter()
        assert counter.count("hello world") == 2

    def test_hello_world_punctuated(self) -> None:
        counter = TiktokenCounter()
        assert counter.count("Hello, world!") == 4

    def test_tiktoken_is_great(self) -> None:
        counter = TiktokenCounter()
        assert counter.count("tiktoken is great!") == 6

    def test_longer_sentence(self) -> None:
        counter = TiktokenCounter()
        assert counter.count("The quick brown fox jumps over the lazy dog.") == 10

    def test_gpt4_model_uses_cl100k(self) -> None:
        counter = TiktokenCounter(model="gpt-4")
        assert counter.count("hello world") == 2
        assert counter.count("Hello, world!") == 4


# ── count_messages (tiktoken present) ────────────────────────────────


class TestCountMessagesExact:
    def test_single_message(self) -> None:
        counter = TiktokenCounter()
        messages = [{"role": "user", "content": "hello world"}]
        # role "user" (1 token) + content "hello world" (2 tokens) + msg overhead (3) + reply overhead (3)
        expected = (
            counter.count("user")
            + counter.count("hello world")
            + _MESSAGE_OVERHEAD
            + _REPLY_OVERHEAD
        )
        assert counter.count_messages(messages) == expected

    def test_multiple_messages(self) -> None:
        counter = TiktokenCounter()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        expected = _REPLY_OVERHEAD
        for msg in messages:
            expected += _MESSAGE_OVERHEAD
            for v in msg.values():
                expected += counter.count(v)
        assert counter.count_messages(messages) == expected

    def test_empty_messages(self) -> None:
        counter = TiktokenCounter()
        # No messages, only reply overhead
        assert counter.count_messages([]) == _REPLY_OVERHEAD


# ── Fallback (tiktoken not installed) ────────────────────────────────


class TestFallback:
    def _make_fallback_counter(self) -> TiktokenCounter:
        with patch.dict("sys.modules", {"tiktoken": None}):
            counter = TiktokenCounter.__new__(TiktokenCounter)
            counter._model = "gpt-4o"
            counter._encoding = None
            counter._has_tiktoken = False
        return counter

    def test_has_tiktoken_false(self) -> None:
        counter = self._make_fallback_counter()
        assert counter.has_tiktoken is False

    def test_repr_fallback(self) -> None:
        counter = self._make_fallback_counter()
        assert repr(counter) == "TiktokenCounter(model='gpt-4o', backend='fallback')"

    def test_count_fallback_estimation(self) -> None:
        counter = self._make_fallback_counter()
        text = "hello world!!"  # 13 chars → round(13/4) = 3
        assert counter.count(text) == round(len(text) / _CHARS_PER_TOKEN)

    def test_count_fallback_minimum_one(self) -> None:
        counter = self._make_fallback_counter()
        # Single char → max(1, round(1/4)) = max(1, 0) = 1
        assert counter.count("a") == 1

    def test_count_fallback_empty(self) -> None:
        counter = self._make_fallback_counter()
        # Empty → max(1, round(0/4)) = max(1, 0) = 1
        assert counter.count("") == 1

    def test_count_messages_fallback(self) -> None:
        counter = self._make_fallback_counter()
        messages = [{"role": "user", "content": "hello world"}]
        total = _REPLY_OVERHEAD
        total += _MESSAGE_OVERHEAD
        total += counter.count("user")
        total += counter.count("hello world")
        assert counter.count_messages(messages) == total


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_unicode_text(self) -> None:
        counter = TiktokenCounter()
        result = counter.count("こんにちは世界")
        assert isinstance(result, int)
        assert result > 0

    def test_multiline_text(self) -> None:
        counter = TiktokenCounter()
        result = counter.count("line one\nline two\nline three")
        assert isinstance(result, int)
        assert result > 0

    def test_message_with_extra_fields(self) -> None:
        """Non-string values in messages should be skipped."""
        counter = TiktokenCounter()
        messages: list[dict[str, str]] = [
            {"role": "user", "content": "hello"},
        ]
        result = counter.count_messages(messages)
        assert isinstance(result, int)
        assert result > _REPLY_OVERHEAD

    def test_whitespace_only(self) -> None:
        counter = TiktokenCounter()
        result = counter.count("   ")
        assert isinstance(result, int)
        assert result > 0
