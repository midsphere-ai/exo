"""Tests for exo.token_counter — TokenCounter and count_tokens."""

from __future__ import annotations

from exo.token_counter import (
    _CHARS_PER_TOKEN_BY_ENCODING,
    _CHARS_PER_TOKEN_FALLBACK,
    _MESSAGE_OVERHEAD,
    _REPLY_OVERHEAD,
    TokenCounter,
    _counter_cache,
    _resolve_encoding_name,
    count_tokens,
)

# ── Encoding resolution ──────────────────────────────────────────────


class TestEncodingResolution:
    def test_openai_gpt4o(self) -> None:
        _, _, enc = _resolve_encoding_name("openai:gpt-4o")
        assert enc == "o200k_base"

    def test_openai_gpt4o_mini(self) -> None:
        _, _, enc = _resolve_encoding_name("openai:gpt-4o-mini")
        assert enc == "o200k_base"

    def test_openai_gpt4(self) -> None:
        _, _, enc = _resolve_encoding_name("openai:gpt-4")
        assert enc == "cl100k_base"

    def test_openai_gpt4_turbo(self) -> None:
        _, _, enc = _resolve_encoding_name("openai:gpt-4-turbo")
        assert enc == "cl100k_base"

    def test_openai_gpt35(self) -> None:
        _, _, enc = _resolve_encoding_name("openai:gpt-3.5-turbo")
        assert enc == "cl100k_base"

    def test_openai_o1(self) -> None:
        _, _, enc = _resolve_encoding_name("openai:o1")
        assert enc == "o200k_base"

    def test_openai_o3_mini(self) -> None:
        _, _, enc = _resolve_encoding_name("openai:o3-mini")
        assert enc == "o200k_base"

    def test_openai_o4_mini(self) -> None:
        _, _, enc = _resolve_encoding_name("openai:o4-mini")
        assert enc == "o200k_base"

    def test_openai_unknown_future_model(self) -> None:
        _, _, enc = _resolve_encoding_name("openai:gpt-99")
        assert enc == "o200k_base"

    def test_anthropic_claude(self) -> None:
        _, _, enc = _resolve_encoding_name("anthropic:claude-sonnet-4-6")
        assert enc == "cl100k_base"

    def test_anthropic_any_model(self) -> None:
        _, _, enc = _resolve_encoding_name("anthropic:some-future-model")
        assert enc == "cl100k_base"

    def test_gemini(self) -> None:
        _, _, enc = _resolve_encoding_name("gemini:gemini-2.0-flash")
        assert enc == "o200k_base"

    def test_vertex(self) -> None:
        _, _, enc = _resolve_encoding_name("vertex:gemini-2.0-flash")
        assert enc == "o200k_base"

    def test_unknown_provider(self) -> None:
        _, _, enc = _resolve_encoding_name("deepseek:deepseek-v3")
        assert enc == "o200k_base"

    def test_bare_model_defaults_to_openai(self) -> None:
        provider, model_name, enc = _resolve_encoding_name("gpt-4o")
        assert provider == "openai"
        assert model_name == "gpt-4o"
        assert enc == "o200k_base"

    def test_bare_unknown_model(self) -> None:
        provider, _, enc = _resolve_encoding_name("some-model")
        assert provider == "openai"
        assert enc == "o200k_base"


# ── Construction ─────────────────────────────────────────────────────


class TestConstruction:
    def test_default_model(self) -> None:
        counter = TokenCounter()
        assert counter.model == "gpt-4o"
        assert counter.provider == "openai"

    def test_provider_model_string(self) -> None:
        counter = TokenCounter("anthropic:claude-sonnet-4-6")
        assert counter.model == "claude-sonnet-4-6"
        assert counter.provider == "anthropic"
        assert counter.encoding_name == "cl100k_base"

    def test_bare_model_string(self) -> None:
        counter = TokenCounter("gpt-4")
        assert counter.model == "gpt-4"
        assert counter.provider == "openai"
        assert counter.encoding_name == "cl100k_base"

    def test_has_tiktoken_true(self) -> None:
        counter = TokenCounter()
        assert counter.has_tiktoken is True

    def test_repr(self) -> None:
        counter = TokenCounter("anthropic:claude-sonnet-4-6")
        r = repr(counter)
        assert "anthropic" in r
        assert "claude-sonnet-4-6" in r
        assert "cl100k_base" in r
        assert "tiktoken" in r


# ── Exact token counting (tiktoken present) ──────────────────────────


class TestCountExact:
    def test_empty_string(self) -> None:
        counter = TokenCounter()
        assert counter.count("") == 0

    def test_single_char(self) -> None:
        counter = TokenCounter()
        assert counter.count("a") == 1

    def test_hello_world(self) -> None:
        counter = TokenCounter()
        assert counter.count("hello world") == 2

    def test_hello_world_punctuated(self) -> None:
        counter = TokenCounter()
        assert counter.count("Hello, world!") == 4

    def test_tiktoken_is_great(self) -> None:
        counter = TokenCounter()
        assert counter.count("tiktoken is great!") == 6

    def test_longer_sentence(self) -> None:
        counter = TokenCounter()
        assert counter.count("The quick brown fox jumps over the lazy dog.") == 10

    def test_gpt4_cl100k(self) -> None:
        counter = TokenCounter("openai:gpt-4")
        assert counter.count("hello world") == 2
        assert counter.count("Hello, world!") == 4

    def test_anthropic_counts(self) -> None:
        counter = TokenCounter("anthropic:claude-sonnet-4-6")
        result = counter.count("hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_gemini_counts(self) -> None:
        counter = TokenCounter("gemini:gemini-2.0-flash")
        result = counter.count("hello world")
        assert isinstance(result, int)
        assert result > 0


# ── count_messages ───────────────────────────────────────────────────


class TestCountMessages:
    def test_single_message(self) -> None:
        counter = TokenCounter()
        messages = [{"role": "user", "content": "hello world"}]
        expected = (
            counter.count("user")
            + counter.count("hello world")
            + _MESSAGE_OVERHEAD
            + _REPLY_OVERHEAD
        )
        assert counter.count_messages(messages) == expected

    def test_multiple_messages(self) -> None:
        counter = TokenCounter()
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
        counter = TokenCounter()
        assert counter.count_messages([]) == _REPLY_OVERHEAD


# ── Conversion methods ───────────────────────────────────────────────


class TestConversions:
    def test_tokens_to_chars_o200k(self) -> None:
        counter = TokenCounter("openai:gpt-4o")
        assert counter.tokens_to_chars(100) == round(100 * 4.2)

    def test_tokens_to_chars_cl100k(self) -> None:
        counter = TokenCounter("openai:gpt-4")
        assert counter.tokens_to_chars(100) == round(100 * 3.7)

    def test_chars_to_tokens_o200k(self) -> None:
        counter = TokenCounter("openai:gpt-4o")
        assert counter.chars_to_tokens(420) == round(420 / 4.2)

    def test_chars_to_tokens_cl100k(self) -> None:
        counter = TokenCounter("openai:gpt-4")
        assert counter.chars_to_tokens(370) == round(370 / 3.7)

    def test_chars_to_tokens_minimum_one(self) -> None:
        counter = TokenCounter()
        assert counter.chars_to_tokens(0) == 1

    def test_roundtrip_approximate(self) -> None:
        counter = TokenCounter()
        tokens = 1000
        chars = counter.tokens_to_chars(tokens)
        back = counter.chars_to_tokens(chars)
        assert abs(back - tokens) <= 1


# ── count_tokens convenience function ────────────────────────────────


class TestCountTokensFunction:
    def test_basic(self) -> None:
        result = count_tokens("hello world", model="openai:gpt-4o")
        assert result == 2

    def test_different_model(self) -> None:
        result = count_tokens("hello world", model="anthropic:claude-sonnet-4-6")
        assert isinstance(result, int)
        assert result > 0

    def test_caching(self) -> None:
        _counter_cache.clear()
        count_tokens("hello", model="openai:gpt-4o")
        assert "openai:gpt-4o" in _counter_cache
        # Second call should reuse cached counter
        cached = _counter_cache["openai:gpt-4o"]
        count_tokens("world", model="openai:gpt-4o")
        assert _counter_cache["openai:gpt-4o"] is cached

    def test_default_model(self) -> None:
        result = count_tokens("hello world")
        assert result == 2


# ── Fallback (tiktoken not installed) ────────────────────────────────


class TestFallback:
    def _make_fallback_counter(self, model: str = "openai:gpt-4o") -> TokenCounter:
        counter = TokenCounter.__new__(TokenCounter)
        counter._provider = "openai"
        counter._model = "gpt-4o"
        counter._encoding_name = "o200k_base"
        counter._encoding = None
        counter._has_tiktoken = False
        counter._chars_per_token = _CHARS_PER_TOKEN_BY_ENCODING.get(
            "o200k_base", _CHARS_PER_TOKEN_FALLBACK
        )
        return counter

    def test_has_tiktoken_false(self) -> None:
        counter = self._make_fallback_counter()
        assert counter.has_tiktoken is False

    def test_count_fallback_estimation(self) -> None:
        counter = self._make_fallback_counter()
        text = "hello world!!"  # 13 chars
        expected = round(len(text) / _CHARS_PER_TOKEN_BY_ENCODING["o200k_base"])
        assert counter.count(text) == expected

    def test_count_fallback_minimum_one(self) -> None:
        counter = self._make_fallback_counter()
        assert counter.count("a") == 1

    def test_count_fallback_empty(self) -> None:
        counter = self._make_fallback_counter()
        assert counter.count("") == 1

    def test_repr_fallback(self) -> None:
        counter = self._make_fallback_counter()
        assert "fallback" in repr(counter)


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_unicode_text(self) -> None:
        counter = TokenCounter()
        result = counter.count("こんにちは世界")
        assert isinstance(result, int)
        assert result > 0

    def test_multiline_text(self) -> None:
        counter = TokenCounter()
        result = counter.count("line one\nline two\nline three")
        assert isinstance(result, int)
        assert result > 0

    def test_whitespace_only(self) -> None:
        counter = TokenCounter()
        result = counter.count("   ")
        assert isinstance(result, int)
        assert result > 0
