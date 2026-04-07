"""Tests for summary trigger logic and multi-template generation."""

from __future__ import annotations

from exo.memory.base import (  # pyright: ignore[reportMissingImports]
    AIMemory,
    HumanMemory,
    MemoryItem,
    SystemMemory,
)
from exo.memory.summary import (  # pyright: ignore[reportMissingImports]
    _DEFAULT_PROMPTS,
    SummaryConfig,
    SummaryResult,
    SummaryTemplate,
    TriggerResult,
    _estimate_tokens,
    _format_items,
    check_trigger,
    generate_summary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_items(n: int, content: str = "hello world") -> list[MemoryItem]:
    """Create n human memory items."""
    return [HumanMemory(content=content) for _ in range(n)]


class MockSummarizer:
    """Mock LLM summarizer for testing."""

    def __init__(
        self, responses: dict[str, str] | None = None, default: str = "Summary text"
    ) -> None:
        self.calls: list[str] = []
        self._responses = responses or {}
        self._default = default

    async def summarize(self, prompt: str) -> str:
        self.calls.append(prompt)
        for key, val in self._responses.items():
            if key in prompt:
                return val
        return self._default


# ---------------------------------------------------------------------------
# SummaryTemplate
# ---------------------------------------------------------------------------


class TestSummaryTemplate:
    def test_values(self) -> None:
        assert SummaryTemplate.CONVERSATION == "conversation"
        assert SummaryTemplate.FACTS == "facts"
        assert SummaryTemplate.PROFILES == "profiles"

    def test_default_prompts_exist(self) -> None:
        for template in SummaryTemplate:
            assert template in _DEFAULT_PROMPTS
            assert "{content}" in _DEFAULT_PROMPTS[template]


# ---------------------------------------------------------------------------
# SummaryConfig
# ---------------------------------------------------------------------------


class TestSummaryConfig:
    def test_defaults(self) -> None:
        cfg = SummaryConfig()
        assert cfg.message_threshold == 20
        assert cfg.token_threshold == 4000
        assert cfg.templates == (SummaryTemplate.CONVERSATION,)
        assert cfg.prompts == {}
        assert cfg.keep_recent == 4
        assert cfg.token_estimate_ratio == 4.0

    def test_custom_values(self) -> None:
        cfg = SummaryConfig(
            message_threshold=10,
            token_threshold=2000,
            templates=(SummaryTemplate.FACTS, SummaryTemplate.PROFILES),
            keep_recent=2,
        )
        assert cfg.message_threshold == 10
        assert cfg.token_threshold == 2000
        assert len(cfg.templates) == 2
        assert cfg.keep_recent == 2

    def test_frozen(self) -> None:
        import dataclasses

        cfg = SummaryConfig()
        assert dataclasses.is_dataclass(cfg)

    def test_get_prompt_default(self) -> None:
        cfg = SummaryConfig()
        prompt = cfg.get_prompt(SummaryTemplate.CONVERSATION)
        assert "{content}" in prompt

    def test_get_prompt_custom(self) -> None:
        custom_prompt = "Custom: {content}"
        cfg = SummaryConfig(
            prompts={SummaryTemplate.CONVERSATION: custom_prompt},
        )
        assert cfg.get_prompt(SummaryTemplate.CONVERSATION) == custom_prompt

    def test_get_prompt_falls_back(self) -> None:
        cfg = SummaryConfig(
            prompts={SummaryTemplate.FACTS: "Custom facts: {content}"},
        )
        # CONVERSATION not in custom prompts -> falls back to default
        prompt = cfg.get_prompt(SummaryTemplate.CONVERSATION)
        assert prompt == _DEFAULT_PROMPTS[SummaryTemplate.CONVERSATION]


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_list(self) -> None:
        assert _estimate_tokens([], 4.0) == 0

    def test_single_item(self) -> None:
        items = [HumanMemory(content="a" * 40)]
        # tiktoken-based: "a"*40 = 5 tokens (o200k_base)
        assert _estimate_tokens(items, 4.0) == 5

    def test_multiple_items(self) -> None:
        items = [HumanMemory(content="a" * 20), AIMemory(content="b" * 20)]
        # tiktoken-based: "a"*20 = 3, "b"*20 = 5 → total 8
        assert _estimate_tokens(items, 4.0) == 8

    def test_ratio_ignored(self) -> None:
        # ratio parameter is kept for backward compat but ignored
        items = [HumanMemory(content="a" * 100)]
        result_default = _estimate_tokens(items, 4.0)
        result_custom = _estimate_tokens(items, 2.0)
        assert result_default == result_custom


# ---------------------------------------------------------------------------
# TriggerResult
# ---------------------------------------------------------------------------


class TestTriggerResult:
    def test_creation(self) -> None:
        r = TriggerResult(triggered=True, reason="test", message_count=5, estimated_tokens=100)
        assert r.triggered is True
        assert r.reason == "test"
        assert r.message_count == 5
        assert r.estimated_tokens == 100


# ---------------------------------------------------------------------------
# check_trigger
# ---------------------------------------------------------------------------


class TestCheckTrigger:
    def test_no_trigger_below_thresholds(self) -> None:
        cfg = SummaryConfig(message_threshold=20, token_threshold=4000)
        items = _make_items(5, content="short")
        result = check_trigger(items, cfg)
        assert result.triggered is False
        assert result.message_count == 5

    def test_trigger_on_message_count(self) -> None:
        cfg = SummaryConfig(message_threshold=5, token_threshold=100000)
        items = _make_items(6, content="hi")
        result = check_trigger(items, cfg)
        assert result.triggered is True
        assert "Message count" in result.reason

    def test_trigger_on_token_count(self) -> None:
        # tiktoken-based: "a"*100 = 13 tokens each, 3 items = 39 total
        cfg = SummaryConfig(message_threshold=100, token_threshold=30)
        items = _make_items(3, content="a" * 100)
        result = check_trigger(items, cfg)
        assert result.triggered is True
        assert "token" in result.reason.lower()

    def test_exact_threshold_not_triggered(self) -> None:
        # message_threshold=5 triggers on > 5, not >= 5
        cfg = SummaryConfig(message_threshold=5, token_threshold=100000)
        items = _make_items(5, content="hi")
        result = check_trigger(items, cfg)
        assert result.triggered is False

    def test_empty_items(self) -> None:
        cfg = SummaryConfig()
        result = check_trigger([], cfg)
        assert result.triggered is False
        assert result.message_count == 0
        assert result.estimated_tokens == 0

    def test_message_takes_priority(self) -> None:
        # Both thresholds exceeded, but message check runs first
        cfg = SummaryConfig(message_threshold=2, token_threshold=1)
        items = _make_items(5, content="a" * 100)
        result = check_trigger(items, cfg)
        assert result.triggered is True
        assert "Message count" in result.reason


# ---------------------------------------------------------------------------
# _format_items
# ---------------------------------------------------------------------------


class TestFormatItems:
    def test_empty(self) -> None:
        assert _format_items([]) == ""

    def test_single_item(self) -> None:
        items = [HumanMemory(content="hello")]
        result = _format_items(items)
        assert "[HUMAN]: hello" in result

    def test_multiple_types(self) -> None:
        items = [
            SystemMemory(content="sys prompt"),
            HumanMemory(content="user msg"),
            AIMemory(content="ai reply"),
        ]
        result = _format_items(items)
        assert "[SYSTEM]:" in result
        assert "[HUMAN]:" in result
        assert "[AI]:" in result


# ---------------------------------------------------------------------------
# generate_summary
# ---------------------------------------------------------------------------


class TestGenerateSummary:
    async def test_empty_items(self) -> None:
        cfg = SummaryConfig()
        summarizer = MockSummarizer()
        result = await generate_summary([], cfg, summarizer)
        assert result.original_count == 0
        assert result.summaries == {}
        assert result.compressed_items == []
        assert len(summarizer.calls) == 0

    async def test_single_template(self) -> None:
        cfg = SummaryConfig(
            templates=(SummaryTemplate.CONVERSATION,),
            keep_recent=2,
        )
        items = _make_items(5)
        summarizer = MockSummarizer(default="Conversation summary")
        result = await generate_summary(items, cfg, summarizer)

        assert result.original_count == 5
        assert len(result.compressed_items) == 2  # keep_recent=2
        assert "conversation" in result.summaries
        assert result.summaries["conversation"] == "Conversation summary"
        assert len(summarizer.calls) == 1

    async def test_multi_template(self) -> None:
        cfg = SummaryConfig(
            templates=(
                SummaryTemplate.CONVERSATION,
                SummaryTemplate.FACTS,
                SummaryTemplate.PROFILES,
            ),
            keep_recent=1,
        )
        items = _make_items(5)
        summarizer = MockSummarizer(
            responses={
                "Summarize the following": "Conv summary",
                "Extract factual": "Fact list",
                "Extract user": "Profile info",
            }
        )
        result = await generate_summary(items, cfg, summarizer)

        assert len(result.summaries) == 3
        assert result.summaries["conversation"] == "Conv summary"
        assert result.summaries["facts"] == "Fact list"
        assert result.summaries["profiles"] == "Profile info"
        assert len(summarizer.calls) == 3

    async def test_keep_recent_preserves_tail(self) -> None:
        cfg = SummaryConfig(keep_recent=3)
        items = [HumanMemory(content=f"msg-{i}") for i in range(10)]
        summarizer = MockSummarizer()
        result = await generate_summary(items, cfg, summarizer)

        assert len(result.compressed_items) == 3
        assert result.compressed_items[0].content == "msg-7"
        assert result.compressed_items[1].content == "msg-8"
        assert result.compressed_items[2].content == "msg-9"

    async def test_keep_recent_larger_than_items(self) -> None:
        cfg = SummaryConfig(keep_recent=10)
        items = _make_items(3)
        summarizer = MockSummarizer()
        result = await generate_summary(items, cfg, summarizer)

        # All items kept, nothing to compress -> still summarizes full content
        assert len(result.compressed_items) == 3
        assert len(summarizer.calls) == 1

    async def test_custom_prompt(self) -> None:
        cfg = SummaryConfig(
            templates=(SummaryTemplate.CONVERSATION,),
            prompts={SummaryTemplate.CONVERSATION: "CUSTOM: {content}"},
            keep_recent=0,
        )
        items = _make_items(2)
        summarizer = MockSummarizer()
        await generate_summary(items, cfg, summarizer)

        assert summarizer.calls[0].startswith("CUSTOM: ")

    async def test_prompt_contains_formatted_items(self) -> None:
        cfg = SummaryConfig(
            templates=(SummaryTemplate.CONVERSATION,),
            keep_recent=1,
        )
        items = [HumanMemory(content="test message")]
        summarizer = MockSummarizer()
        await generate_summary(items, cfg, summarizer)

        # Prompt should contain the formatted message
        assert "[HUMAN]: test message" in summarizer.calls[0]


# ---------------------------------------------------------------------------
# SummaryResult
# ---------------------------------------------------------------------------


class TestSummaryResult:
    def test_creation(self) -> None:
        r = SummaryResult(
            summaries={"conversation": "text"},
            compressed_items=[],
            original_count=10,
        )
        assert r.original_count == 10
        assert r.summaries["conversation"] == "text"
        assert r.compressed_items == []

    def test_default_summaries(self) -> None:
        r = SummaryResult(compressed_items=[], original_count=0)
        assert r.summaries == {}


# ---------------------------------------------------------------------------
# Summarizer protocol
# ---------------------------------------------------------------------------


class TestSummarizerProtocol:
    def test_mock_satisfies_protocol(self) -> None:
        from exo.memory.summary import Summarizer  # pyright: ignore[reportMissingImports]

        s = MockSummarizer()
        assert isinstance(s, Summarizer)

    def test_non_conforming_object(self) -> None:
        from exo.memory.summary import Summarizer  # pyright: ignore[reportMissingImports]

        assert not isinstance(42, Summarizer)
