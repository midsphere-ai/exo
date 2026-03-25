"""Tests for the prompt execution logger."""

from __future__ import annotations

import logging

import pytest

from exo.observability.prompt_logger import (  # pyright: ignore[reportMissingImports]
    DEFAULT_CHAR_TOKEN_RATIO,
    ExecutionLogEntry,
    PromptLogger,
    TokenBreakdown,
    compute_token_breakdown,
    estimate_tokens,
)

# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0

    def test_short_string(self) -> None:
        assert estimate_tokens("hi") >= 1

    def test_long_string(self) -> None:
        text = "a" * 400
        result = estimate_tokens(text)
        assert result == int(400 / DEFAULT_CHAR_TOKEN_RATIO)

    def test_custom_ratio(self) -> None:
        text = "a" * 100
        assert estimate_tokens(text, ratio=2.0) == 50


# ---------------------------------------------------------------------------
# TokenBreakdown
# ---------------------------------------------------------------------------


class TestTokenBreakdown:
    def test_defaults(self) -> None:
        bd = TokenBreakdown()
        assert bd.total == 0

    def test_total(self) -> None:
        bd = TokenBreakdown(system=10, user=20, assistant=30, tool=40, other=5)
        assert bd.total == 105

    def test_percentages(self) -> None:
        bd = TokenBreakdown(system=100, user=0, assistant=0, tool=0, other=0)
        pcts = bd.percentages(1000)
        assert pcts["system"] == 10.0
        assert pcts["free"] == 90.0

    def test_percentages_zero_window(self) -> None:
        bd = TokenBreakdown(system=10)
        pcts = bd.percentages(0)
        assert all(v == 0.0 for v in pcts.values())

    def test_frozen(self) -> None:
        bd = TokenBreakdown()
        with pytest.raises(AttributeError):
            bd.system = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compute_token_breakdown
# ---------------------------------------------------------------------------


class TestComputeTokenBreakdown:
    def test_empty_messages(self) -> None:
        bd = compute_token_breakdown([])
        assert bd.total == 0

    def test_single_system(self) -> None:
        msgs = [{"role": "system", "content": "You are helpful."}]
        bd = compute_token_breakdown(msgs)
        assert bd.system > 0
        assert bd.user == 0

    def test_multiple_roles(self) -> None:
        msgs = [
            {"role": "system", "content": "Be nice."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        bd = compute_token_breakdown(msgs)
        assert bd.system > 0
        assert bd.user > 0
        assert bd.assistant > 0
        assert bd.tool == 0

    def test_tool_role(self) -> None:
        msgs = [{"role": "tool", "content": "result of tool call"}]
        bd = compute_token_breakdown(msgs)
        assert bd.tool > 0

    def test_unknown_role(self) -> None:
        msgs = [{"role": "custom_role", "content": "data"}]
        bd = compute_token_breakdown(msgs)
        assert bd.other > 0

    def test_tool_calls_in_assistant(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": '{"city": "London"}'}},
                ],
            }
        ]
        bd = compute_token_breakdown(msgs)
        assert bd.assistant > 0

    def test_none_content(self) -> None:
        msgs = [{"role": "assistant", "content": None}]
        bd = compute_token_breakdown(msgs)
        assert bd.assistant == 0

    def test_custom_ratio(self) -> None:
        msgs = [{"role": "user", "content": "a" * 100}]
        bd_default = compute_token_breakdown(msgs)
        bd_custom = compute_token_breakdown(msgs, ratio=2.0)
        assert bd_custom.user > bd_default.user


# ---------------------------------------------------------------------------
# Multi-modal content
# ---------------------------------------------------------------------------


class TestMultiModalContent:
    def test_text_item(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Describe this image."}],
            }
        ]
        bd = compute_token_breakdown(msgs)
        assert bd.user > 0

    def test_image_url_item(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                ],
            }
        ]
        bd = compute_token_breakdown(msgs)
        assert bd.user == 85  # fixed image estimate

    def test_tool_use_item(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "tool_use": {"name": "calc", "args": "{}"}}],
            }
        ]
        bd = compute_token_breakdown(msgs)
        assert bd.assistant > 0

    def test_mixed_content(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this:"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            }
        ]
        bd = compute_token_breakdown(msgs)
        assert bd.user > 85  # text + image


# ---------------------------------------------------------------------------
# ExecutionLogEntry
# ---------------------------------------------------------------------------


class TestExecutionLogEntry:
    def test_defaults(self) -> None:
        entry = ExecutionLogEntry()
        assert entry.agent_name == ""
        assert entry.message_count == 0

    def test_format_summary_with_context_window(self) -> None:
        entry = ExecutionLogEntry(
            agent_name="test-agent",
            model_name="gpt-4",
            message_count=5,
            breakdown=TokenBreakdown(system=1000, user=500, assistant=300, tool=200),
            context_window=8192,
            duration_s=1.234,
            tool_names=["search", "calc"],
        )
        summary = entry.format_summary()
        assert "test-agent" in summary
        assert "gpt-4" in summary
        assert "1.234" in summary
        assert "system=" in summary
        assert "search" in summary

    def test_format_summary_without_context_window(self) -> None:
        entry = ExecutionLogEntry(
            agent_name="a",
            breakdown=TokenBreakdown(system=10, user=20),
        )
        summary = entry.format_summary()
        assert "system=10" in summary
        assert "user=20" in summary

    def test_format_summary_no_tools(self) -> None:
        entry = ExecutionLogEntry(agent_name="a")
        summary = entry.format_summary()
        assert "Tools:" not in summary


# ---------------------------------------------------------------------------
# PromptLogger
# ---------------------------------------------------------------------------


class TestPromptLogger:
    def test_init_default(self) -> None:
        pl = PromptLogger()
        assert pl._ratio == DEFAULT_CHAR_TOKEN_RATIO

    def test_init_custom(self) -> None:
        custom_logger = logging.getLogger("test_custom")
        pl = PromptLogger(log=custom_logger, ratio=2.0)
        assert pl._ratio == 2.0

    def test_log_execution_returns_entry(self) -> None:
        pl = PromptLogger()
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello world"},
        ]
        entry = pl.log_execution(
            msgs,
            agent_name="agent-1",
            model_name="gpt-4",
            context_window=8192,
        )
        assert isinstance(entry, ExecutionLogEntry)
        assert entry.agent_name == "agent-1"
        assert entry.message_count == 2
        assert entry.breakdown.system > 0
        assert entry.breakdown.user > 0

    def test_log_execution_emits_log(self, caplog: pytest.LogCaptureFixture) -> None:
        pl = PromptLogger()
        msgs = [{"role": "user", "content": "test"}]
        with caplog.at_level(logging.INFO, logger="exo.prompt"):
            pl.log_execution(msgs, agent_name="a")
        assert "LLM Execution" in caplog.text

    def test_log_execution_custom_level(self, caplog: pytest.LogCaptureFixture) -> None:
        pl = PromptLogger()
        msgs = [{"role": "user", "content": "test"}]
        with caplog.at_level(logging.DEBUG, logger="exo.prompt"):
            pl.log_execution(msgs, agent_name="a", level=logging.DEBUG)
        assert "LLM Execution" in caplog.text

    def test_log_execution_with_tools(self) -> None:
        pl = PromptLogger()
        msgs = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_weather", "arguments": '{"city": "NYC"}'}},
                ],
            },
            {"role": "tool", "content": "Sunny 72F"},
        ]
        entry = pl.log_execution(
            msgs,
            agent_name="weather-bot",
            tool_names=["get_weather"],
        )
        assert entry.breakdown.tool > 0
        assert entry.tool_names == ["get_weather"]

    def test_log_execution_empty(self) -> None:
        pl = PromptLogger()
        entry = pl.log_execution([])
        assert entry.message_count == 0
        assert entry.breakdown.total == 0


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_flow(self) -> None:
        """End-to-end: messages -> breakdown -> entry -> formatted summary."""
        messages = [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."},
        ]
        pl = PromptLogger()
        entry = pl.log_execution(
            messages,
            agent_name="math-agent",
            model_name="gpt-4",
            context_window=128_000,
            duration_s=0.5,
        )
        summary = entry.format_summary()
        assert "math-agent" in summary
        assert "gpt-4" in summary
        # All roles should have some tokens
        assert entry.breakdown.system > 0
        assert entry.breakdown.user > 0
        assert entry.breakdown.assistant > 0

    def test_multimodal_flow(self) -> None:
        """Multi-modal messages compute breakdown correctly."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            },
            {"role": "assistant", "content": "I see a cat."},
        ]
        bd = compute_token_breakdown(messages)
        assert bd.user > 85  # text + image
        assert bd.assistant > 0

    def test_breakdown_percentages_sum(self) -> None:
        """Percentages of all roles + free should roughly sum to 100."""
        bd = TokenBreakdown(system=100, user=200, assistant=150, tool=50)
        pcts = bd.percentages(1000)
        total_pct = sum(pcts.values())
        assert abs(total_pct - 100.0) < 0.01
