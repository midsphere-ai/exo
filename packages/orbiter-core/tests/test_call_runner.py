"""Tests for orbiter._internal.call_runner — core execution loop."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from orbiter._internal.call_runner import CallRunnerError, call_runner
from orbiter._internal.state import RunNodeStatus, RunState
from orbiter.agent import Agent
from orbiter.tool import tool
from orbiter.types import AgentOutput, RunResult, ToolCall, Usage

# ---------------------------------------------------------------------------
# Fixtures: mock provider
# ---------------------------------------------------------------------------


def _make_provider(responses: list[AgentOutput]) -> Any:
    """Create a mock provider that returns pre-defined AgentOutput values.

    Since call_runner delegates to Agent.run() which internally calls
    provider.complete(), we mock at the provider level to simulate
    the full pipeline.
    """
    mock = AsyncMock()
    # Agent.run() calls provider.complete() which returns a ModelResponse-like
    # object. We mock it to return objects with the fields parse_response needs.
    call_count = 0

    async def complete(messages: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1

        class FakeResponse:
            content = resp.text
            tool_calls = resp.tool_calls
            usage = resp.usage
            reasoning_content = ""
            thought_signatures: list[bytes] = []

        return FakeResponse()

    mock.complete = complete
    return mock


# ---------------------------------------------------------------------------
# Single-turn execution
# ---------------------------------------------------------------------------


class TestCallRunnerSingleTurn:
    async def test_simple_text_response(self) -> None:
        """Agent returns text on first call — single step, success."""
        agent = Agent(name="bot", instructions="You are helpful.")
        output = AgentOutput(
            text="Hello!",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        provider = _make_provider([output])

        result = await call_runner(agent, "Hi", provider=provider)

        assert isinstance(result, RunResult)
        assert result.output == "Hello!"
        assert result.steps == 1
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.usage.total_tokens == 15

    async def test_returns_run_result(self) -> None:
        """call_runner always returns a RunResult."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="ok")])

        result = await call_runner(agent, "test", provider=provider)

        assert isinstance(result, RunResult)
        assert result.output == "ok"

    async def test_messages_included(self) -> None:
        """Result includes message history."""
        agent = Agent(name="bot", instructions="Be nice.")
        provider = _make_provider([AgentOutput(text="Sure!")])

        result = await call_runner(agent, "hello", provider=provider)

        assert len(result.messages) > 0


# ---------------------------------------------------------------------------
# Multi-turn execution (tool calls)
# ---------------------------------------------------------------------------


class TestCallRunnerMultiTurn:
    async def test_tool_call_and_text(self) -> None:
        """Agent calls a tool then returns text — multi-step execution."""

        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Sunny in {city}"

        agent = Agent(name="weather-bot", tools=[get_weather])

        # First response: tool call. Second: text.
        responses = [
            AgentOutput(
                text="",
                tool_calls=[ToolCall(id="tc1", name="get_weather", arguments='{"city":"NYC"}')],
                usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
            AgentOutput(
                text="It's sunny in NYC!",
                usage=Usage(input_tokens=30, output_tokens=15, total_tokens=45),
            ),
        ]
        provider = _make_provider(responses)

        result = await call_runner(agent, "What's the weather?", provider=provider)

        assert result.output == "It's sunny in NYC!"
        assert result.steps == 1  # call_runner wraps a single Agent.run() call


# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------


class TestCallRunnerState:
    async def test_creates_state_if_none(self) -> None:
        """When no state is passed, call_runner creates one internally."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="done")])

        result = await call_runner(agent, "go", provider=provider)

        assert result.steps == 1

    async def test_uses_provided_state(self) -> None:
        """When state is passed, call_runner uses it."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="done")])
        state = RunState(agent_name="bot")

        result = await call_runner(agent, "go", state=state, provider=provider)

        assert state.status == RunNodeStatus.SUCCESS
        assert len(state.nodes) == 1
        assert state.nodes[0].status == RunNodeStatus.SUCCESS
        assert result.steps == state.iterations

    async def test_state_records_usage(self) -> None:
        """State accumulates usage from agent output."""
        agent = Agent(name="bot")
        usage = Usage(input_tokens=50, output_tokens=25, total_tokens=75)
        provider = _make_provider([AgentOutput(text="ok", usage=usage)])
        state = RunState(agent_name="bot")

        await call_runner(agent, "test", state=state, provider=provider)

        assert state.total_usage.input_tokens == 50
        assert state.total_usage.output_tokens == 25
        assert state.total_usage.total_tokens == 75

    async def test_node_lifecycle(self) -> None:
        """RunNode goes through INIT -> RUNNING -> SUCCESS."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="ok")])
        state = RunState(agent_name="bot")

        await call_runner(agent, "test", state=state, provider=provider)

        node = state.nodes[0]
        assert node.status == RunNodeStatus.SUCCESS
        assert node.started_at is not None
        assert node.ended_at is not None
        assert node.duration is not None
        assert node.duration >= 0


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------


class TestCallRunnerLoopDetection:
    async def test_no_loop_on_text_only(self) -> None:
        """Text-only responses don't trigger loop detection."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="ok")])

        result = await call_runner(agent, "test", provider=provider, loop_threshold=1)

        assert result.output == "ok"

    async def test_loop_detection_needs_threshold(self) -> None:
        """Loop detection only fires after threshold consecutive repeats."""

        @tool
        def noop() -> str:
            """Do nothing."""
            return "done"

        agent = Agent(name="bot", tools=[noop])

        # The tool calls happen within Agent.run() loop, so we need
        # repeated call_runner invocations with shared state to test
        # loop detection across calls.
        state = RunState(agent_name="bot")

        # First call — tool then text, succeeds
        responses1 = [
            AgentOutput(
                text="",
                tool_calls=[ToolCall(id="tc1", name="noop", arguments="")],
            ),
            AgentOutput(text="done1"),
        ]
        provider1 = _make_provider(responses1)
        await call_runner(agent, "go", state=state, provider=provider1, loop_threshold=5)
        assert state.status == RunNodeStatus.SUCCESS


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestCallRunnerErrors:
    async def test_agent_error_propagates(self) -> None:
        """Agent errors are wrapped in CallRunnerError."""
        agent = Agent(name="bot")
        # No provider — Agent.run() will raise AgentError
        with pytest.raises(CallRunnerError, match="Call runner failed"):
            await call_runner(agent, "test")

    async def test_state_marked_failed_on_error(self) -> None:
        """State is marked FAILED when agent raises."""
        agent = Agent(name="bot")
        state = RunState(agent_name="bot")

        with pytest.raises(CallRunnerError):
            await call_runner(agent, "test", state=state)

        assert state.status == RunNodeStatus.FAILED
        assert state.nodes[0].status == RunNodeStatus.FAILED

    async def test_node_records_error_message(self) -> None:
        """Failed node records the error message."""
        agent = Agent(name="bot")
        state = RunState(agent_name="bot")

        with pytest.raises(CallRunnerError):
            await call_runner(agent, "test", state=state)

        assert state.nodes[0].error is not None
        assert len(state.nodes[0].error) > 0

    async def test_provider_error_wrapped(self) -> None:
        """Provider errors are caught and wrapped."""
        agent = Agent(name="bot")
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=RuntimeError("API down"))

        with pytest.raises(CallRunnerError, match="Call runner failed"):
            await call_runner(agent, "test", provider=provider)


# ---------------------------------------------------------------------------
# Messages parameter passthrough
# ---------------------------------------------------------------------------


class TestCallRunnerMessages:
    async def test_prior_messages_passed(self) -> None:
        """Prior conversation history is forwarded to Agent.run()."""
        from orbiter.types import UserMessage

        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="continued")])

        prior = [UserMessage(content="Earlier message")]
        result = await call_runner(agent, "Continue", messages=prior, provider=provider)

        assert result.output == "continued"
