"""Tests for live message injection into a running agent."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from exo.agent import Agent
from exo.runner import run
from exo.tool import tool
from exo.types import (
    AgentOutput,
    MessageInjectedEvent,
    StreamEvent,
    TextEvent,
    ToolCall,
    Usage,
    UserMessage,
)

# ---------------------------------------------------------------------------
# Fixtures: mock providers
# ---------------------------------------------------------------------------


def _make_provider(responses: list[AgentOutput]) -> Any:
    """Create a mock provider returning pre-defined AgentOutput values."""
    call_count = 0

    async def complete(messages: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1

        class FakeResponse:
            content = resp.text
            tool_calls = resp.tool_calls
            usage = resp.usage

        return FakeResponse()

    mock = AsyncMock()
    mock.complete = complete
    return mock


class _FakeStreamChunk:
    """Lightweight stream chunk for testing (mirrors StreamChunk fields)."""

    def __init__(
        self,
        delta: str = "",
        tool_call_deltas: list[Any] | None = None,
        finish_reason: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.delta = delta
        self.tool_call_deltas = tool_call_deltas or []
        self.finish_reason = finish_reason
        self.usage = usage or Usage()


class _FakeToolCallDelta:
    """Mirrors ToolCallDelta fields for testing."""

    def __init__(
        self,
        index: int = 0,
        id: str | None = None,
        name: str | None = None,
        arguments: str = "",
    ) -> None:
        self.index = index
        self.id = id
        self.name = name
        self.arguments = arguments


def _make_stream_provider(
    stream_rounds: list[list[_FakeStreamChunk]],
) -> Any:
    """Create a mock provider with stream() returning pre-defined chunks."""
    call_count = 0

    async def stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        nonlocal call_count
        chunks = stream_rounds[min(call_count, len(stream_rounds) - 1)]
        call_count += 1
        for c in chunks:
            yield c

    mock = AsyncMock()
    mock.stream = stream
    mock.complete = AsyncMock()
    return mock


# ---------------------------------------------------------------------------
# Tests: inject_message on Agent (non-streaming run path)
# ---------------------------------------------------------------------------


class TestInjectMessageBasic:
    async def test_inject_message_basic(self) -> None:
        """Inject during tool execution — verify injected UserMessage appears."""
        injected_text = "Also check the weather"

        @tool
        def slow_tool(query: str) -> str:
            """A tool that triggers injection as a side-effect."""
            # Simulate external code injecting a message while tool runs
            agent_ref.inject_message(injected_text)
            return f"result for {query}"

        agent_ref = Agent(name="bot", instructions="Be helpful.", tools=[slow_tool])

        # Track messages seen by the second LLM call
        seen_messages: list[Any] = []

        call_count = 0

        async def complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: trigger tool
                class R1:
                    content = ""
                    tool_calls = [
                        ToolCall(id="tc1", name="slow_tool", arguments='{"query":"test"}')
                    ]
                    usage = Usage()

                return R1()
            else:
                # Second call: capture messages for assertion, return final text
                seen_messages.extend(messages)

                class R2:
                    content = "Done!"
                    tool_calls = []
                    usage = Usage()

                return R2()

        provider = AsyncMock()
        provider.complete = complete

        result = await run(agent_ref, "Search for something", provider=provider)

        assert result.output == "Done!"
        # Verify the injected message appears as a UserMessage in the second call
        user_msgs = [m for m in seen_messages if isinstance(m, UserMessage)]
        injected = [m for m in user_msgs if m.content == injected_text]
        assert len(injected) == 1, f"Expected 1 injected message, found {len(injected)}"

    async def test_inject_message_empty_raises(self) -> None:
        """inject_message('') raises ValueError."""
        agent = Agent(name="bot")
        with pytest.raises(ValueError, match="non-empty"):
            agent.inject_message("")

    async def test_inject_message_multiple_fifo(self) -> None:
        """Multiple injected messages appear in FIFO order."""
        messages_a = "First injected"
        messages_b = "Second injected"
        messages_c = "Third injected"

        @tool
        def trigger(query: str) -> str:
            """Trigger injection of three messages."""
            agent_ref.inject_message(messages_a)
            agent_ref.inject_message(messages_b)
            agent_ref.inject_message(messages_c)
            return "done"

        agent_ref = Agent(name="bot", tools=[trigger])

        seen_messages: list[Any] = []
        call_count = 0

        async def complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:

                class R1:
                    content = ""
                    tool_calls = [ToolCall(id="tc1", name="trigger", arguments='{"query":"go"}')]
                    usage = Usage()

                return R1()
            else:
                seen_messages.extend(messages)

                class R2:
                    content = "All done."
                    tool_calls = []
                    usage = Usage()

                return R2()

        provider = AsyncMock()
        provider.complete = complete

        result = await run(agent_ref, "Go", provider=provider)
        assert result.output == "All done."

        # Extract injected UserMessages (skip the original user query)
        user_msgs = [m for m in seen_messages if isinstance(m, UserMessage)]
        injected_contents = [
            m.content for m in user_msgs if m.content in (messages_a, messages_b, messages_c)
        ]
        assert injected_contents == [messages_a, messages_b, messages_c]

    async def test_inject_no_messages_noop(self) -> None:
        """No injection results in no extra UserMessages."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="Hello!")])

        result = await run(agent, "Hi", provider=provider)
        assert result.output == "Hello!"


# ---------------------------------------------------------------------------
# Tests: streaming path — MessageInjectedEvent
# ---------------------------------------------------------------------------


class TestInjectMessageStream:
    async def test_inject_message_stream_yields_event(self) -> None:
        """run.stream() yields MessageInjectedEvent when message is injected."""
        injected_text = "Check this too"

        @tool
        def do_work(task: str) -> str:
            """Tool that injects a message as side-effect."""
            agent_ref.inject_message(injected_text)
            return "work done"

        agent_ref = Agent(name="streamer", tools=[do_work])

        # Round 1: tool call chunks
        round1 = [
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, id="tc1", name="do_work", arguments='{"task":'),
                ],
            ),
            _FakeStreamChunk(
                tool_call_deltas=[
                    _FakeToolCallDelta(index=0, arguments='"go"}'),
                ],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        ]
        # Round 2: final text
        round2 = [
            _FakeStreamChunk(delta="Done!", finish_reason="stop"),
        ]
        provider = _make_stream_provider([round1, round2])

        events: list[StreamEvent] = []
        async for ev in run.stream(agent_ref, "Do it", provider=provider):
            events.append(ev)

        injected_events = [e for e in events if isinstance(e, MessageInjectedEvent)]
        assert len(injected_events) == 1
        assert injected_events[0].content == injected_text
        assert injected_events[0].agent_name == "streamer"
        assert injected_events[0].type == "message_injected"

    async def test_inject_no_messages_no_event_in_stream(self) -> None:
        """No injection → no MessageInjectedEvent in stream."""
        agent = Agent(name="bot")
        chunks = [_FakeStreamChunk(delta="Hello", finish_reason="stop")]
        provider = _make_stream_provider([chunks])

        events: list[StreamEvent] = []
        async for ev in run.stream(agent, "Hi", provider=provider):
            events.append(ev)

        injected_events = [e for e in events if isinstance(e, MessageInjectedEvent)]
        assert len(injected_events) == 0
        # Should have at least one text event
        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert len(text_events) >= 1


# ---------------------------------------------------------------------------
# Tests: MessageInjectedEvent Pydantic model
# ---------------------------------------------------------------------------


class TestMessageInjectedEventModel:
    def test_defaults(self) -> None:
        e = MessageInjectedEvent(content="hello")
        assert e.type == "message_injected"
        assert e.content == "hello"
        assert e.agent_name == ""

    def test_with_agent_name(self) -> None:
        e = MessageInjectedEvent(content="test", agent_name="bot")
        assert e.agent_name == "bot"

    def test_frozen(self) -> None:
        e = MessageInjectedEvent(content="test")
        with pytest.raises(ValidationError):
            e.content = "other"  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        e = MessageInjectedEvent(content="injected text", agent_name="bot")
        data = e.model_dump()
        e2 = MessageInjectedEvent.model_validate(data)
        assert e == e2
        assert data["type"] == "message_injected"
        assert data["content"] == "injected text"
        assert data["agent_name"] == "bot"
