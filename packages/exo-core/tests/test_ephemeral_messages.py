"""Tests for ephemeral message injection — visible for one LLM call only."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from exo.agent import Agent
from exo.runner import run
from exo.tool import tool
from exo.types import (
    AgentOutput,
    StreamEvent,
    SystemMessage,
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
    """Lightweight stream chunk for testing."""

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


def _make_stream_provider(stream_rounds: list[list[_FakeStreamChunk]]) -> Any:
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
# Tests: inject_ephemeral on Agent (non-streaming run path)
# ---------------------------------------------------------------------------


class TestEphemeralBasic:
    async def test_ephemeral_present_in_llm_call(self) -> None:
        """Ephemeral content appears in messages passed to the LLM."""
        ephemeral_text = "Temporary guidance"
        seen_messages: list[Any] = []

        async def complete(messages: Any, **kwargs: Any) -> Any:
            seen_messages.extend(messages)

            class R:
                content = "OK"
                tool_calls = []
                usage = Usage()

            return R()

        provider = AsyncMock()
        provider.complete = complete

        agent = Agent(name="bot", instructions="Be helpful.")
        agent.inject_ephemeral(ephemeral_text)

        await run(agent, "Hello", provider=provider)

        user_msgs = [m for m in seen_messages if isinstance(m, UserMessage)]
        ephemeral = [m for m in user_msgs if m.content == ephemeral_text]
        assert len(ephemeral) == 1

    async def test_ephemeral_removed_after_call(self) -> None:
        """Ephemeral message is NOT present in the second LLM call."""
        ephemeral_text = "One-shot context"
        calls: list[list[Any]] = []

        @tool
        def ping(msg: str) -> str:
            """Simple tool."""
            return "pong"

        call_count = 0

        async def complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            calls.append(list(messages))
            call_count += 1
            if call_count == 1:
                class R1:
                    content = ""
                    tool_calls = [
                        ToolCall(id="tc1", name="ping", arguments='{"msg":"hi"}')
                    ]
                    usage = Usage()
                return R1()
            else:
                class R2:
                    content = "Done"
                    tool_calls = []
                    usage = Usage()
                return R2()

        provider = AsyncMock()
        provider.complete = complete

        agent = Agent(name="bot", tools=[ping])
        agent.inject_ephemeral(ephemeral_text)

        result = await run(agent, "Go", provider=provider)
        assert result.output == "Done"
        assert len(calls) == 2

        # First call: ephemeral present
        first_user_msgs = [m for m in calls[0] if isinstance(m, UserMessage)]
        assert any(m.content == ephemeral_text for m in first_user_msgs)

        # Second call: ephemeral gone
        second_user_msgs = [m for m in calls[1] if isinstance(m, UserMessage)]
        assert not any(m.content == ephemeral_text for m in second_user_msgs)

    async def test_ephemeral_empty_string_raises(self) -> None:
        """inject_ephemeral('') raises ValueError."""
        agent = Agent(name="bot")
        with pytest.raises(ValueError, match="non-empty"):
            agent.inject_ephemeral("")

    async def test_ephemeral_accepts_message_object(self) -> None:
        """Can pass a SystemMessage directly."""
        sys_msg = SystemMessage(content="Ephemeral system instruction")
        seen_messages: list[Any] = []

        async def complete(messages: Any, **kwargs: Any) -> Any:
            seen_messages.extend(messages)

            class R:
                content = "OK"
                tool_calls = []
                usage = Usage()

            return R()

        provider = AsyncMock()
        provider.complete = complete

        agent = Agent(name="bot")
        agent.inject_ephemeral(sys_msg)

        await run(agent, "Hello", provider=provider)

        system_msgs = [m for m in seen_messages if isinstance(m, SystemMessage)]
        assert any(m.content == "Ephemeral system instruction" for m in system_msgs)

    async def test_ephemeral_multiple_fifo(self) -> None:
        """Multiple ephemeral messages appear in FIFO order, all removed after."""
        texts = ["First eph", "Second eph", "Third eph"]
        calls: list[list[Any]] = []

        @tool
        def noop(x: str) -> str:
            """No-op tool."""
            return "ok"

        call_count = 0

        async def complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            calls.append(list(messages))
            call_count += 1
            if call_count == 1:
                class R1:
                    content = ""
                    tool_calls = [
                        ToolCall(id="tc1", name="noop", arguments='{"x":"go"}')
                    ]
                    usage = Usage()
                return R1()
            else:
                class R2:
                    content = "Done"
                    tool_calls = []
                    usage = Usage()
                return R2()

        provider = AsyncMock()
        provider.complete = complete

        agent = Agent(name="bot", tools=[noop])
        for t in texts:
            agent.inject_ephemeral(t)

        await run(agent, "Go", provider=provider)

        # First call: all three present in order
        first_user = [m for m in calls[0] if isinstance(m, UserMessage)]
        eph_contents = [m.content for m in first_user if m.content in texts]
        assert eph_contents == texts

        # Second call: none present
        second_user = [m for m in calls[1] if isinstance(m, UserMessage)]
        assert not any(m.content in texts for m in second_user)

    async def test_ephemeral_with_inject_message_coexist(self) -> None:
        """inject_message persists; inject_ephemeral does not."""
        persistent_text = "I persist"
        ephemeral_text = "I vanish"
        calls: list[list[Any]] = []

        @tool
        def noop(x: str) -> str:
            """No-op tool."""
            return "ok"

        call_count = 0

        async def complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            calls.append(list(messages))
            call_count += 1
            if call_count == 1:
                class R1:
                    content = ""
                    tool_calls = [
                        ToolCall(id="tc1", name="noop", arguments='{"x":"go"}')
                    ]
                    usage = Usage()
                return R1()
            else:
                class R2:
                    content = "Done"
                    tool_calls = []
                    usage = Usage()
                return R2()

        provider = AsyncMock()
        provider.complete = complete

        agent = Agent(name="bot", tools=[noop])
        agent.inject_message(persistent_text)
        agent.inject_ephemeral(ephemeral_text)

        await run(agent, "Go", provider=provider)

        # First call: both present
        first_user = [m.content for m in calls[0] if isinstance(m, UserMessage)]
        assert persistent_text in first_user
        assert ephemeral_text in first_user

        # Second call: persistent stays, ephemeral gone
        second_user = [m.content for m in calls[1] if isinstance(m, UserMessage)]
        assert persistent_text in second_user
        assert ephemeral_text not in second_user

    async def test_ephemeral_persists_across_retries(self) -> None:
        """Ephemeral message is present on both the failed and retried attempt."""
        ephemeral_text = "Retry context"
        attempt_messages: list[list[Any]] = []
        call_count = 0

        async def complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            attempt_messages.append(list(messages))
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient failure")

            class R:
                content = "OK"
                tool_calls = []
                usage = Usage()

            return R()

        provider = AsyncMock()
        provider.complete = complete

        agent = Agent(name="bot")
        agent.inject_ephemeral(ephemeral_text)

        result = await run(agent, "Go", provider=provider, max_retries=2)
        assert result.output == "OK"
        assert len(attempt_messages) == 2

        # Both attempts should have the ephemeral message
        for i, msgs in enumerate(attempt_messages):
            user_msgs = [m for m in msgs if isinstance(m, UserMessage)]
            assert any(m.content == ephemeral_text for m in user_msgs), (
                f"Ephemeral missing in attempt {i + 1}"
            )


# ---------------------------------------------------------------------------
# Tests: streaming path
# ---------------------------------------------------------------------------


class TestEphemeralStream:
    async def test_ephemeral_stream_present_then_removed(self) -> None:
        """In streaming, ephemeral is present for first call, gone for second."""
        ephemeral_text = "Stream ephemeral"
        calls: list[list[Any]] = []

        @tool
        def ping(msg: str) -> str:
            """Simple tool."""
            return "pong"

        call_count = 0

        async def stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
            nonlocal call_count
            calls.append(list(messages))
            call_count += 1
            if call_count == 1:
                yield _FakeStreamChunk(
                    tool_call_deltas=[
                        _FakeToolCallDelta(
                            index=0, id="tc1", name="ping", arguments='{"msg":"hi"}'
                        ),
                    ],
                    usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
                )
            else:
                yield _FakeStreamChunk(
                    delta="Done!",
                    finish_reason="stop",
                    usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
                )

        provider = AsyncMock()
        provider.stream = stream
        provider.complete = AsyncMock()

        agent = Agent(name="bot", tools=[ping])
        agent.inject_ephemeral(ephemeral_text)

        events: list[StreamEvent] = []
        async for ev in run.stream(agent, "Go", provider=provider):
            events.append(ev)

        assert len(calls) == 2

        # First call: ephemeral present
        first_user = [m for m in calls[0] if isinstance(m, UserMessage)]
        assert any(m.content == ephemeral_text for m in first_user)

        # Second call: ephemeral gone
        second_user = [m for m in calls[1] if isinstance(m, UserMessage)]
        assert not any(m.content == ephemeral_text for m in second_user)

        # Verify we got text output
        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert any(e.text == "Done!" for e in text_events)
