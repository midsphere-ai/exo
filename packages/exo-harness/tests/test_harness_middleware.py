"""Tests for harness middleware."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

from exo.agent import Agent
from exo.harness.base import Harness, HarnessContext
from exo.harness.middleware import CostTrackingMiddleware, Middleware, TimeoutMiddleware
from exo.types import (
    ErrorEvent,
    StreamEvent,
    TextEvent,
    Usage,
    UsageEvent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeStreamChunk:
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


def _make_stream_provider(stream_rounds: list[list[_FakeStreamChunk]]) -> Any:
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


class PassthroughHarness(Harness):
    async def execute(self, ctx: HarnessContext) -> AsyncIterator[StreamEvent]:
        agent = next(iter(self.agents.values()))
        async for event in ctx.stream_agent(agent, ctx.input):
            yield event


class UsageEmittingHarness(Harness):
    """Harness that streams with detailed=True to get UsageEvents."""

    async def execute(self, ctx: HarnessContext) -> AsyncIterator[StreamEvent]:
        agent = next(iter(self.agents.values()))
        async for event in ctx.stream_agent(agent, ctx.input, detailed=True):
            yield event


# ---------------------------------------------------------------------------
# Tests: Middleware ABC
# ---------------------------------------------------------------------------


class TestMiddlewareABC:
    async def test_default_wrap_is_passthrough(self) -> None:
        """Concrete subclass with no-op wrap passes events through."""

        class NoOpMiddleware(Middleware):
            async def wrap(self, stream, ctx):
                async for event in stream:
                    yield event

        agent = Agent(name="bot")
        chunks = [_FakeStreamChunk(delta="hi")]
        provider = _make_stream_provider([chunks])

        h = PassthroughHarness(name="h", agents=[agent], middleware=[NoOpMiddleware()])
        events = [ev async for ev in h.stream("test", provider=provider)]

        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert len(text_events) == 1
        assert text_events[0].text == "hi"


# ---------------------------------------------------------------------------
# Tests: TimeoutMiddleware
# ---------------------------------------------------------------------------


class TestTimeoutMiddleware:
    async def test_no_timeout_when_fast(self) -> None:
        agent = Agent(name="bot")
        chunks = [_FakeStreamChunk(delta="ok")]
        provider = _make_stream_provider([chunks])

        h = PassthroughHarness(
            name="h",
            agents=[agent],
            middleware=[TimeoutMiddleware(10.0)],
        )
        events = [ev async for ev in h.stream("Hi", provider=provider)]

        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert len(text_events) == 1

    async def test_timeout_emits_error_event(self) -> None:
        """TimeoutMiddleware emits ErrorEvent when deadline exceeded."""

        class SlowHarness(Harness):
            async def execute(self, ctx):
                # Yield one event, then simulate slow processing
                yield TextEvent(text="first", agent_name="bot")
                # Move time forward by manipulating monotonic (not possible)
                # Instead, use a very short timeout
                import asyncio

                await asyncio.sleep(0.05)
                yield TextEvent(text="second", agent_name="bot")

        agent = Agent(name="bot")
        h = SlowHarness(
            name="h",
            agents=[agent],
            middleware=[TimeoutMiddleware(0.01)],
        )

        events = [ev async for ev in h.stream("Hi")]

        # First event should pass, then timeout error
        has_error = any(isinstance(e, ErrorEvent) for e in events)
        # The first event was yielded before timeout check
        assert events[0].type == "text"
        assert has_error


# ---------------------------------------------------------------------------
# Tests: CostTrackingMiddleware
# ---------------------------------------------------------------------------


class TestCostTrackingMiddleware:
    async def test_accumulates_usage(self) -> None:
        """CostTrackingMiddleware writes cumulative usage to state."""

        class UsageHarness(Harness):
            async def execute(self, ctx):
                yield UsageEvent(
                    usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
                    agent_name="a",
                    step_number=1,
                    model="test",
                )
                yield UsageEvent(
                    usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
                    agent_name="a",
                    step_number=2,
                    model="test",
                )

        h = UsageHarness(
            name="h",
            middleware=[CostTrackingMiddleware()],
        )
        events = [ev async for ev in h.stream("Hi")]

        assert len(events) == 2
        cost = h.session["_cost"]
        assert cost["input_tokens"] == 30
        assert cost["output_tokens"] == 15
        assert cost["total_tokens"] == 45

    async def test_no_usage_events_no_cost(self) -> None:
        """No UsageEvents means no _cost key in state."""

        class EmptyHarness(Harness):
            async def execute(self, ctx):
                yield TextEvent(text="hi", agent_name="bot")

        h = EmptyHarness(name="h", middleware=[CostTrackingMiddleware()])
        events = [ev async for ev in h.stream("Hi")]

        assert len(events) == 1
        assert "_cost" not in h.session


# ---------------------------------------------------------------------------
# Tests: Middleware chaining
# ---------------------------------------------------------------------------


class TestMiddlewareChaining:
    async def test_multiple_middleware(self) -> None:
        """Multiple middleware are applied in order."""

        class TagMiddleware(Middleware):
            def __init__(self, tag: str) -> None:
                self._tag = tag

            async def wrap(self, stream, ctx):
                ctx.state[f"_seen_{self._tag}"] = True
                async for event in stream:
                    yield event

        class SimpleHarness(Harness):
            async def execute(self, ctx):
                yield TextEvent(text="hi", agent_name="bot")

        h = SimpleHarness(
            name="h",
            middleware=[TagMiddleware("first"), TagMiddleware("second")],
        )
        events = [ev async for ev in h.stream("Hi")]

        assert len(events) == 1
        assert h.session["_seen_first"] is True
        assert h.session["_seen_second"] is True
