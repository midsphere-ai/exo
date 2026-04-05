"""Middleware for harness event streams.

Middleware wraps the event stream produced by :meth:`Harness.execute`,
transforming, filtering, or augmenting events as they pass through.
Each middleware is an async generator transformer — it receives an
upstream ``AsyncIterator[StreamEvent]`` and yields events downstream.

Usage::

    harness = MyHarness(
        name="bot",
        agents=[agent_a],
        middleware=[TimeoutMiddleware(30.0), CostTrackingMiddleware()],
    )
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from exo.types import ErrorEvent, StreamEvent, UsageEvent

if TYPE_CHECKING:
    from exo.harness.base import HarnessContext


class Middleware(ABC):
    """Base class for harness middleware.

    Subclass and implement :meth:`wrap` to intercept or transform
    the event stream.
    """

    @abstractmethod
    async def wrap(
        self,
        stream: AsyncIterator[StreamEvent],
        ctx: HarnessContext,
    ) -> AsyncIterator[StreamEvent]:
        """Wrap an event stream, yielding modified/filtered events.

        Args:
            stream: Upstream event iterator.
            ctx: The harness runtime context.

        Yields:
            ``StreamEvent`` instances (passed through, modified,
            filtered, or newly created).
        """
        async for event in stream:
            yield event


class TimeoutMiddleware(Middleware):
    """Aborts execution after a wall-clock timeout.

    Emits an ``ErrorEvent`` and stops iteration once the deadline
    is exceeded.

    Args:
        timeout_seconds: Maximum wall-clock seconds before timeout.
    """

    def __init__(self, timeout_seconds: float) -> None:
        self._timeout = timeout_seconds

    async def wrap(
        self,
        stream: AsyncIterator[StreamEvent],
        ctx: HarnessContext,
    ) -> AsyncIterator[StreamEvent]:
        deadline = time.monotonic() + self._timeout
        async for event in stream:
            if time.monotonic() > deadline:
                yield ErrorEvent(
                    error=f"Harness timed out after {self._timeout}s",
                    error_type="TimeoutError",
                    agent_name=ctx._harness.name,
                    recoverable=False,
                )
                return
            yield event


class CostTrackingMiddleware(Middleware):
    """Accumulates token usage from ``UsageEvent`` instances.

    Writes cumulative totals to ``ctx.state["_cost"]`` as a dict
    with keys ``input_tokens``, ``output_tokens``, ``total_tokens``.

    All events pass through unmodified.
    """

    async def wrap(
        self,
        stream: AsyncIterator[Any],
        ctx: HarnessContext,
    ) -> AsyncIterator[StreamEvent]:
        total_input = 0
        total_output = 0
        total_total = 0
        async for event in stream:
            if isinstance(event, UsageEvent):
                total_input += event.usage.input_tokens
                total_output += event.usage.output_tokens
                total_total += event.usage.total_tokens
                ctx.state["_cost"] = {
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "total_tokens": total_total,
                }
            yield event
