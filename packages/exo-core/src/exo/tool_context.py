"""ToolContext: injectable context for tools that emit streaming events."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exo.types import StreamEvent


class ToolContext:
    """Injected into tools that declare a ToolContext parameter.

    Provides a way for tools to push streaming events onto the parent
    agent's event queue.  Events pushed here are drained by
    ``run.stream()`` after tool execution completes.

    Args:
        agent_name: Name of the parent agent.
        queue: The parent agent's event queue.
    """

    __slots__ = ("_queue", "agent_name")

    def __init__(self, agent_name: str, queue: asyncio.Queue[StreamEvent]) -> None:
        self._queue = queue
        self.agent_name = agent_name

    def emit(self, event: StreamEvent) -> None:
        """Push an event to the parent agent's stream.

        This is non-blocking. Events are buffered in the queue and
        drained by the runner after tool execution completes.

        Args:
            event: The streaming event to forward.
        """
        self._queue.put_nowait(event)
