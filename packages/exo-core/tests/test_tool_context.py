"""Tests for exo.tool_context — ToolContext injection and event emission."""

from __future__ import annotations

import asyncio

from exo.tool_context import ToolContext
from exo.types import TextEvent


class TestToolContextEmit:
    def test_emit_puts_event_on_queue(self) -> None:
        """emit() puts an event onto the backing queue."""
        queue: asyncio.Queue = asyncio.Queue()
        ctx = ToolContext(agent_name="parent", queue=queue)

        event = TextEvent(text="hello", agent_name="inner")
        ctx.emit(event)

        assert not queue.empty()
        assert queue.get_nowait() is event

    def test_multiple_emits(self) -> None:
        """Multiple emit() calls enqueue events in order."""
        queue: asyncio.Queue = asyncio.Queue()
        ctx = ToolContext(agent_name="parent", queue=queue)

        events = [TextEvent(text=f"msg-{i}", agent_name="inner") for i in range(3)]
        for ev in events:
            ctx.emit(ev)

        collected = []
        while not queue.empty():
            collected.append(queue.get_nowait())
        assert collected == events

    def test_agent_name_stored(self) -> None:
        """ToolContext stores the parent agent name."""
        queue: asyncio.Queue = asyncio.Queue()
        ctx = ToolContext(agent_name="my_agent", queue=queue)
        assert ctx.agent_name == "my_agent"

    def test_unused_queue_stays_empty(self) -> None:
        """Queue remains empty when emit() is never called."""
        queue: asyncio.Queue = asyncio.Queue()
        ToolContext(agent_name="parent", queue=queue)
        assert queue.empty()
