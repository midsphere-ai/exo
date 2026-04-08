"""ToolContext: injectable context for tools that emit streaming events."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from exo.tool import ToolError

if TYPE_CHECKING:
    from exo.types import StreamEvent


class ToolContext:
    """Injected into tools that declare a ToolContext parameter.

    Provides a way for tools to push streaming events onto the parent
    agent's event queue and to request human approval before sensitive
    operations.

    Args:
        agent_name: Name of the parent agent.
        queue: The parent agent's event queue.
        human_input_handler: Optional handler for HITL approval prompts.
    """

    __slots__ = ("_human_input_handler", "_queue", "agent_name")

    def __init__(
        self,
        agent_name: str,
        queue: asyncio.Queue[StreamEvent],
        human_input_handler: Any = None,
    ) -> None:
        self._queue = queue
        self._human_input_handler = human_input_handler
        self.agent_name = agent_name

    def emit(self, event: StreamEvent) -> None:
        """Push an event to the parent agent's stream.

        This is non-blocking. Events are buffered in the queue and
        drained by the runner after tool execution completes.

        Args:
            event: The streaming event to forward.
        """
        self._queue.put_nowait(event)

    async def require_approval(self, message: str = "Approve this operation?") -> None:
        """Block until a human approves this operation.

        Call this inside a tool's ``execute()`` to gate sensitive
        operations on human approval.  The handler is sourced from
        the agent's ``human_input_handler``.

        Args:
            message: Prompt shown to the human.

        Raises:
            ToolError: If no handler is configured or the human denies.
        """
        if self._human_input_handler is None:
            raise ToolError(
                "require_approval() called but no human_input_handler is set on the agent"
            )
        response = await self._human_input_handler.get_input(
            message, choices=["yes", "no"]
        )
        if response.strip().lower() not in ("yes", "y"):
            raise ToolError("Tool execution denied by human")
