"""Task loop tools — steer and abort tools that push events to a TaskLoopQueue."""

from __future__ import annotations

from typing import Any

from orbiter.tool import Tool, ToolError, _extract_description, _generate_schema

from .task_loop_queue import TaskLoopEvent, TaskLoopEventType, TaskLoopQueue


class _QueueTool(Tool):
    """A tool whose ``execute`` pushes events to a bound :class:`TaskLoopQueue`.

    The ``queue`` parameter is NOT part of the JSON schema — it is bound
    by the caller (runner / agent) at setup time.
    """

    __slots__ = ("_fn", "_queue", "description", "name", "parameters")

    def __init__(
        self,
        fn: Any,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self._fn = fn
        self.name = name or fn.__name__
        self.description = description or _extract_description(fn)
        schema = _generate_schema(fn)
        props = schema.get("properties", {})
        props.pop("queue", None)
        req = schema.get("required", [])
        schema["required"] = [r for r in req if r != "queue"]
        if not schema["required"]:
            del schema["required"]
        schema["properties"] = props
        self.parameters = schema
        self._queue: TaskLoopQueue | None = None

    def bind(self, queue: TaskLoopQueue) -> _QueueTool:
        """Bind a :class:`TaskLoopQueue` for subsequent calls."""
        self._queue = queue
        return self

    async def execute(self, **kwargs: Any) -> str | dict[str, Any]:
        if self._queue is None:
            raise ToolError(f"Tool '{self.name}' requires a bound queue (call .bind(queue) first)")
        return await self._fn(queue=self._queue, **kwargs)


async def _steer_agent(queue: TaskLoopQueue, content: str) -> str:
    """Push a STEER event to redirect the agent's current task.

    Use this to change the agent's direction without aborting the run.

    Args:
        content: Instruction describing the new direction for the agent.
    """
    event = TaskLoopEvent(type=TaskLoopEventType.STEER, content=content)
    queue.push(event)
    return f"Steering event queued: {content}"


async def _abort_agent(queue: TaskLoopQueue, reason: str) -> str:
    """Push an ABORT event to stop the agent's current task.

    Use this when the agent must stop immediately.

    Args:
        reason: Reason for aborting the agent run.
    """
    event = TaskLoopEvent(type=TaskLoopEventType.ABORT, content=reason)
    queue.push(event)
    return f"Abort event queued: {reason}"


steer_agent_tool = _QueueTool(
    _steer_agent,
    name="steer_agent",
    description="Push a STEER event to redirect the agent's current task.",
)

abort_agent_tool = _QueueTool(
    _abort_agent,
    name="abort_agent",
    description="Push an ABORT event to stop the agent's current task.",
)


def get_task_loop_tools() -> list[Tool]:
    """Return all task loop tools (steer + abort)."""
    return [steer_agent_tool, abort_agent_tool]
