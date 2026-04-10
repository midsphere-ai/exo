"""Task loop tools — steer and abort tools that push events to a TaskLoopQueue."""

from __future__ import annotations

from typing import Any

from exo.tool import Tool, ToolError, _extract_description, _generate_schema
from exo.tool_result import tool_error, tool_ok

from .task_loop_queue import TaskLoopEvent, TaskLoopEventType, TaskLoopQueue


class _QueueTool(Tool):
    """A tool whose ``execute`` pushes events to a bound :class:`TaskLoopQueue`.

    The ``queue`` parameter is NOT part of the JSON schema — it is bound
    by the caller (runner / agent) at setup time.

    ``_ptc_exclude=True`` keeps task-loop tools (``steer_agent``,
    ``abort_agent``) as *direct* schemas even when the owning agent has
    ``ptc=True``.  Steer/abort are control-flow signals for another
    agent's task loop — wrapping them in a PTC ``code`` payload would
    add latency and break the immediacy contract the task controller
    depends on.
    """

    _ptc_exclude: bool = True

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
    try:
        event = TaskLoopEvent(type=TaskLoopEventType.STEER, content=content)
        queue.push(event)
        return tool_ok(f"Steering event queued: {content}")
    except Exception as exc:
        return tool_error(
            f"Failed to push steer event: {exc}",
            hint="Retry the steer_agent call.",
        )


async def _abort_agent(queue: TaskLoopQueue, reason: str) -> str:
    """Push an ABORT event to stop the agent's current task.

    Use this when the agent must stop immediately.

    Args:
        reason: Reason for aborting the agent run.
    """
    try:
        event = TaskLoopEvent(type=TaskLoopEventType.ABORT, content=reason)
        queue.push(event)
        return tool_ok(f"Abort event queued: {reason}")
    except Exception as exc:
        return tool_error(
            f"Failed to push abort event: {exc}",
            hint="Retry the abort_agent call.",
        )


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
