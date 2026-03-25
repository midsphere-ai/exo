"""Async hook system for agent lifecycle interception."""

from __future__ import annotations

import contextlib
import enum
from collections import defaultdict
from collections.abc import Callable, Coroutine
from typing import Any

Hook = Callable[..., Coroutine[Any, Any, None]]
"""Type alias for async hook functions."""


class HookPoint(enum.Enum):
    """Points in the agent lifecycle where hooks can be attached.

    Each value is a descriptive string for readable debug output.
    """

    START = "start"
    FINISHED = "finished"
    ERROR = "error"
    PRE_LLM_CALL = "pre_llm_call"
    POST_LLM_CALL = "post_llm_call"
    PRE_TOOL_CALL = "pre_tool_call"
    POST_TOOL_CALL = "post_tool_call"


class HookManager:
    """Manages async hooks attached to lifecycle points.

    Hooks are called sequentially in registration order. Unlike
    ``EventBus``, exceptions from hooks are **not** suppressed —
    a failing hook aborts the run.
    """

    def __init__(self) -> None:
        self._hooks: defaultdict[HookPoint, list[Hook]] = defaultdict(list)

    def add(self, point: HookPoint, hook: Hook) -> None:
        """Register a hook at a lifecycle point.

        Args:
            point: The lifecycle point to attach to.
            hook: Async callable to invoke at that point.
        """
        self._hooks[point].append(hook)

    def remove(self, point: HookPoint, hook: Hook) -> None:
        """Remove a hook from a lifecycle point.

        Silently does nothing if the hook is not registered.
        Removes the first occurrence only.

        Args:
            point: The lifecycle point.
            hook: The hook to remove.
        """
        with contextlib.suppress(ValueError):
            self._hooks[point].remove(hook)

    async def run(self, point: HookPoint, **data: Any) -> None:
        """Run all hooks for a lifecycle point sequentially.

        Exceptions from hooks propagate immediately — they are never
        silently swallowed.

        Args:
            point: The lifecycle point to fire.
            **data: Keyword arguments passed to each hook.
        """
        for hook in self._hooks[point]:
            await hook(**data)

    def has_hooks(self, point: HookPoint) -> bool:
        """Check whether any hooks are registered for a point.

        Args:
            point: The lifecycle point to check.

        Returns:
            True if at least one hook is registered.
        """
        return len(self._hooks[point]) > 0

    def clear(self) -> None:
        """Remove all hooks for all lifecycle points."""
        self._hooks.clear()


async def run_hooks(manager: HookManager, point: HookPoint, **data: Any) -> None:
    """Convenience function to run hooks on a manager.

    Args:
        manager: The hook manager to use.
        point: The lifecycle point to fire.
        **data: Keyword arguments passed to each hook.
    """
    await manager.run(point, **data)
