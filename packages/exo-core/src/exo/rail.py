"""Rail ABC with priority ordering and retry capability.

Rails are structured lifecycle guards that attach to agent hook points.
Each rail has a priority (lower runs first) and returns a ``RailAction``
to control execution flow: continue, skip, retry, or abort.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from exo.hooks import Hook, HookPoint
from exo.types import ExoError

if TYPE_CHECKING:
    from exo.rail_types import InvokeInputs, ModelCallInputs, RailContext, ToolCallInputs


class RailAction(enum.StrEnum):
    """Action returned by a rail to control execution flow.

    Attributes:
        CONTINUE: Proceed to the next rail (or the guarded operation).
        SKIP: Skip the guarded operation entirely.
        RETRY: Retry the guarded operation (see ``RetryRequest``).
        ABORT: Abort the agent run immediately.
    """

    CONTINUE = "continue"
    SKIP = "skip"
    RETRY = "retry"
    ABORT = "abort"


@dataclass(frozen=True)
class RetryRequest:
    """Parameters for a RETRY action.

    Attach an instance to the rail's return context when returning
    ``RailAction.RETRY`` so the caller knows how to retry.

    Args:
        delay: Seconds to wait before retrying.
        max_retries: Maximum number of retry attempts.
        reason: Human-readable explanation for the retry.
    """

    delay: float = 0.0
    max_retries: int = 1
    reason: str = ""


class RailAbortError(ExoError):
    """Raised when a rail returns ``RailAction.ABORT``.

    Args:
        rail_name: Name of the rail that triggered the abort.
        reason: Human-readable reason for the abort.
    """

    def __init__(self, rail_name: str, reason: str = "") -> None:
        self.rail_name = rail_name
        self.reason = reason
        msg = f"Rail '{rail_name}' aborted"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class Rail(abc.ABC):
    """Abstract base class for agent rails.

    A rail is a lifecycle guard that inspects and optionally alters
    agent execution at specific hook points.  Rails are run in
    ascending ``priority`` order (lower numbers run first).

    Args:
        name: Unique identifier for this rail.
        priority: Execution order (lower = earlier). Defaults to 50.

    Example:
        >>> class BlockDangerousTool(Rail):
        ...     async def handle(self, ctx):
        ...         if ctx.inputs.tool_name == "rm_rf":
        ...             return RailAction.ABORT
        ...         return RailAction.CONTINUE
    """

    def __init__(self, name: str, *, priority: int = 50) -> None:
        self.name = name
        self.priority = priority

    @abc.abstractmethod
    async def handle(self, ctx: RailContext) -> RailAction | None:
        """Inspect and optionally act on a lifecycle event.

        Args:
            ctx: The rail context containing agent, event, typed
                inputs, and cross-rail extra dict.

        Returns:
            A ``RailAction`` indicating what should happen next,
            or ``None`` (treated as ``CONTINUE``).
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, priority={self.priority})"


def _build_inputs(
    event: HookPoint, data: dict[str, Any]
) -> InvokeInputs | ModelCallInputs | ToolCallInputs:
    """Build the appropriate typed inputs model from raw hook data.

    Args:
        event: The lifecycle hook point (determines which model to use).
        data: Keyword arguments originally passed to the hook.

    Returns:
        A typed inputs model matching the event kind.
    """
    from exo.rail_types import InvokeInputs, ModelCallInputs, ToolCallInputs

    if event in (HookPoint.PRE_TOOL_CALL, HookPoint.POST_TOOL_CALL):
        return ToolCallInputs(
            tool_name=data.get("tool_name", ""),
            arguments=data.get("arguments", {}),
            result=data.get("result"),
            metadata=data.get("metadata"),
        )
    if event in (HookPoint.PRE_LLM_CALL, HookPoint.POST_LLM_CALL):
        return ModelCallInputs(
            messages=data.get("messages", []),
            tools=data.get("tools"),
            response=data.get("response"),
            usage=data.get("usage"),
        )
    return InvokeInputs(
        input=data.get("input", ""),
        messages=data.get("messages"),
        result=data.get("result"),
    )


class RailManager:
    """Manages rails with priority ordering and cross-rail state.

    Rails are run in ascending priority order (lower numbers first).
    A shared ``extra`` dict is passed through all rails within a single
    invocation, enabling cross-rail coordination.

    The manager can be registered as a single hook on
    :class:`~exo.hooks.HookManager` via :meth:`hook_for`.

    Example:
        >>> manager = RailManager()
        >>> manager.add(my_safety_rail)
        >>> action = await manager.run(HookPoint.PRE_LLM_CALL, agent=agent, messages=msgs)
    """

    def __init__(self) -> None:
        self._rails: list[Rail] = []

    def add(self, rail: Rail) -> None:
        """Add a rail to the manager.

        Args:
            rail: The rail instance to add.
        """
        self._rails.append(rail)

    def remove(self, rail: Rail) -> None:
        """Remove a rail from the manager.

        Args:
            rail: The rail instance to remove.

        Raises:
            ValueError: If the rail is not in the manager.
        """
        self._rails.remove(rail)

    def clear(self) -> None:
        """Remove all rails."""
        self._rails.clear()

    async def run(self, event: HookPoint, **data: Any) -> RailAction:
        """Run all rails for an event in priority order.

        Builds a :class:`~exo.rail_types.RailContext` from the keyword
        arguments, then executes each rail sorted by ascending priority.
        Returns the first non-CONTINUE action, or CONTINUE if all rails pass.

        The ``extra`` dict on the context is shared across all rails in
        the same invocation, allowing cross-rail state sharing.

        Args:
            event: The lifecycle hook point.
            **data: Keyword arguments (``agent`` plus event-specific data).

        Returns:
            The resulting :class:`RailAction`.
        """
        from exo.rail_types import RailContext

        agent = data.get("agent")
        inputs = _build_inputs(event, data)
        extra: dict[str, Any] = {}
        ctx = RailContext(agent=agent, event=event, inputs=inputs, extra=extra)

        for rail in sorted(self._rails, key=lambda r: r.priority):
            action = await rail.handle(ctx)
            if action is not None and action != RailAction.CONTINUE:
                return action

        return RailAction.CONTINUE

    def hook_for(self, event: HookPoint) -> Hook:
        """Create a hook callable for a specific lifecycle event.

        The returned async callable can be registered on a
        :class:`~exo.hooks.HookManager`.  If any rail returns
        :attr:`RailAction.ABORT`, a :class:`RailAbortError` is raised
        so that the agent run is halted.

        Args:
            event: The lifecycle hook point.

        Returns:
            An async callable compatible with the Hook type.
        """

        async def _hook(**hook_data: Any) -> None:
            action = await self.run(event, **hook_data)
            if action == RailAction.ABORT:
                raise RailAbortError("RailManager", reason=f"Rail aborted at {event.value}")

        return _hook
