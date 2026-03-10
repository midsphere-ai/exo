"""Rail ABC with priority ordering and retry capability.

Rails are structured lifecycle guards that attach to agent hook points.
Each rail has a priority (lower runs first) and returns a ``RailAction``
to control execution flow: continue, skip, retry, or abort.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from orbiter.types import OrbiterError

if TYPE_CHECKING:
    from orbiter.rail_types import RailContext


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


class RailAbortError(OrbiterError):
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
