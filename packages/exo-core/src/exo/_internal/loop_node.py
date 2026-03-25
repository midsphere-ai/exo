"""Iteration node for workflow loops.

A :class:`LoopNode` repeats one or more body agents based on a count,
an array of items, or a while-style expression condition.

Three loop modes:

- **count**: ``LoopNode(name="repeat", count=3, body="worker")``
  Runs the body agent(s) exactly *count* times.
- **items**: ``LoopNode(name="each", items="documents", body="processor")``
  Iterates over the array stored under the given key in workflow state.
- **condition**: ``LoopNode(name="poll", condition="status != 'done'", body="checker")``
  Repeats while the expression evaluates to truthy.

All modes enforce ``max_iterations`` (default 100) as a safety limit.
If a body agent's output contains ``[BREAK]``, the loop terminates early.

Usage::

    loop = LoopNode(
        name="process_items",
        items="tasks",
        body="worker",
    )
    swarm = Swarm(
        agents=[loop, worker_agent],
        flow="process_items >> done",
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from exo._internal.expression import ExpressionError, evaluate_expression
from exo.types import ExoError, Message, RunResult

BREAK_SENTINEL = "[BREAK]"


class LoopError(ExoError):
    """Raised for loop node errors."""


class LoopNode:
    """Iteration node for Swarm workflows.

    Repeats body agent(s) based on count, items, or condition.
    The Swarm's workflow loop detects ``is_loop`` and delegates
    execution to :meth:`execute`.

    Args:
        name: Node name for the Swarm's flow DSL.
        body: Agent name or list of agent names to execute each iteration.
        count: Fixed number of iterations (mutually exclusive with
            *items* and *condition*).
        items: State key containing a list to iterate over (mutually
            exclusive with *count* and *condition*).
        condition: Expression string evaluated each iteration — loop
            continues while truthy (mutually exclusive with *count*
            and *items*).
        max_iterations: Safety limit to prevent infinite loops.
    """

    def __init__(
        self,
        *,
        name: str,
        body: str | list[str],
        count: int | None = None,
        items: str | None = None,
        condition: str | None = None,
        max_iterations: int = 100,
    ) -> None:
        if not name:
            raise LoopError("LoopNode requires a non-empty name")
        if not body or (isinstance(body, list) and len(body) == 0):
            raise LoopError("LoopNode requires at least one body agent")

        modes = sum(x is not None for x in (count, items, condition))
        if modes == 0:
            raise LoopError("LoopNode requires exactly one of: count, items, or condition")
        if modes > 1:
            raise LoopError("LoopNode requires exactly one of: count, items, or condition")

        if count is not None and count < 0:
            raise LoopError("count must be non-negative")
        if max_iterations < 1:
            raise LoopError("max_iterations must be at least 1")

        self.name = name
        self.body = [body] if isinstance(body, str) else list(body)
        self.count = count
        self.items = items
        self.condition = condition
        self.max_iterations = max_iterations

        # Duck-type marker so Swarm can detect loop nodes
        self.is_loop: bool = True

    def _resolve_iterations(self, state: dict[str, Any]) -> int | None:
        """Return the number of iterations, or None for condition-based loops.

        For count mode, returns the count directly.
        For items mode, returns the length of the list at the state key.
        For condition mode, returns None (unbounded, checked per-iteration).

        Raises:
            LoopError: If the items key is missing or not a list.
        """
        if self.count is not None:
            return self.count
        if self.items is not None:
            arr = state.get(self.items)
            if arr is None:
                raise LoopError(f"Loop '{self.name}': items key '{self.items}' not found in state")
            if not isinstance(arr, (list, tuple)):
                raise LoopError(f"Loop '{self.name}': items key '{self.items}' is not a list")
            return len(arr)
        return None  # condition mode

    def _check_condition(self, state: dict[str, Any]) -> bool:
        """Evaluate the condition expression against current state.

        Raises:
            LoopError: If the expression evaluation fails.
        """
        assert self.condition is not None
        try:
            return bool(evaluate_expression(self.condition, state))
        except ExpressionError as exc:
            raise LoopError(f"Loop '{self.name}' condition evaluation failed: {exc}") from exc

    async def run(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Placeholder run — the Swarm handles actual loop execution.

        This method exists so LoopNode has the same interface as other
        node types, but the Swarm should use :meth:`_resolve_iterations`,
        :meth:`_check_condition`, and the body agents for actual execution.

        Returns:
            ``RunResult`` echoing the input (loop logic is in Swarm).
        """
        return RunResult(output=input)

    def describe(self) -> dict[str, Any]:
        """Return a summary of the loop node's configuration."""
        mode = (
            "count"
            if self.count is not None
            else "items"
            if self.items is not None
            else "condition"
        )
        desc: dict[str, Any] = {
            "type": "loop",
            "name": self.name,
            "mode": mode,
            "body": self.body,
            "max_iterations": self.max_iterations,
        }
        if self.count is not None:
            desc["count"] = self.count
        if self.items is not None:
            desc["items"] = self.items
        if self.condition is not None:
            desc["condition"] = self.condition
        return desc

    def __repr__(self) -> str:
        if self.count is not None:
            mode_str = f"count={self.count}"
        elif self.items is not None:
            mode_str = f"items={self.items!r}"
        else:
            mode_str = f"condition={self.condition!r}"
        return (
            f"LoopNode(name={self.name!r}, {mode_str}, "
            f"body={self.body!r}, max_iterations={self.max_iterations})"
        )
