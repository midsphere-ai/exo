"""Conditional branching node for workflow routing.

A :class:`BranchNode` evaluates a condition and routes execution to one
of two named agents (``if_true`` or ``if_false``).  Conditions can be
either a string expression (evaluated via :func:`evaluate_expression`)
or a callable ``(state: dict) -> bool``.

The BranchNode does not execute the target agent itself — it returns the
target agent's name so the Swarm workflow loop can route accordingly.

Usage::

    branch = BranchNode(
        name="check_score",
        condition="score > 80",
        if_true="approve",
        if_false="review",
    )
    swarm = Swarm(
        agents=[branch, approve_agent, review_agent],
        flow="check_score >> approve",  # flow defines topology
    )
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from orbiter._internal.expression import ExpressionError, evaluate_expression
from orbiter.types import Message, OrbiterError, RunResult


class BranchError(OrbiterError):
    """Raised for branch node errors."""


class BranchNode:
    """Conditional routing node for Swarm workflows.

    Evaluates a condition against the current workflow state and routes
    to one of two named agents.  The Swarm's workflow loop detects
    ``is_branch`` and uses the returned target to decide execution path.

    Args:
        name: Node name for the Swarm's flow DSL.
        condition: Either a string expression (evaluated safely via
            :func:`evaluate_expression`) or a callable ``(state: dict) -> Any``
            (truthy/falsy evaluation).
        if_true: Name of the agent to route to when condition is truthy.
        if_false: Name of the agent to route to when condition is falsy.
    """

    def __init__(
        self,
        *,
        name: str,
        condition: str | Callable[[dict[str, Any]], Any],
        if_true: str,
        if_false: str,
    ) -> None:
        if not name:
            raise BranchError("BranchNode requires a non-empty name")
        if not if_true:
            raise BranchError("BranchNode requires a non-empty if_true agent name")
        if not if_false:
            raise BranchError("BranchNode requires a non-empty if_false agent name")

        self.name = name
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false

        # Duck-type marker so Swarm can detect branch nodes
        self.is_branch: bool = True

    def evaluate(self, state: dict[str, Any]) -> str:
        """Evaluate the condition and return the target agent name.

        Args:
            state: Current workflow state dict (contains ``input``,
                ``output``, and any other accumulated state).

        Returns:
            The agent name to route to (``if_true`` or ``if_false``).

        Raises:
            BranchError: If the condition evaluation fails.
        """
        try:
            if callable(self.condition) and not isinstance(self.condition, str):
                result = self.condition(state)
            else:
                result = evaluate_expression(self.condition, state)
        except ExpressionError as exc:
            raise BranchError(
                f"Branch '{self.name}' condition evaluation failed: {exc}"
            ) from exc
        except Exception as exc:
            raise BranchError(
                f"Branch '{self.name}' condition raised an error: {exc}"
            ) from exc

        return self.if_true if result else self.if_false

    async def run(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Evaluate condition and return a RunResult with the target agent name.

        The Swarm workflow loop uses the output to determine routing.
        This method exists so BranchNode has the same interface as other
        node types, but the Swarm should use :meth:`evaluate` directly
        for routing decisions.

        Args:
            input: Current workflow input text.
            messages: Not used by BranchNode.
            provider: Not used by BranchNode.
            max_retries: Not used by BranchNode.

        Returns:
            ``RunResult`` with the target agent name as output.
        """
        state = {"input": input}
        target = self.evaluate(state)
        return RunResult(output=target)

    def describe(self) -> dict[str, Any]:
        """Return a summary of the branch node's configuration."""
        return {
            "type": "branch",
            "name": self.name,
            "condition": self.condition if isinstance(self.condition, str) else repr(self.condition),
            "if_true": self.if_true,
            "if_false": self.if_false,
        }

    def __repr__(self) -> str:
        cond = self.condition if isinstance(self.condition, str) else "<callable>"
        return f"BranchNode(name={self.name!r}, condition={cond!r}, if_true={self.if_true!r}, if_false={self.if_false!r})"
