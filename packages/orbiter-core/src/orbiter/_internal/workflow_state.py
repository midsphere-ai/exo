"""Workflow state container for passing context through workflow execution.

:class:`WorkflowState` accumulates agent outputs and arbitrary key-value
data as a workflow executes.  It is passed through
:class:`~orbiter.swarm.Swarm` workflow mode so that :class:`BranchNode`
and :class:`LoopNode` can evaluate conditions against the full workflow
context (not just the current ``input`` string).

Usage::

    state = WorkflowState()
    state.set("score", 95)
    assert state.get("score") == 95
    assert state.to_dict() == {"score": 95}
"""

from __future__ import annotations

from typing import Any


class WorkflowState:
    """Shared state container for workflow execution.

    Stores key-value pairs accumulated during a workflow run.
    Each agent's output is stored via ``set(agent_name, output)``
    and can be read by downstream nodes (including :class:`BranchNode`
    and :class:`LoopNode`) for condition evaluation.

    Args:
        initial: Optional dict of initial state values.
    """

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(initial) if initial else {}

    @property
    def data(self) -> dict[str, Any]:
        """Read-only view of the internal state data."""
        return self._data

    def set(self, key: str, value: Any) -> None:
        """Set a key-value pair in the workflow state.

        Args:
            key: State key (typically an agent name or a domain key).
            value: Value to store.
        """
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the workflow state.

        Args:
            key: State key to look up.
            default: Value to return if the key is not found.

        Returns:
            The stored value, or *default* if absent.
        """
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the state as a plain dict.

        Returns:
            Dict snapshot of the current state.
        """
        return dict(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"WorkflowState({self._data!r})"
