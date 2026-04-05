"""Leaf types for harness orchestration.

:class:`HarnessEvent` is a custom streaming event emitted alongside
standard ``StreamEvent`` instances.  :class:`SessionState` wraps
mutable orchestration state with dirty-tracking for checkpoint
optimization.  :class:`HarnessCheckpoint` is an immutable snapshot
for persistence and resumption.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field


class HarnessEvent(BaseModel):
    """Custom event emitted by harness orchestration logic.

    Follows the same pattern as ``StreamEvent`` types — frozen
    ``BaseModel`` with a ``type`` discriminator and ``agent_name``
    field — but is not part of the ``StreamEvent`` union to avoid
    coupling ``exo-core`` to ``exo-harness``.

    Args:
        type: Discriminator literal, always ``"harness"``.
        kind: Developer-defined sub-kind for filtering/routing.
        agent_name: Name of the harness or agent that emitted this event.
        data: Arbitrary payload.
    """

    model_config = {"frozen": True}

    type: Literal["harness"] = "harness"
    kind: str
    agent_name: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


@dataclass
class SessionState:
    """Mutable session state persisted across harness runs.

    Wraps a plain ``dict`` with dirty-tracking so checkpoint logic
    can skip persistence when nothing has changed.

    Args:
        data: The underlying state dictionary.
    """

    data: dict[str, Any] = field(default_factory=dict)
    _dirty: bool = field(default=False, repr=False)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value
        self._dirty = True

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if absent."""
        return self.data.get(key, default)

    def mark_clean(self) -> None:
        """Reset the dirty flag after a successful checkpoint."""
        self._dirty = False

    @property
    def dirty(self) -> bool:
        """Whether the state has been modified since the last checkpoint."""
        return self._dirty


@dataclass(frozen=True)
class HarnessCheckpoint:
    """Immutable snapshot of harness execution state.

    Captures session state, progress, and message history for
    persistence and resumption after process restart.

    Args:
        harness_name: Name of the harness that created this checkpoint.
        session_state: Serialized ``SessionState.data`` dict.
        completed_agents: Agent names that have finished executing.
        pending_agent: Agent name about to execute (if any).
        messages: Serialized message history.
        timestamp: Unix timestamp when the checkpoint was created.
        metadata: Arbitrary metadata for downstream consumers.
    """

    harness_name: str
    session_state: dict[str, Any]
    completed_agents: list[str]
    pending_agent: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
