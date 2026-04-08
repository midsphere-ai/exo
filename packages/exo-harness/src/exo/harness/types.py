"""Leaf types for harness orchestration.

:class:`HarnessEvent` is a custom streaming event emitted alongside
standard ``StreamEvent`` instances.  :class:`SessionState` wraps
mutable orchestration state with dirty-tracking for checkpoint
optimization.  :class:`HarnessCheckpoint` is an immutable snapshot
for persistence and resumption.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
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


class SubAgentStatus(StrEnum):
    """Terminal status of a sub-agent execution."""

    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class SubAgentTask:
    """Specification for one parallel sub-agent invocation.

    Args:
        agent: An ``Agent``, ``Swarm``, or ``Harness`` instance.
        input: User query string or content blocks.
        name: Label override for event tagging; defaults to ``agent.name``.
        messages: Prior conversation history.  ``None`` means fresh.
        provider: LLM provider override for this agent.
        timeout: Per-agent timeout in seconds.  ``None`` means no timeout.
    """

    agent: Any
    input: Any  # MessageContent
    name: str | None = None
    messages: Sequence[Any] | None = None
    provider: Any = None
    timeout: float | None = None


@dataclass(frozen=True)
class SubAgentResult:
    """Result from one parallel sub-agent.

    Always populated, even on failure.  On error, ``output`` is empty
    and ``error`` holds the original exception.

    Args:
        agent_name: Name of the agent that produced this result.
        status: Terminal status (success, failed, cancelled, timed_out).
        output: Agent's text output (empty on failure).
        result: Full ``RunResult`` on success, ``None`` on failure.
        error: Original exception on failure, ``None`` on success.
        elapsed_seconds: Wall-clock seconds the agent ran.
        log_path: Path to the sub-agent's event log file in ``/tmp/``.
    """

    agent_name: str
    status: SubAgentStatus
    output: str = ""
    result: Any = None  # RunResult | None
    error: BaseException | None = None
    elapsed_seconds: float = 0.0
    log_path: str | None = None


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
    pending_agents: list[str] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
