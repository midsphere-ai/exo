"""Task model, status enum, and transition rules."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from orbiter.types import OrbiterError


class TaskError(OrbiterError):
    """Base exception for task-related errors."""


class InvalidTransitionError(TaskError):
    """Raised when a status transition violates the allowed rules."""


class TaskStatus(StrEnum):
    """Lifecycle states for a managed task."""

    SUBMITTED = "submitted"
    WORKING = "working"
    PAUSED = "paused"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    WAITING = "waiting"


# Allowed transitions: source -> {valid targets}
_VALID_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.SUBMITTED: {TaskStatus.WORKING, TaskStatus.CANCELED},
    TaskStatus.WORKING: {
        TaskStatus.PAUSED,
        TaskStatus.INPUT_REQUIRED,
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELED,
        TaskStatus.WAITING,
    },
    TaskStatus.PAUSED: {TaskStatus.WORKING, TaskStatus.CANCELED, TaskStatus.SUBMITTED},
    TaskStatus.INPUT_REQUIRED: {TaskStatus.WORKING, TaskStatus.CANCELED},
    TaskStatus.WAITING: {TaskStatus.WORKING, TaskStatus.CANCELED},
    TaskStatus.COMPLETED: set(),
    TaskStatus.CANCELED: set(),
    TaskStatus.FAILED: {TaskStatus.SUBMITTED},
}

TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.CANCELED}
)


def validate_transition(current: TaskStatus, target: TaskStatus) -> None:
    """Raise ``InvalidTransitionError`` if *current* -> *target* is not allowed."""
    allowed = _VALID_TRANSITIONS.get(current, set())
    if target not in allowed:
        msg = f"Cannot transition from {current!r} to {target!r}"
        raise InvalidTransitionError(msg)


def _now() -> datetime:
    return datetime.now()


def _uuid() -> str:
    return str(uuid.uuid4())


class Task(BaseModel):
    """A user-facing work unit with lifecycle management.

    Args:
        id: Unique identifier (UUID string).
        name: Human-readable task name.
        description: Optional longer description.
        status: Current lifecycle status.
        priority: Higher values = more important.
        parent_id: ID of the parent task (for hierarchical decomposition).
        metadata: Arbitrary key-value metadata.
        created_at: When the task was created.
        updated_at: When the task was last modified.
    """

    id: str = Field(default_factory=_uuid)
    name: str
    description: str = ""
    status: TaskStatus = TaskStatus.SUBMITTED
    priority: int = 0
    parent_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    def transition(self, target: TaskStatus) -> None:
        """Transition to *target* status, enforcing valid transition rules.

        Raises:
            InvalidTransitionError: If the transition is not allowed.
        """
        validate_transition(self.status, target)
        self.status = target
        self.updated_at = _now()

    @property
    def is_terminal(self) -> bool:
        """Whether the task is in a terminal state."""
        return self.status in TERMINAL_STATUSES
