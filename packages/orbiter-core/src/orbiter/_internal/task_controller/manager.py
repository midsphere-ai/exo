"""TaskManager — CRUD operations with status-transition enforcement."""

from __future__ import annotations

from typing import Any

from orbiter._internal.task_controller.types import (
    Task,
    TaskError,
    TaskStatus,
    _now,
)


class TaskNotFoundError(TaskError):
    """Raised when a task ID does not exist in the store."""


class TaskManager:
    """In-memory task store with CRUD, status enforcement, and filtering.

    Args:
        auto_complete_parent: When ``True``, a parent task auto-transitions
            to COMPLETED when all of its children reach COMPLETED.
    """

    def __init__(self, *, auto_complete_parent: bool = False) -> None:
        self._tasks: dict[str, Task] = {}
        self._auto_complete_parent = auto_complete_parent

    # -- helpers --------------------------------------------------------------

    def _require(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task is None:
            msg = f"Task {task_id!r} not found"
            raise TaskNotFoundError(msg)
        return task

    # -- CRUD -----------------------------------------------------------------

    def create(
        self,
        name: str,
        *,
        description: str = "",
        priority: int = 0,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """Create a new task and add it to the store.

        Args:
            name: Human-readable task name.
            description: Optional longer description.
            priority: Higher values = more important.
            parent_id: ID of the parent task.
            metadata: Arbitrary key-value metadata.

        Returns:
            The newly created Task.

        Raises:
            TaskNotFoundError: If *parent_id* is given but does not exist.
        """
        if parent_id is not None:
            self._require(parent_id)

        task = Task(
            name=name,
            description=description,
            priority=priority,
            parent_id=parent_id,
            metadata=metadata or {},
        )
        self._tasks[task.id] = task
        return task

    def get(self, task_id: str) -> Task | None:
        """Return a task by ID, or ``None`` if not found."""
        return self._tasks.get(task_id)

    def update(
        self,
        task_id: str,
        *,
        status: TaskStatus | None = None,
        name: str | None = None,
        description: str | None = None,
        priority: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """Update a task's fields.

        Status transitions are validated. After a status update, cascading
        side-effects are applied (cancel cascade, auto-complete parent).

        Args:
            task_id: The task to update.
            status: New status (validated against transition rules).
            name: New name.
            description: New description.
            priority: New priority.
            metadata: New metadata (replaces existing).

        Returns:
            The updated Task.

        Raises:
            TaskNotFoundError: If the task does not exist.
            InvalidTransitionError: If the status transition is invalid.
        """
        task = self._require(task_id)

        if status is not None:
            task.transition(status)

        if name is not None:
            task.name = name
            task.updated_at = _now()
        if description is not None:
            task.description = description
            task.updated_at = _now()
        if priority is not None:
            task.priority = priority
            task.updated_at = _now()
        if metadata is not None:
            task.metadata = metadata
            task.updated_at = _now()

        # Cascading side-effects
        if status == TaskStatus.CANCELED:
            self._cascade_cancel(task_id)
        if status == TaskStatus.COMPLETED and self._auto_complete_parent and task.parent_id:
            self._maybe_complete_parent(task.parent_id)

        return task

    def delete(self, task_id: str) -> bool:
        """Remove a task from the store.

        Returns:
            ``True`` if the task existed and was removed, ``False`` otherwise.
        """
        return self._tasks.pop(task_id, None) is not None

    def list(self, *, status: TaskStatus | None = None) -> list[Task]:
        """Return tasks sorted by priority (descending) then created_at (ascending).

        Args:
            status: If given, only return tasks with this status.
        """
        tasks = [t for t in self._tasks.values() if t.status == status] if status is not None else list(self._tasks.values())
        tasks.sort(key=lambda t: (-t.priority, t.created_at))
        return tasks

    # -- cascading effects ----------------------------------------------------

    def _cascade_cancel(self, parent_id: str) -> None:
        """Cancel all descendants of *parent_id*."""
        for task in list(self._tasks.values()):
            if task.parent_id == parent_id and not task.is_terminal:
                task.transition(TaskStatus.CANCELED)
                self._cascade_cancel(task.id)

    def _maybe_complete_parent(self, parent_id: str) -> None:
        """Auto-complete parent if all children are COMPLETED."""
        parent = self._tasks.get(parent_id)
        if parent is None or parent.is_terminal:
            return
        children = [t for t in self._tasks.values() if t.parent_id == parent_id]
        if children and all(c.status == TaskStatus.COMPLETED for c in children):
            parent.transition(TaskStatus.COMPLETED)
