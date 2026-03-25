"""TaskScheduler — concurrent task execution with semaphore-based throttling."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from orbiter._internal.task_controller.manager import TaskManager
from orbiter._internal.task_controller.types import Task, TaskStatus


class TaskScheduler:
    """Run eligible tasks concurrently up to a configurable limit.

    Args:
        task_manager: The task manager to read/update tasks from.
        max_concurrent: Maximum number of tasks executing simultaneously.
    """

    def __init__(self, task_manager: TaskManager, *, max_concurrent: int = 3) -> None:
        self._manager = task_manager
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running: dict[str, asyncio.Task[None]] = {}

    @property
    def task_manager(self) -> TaskManager:
        """The underlying task manager."""
        return self._manager

    async def schedule(
        self,
        executor: Callable[[Task], Coroutine[Any, Any, Any]],
    ) -> None:
        """Pick up all SUBMITTED tasks and execute them concurrently.

        Each task is transitioned to WORKING before the executor is called.
        On success the task moves to COMPLETED; on failure it moves to FAILED.

        The semaphore ensures at most ``max_concurrent`` tasks run at once.

        Args:
            executor: An async callable that performs the actual work for a task.
        """
        eligible = self._manager.list(status=TaskStatus.SUBMITTED)
        if not eligible:
            return

        async def _run(task: Task) -> None:
            async with self._semaphore:
                # Re-check status — may have been paused/canceled while waiting
                current = self._manager.get(task.id)
                if current is None or current.status != TaskStatus.SUBMITTED:
                    return

                self._manager.update(task.id, status=TaskStatus.WORKING)
                try:
                    await executor(task)
                    # Re-check: task may have been canceled during execution
                    current = self._manager.get(task.id)
                    if current is not None and current.status == TaskStatus.WORKING:
                        self._manager.update(task.id, status=TaskStatus.COMPLETED)
                except Exception:
                    current = self._manager.get(task.id)
                    if current is not None and current.status == TaskStatus.WORKING:
                        self._manager.update(task.id, status=TaskStatus.FAILED)
                finally:
                    self._running.pop(task.id, None)

        tasks: list[asyncio.Task[None]] = []
        for task in eligible:
            t = asyncio.create_task(_run(task))
            self._running[task.id] = t
            tasks.append(t)

        await asyncio.gather(*tasks, return_exceptions=True)

    def pause(self, task_id: str) -> Task:
        """Pause a running or submitted task.

        Args:
            task_id: The task to pause.

        Returns:
            The updated task.

        Raises:
            TaskNotFoundError: If the task does not exist.
            InvalidTransitionError: If the task cannot be paused from its current state.
        """
        return self._manager.update(task_id, status=TaskStatus.PAUSED)

    def resume(self, task_id: str) -> Task:
        """Resume a paused task by transitioning it back to SUBMITTED.

        The task becomes eligible for the next ``schedule()`` call.

        Args:
            task_id: The task to resume.

        Returns:
            The updated task.

        Raises:
            TaskNotFoundError: If the task does not exist.
            InvalidTransitionError: If the task cannot be resumed from its current state.
        """
        return self._manager.update(task_id, status=TaskStatus.SUBMITTED)

    def cancel(self, task_id: str) -> Task:
        """Cancel a task.

        If the task has a running asyncio task, it will be cancelled.

        Args:
            task_id: The task to cancel.

        Returns:
            The updated task.

        Raises:
            TaskNotFoundError: If the task does not exist.
            InvalidTransitionError: If the task cannot be canceled from its current state.
        """
        result = self._manager.update(task_id, status=TaskStatus.CANCELED)
        # Cancel the asyncio task if it's currently running
        running = self._running.pop(task_id, None)
        if running is not None and not running.done():
            running.cancel()
        return result
