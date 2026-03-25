"""Tests for TaskScheduler — concurrent execution, pause/resume/cancel."""

from __future__ import annotations

import asyncio

import pytest

from exo.task_controller import (
    InvalidTransitionError,
    TaskManager,
    TaskNotFoundError,
    TaskScheduler,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mgr() -> TaskManager:
    return TaskManager()


@pytest.fixture()
def scheduler(mgr: TaskManager) -> TaskScheduler:
    return TaskScheduler(mgr, max_concurrent=3)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self, mgr: TaskManager) -> None:
        sched = TaskScheduler(mgr)
        assert sched.task_manager is mgr

    def test_custom_concurrency(self, mgr: TaskManager) -> None:
        sched = TaskScheduler(mgr, max_concurrent=5)
        assert sched._semaphore._value == 5


# ---------------------------------------------------------------------------
# Schedule — basic execution
# ---------------------------------------------------------------------------


class TestScheduleBasic:
    @pytest.mark.asyncio()
    async def test_schedule_runs_submitted_tasks(
        self, scheduler: TaskScheduler, mgr: TaskManager
    ) -> None:
        t1 = mgr.create("task-1")
        t2 = mgr.create("task-2")
        executed: list[str] = []

        async def executor(task):
            executed.append(task.id)

        await scheduler.schedule(executor)

        assert set(executed) == {t1.id, t2.id}
        assert mgr.get(t1.id).status == TaskStatus.COMPLETED  # type: ignore[union-attr]
        assert mgr.get(t2.id).status == TaskStatus.COMPLETED  # type: ignore[union-attr]

    @pytest.mark.asyncio()
    async def test_schedule_skips_non_submitted(
        self, scheduler: TaskScheduler, mgr: TaskManager
    ) -> None:
        t1 = mgr.create("task-1")
        mgr.update(t1.id, status=TaskStatus.WORKING)
        mgr.update(t1.id, status=TaskStatus.COMPLETED)

        executed: list[str] = []

        async def executor(task):
            executed.append(task.id)

        await scheduler.schedule(executor)

        assert executed == []

    @pytest.mark.asyncio()
    async def test_schedule_no_tasks(self, scheduler: TaskScheduler) -> None:
        """schedule() with no eligible tasks should be a no-op."""
        call_count = 0

        async def executor(task):
            nonlocal call_count
            call_count += 1

        await scheduler.schedule(executor)
        assert call_count == 0

    @pytest.mark.asyncio()
    async def test_schedule_marks_failed_on_exception(
        self, scheduler: TaskScheduler, mgr: TaskManager
    ) -> None:
        t1 = mgr.create("failing-task")

        async def executor(task):
            msg = "boom"
            raise RuntimeError(msg)

        await scheduler.schedule(executor)

        assert mgr.get(t1.id).status == TaskStatus.FAILED  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Schedule — concurrency limit
# ---------------------------------------------------------------------------


class TestConcurrencyLimit:
    @pytest.mark.asyncio()
    async def test_respects_max_concurrent(self, mgr: TaskManager) -> None:
        sched = TaskScheduler(mgr, max_concurrent=2)
        for i in range(5):
            mgr.create(f"task-{i}")

        max_running = 0
        current_running = 0
        lock = asyncio.Lock()

        async def executor(task):
            nonlocal max_running, current_running
            async with lock:
                current_running += 1
                if current_running > max_running:
                    max_running = current_running
            await asyncio.sleep(0.01)
            async with lock:
                current_running -= 1

        await sched.schedule(executor)

        assert max_running <= 2
        # All 5 tasks should still complete
        completed = mgr.list(status=TaskStatus.COMPLETED)
        assert len(completed) == 5


# ---------------------------------------------------------------------------
# Pause / Resume / Cancel
# ---------------------------------------------------------------------------


class TestPause:
    def test_pause_working_task(self, scheduler: TaskScheduler, mgr: TaskManager) -> None:
        t = mgr.create("task")
        mgr.update(t.id, status=TaskStatus.WORKING)

        result = scheduler.pause(t.id)
        assert result.status == TaskStatus.PAUSED

    def test_pause_invalid_status_raises(self, scheduler: TaskScheduler, mgr: TaskManager) -> None:
        t = mgr.create("task")
        mgr.update(t.id, status=TaskStatus.WORKING)
        mgr.update(t.id, status=TaskStatus.COMPLETED)

        with pytest.raises(InvalidTransitionError):
            scheduler.pause(t.id)

    def test_pause_nonexistent_raises(self, scheduler: TaskScheduler) -> None:
        with pytest.raises(TaskNotFoundError):
            scheduler.pause("no-such-id")


class TestResume:
    def test_resume_paused_task(self, scheduler: TaskScheduler, mgr: TaskManager) -> None:
        t = mgr.create("task")
        mgr.update(t.id, status=TaskStatus.WORKING)
        mgr.update(t.id, status=TaskStatus.PAUSED)

        result = scheduler.resume(t.id)
        assert result.status == TaskStatus.SUBMITTED

    def test_resume_invalid_status_raises(self, scheduler: TaskScheduler, mgr: TaskManager) -> None:
        t = mgr.create("task")
        mgr.update(t.id, status=TaskStatus.WORKING)
        mgr.update(t.id, status=TaskStatus.COMPLETED)

        with pytest.raises(InvalidTransitionError):
            scheduler.resume(t.id)

    def test_resume_nonexistent_raises(self, scheduler: TaskScheduler) -> None:
        with pytest.raises(TaskNotFoundError):
            scheduler.resume("no-such-id")

    @pytest.mark.asyncio()
    async def test_resumed_task_picked_up_by_schedule(
        self, scheduler: TaskScheduler, mgr: TaskManager
    ) -> None:
        t = mgr.create("task")
        mgr.update(t.id, status=TaskStatus.WORKING)
        mgr.update(t.id, status=TaskStatus.PAUSED)
        scheduler.resume(t.id)

        executed: list[str] = []

        async def executor(task):
            executed.append(task.id)

        await scheduler.schedule(executor)
        assert t.id in executed


class TestCancel:
    def test_cancel_submitted_task(self, scheduler: TaskScheduler, mgr: TaskManager) -> None:
        t = mgr.create("task")

        result = scheduler.cancel(t.id)
        assert result.status == TaskStatus.CANCELED

    def test_cancel_working_task(self, scheduler: TaskScheduler, mgr: TaskManager) -> None:
        t = mgr.create("task")
        mgr.update(t.id, status=TaskStatus.WORKING)

        result = scheduler.cancel(t.id)
        assert result.status == TaskStatus.CANCELED

    def test_cancel_already_completed_raises(
        self, scheduler: TaskScheduler, mgr: TaskManager
    ) -> None:
        t = mgr.create("task")
        mgr.update(t.id, status=TaskStatus.WORKING)
        mgr.update(t.id, status=TaskStatus.COMPLETED)

        with pytest.raises(InvalidTransitionError):
            scheduler.cancel(t.id)

    def test_cancel_nonexistent_raises(self, scheduler: TaskScheduler) -> None:
        with pytest.raises(TaskNotFoundError):
            scheduler.cancel("no-such-id")

    @pytest.mark.asyncio()
    async def test_cancel_during_schedule_skips_execution(
        self, scheduler: TaskScheduler, mgr: TaskManager
    ) -> None:
        t = mgr.create("task")
        executed: list[str] = []

        # Cancel the task before schedule runs
        scheduler.cancel(t.id)

        async def executor(task):
            executed.append(task.id)

        await scheduler.schedule(executor)
        assert executed == []


# ---------------------------------------------------------------------------
# Integration — pause/resume/cancel during scheduling
# ---------------------------------------------------------------------------


class TestIntegration:
    @pytest.mark.asyncio()
    async def test_pause_resume_cycle(self, scheduler: TaskScheduler, mgr: TaskManager) -> None:
        t = mgr.create("task")
        mgr.update(t.id, status=TaskStatus.WORKING)
        scheduler.pause(t.id)
        assert mgr.get(t.id).status == TaskStatus.PAUSED  # type: ignore[union-attr]

        scheduler.resume(t.id)
        assert mgr.get(t.id).status == TaskStatus.SUBMITTED  # type: ignore[union-attr]

        executed: list[str] = []

        async def executor(task):
            executed.append(task.id)

        await scheduler.schedule(executor)
        assert t.id in executed
        assert mgr.get(t.id).status == TaskStatus.COMPLETED  # type: ignore[union-attr]
