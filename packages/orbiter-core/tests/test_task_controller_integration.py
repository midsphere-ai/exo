"""Integration tests for the task controller — end-to-end flows combining
TaskManager, TaskScheduler, and TaskEventBus."""

from __future__ import annotations

import asyncio

import pytest

from orbiter.task_controller import (
    TaskEvent,
    TaskEventBus,
    TaskEventType,
    TaskManager,
    TaskScheduler,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event_collector(bus: TaskEventBus) -> list[TaskEvent]:
    """Subscribe to all event types and return the collection list."""
    events: list[TaskEvent] = []

    async def _collect(event: TaskEvent) -> None:
        events.append(event)

    for evt_type in TaskEventType:
        bus.subscribe(evt_type, _collect)
    return events


# ---------------------------------------------------------------------------
# Test: create → schedule → agent runs → task completes
# ---------------------------------------------------------------------------


class TestCreateScheduleComplete:
    """Full lifecycle: create a task, schedule it, verify the executor runs,
    and confirm the task reaches COMPLETED with correct events."""

    @pytest.mark.asyncio()
    async def test_single_task_lifecycle(self) -> None:
        bus = TaskEventBus()
        events = _event_collector(bus)
        mgr = TaskManager(event_bus=bus)
        scheduler = TaskScheduler(mgr, max_concurrent=3)

        task = mgr.create("integration-task")
        work_done: list[str] = []

        async def agent_executor(t):
            work_done.append(t.id)
            await asyncio.sleep(0.01)

        await scheduler.schedule(agent_executor)
        await asyncio.sleep(0)  # Let fire-and-forget events complete

        # Task completed
        assert mgr.get(task.id).status == TaskStatus.COMPLETED  # type: ignore[union-attr]
        assert task.id in work_done

        # Events: created → started → completed
        event_types = [e.event_type for e in events if e.task_id == task.id]
        assert TaskEventType.CREATED in event_types
        assert TaskEventType.STARTED in event_types
        assert TaskEventType.COMPLETED in event_types

    @pytest.mark.asyncio()
    async def test_multiple_tasks_all_complete(self) -> None:
        bus = TaskEventBus()
        events = _event_collector(bus)
        mgr = TaskManager(event_bus=bus)
        scheduler = TaskScheduler(mgr, max_concurrent=5)

        tasks = [mgr.create(f"task-{i}", priority=i) for i in range(5)]
        executed: list[str] = []

        async def agent_executor(t):
            executed.append(t.id)
            await asyncio.sleep(0.005)

        await scheduler.schedule(agent_executor)
        await asyncio.sleep(0)

        # All tasks completed
        for task in tasks:
            assert mgr.get(task.id).status == TaskStatus.COMPLETED  # type: ignore[union-attr]
        assert set(executed) == {t.id for t in tasks}

        # Each task should have created + started + completed events
        for task in tasks:
            task_events = [e.event_type for e in events if e.task_id == task.id]
            assert TaskEventType.CREATED in task_events
            assert TaskEventType.COMPLETED in task_events

    @pytest.mark.asyncio()
    async def test_failing_executor_marks_task_failed(self) -> None:
        bus = TaskEventBus()
        events = _event_collector(bus)
        mgr = TaskManager(event_bus=bus)
        scheduler = TaskScheduler(mgr, max_concurrent=3)

        task = mgr.create("doomed-task")

        async def failing_executor(t):
            msg = "simulated agent failure"
            raise RuntimeError(msg)

        await scheduler.schedule(failing_executor)
        await asyncio.sleep(0)

        assert mgr.get(task.id).status == TaskStatus.FAILED  # type: ignore[union-attr]

        task_events = [e.event_type for e in events if e.task_id == task.id]
        assert TaskEventType.FAILED in task_events


# ---------------------------------------------------------------------------
# Test: parent with 3 children → all complete → parent auto-completes
# ---------------------------------------------------------------------------


class TestParentAutoComplete:
    """Hierarchical tasks: parent auto-completes when all children finish."""

    @pytest.mark.asyncio()
    async def test_parent_auto_completes_after_all_children(self) -> None:
        bus = TaskEventBus()
        events = _event_collector(bus)
        mgr = TaskManager(auto_complete_parent=True, event_bus=bus)
        scheduler = TaskScheduler(mgr, max_concurrent=5)

        parent = mgr.create("parent-task")
        children = [
            mgr.create(f"child-{i}", parent_id=parent.id, priority=i)
            for i in range(3)
        ]

        # Move parent to WORKING so auto-complete can trigger
        mgr.update(parent.id, status=TaskStatus.WORKING)

        executed_children: list[str] = []

        async def child_executor(t):
            executed_children.append(t.id)
            await asyncio.sleep(0.005)

        await scheduler.schedule(child_executor)
        await asyncio.sleep(0)

        # All 3 children completed
        for child in children:
            assert mgr.get(child.id).status == TaskStatus.COMPLETED  # type: ignore[union-attr]
        assert set(executed_children) == {c.id for c in children}

        # Parent auto-completed (via direct transition, no event emitted)
        assert mgr.get(parent.id).status == TaskStatus.COMPLETED  # type: ignore[union-attr]

        # Children should have completed events
        for child in children:
            child_events = [e.event_type for e in events if e.task_id == child.id]
            assert TaskEventType.COMPLETED in child_events

    @pytest.mark.asyncio()
    async def test_parent_stays_working_if_child_incomplete(self) -> None:
        mgr = TaskManager(auto_complete_parent=True)
        scheduler = TaskScheduler(mgr, max_concurrent=5)

        parent = mgr.create("parent")
        mgr.create("child-1", parent_id=parent.id)
        child2 = mgr.create("child-2", parent_id=parent.id)

        mgr.update(parent.id, status=TaskStatus.WORKING)

        # Only schedule child-2 by manually completing child-1's flow
        # but leave child-1 as SUBMITTED
        mgr.update(child2.id, status=TaskStatus.WORKING)
        mgr.update(child2.id, status=TaskStatus.COMPLETED)

        # Parent should NOT auto-complete — child-1 is still SUBMITTED
        assert mgr.get(parent.id).status == TaskStatus.WORKING  # type: ignore[union-attr]

    @pytest.mark.asyncio()
    async def test_parent_does_not_auto_complete_when_disabled(self) -> None:
        mgr = TaskManager(auto_complete_parent=False)
        scheduler = TaskScheduler(mgr, max_concurrent=5)

        parent = mgr.create("parent")
        children = [
            mgr.create(f"child-{i}", parent_id=parent.id) for i in range(3)
        ]

        mgr.update(parent.id, status=TaskStatus.WORKING)

        async def executor(t):
            await asyncio.sleep(0.001)

        await scheduler.schedule(executor)

        for child in children:
            assert mgr.get(child.id).status == TaskStatus.COMPLETED  # type: ignore[union-attr]

        # Parent stays WORKING — auto_complete_parent is disabled
        assert mgr.get(parent.id).status == TaskStatus.WORKING  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Test: concurrent scheduling respects limit
# ---------------------------------------------------------------------------


class TestConcurrentSchedulingLimit:
    """Verify the semaphore-based concurrency control with event tracking."""

    @pytest.mark.asyncio()
    async def test_respects_max_concurrent_with_events(self) -> None:
        bus = TaskEventBus()
        events = _event_collector(bus)
        mgr = TaskManager(event_bus=bus)
        scheduler = TaskScheduler(mgr, max_concurrent=2)

        for i in range(6):
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
            await asyncio.sleep(0.02)
            async with lock:
                current_running -= 1

        await scheduler.schedule(executor)
        await asyncio.sleep(0)

        # Concurrency limit respected
        assert max_running <= 2

        # All tasks completed
        completed = mgr.list(status=TaskStatus.COMPLETED)
        assert len(completed) == 6

        # Events were emitted for all tasks
        completed_events = [
            e for e in events if e.event_type == TaskEventType.COMPLETED
        ]
        assert len(completed_events) == 6

    @pytest.mark.asyncio()
    async def test_max_concurrent_one_serializes(self) -> None:
        mgr = TaskManager()
        scheduler = TaskScheduler(mgr, max_concurrent=1)

        execution_order: list[str] = []
        for i in range(4):
            mgr.create(f"serial-{i}")

        async def executor(task):
            execution_order.append(task.name)
            await asyncio.sleep(0.01)

        await scheduler.schedule(executor)

        assert len(execution_order) == 4
        completed = mgr.list(status=TaskStatus.COMPLETED)
        assert len(completed) == 4


# ---------------------------------------------------------------------------
# Test: pause/resume/cancel lifecycle
# ---------------------------------------------------------------------------


class TestPauseResumeCancelLifecycle:
    """End-to-end pause, resume, and cancel flows with events."""

    @pytest.mark.asyncio()
    async def test_full_pause_resume_complete_lifecycle(self) -> None:
        bus = TaskEventBus()
        events = _event_collector(bus)
        mgr = TaskManager(event_bus=bus)
        scheduler = TaskScheduler(mgr, max_concurrent=3)

        task = mgr.create("pausable-task")
        mgr.update(task.id, status=TaskStatus.WORKING)

        # Pause
        scheduler.pause(task.id)
        assert mgr.get(task.id).status == TaskStatus.PAUSED  # type: ignore[union-attr]

        # Resume — goes back to SUBMITTED
        scheduler.resume(task.id)
        assert mgr.get(task.id).status == TaskStatus.SUBMITTED  # type: ignore[union-attr]

        # Now schedule — task should be picked up and completed
        executed: list[str] = []

        async def executor(t):
            executed.append(t.id)

        await scheduler.schedule(executor)
        await asyncio.sleep(0)

        assert task.id in executed
        assert mgr.get(task.id).status == TaskStatus.COMPLETED  # type: ignore[union-attr]

        # Events: created → started → paused → started(via resume→submitted→working) → completed
        task_events = [e.event_type for e in events if e.task_id == task.id]
        assert TaskEventType.CREATED in task_events
        assert TaskEventType.PAUSED in task_events
        assert TaskEventType.COMPLETED in task_events

    @pytest.mark.asyncio()
    async def test_cancel_before_schedule_prevents_execution(self) -> None:
        bus = TaskEventBus()
        events = _event_collector(bus)
        mgr = TaskManager(event_bus=bus)
        scheduler = TaskScheduler(mgr, max_concurrent=3)

        task = mgr.create("cancel-me")
        scheduler.cancel(task.id)

        executed: list[str] = []

        async def executor(t):
            executed.append(t.id)

        await scheduler.schedule(executor)
        await asyncio.sleep(0)

        assert executed == []
        assert mgr.get(task.id).status == TaskStatus.CANCELED  # type: ignore[union-attr]

        task_events = [e.event_type for e in events if e.task_id == task.id]
        assert TaskEventType.CANCELED in task_events

    @pytest.mark.asyncio()
    async def test_cancel_cascades_to_children_with_events(self) -> None:
        bus = TaskEventBus()
        events = _event_collector(bus)
        mgr = TaskManager(event_bus=bus)
        scheduler = TaskScheduler(mgr, max_concurrent=3)

        parent = mgr.create("parent")
        children = [
            mgr.create(f"child-{i}", parent_id=parent.id) for i in range(3)
        ]

        mgr.update(parent.id, status=TaskStatus.WORKING)
        scheduler.cancel(parent.id)

        await asyncio.sleep(0)

        assert mgr.get(parent.id).status == TaskStatus.CANCELED  # type: ignore[union-attr]
        for child in children:
            assert mgr.get(child.id).status == TaskStatus.CANCELED  # type: ignore[union-attr]

        # Parent has canceled event (children are cascade-canceled via
        # direct transition, so no individual events for them)
        canceled_task_ids = {
            e.task_id for e in events if e.event_type == TaskEventType.CANCELED
        }
        assert parent.id in canceled_task_ids

    @pytest.mark.asyncio()
    async def test_cancel_during_schedule_skips_waiting_task(self) -> None:
        """A task canceled while waiting for the semaphore is skipped."""
        mgr = TaskManager()
        scheduler = TaskScheduler(mgr, max_concurrent=1)

        t1 = mgr.create("blocker", priority=10)
        t2 = mgr.create("cancellable", priority=0)

        t2_executed = False

        async def executor(task):
            nonlocal t2_executed
            if task.id == t1.id:
                # While t1 is running, cancel t2 before it gets to run
                scheduler.cancel(t2.id)
                await asyncio.sleep(0.02)
            else:
                t2_executed = True

        await scheduler.schedule(executor)

        assert mgr.get(t1.id).status == TaskStatus.COMPLETED  # type: ignore[union-attr]
        assert mgr.get(t2.id).status == TaskStatus.CANCELED  # type: ignore[union-attr]
        assert not t2_executed

    @pytest.mark.asyncio()
    async def test_resume_and_reschedule(self) -> None:
        """Resumed task is picked up by the next schedule() call."""
        mgr = TaskManager()
        scheduler = TaskScheduler(mgr, max_concurrent=3)

        task = mgr.create("retry-me")
        mgr.update(task.id, status=TaskStatus.WORKING)
        mgr.update(task.id, status=TaskStatus.PAUSED)

        # First schedule — nothing eligible (task is paused)
        first_run: list[str] = []

        async def noop(t):
            first_run.append(t.id)

        await scheduler.schedule(noop)
        assert first_run == []

        # Resume → SUBMITTED
        scheduler.resume(task.id)
        assert mgr.get(task.id).status == TaskStatus.SUBMITTED  # type: ignore[union-attr]

        # Second schedule — now picks it up
        second_run: list[str] = []

        async def executor(t):
            second_run.append(t.id)

        await scheduler.schedule(executor)
        assert task.id in second_run
        assert mgr.get(task.id).status == TaskStatus.COMPLETED  # type: ignore[union-attr]
