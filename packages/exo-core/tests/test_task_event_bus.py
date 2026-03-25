"""Tests for TaskEventBus — pub/sub for task lifecycle events."""

from __future__ import annotations

import asyncio

import pytest

from exo.task_controller import (
    TaskEvent,
    TaskEventBus,
    TaskEventType,
    TaskManager,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# TaskEvent tests
# ---------------------------------------------------------------------------


class TestTaskEvent:
    def test_defaults(self) -> None:
        evt = TaskEvent(event_type=TaskEventType.CREATED, task_id="t-1")
        assert evt.event_type == TaskEventType.CREATED
        assert evt.task_id == "t-1"
        assert evt.data == {}
        assert evt.timestamp is not None

    def test_with_data(self) -> None:
        evt = TaskEvent(
            event_type=TaskEventType.COMPLETED,
            task_id="t-2",
            data={"result": "ok"},
        )
        assert evt.data == {"result": "ok"}


# ---------------------------------------------------------------------------
# TaskEventType tests
# ---------------------------------------------------------------------------


class TestTaskEventType:
    def test_all_event_types(self) -> None:
        expected = {
            "task.created",
            "task.started",
            "task.completed",
            "task.failed",
            "task.paused",
            "task.canceled",
        }
        assert {e.value for e in TaskEventType} == expected


# ---------------------------------------------------------------------------
# TaskEventBus tests
# ---------------------------------------------------------------------------


class TestTaskEventBus:
    @pytest.mark.asyncio()
    async def test_subscribe_and_emit(self) -> None:
        bus = TaskEventBus()
        received: list[TaskEvent] = []

        async def handler(event: TaskEvent) -> None:
            received.append(event)

        bus.subscribe(TaskEventType.CREATED, handler)

        evt = TaskEvent(event_type=TaskEventType.CREATED, task_id="t-1")
        await bus.emit(evt)

        assert len(received) == 1
        assert received[0].task_id == "t-1"

    @pytest.mark.asyncio()
    async def test_multiple_handlers(self) -> None:
        bus = TaskEventBus()
        log1: list[str] = []
        log2: list[str] = []

        async def handler1(event: TaskEvent) -> None:
            log1.append(event.task_id)

        async def handler2(event: TaskEvent) -> None:
            log2.append(event.task_id)

        bus.subscribe(TaskEventType.COMPLETED, handler1)
        bus.subscribe(TaskEventType.COMPLETED, handler2)

        evt = TaskEvent(event_type=TaskEventType.COMPLETED, task_id="t-5")
        await bus.emit(evt)

        assert log1 == ["t-5"]
        assert log2 == ["t-5"]

    @pytest.mark.asyncio()
    async def test_unsubscribe(self) -> None:
        bus = TaskEventBus()
        received: list[TaskEvent] = []

        async def handler(event: TaskEvent) -> None:
            received.append(event)

        bus.subscribe(TaskEventType.FAILED, handler)
        bus.unsubscribe(TaskEventType.FAILED, handler)

        evt = TaskEvent(event_type=TaskEventType.FAILED, task_id="t-1")
        await bus.emit(evt)

        assert len(received) == 0

    @pytest.mark.asyncio()
    async def test_unsubscribe_unknown_handler(self) -> None:
        """Unsubscribing a handler that was never subscribed is a no-op."""
        bus = TaskEventBus()

        async def handler(event: TaskEvent) -> None:
            pass  # pragma: no cover

        # Should not raise
        bus.unsubscribe(TaskEventType.CREATED, handler)

    @pytest.mark.asyncio()
    async def test_emit_no_subscribers(self) -> None:
        """Emitting an event with no subscribers is a no-op."""
        bus = TaskEventBus()
        evt = TaskEvent(event_type=TaskEventType.STARTED, task_id="t-1")
        # Should not raise
        await bus.emit(evt)

    @pytest.mark.asyncio()
    async def test_different_event_types_isolated(self) -> None:
        bus = TaskEventBus()
        created_log: list[str] = []
        completed_log: list[str] = []

        async def on_created(event: TaskEvent) -> None:
            created_log.append(event.task_id)

        async def on_completed(event: TaskEvent) -> None:
            completed_log.append(event.task_id)

        bus.subscribe(TaskEventType.CREATED, on_created)
        bus.subscribe(TaskEventType.COMPLETED, on_completed)

        await bus.emit(TaskEvent(event_type=TaskEventType.CREATED, task_id="t-1"))
        await bus.emit(TaskEvent(event_type=TaskEventType.COMPLETED, task_id="t-2"))

        assert created_log == ["t-1"]
        assert completed_log == ["t-2"]

    @pytest.mark.asyncio()
    async def test_handler_execution_order(self) -> None:
        """Handlers fire in subscription order."""
        bus = TaskEventBus()
        order: list[int] = []

        async def first(event: TaskEvent) -> None:
            order.append(1)

        async def second(event: TaskEvent) -> None:
            order.append(2)

        async def third(event: TaskEvent) -> None:
            order.append(3)

        bus.subscribe(TaskEventType.STARTED, first)
        bus.subscribe(TaskEventType.STARTED, second)
        bus.subscribe(TaskEventType.STARTED, third)

        await bus.emit(TaskEvent(event_type=TaskEventType.STARTED, task_id="t-1"))
        assert order == [1, 2, 3]


# ---------------------------------------------------------------------------
# TaskManager + EventBus integration tests
# ---------------------------------------------------------------------------


class TestTaskManagerEventBusIntegration:
    @pytest.mark.asyncio()
    async def test_create_emits_created_event(self) -> None:
        bus = TaskEventBus()
        received: list[TaskEvent] = []

        async def handler(event: TaskEvent) -> None:
            received.append(event)

        bus.subscribe(TaskEventType.CREATED, handler)

        mgr = TaskManager(event_bus=bus)
        task = mgr.create("Test task")

        # Let the fire-and-forget task run
        await asyncio.sleep(0)

        assert len(received) == 1
        assert received[0].event_type == TaskEventType.CREATED
        assert received[0].task_id == task.id

    @pytest.mark.asyncio()
    async def test_update_to_working_emits_started(self) -> None:
        bus = TaskEventBus()
        received: list[TaskEvent] = []

        async def handler(event: TaskEvent) -> None:
            received.append(event)

        bus.subscribe(TaskEventType.STARTED, handler)

        mgr = TaskManager(event_bus=bus)
        task = mgr.create("Test task")
        mgr.update(task.id, status=TaskStatus.WORKING)

        await asyncio.sleep(0)

        assert len(received) == 1
        assert received[0].event_type == TaskEventType.STARTED

    @pytest.mark.asyncio()
    async def test_update_to_completed_emits_completed(self) -> None:
        bus = TaskEventBus()
        received: list[TaskEvent] = []

        async def handler(event: TaskEvent) -> None:
            received.append(event)

        bus.subscribe(TaskEventType.COMPLETED, handler)

        mgr = TaskManager(event_bus=bus)
        task = mgr.create("Test task")
        mgr.update(task.id, status=TaskStatus.WORKING)
        mgr.update(task.id, status=TaskStatus.COMPLETED)

        await asyncio.sleep(0)

        assert len(received) == 1
        assert received[0].event_type == TaskEventType.COMPLETED

    @pytest.mark.asyncio()
    async def test_update_to_failed_emits_failed(self) -> None:
        bus = TaskEventBus()
        received: list[TaskEvent] = []

        async def handler(event: TaskEvent) -> None:
            received.append(event)

        bus.subscribe(TaskEventType.FAILED, handler)

        mgr = TaskManager(event_bus=bus)
        task = mgr.create("Test task")
        mgr.update(task.id, status=TaskStatus.WORKING)
        mgr.update(task.id, status=TaskStatus.FAILED)

        await asyncio.sleep(0)

        assert len(received) == 1
        assert received[0].event_type == TaskEventType.FAILED

    @pytest.mark.asyncio()
    async def test_update_to_paused_emits_paused(self) -> None:
        bus = TaskEventBus()
        received: list[TaskEvent] = []

        async def handler(event: TaskEvent) -> None:
            received.append(event)

        bus.subscribe(TaskEventType.PAUSED, handler)

        mgr = TaskManager(event_bus=bus)
        task = mgr.create("Test task")
        mgr.update(task.id, status=TaskStatus.WORKING)
        mgr.update(task.id, status=TaskStatus.PAUSED)

        await asyncio.sleep(0)

        assert len(received) == 1
        assert received[0].event_type == TaskEventType.PAUSED

    @pytest.mark.asyncio()
    async def test_update_to_canceled_emits_canceled(self) -> None:
        bus = TaskEventBus()
        received: list[TaskEvent] = []

        async def handler(event: TaskEvent) -> None:
            received.append(event)

        bus.subscribe(TaskEventType.CANCELED, handler)

        mgr = TaskManager(event_bus=bus)
        task = mgr.create("Test task")
        mgr.update(task.id, status=TaskStatus.WORKING)
        mgr.update(task.id, status=TaskStatus.CANCELED)

        await asyncio.sleep(0)

        assert len(received) == 1
        assert received[0].event_type == TaskEventType.CANCELED

    @pytest.mark.asyncio()
    async def test_no_event_bus_no_error(self) -> None:
        """TaskManager without event_bus works as before."""
        mgr = TaskManager()
        task = mgr.create("Test task")
        mgr.update(task.id, status=TaskStatus.WORKING)
        mgr.update(task.id, status=TaskStatus.COMPLETED)
        assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio()
    async def test_full_lifecycle_events(self) -> None:
        """Track all events through a full task lifecycle."""
        bus = TaskEventBus()
        events: list[TaskEvent] = []

        async def collector(event: TaskEvent) -> None:
            events.append(event)

        for evt_type in TaskEventType:
            bus.subscribe(evt_type, collector)

        mgr = TaskManager(event_bus=bus)
        task = mgr.create("Lifecycle task")
        mgr.update(task.id, status=TaskStatus.WORKING)
        mgr.update(task.id, status=TaskStatus.PAUSED)
        mgr.update(task.id, status=TaskStatus.WORKING)
        mgr.update(task.id, status=TaskStatus.COMPLETED)

        await asyncio.sleep(0)

        event_types = [e.event_type for e in events]
        assert event_types == [
            TaskEventType.CREATED,
            TaskEventType.STARTED,
            TaskEventType.PAUSED,
            TaskEventType.STARTED,
            TaskEventType.COMPLETED,
        ]
