"""Tests for the BackgroundTaskHandler with hot-merge and wake-up-merge."""

from __future__ import annotations

from typing import Any

import pytest

from exo._internal.background import (
    BackgroundTask,
    BackgroundTaskError,
    BackgroundTaskHandler,
    MergeMode,
    PendingQueue,
)
from exo._internal.state import RunNodeStatus, RunState

# ---------------------------------------------------------------------------
# BackgroundTask lifecycle
# ---------------------------------------------------------------------------


class TestBackgroundTask:
    def test_init_defaults(self) -> None:
        t = BackgroundTask("t1", "parent1")
        assert t.task_id == "t1"
        assert t.parent_task_id == "parent1"
        assert t.payload is None
        assert t.result is None
        assert t.error is None
        assert t.status == RunNodeStatus.INIT
        assert t.merge_mode is None
        assert not t.is_complete

    def test_start(self) -> None:
        t = BackgroundTask("t1", "p1")
        t.start()
        assert t.status == RunNodeStatus.RUNNING
        assert not t.is_complete

    def test_complete(self) -> None:
        t = BackgroundTask("t1", "p1")
        t.start()
        t.complete({"key": "value"})
        assert t.status == RunNodeStatus.SUCCESS
        assert t.result == {"key": "value"}
        assert t.is_complete

    def test_fail(self) -> None:
        t = BackgroundTask("t1", "p1")
        t.start()
        t.fail("something went wrong")
        assert t.status == RunNodeStatus.FAILED
        assert t.error == "something went wrong"
        assert t.is_complete

    def test_payload(self) -> None:
        t = BackgroundTask("t1", "p1", payload="initial data")
        assert t.payload == "initial data"


# ---------------------------------------------------------------------------
# PendingQueue
# ---------------------------------------------------------------------------


class TestPendingQueue:
    def test_push_and_pop(self) -> None:
        q = PendingQueue()
        assert q.empty
        assert q.size == 0

        t1 = BackgroundTask("t1", "p1")
        t2 = BackgroundTask("t2", "p1")
        q.push(t1)
        q.push(t2)

        assert q.size == 2
        assert not q.empty

        items = q.pop_all()
        assert len(items) == 2
        assert items[0].task_id == "t1"
        assert items[1].task_id == "t2"
        assert q.empty

    def test_pop_all_clears(self) -> None:
        q = PendingQueue()
        q.push(BackgroundTask("t1", "p1"))
        q.pop_all()
        assert q.pop_all() == []

    async def test_wait_returns_true_when_items_available(self) -> None:
        q = PendingQueue()
        q.push(BackgroundTask("t1", "p1"))
        result = await q.wait(timeout=0.1)
        assert result is True

    async def test_wait_returns_false_on_timeout(self) -> None:
        q = PendingQueue()
        result = await q.wait(timeout=0.01)
        assert result is False


# ---------------------------------------------------------------------------
# BackgroundTaskHandler — submit
# ---------------------------------------------------------------------------


class TestBackgroundTaskHandlerSubmit:
    def test_submit_creates_task(self) -> None:
        handler = BackgroundTaskHandler()
        task = handler.submit("bg1", "parent1", payload="data")
        assert task.task_id == "bg1"
        assert task.parent_task_id == "parent1"
        assert task.payload == "data"
        assert task.status == RunNodeStatus.RUNNING

    def test_submit_duplicate_raises(self) -> None:
        handler = BackgroundTaskHandler()
        handler.submit("bg1", "parent1")
        with pytest.raises(BackgroundTaskError, match="already exists"):
            handler.submit("bg1", "parent1")

    def test_submit_with_state_creates_node(self) -> None:
        state = RunState(agent_name="main")
        handler = BackgroundTaskHandler(state=state)
        handler.submit("bg1", "parent1")
        assert len(state.nodes) == 1
        assert state.nodes[0].agent_name == "bg:bg1"
        assert state.nodes[0].status == RunNodeStatus.RUNNING


# ---------------------------------------------------------------------------
# BackgroundTaskHandler — hot-merge
# ---------------------------------------------------------------------------


class TestHotMerge:
    async def test_hot_merge_returns_hot_mode(self) -> None:
        handler = BackgroundTaskHandler()
        handler.submit("bg1", "parent1")
        mode = await handler.handle_result("bg1", "result-data", is_main_running=True)
        assert mode == MergeMode.HOT

    async def test_hot_merge_sets_task_result(self) -> None:
        handler = BackgroundTaskHandler()
        handler.submit("bg1", "parent1")
        await handler.handle_result("bg1", "result-data", is_main_running=True)
        task = handler.get_task("bg1")
        assert task is not None
        assert task.result == "result-data"
        assert task.status == RunNodeStatus.SUCCESS
        assert task.merge_mode == MergeMode.HOT

    async def test_hot_merge_fires_callback(self) -> None:
        handler = BackgroundTaskHandler()
        received: list[tuple[BackgroundTask, MergeMode]] = []

        async def cb(task: BackgroundTask, mode: MergeMode) -> None:
            received.append((task, mode))

        handler.on_merge(cb)
        handler.submit("bg1", "parent1")
        await handler.handle_result("bg1", "data", is_main_running=True)

        assert len(received) == 1
        assert received[0][0].task_id == "bg1"
        assert received[0][1] == MergeMode.HOT

    async def test_hot_merge_completes_state_node(self) -> None:
        state = RunState(agent_name="main")
        handler = BackgroundTaskHandler(state=state)
        handler.submit("bg1", "parent1")
        await handler.handle_result("bg1", "data", is_main_running=True)
        assert state.nodes[0].status == RunNodeStatus.SUCCESS


# ---------------------------------------------------------------------------
# BackgroundTaskHandler — wake-up-merge
# ---------------------------------------------------------------------------


class TestWakeupMerge:
    async def test_wakeup_merge_returns_wakeup_mode(self) -> None:
        handler = BackgroundTaskHandler()
        handler.submit("bg1", "parent1")
        mode = await handler.handle_result("bg1", "result", is_main_running=False)
        assert mode == MergeMode.WAKEUP

    async def test_wakeup_merge_queues_in_pending(self) -> None:
        handler = BackgroundTaskHandler()
        handler.submit("bg1", "parent1")
        await handler.handle_result("bg1", "result", is_main_running=False)
        assert handler.pending_queue.size == 1

    async def test_drain_pending_yields_tasks(self) -> None:
        handler = BackgroundTaskHandler()
        handler.submit("bg1", "parent1")
        handler.submit("bg2", "parent1")
        await handler.handle_result("bg1", "r1", is_main_running=False)
        await handler.handle_result("bg2", "r2", is_main_running=False)

        tasks: list[BackgroundTask] = []
        async for task in handler.drain_pending():
            tasks.append(task)

        assert len(tasks) == 2
        assert tasks[0].task_id == "bg1"
        assert tasks[1].task_id == "bg2"
        assert handler.pending_queue.empty

    async def test_wakeup_merge_does_not_fire_hot_callback(self) -> None:
        handler = BackgroundTaskHandler()
        hot_calls: list[Any] = []

        async def cb(task: BackgroundTask, mode: MergeMode) -> None:
            hot_calls.append(mode)

        handler.on_merge(cb)
        handler.submit("bg1", "parent1")
        await handler.handle_result("bg1", "data", is_main_running=False)
        # callback should not be called for wake-up merge
        assert len(hot_calls) == 0


# ---------------------------------------------------------------------------
# BackgroundTaskHandler — error handling
# ---------------------------------------------------------------------------


class TestBackgroundTaskHandlerErrors:
    async def test_handle_result_unknown_task(self) -> None:
        handler = BackgroundTaskHandler()
        with pytest.raises(BackgroundTaskError, match="not found"):
            await handler.handle_result("unknown", "data")

    def test_handle_error_marks_task_failed(self) -> None:
        handler = BackgroundTaskHandler()
        handler.submit("bg1", "parent1")
        handler.handle_error("bg1", "crash")
        task = handler.get_task("bg1")
        assert task is not None
        assert task.status == RunNodeStatus.FAILED
        assert task.error == "crash"

    def test_handle_error_unknown_task(self) -> None:
        handler = BackgroundTaskHandler()
        with pytest.raises(BackgroundTaskError, match="not found"):
            handler.handle_error("unknown", "err")

    def test_handle_error_updates_state_node(self) -> None:
        state = RunState(agent_name="main")
        handler = BackgroundTaskHandler(state=state)
        handler.submit("bg1", "parent1")
        handler.handle_error("bg1", "boom")
        assert state.nodes[0].status == RunNodeStatus.FAILED
        assert state.nodes[0].error == "boom"


# ---------------------------------------------------------------------------
# BackgroundTaskHandler — list and get
# ---------------------------------------------------------------------------


class TestBackgroundTaskHandlerListing:
    def test_get_task(self) -> None:
        handler = BackgroundTaskHandler()
        handler.submit("bg1", "p1")
        assert handler.get_task("bg1") is not None
        assert handler.get_task("nonexistent") is None

    def test_list_all_tasks(self) -> None:
        handler = BackgroundTaskHandler()
        handler.submit("bg1", "p1")
        handler.submit("bg2", "p1")
        assert len(handler.list_tasks()) == 2

    def test_list_tasks_by_status(self) -> None:
        handler = BackgroundTaskHandler()
        handler.submit("bg1", "p1")
        handler.submit("bg2", "p1")
        handler.handle_error("bg2", "err")
        running = handler.list_tasks(status=RunNodeStatus.RUNNING)
        failed = handler.list_tasks(status=RunNodeStatus.FAILED)
        assert len(running) == 1
        assert running[0].task_id == "bg1"
        assert len(failed) == 1
        assert failed[0].task_id == "bg2"


# ---------------------------------------------------------------------------
# Multiple callbacks
# ---------------------------------------------------------------------------


class TestMultipleCallbacks:
    async def test_multiple_merge_callbacks(self) -> None:
        handler = BackgroundTaskHandler()
        calls_a: list[str] = []
        calls_b: list[str] = []

        async def cb_a(task: BackgroundTask, mode: MergeMode) -> None:
            calls_a.append(task.task_id)

        async def cb_b(task: BackgroundTask, mode: MergeMode) -> None:
            calls_b.append(task.task_id)

        handler.on_merge(cb_a)
        handler.on_merge(cb_b)
        handler.submit("bg1", "p1")
        await handler.handle_result("bg1", "data", is_main_running=True)

        assert calls_a == ["bg1"]
        assert calls_b == ["bg1"]
