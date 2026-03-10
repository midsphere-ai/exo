"""Tests for TaskManager CRUD, status transitions, and cascading effects."""

from __future__ import annotations

import pytest

from orbiter.task_controller import (
    InvalidTransitionError,
    TaskManager,
    TaskNotFoundError,
    TaskStatus,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mgr() -> TaskManager:
    return TaskManager()


@pytest.fixture()
def mgr_auto() -> TaskManager:
    return TaskManager(auto_complete_parent=True)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


class TestCreate:
    def test_basic_create(self, mgr: TaskManager) -> None:
        task = mgr.create("my task")
        assert task.name == "my task"
        assert task.status == TaskStatus.SUBMITTED
        assert task.id

    def test_create_with_all_fields(self, mgr: TaskManager) -> None:
        task = mgr.create(
            "full task",
            description="desc",
            priority=5,
            metadata={"k": "v"},
        )
        assert task.description == "desc"
        assert task.priority == 5
        assert task.metadata == {"k": "v"}

    def test_create_with_parent(self, mgr: TaskManager) -> None:
        parent = mgr.create("parent")
        child = mgr.create("child", parent_id=parent.id)
        assert child.parent_id == parent.id

    def test_create_with_invalid_parent_raises(self, mgr: TaskManager) -> None:
        with pytest.raises(TaskNotFoundError):
            mgr.create("orphan", parent_id="nonexistent")


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_existing(self, mgr: TaskManager) -> None:
        task = mgr.create("t")
        assert mgr.get(task.id) is task

    def test_get_missing_returns_none(self, mgr: TaskManager) -> None:
        assert mgr.get("missing") is None


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_name(self, mgr: TaskManager) -> None:
        task = mgr.create("old")
        updated = mgr.update(task.id, name="new")
        assert updated.name == "new"

    def test_update_description(self, mgr: TaskManager) -> None:
        task = mgr.create("t")
        mgr.update(task.id, description="updated desc")
        assert task.description == "updated desc"

    def test_update_priority(self, mgr: TaskManager) -> None:
        task = mgr.create("t")
        mgr.update(task.id, priority=99)
        assert task.priority == 99

    def test_update_metadata(self, mgr: TaskManager) -> None:
        task = mgr.create("t", metadata={"a": 1})
        mgr.update(task.id, metadata={"b": 2})
        assert task.metadata == {"b": 2}

    def test_update_status(self, mgr: TaskManager) -> None:
        task = mgr.create("t")
        mgr.update(task.id, status=TaskStatus.WORKING)
        assert task.status == TaskStatus.WORKING

    def test_update_invalid_status_raises(self, mgr: TaskManager) -> None:
        task = mgr.create("t")
        with pytest.raises(InvalidTransitionError):
            mgr.update(task.id, status=TaskStatus.COMPLETED)

    def test_update_missing_task_raises(self, mgr: TaskManager) -> None:
        with pytest.raises(TaskNotFoundError):
            mgr.update("missing", name="nope")

    def test_update_updates_timestamp(self, mgr: TaskManager) -> None:
        task = mgr.create("t")
        original = task.updated_at
        mgr.update(task.id, name="renamed")
        assert task.updated_at >= original


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_existing(self, mgr: TaskManager) -> None:
        task = mgr.create("t")
        assert mgr.delete(task.id) is True
        assert mgr.get(task.id) is None

    def test_delete_missing(self, mgr: TaskManager) -> None:
        assert mgr.delete("missing") is False


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


class TestList:
    def test_list_empty(self, mgr: TaskManager) -> None:
        assert mgr.list() == []

    def test_list_returns_all(self, mgr: TaskManager) -> None:
        mgr.create("a")
        mgr.create("b")
        assert len(mgr.list()) == 2

    def test_list_filter_by_status(self, mgr: TaskManager) -> None:
        t1 = mgr.create("a")
        mgr.create("b")
        mgr.update(t1.id, status=TaskStatus.WORKING)
        working = mgr.list(status=TaskStatus.WORKING)
        assert len(working) == 1
        assert working[0].id == t1.id

    def test_list_sorted_by_priority_desc(self, mgr: TaskManager) -> None:
        mgr.create("low", priority=1)
        mgr.create("high", priority=10)
        mgr.create("mid", priority=5)
        tasks = mgr.list()
        priorities = [t.priority for t in tasks]
        assert priorities == [10, 5, 1]

    def test_list_same_priority_sorted_by_created_at(self, mgr: TaskManager) -> None:
        t1 = mgr.create("first", priority=5)
        t2 = mgr.create("second", priority=5)
        tasks = mgr.list()
        assert tasks[0].id == t1.id
        assert tasks[1].id == t2.id


# ---------------------------------------------------------------------------
# Cancel cascade
# ---------------------------------------------------------------------------


class TestCancelCascade:
    def test_cancel_parent_cancels_children(self, mgr: TaskManager) -> None:
        parent = mgr.create("parent")
        child1 = mgr.create("child1", parent_id=parent.id)
        child2 = mgr.create("child2", parent_id=parent.id)
        mgr.update(parent.id, status=TaskStatus.WORKING)
        mgr.update(parent.id, status=TaskStatus.CANCELED)
        assert child1.status == TaskStatus.CANCELED
        assert child2.status == TaskStatus.CANCELED

    def test_cancel_cascades_to_grandchildren(self, mgr: TaskManager) -> None:
        grandparent = mgr.create("gp")
        parent = mgr.create("parent", parent_id=grandparent.id)
        child = mgr.create("child", parent_id=parent.id)
        mgr.update(grandparent.id, status=TaskStatus.WORKING)
        mgr.update(grandparent.id, status=TaskStatus.CANCELED)
        assert parent.status == TaskStatus.CANCELED
        assert child.status == TaskStatus.CANCELED

    def test_cancel_skips_already_terminal_children(self, mgr: TaskManager) -> None:
        parent = mgr.create("parent")
        child = mgr.create("child", parent_id=parent.id)
        mgr.update(child.id, status=TaskStatus.WORKING)
        mgr.update(child.id, status=TaskStatus.COMPLETED)
        mgr.update(parent.id, status=TaskStatus.WORKING)
        mgr.update(parent.id, status=TaskStatus.CANCELED)
        # Already completed child stays completed
        assert child.status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Auto-complete parent
# ---------------------------------------------------------------------------


class TestAutoCompleteParent:
    def test_auto_complete_when_all_children_done(self, mgr_auto: TaskManager) -> None:
        parent = mgr_auto.create("parent")
        child1 = mgr_auto.create("c1", parent_id=parent.id)
        child2 = mgr_auto.create("c2", parent_id=parent.id)
        mgr_auto.update(parent.id, status=TaskStatus.WORKING)
        mgr_auto.update(child1.id, status=TaskStatus.WORKING)
        mgr_auto.update(child2.id, status=TaskStatus.WORKING)
        mgr_auto.update(child1.id, status=TaskStatus.COMPLETED)
        assert parent.status == TaskStatus.WORKING  # not yet
        mgr_auto.update(child2.id, status=TaskStatus.COMPLETED)
        assert parent.status == TaskStatus.COMPLETED  # now auto-completed

    def test_no_auto_complete_when_disabled(self, mgr: TaskManager) -> None:
        parent = mgr.create("parent")
        child = mgr.create("c", parent_id=parent.id)
        mgr.update(parent.id, status=TaskStatus.WORKING)
        mgr.update(child.id, status=TaskStatus.WORKING)
        mgr.update(child.id, status=TaskStatus.COMPLETED)
        assert parent.status == TaskStatus.WORKING  # stays working
