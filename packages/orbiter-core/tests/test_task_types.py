"""Tests for TaskStatus, Task model, and status transition rules."""

from __future__ import annotations

import pytest

from orbiter.task_controller import (
    InvalidTransitionError,
    Task,
    TaskError,
    TaskStatus,
)
from orbiter.types import OrbiterError

# ---------------------------------------------------------------------------
# TaskStatus enum
# ---------------------------------------------------------------------------


class TestTaskStatus:
    def test_members(self) -> None:
        assert set(TaskStatus) == {
            TaskStatus.SUBMITTED,
            TaskStatus.WORKING,
            TaskStatus.PAUSED,
            TaskStatus.INPUT_REQUIRED,
            TaskStatus.COMPLETED,
            TaskStatus.CANCELED,
            TaskStatus.FAILED,
            TaskStatus.WAITING,
        }

    def test_values(self) -> None:
        assert TaskStatus.SUBMITTED == "submitted"
        assert TaskStatus.WORKING == "working"
        assert TaskStatus.PAUSED == "paused"
        assert TaskStatus.INPUT_REQUIRED == "input_required"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.CANCELED == "canceled"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.WAITING == "waiting"

    def test_is_str(self) -> None:
        for status in TaskStatus:
            assert isinstance(status, str)


# ---------------------------------------------------------------------------
# TaskError hierarchy
# ---------------------------------------------------------------------------


class TestTaskError:
    def test_inherits_from_orbiter_error(self) -> None:
        assert issubclass(TaskError, OrbiterError)

    def test_invalid_transition_inherits_from_task_error(self) -> None:
        assert issubclass(InvalidTransitionError, TaskError)


# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------


class TestTask:
    def test_defaults(self) -> None:
        task = Task(name="test task")
        assert task.name == "test task"
        assert task.description == ""
        assert task.status == TaskStatus.SUBMITTED
        assert task.priority == 0
        assert task.parent_id is None
        assert task.metadata == {}
        assert task.id  # UUID string is non-empty
        assert task.created_at is not None
        assert task.updated_at is not None

    def test_custom_values(self) -> None:
        task = Task(
            name="custom",
            description="detailed",
            priority=10,
            parent_id="parent-1",
            metadata={"key": "value"},
        )
        assert task.name == "custom"
        assert task.description == "detailed"
        assert task.priority == 10
        assert task.parent_id == "parent-1"
        assert task.metadata == {"key": "value"}

    def test_unique_ids(self) -> None:
        t1 = Task(name="a")
        t2 = Task(name="b")
        assert t1.id != t2.id

    def test_is_terminal_completed(self) -> None:
        task = Task(name="t", status=TaskStatus.COMPLETED)
        assert task.is_terminal is True

    def test_is_terminal_canceled(self) -> None:
        task = Task(name="t", status=TaskStatus.CANCELED)
        assert task.is_terminal is True

    def test_is_terminal_false_for_non_terminal(self) -> None:
        for status in TaskStatus:
            if status not in (TaskStatus.COMPLETED, TaskStatus.CANCELED):
                task = Task(name="t", status=status)
                assert task.is_terminal is False, f"{status} should not be terminal"


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------


class TestStatusTransitions:
    def test_submitted_to_working(self) -> None:
        task = Task(name="t")
        task.transition(TaskStatus.WORKING)
        assert task.status == TaskStatus.WORKING

    def test_submitted_to_canceled(self) -> None:
        task = Task(name="t")
        task.transition(TaskStatus.CANCELED)
        assert task.status == TaskStatus.CANCELED

    def test_working_to_paused(self) -> None:
        task = Task(name="t", status=TaskStatus.WORKING)
        task.transition(TaskStatus.PAUSED)
        assert task.status == TaskStatus.PAUSED

    def test_working_to_input_required(self) -> None:
        task = Task(name="t", status=TaskStatus.WORKING)
        task.transition(TaskStatus.INPUT_REQUIRED)
        assert task.status == TaskStatus.INPUT_REQUIRED

    def test_working_to_completed(self) -> None:
        task = Task(name="t", status=TaskStatus.WORKING)
        task.transition(TaskStatus.COMPLETED)
        assert task.status == TaskStatus.COMPLETED

    def test_working_to_failed(self) -> None:
        task = Task(name="t", status=TaskStatus.WORKING)
        task.transition(TaskStatus.FAILED)
        assert task.status == TaskStatus.FAILED

    def test_working_to_canceled(self) -> None:
        task = Task(name="t", status=TaskStatus.WORKING)
        task.transition(TaskStatus.CANCELED)
        assert task.status == TaskStatus.CANCELED

    def test_working_to_waiting(self) -> None:
        task = Task(name="t", status=TaskStatus.WORKING)
        task.transition(TaskStatus.WAITING)
        assert task.status == TaskStatus.WAITING

    def test_paused_to_working(self) -> None:
        task = Task(name="t", status=TaskStatus.PAUSED)
        task.transition(TaskStatus.WORKING)
        assert task.status == TaskStatus.WORKING

    def test_paused_to_canceled(self) -> None:
        task = Task(name="t", status=TaskStatus.PAUSED)
        task.transition(TaskStatus.CANCELED)
        assert task.status == TaskStatus.CANCELED

    def test_input_required_to_working(self) -> None:
        task = Task(name="t", status=TaskStatus.INPUT_REQUIRED)
        task.transition(TaskStatus.WORKING)
        assert task.status == TaskStatus.WORKING

    def test_waiting_to_working(self) -> None:
        task = Task(name="t", status=TaskStatus.WAITING)
        task.transition(TaskStatus.WORKING)
        assert task.status == TaskStatus.WORKING

    def test_failed_to_submitted(self) -> None:
        task = Task(name="t", status=TaskStatus.FAILED)
        task.transition(TaskStatus.SUBMITTED)
        assert task.status == TaskStatus.SUBMITTED

    def test_completed_cannot_transition(self) -> None:
        task = Task(name="t", status=TaskStatus.COMPLETED)
        with pytest.raises(InvalidTransitionError):
            task.transition(TaskStatus.WORKING)

    def test_canceled_cannot_transition(self) -> None:
        task = Task(name="t", status=TaskStatus.CANCELED)
        with pytest.raises(InvalidTransitionError):
            task.transition(TaskStatus.SUBMITTED)

    def test_submitted_cannot_go_to_completed(self) -> None:
        task = Task(name="t")
        with pytest.raises(InvalidTransitionError):
            task.transition(TaskStatus.COMPLETED)

    def test_paused_cannot_go_to_completed(self) -> None:
        task = Task(name="t", status=TaskStatus.PAUSED)
        with pytest.raises(InvalidTransitionError):
            task.transition(TaskStatus.COMPLETED)

    def test_transition_updates_updated_at(self) -> None:
        task = Task(name="t")
        original = task.updated_at
        task.transition(TaskStatus.WORKING)
        assert task.updated_at >= original
