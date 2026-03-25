"""Tests for distributed task models."""

from __future__ import annotations

import json
import time

from exo.distributed.models import (  # pyright: ignore[reportMissingImports]
    TaskPayload,
    TaskResult,
    TaskStatus,
)


class TestTaskStatus:
    def test_values(self) -> None:
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
        assert TaskStatus.CANCELLED == "cancelled"
        assert TaskStatus.RETRYING == "retrying"

    def test_is_str(self) -> None:
        assert isinstance(TaskStatus.PENDING, str)


class TestTaskPayload:
    def test_defaults(self) -> None:
        payload = TaskPayload()
        assert len(payload.task_id) == 32  # uuid4().hex length
        assert payload.agent_config == {}
        assert payload.input == ""
        assert payload.messages == []
        assert payload.model is None
        assert payload.detailed is False
        assert payload.metadata == {}
        assert payload.created_at == 0.0
        assert payload.timeout_seconds == 300.0

    def test_unique_task_ids(self) -> None:
        p1 = TaskPayload()
        p2 = TaskPayload()
        assert p1.task_id != p2.task_id

    def test_custom_fields(self) -> None:
        now = time.time()
        payload = TaskPayload(
            task_id="custom-id",
            agent_config={"name": "test-agent", "model": "gpt-4"},
            input="Hello world",
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4",
            detailed=True,
            metadata={"trace_id": "abc123"},
            created_at=now,
            timeout_seconds=60.0,
        )
        assert payload.task_id == "custom-id"
        assert payload.agent_config["name"] == "test-agent"
        assert payload.input == "Hello world"
        assert payload.messages == [{"role": "user", "content": "Hi"}]
        assert payload.model == "gpt-4"
        assert payload.detailed is True
        assert payload.metadata["trace_id"] == "abc123"
        assert payload.created_at == now
        assert payload.timeout_seconds == 60.0

    def test_frozen(self) -> None:
        payload = TaskPayload()
        try:
            payload.input = "changed"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except Exception:
            pass

    def test_json_serialization_roundtrip(self) -> None:
        now = time.time()
        payload = TaskPayload(
            agent_config={"name": "agent", "tools": ["search"]},
            input="test input",
            messages=[{"role": "user", "content": "msg"}],
            model="gpt-4",
            detailed=True,
            metadata={"key": "value"},
            created_at=now,
            timeout_seconds=120.0,
        )
        data = payload.model_dump()
        json_str = json.dumps(data)
        restored = TaskPayload(**json.loads(json_str))
        assert restored.task_id == payload.task_id
        assert restored.agent_config == payload.agent_config
        assert restored.input == payload.input
        assert restored.messages == payload.messages
        assert restored.model == payload.model
        assert restored.detailed == payload.detailed
        assert restored.metadata == payload.metadata
        assert restored.created_at == payload.created_at
        assert restored.timeout_seconds == payload.timeout_seconds

    def test_model_dump(self) -> None:
        payload = TaskPayload(input="hello")
        data = payload.model_dump()
        assert isinstance(data, dict)
        assert data["input"] == "hello"
        assert "task_id" in data


class TestTaskResult:
    def test_defaults(self) -> None:
        result = TaskResult()
        assert result.task_id == ""
        assert result.status == TaskStatus.PENDING
        assert result.result is None
        assert result.error is None
        assert result.started_at is None
        assert result.completed_at is None
        assert result.worker_id is None
        assert result.retries == 0

    def test_custom_fields(self) -> None:
        now = time.time()
        result = TaskResult(
            task_id="task-123",
            status=TaskStatus.COMPLETED,
            result={"output": "done", "steps": 3},
            error=None,
            started_at=now - 10,
            completed_at=now,
            worker_id="worker-1",
            retries=2,
        )
        assert result.task_id == "task-123"
        assert result.status == TaskStatus.COMPLETED
        assert result.result == {"output": "done", "steps": 3}
        assert result.started_at == now - 10
        assert result.completed_at == now
        assert result.worker_id == "worker-1"
        assert result.retries == 2

    def test_failed_with_error(self) -> None:
        result = TaskResult(
            task_id="task-456",
            status=TaskStatus.FAILED,
            error="Connection timeout",
            retries=3,
        )
        assert result.status == TaskStatus.FAILED
        assert result.error == "Connection timeout"
        assert result.retries == 3

    def test_frozen(self) -> None:
        result = TaskResult()
        try:
            result.status = TaskStatus.RUNNING  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except Exception:
            pass

    def test_json_serialization_roundtrip(self) -> None:
        now = time.time()
        result = TaskResult(
            task_id="task-789",
            status=TaskStatus.RUNNING,
            result={"partial": True},
            started_at=now,
            worker_id="worker-2",
            retries=1,
        )
        data = result.model_dump()
        json_str = json.dumps(data)
        restored = TaskResult(**json.loads(json_str))
        assert restored.task_id == result.task_id
        assert restored.status == result.status
        assert restored.result == result.result
        assert restored.started_at == result.started_at
        assert restored.worker_id == result.worker_id
        assert restored.retries == result.retries

    def test_model_dump(self) -> None:
        result = TaskResult(task_id="t1", status=TaskStatus.CANCELLED)
        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["task_id"] == "t1"
        assert data["status"] == "cancelled"

    def test_all_statuses_in_result(self) -> None:
        for status in TaskStatus:
            result = TaskResult(status=status)
            assert result.status == status
            data = result.model_dump()
            restored = TaskResult(**data)
            assert restored.status == status
