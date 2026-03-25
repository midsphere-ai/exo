"""Integration tests: distributed Worker picks up tasks from Redis, runs agent,
and stores results back in Redis via TaskStore.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


async def _poll_store(store, task_id: str, *, timeout: float = 25.0):  # type: ignore[no-untyped-def]
    """Poll TaskStore until task reaches a terminal state and return the result."""

    async def _wait():
        while True:
            result = await store.get_status(task_id)
            if result is not None and str(result.status) in _TERMINAL_STATUSES:
                return result
            await asyncio.sleep(0.5)

    return await asyncio.wait_for(_wait(), timeout=timeout)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_worker_executes_task_and_stores_result(
    redis_container: str,
    vertex_model: str,
) -> None:
    """Worker processes a task and stores a COMPLETED result in Redis."""
    from exo.distributed.broker import TaskBroker  # pyright: ignore[reportMissingImports]
    from exo.distributed.models import (  # pyright: ignore[reportMissingImports]
        TaskPayload,
        TaskStatus,
    )
    from exo.distributed.store import TaskStore  # pyright: ignore[reportMissingImports]
    from exo.distributed.worker import Worker  # pyright: ignore[reportMissingImports]

    redis_url = redis_container

    broker = TaskBroker(redis_url)
    store = TaskStore(redis_url)
    await broker.connect()
    await store.connect()

    try:
        task = TaskPayload(
            agent_config={
                "name": "dist-int-agent",
                "model": vertex_model,
                "instructions": "You are a helpful assistant. Answer concisely.",
            },
            input="What is 3+3? Respond with just the number.",
        )
        await broker.submit(task)

        worker = Worker(redis_url, worker_id="test-worker-int025-a")
        worker._broker._max_retries = 0  # type: ignore[attr-defined]

        worker_task = asyncio.create_task(worker.start())
        try:
            result = await _poll_store(store, task.task_id, timeout=25.0)
        finally:
            await worker.stop()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(worker_task, timeout=5.0)

        assert result is not None
        assert result.status == TaskStatus.COMPLETED
        assert result.result is not None
        assert "6" in result.result.get("output", "")
    finally:
        await broker.disconnect()
        await store.disconnect()


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_worker_handles_task_failure_gracefully(
    redis_container: str,
) -> None:
    """Worker marks task as FAILED and records an error when agent config is invalid."""
    from exo.distributed.broker import TaskBroker  # pyright: ignore[reportMissingImports]
    from exo.distributed.models import (  # pyright: ignore[reportMissingImports]
        TaskPayload,
        TaskStatus,
    )
    from exo.distributed.store import TaskStore  # pyright: ignore[reportMissingImports]
    from exo.distributed.worker import Worker  # pyright: ignore[reportMissingImports]

    redis_url = redis_container

    broker = TaskBroker(redis_url)
    store = TaskStore(redis_url)
    await broker.connect()
    await store.connect()

    try:
        task = TaskPayload(
            agent_config={
                "name": "fail-agent",
                "model": "invalid_provider:nonexistent-model",
            },
            input="This should fail.",
        )
        await broker.submit(task)

        worker = Worker(redis_url, worker_id="test-worker-int025-b")
        worker._broker._max_retries = 0  # type: ignore[attr-defined]

        worker_task = asyncio.create_task(worker.start())
        try:
            result = await _poll_store(store, task.task_id, timeout=15.0)
        finally:
            await worker.stop()
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(worker_task, timeout=5.0)

        assert result is not None
        assert result.status == TaskStatus.FAILED
        assert result.error is not None
        assert len(result.error) > 0
    finally:
        await broker.disconnect()
        await store.disconnect()
