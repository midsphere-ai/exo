"""Tests for cooperative cancellation: CancellationToken, TaskBroker.cancel(), Worker integration."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.distributed.broker import TaskBroker  # pyright: ignore[reportMissingImports]
from exo.distributed.cancel import CancellationToken  # pyright: ignore[reportMissingImports]
from exo.distributed.models import (  # pyright: ignore[reportMissingImports]
    TaskPayload,
    TaskStatus,
)
from exo.distributed.worker import Worker  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# CancellationToken
# ---------------------------------------------------------------------------


class TestCancellationToken:
    def test_initial_state(self) -> None:
        token = CancellationToken()
        assert token.cancelled is False

    def test_cancel_sets_flag(self) -> None:
        token = CancellationToken()
        token.cancel()
        assert token.cancelled is True

    def test_cancel_idempotent(self) -> None:
        token = CancellationToken()
        token.cancel()
        token.cancel()
        assert token.cancelled is True


# ---------------------------------------------------------------------------
# TaskBroker.cancel
# ---------------------------------------------------------------------------


class TestTaskBrokerCancel:
    @pytest.mark.asyncio
    async def test_cancel_publishes_signal(self) -> None:
        broker = TaskBroker("redis://localhost:6379")
        with patch("exo.distributed.broker.aioredis.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_from_url.return_value = mock_redis
            mock_redis.xgroup_create = AsyncMock()
            await broker.connect()

            await broker.cancel("task-123")

            mock_redis.publish.assert_called_once_with("exo:cancel:task-123", "cancel")

    @pytest.mark.asyncio
    async def test_cancel_sets_status_cancelled(self) -> None:
        broker = TaskBroker("redis://localhost:6379")
        with patch("exo.distributed.broker.aioredis.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_from_url.return_value = mock_redis
            mock_redis.xgroup_create = AsyncMock()
            await broker.connect()

            await broker.cancel("task-456")

            mock_redis.hset.assert_called_once_with(
                "exo:task:task-456",
                mapping={"status": str(TaskStatus.CANCELLED)},
            )

    @pytest.mark.asyncio
    async def test_cancel_with_fakeredis(self) -> None:
        import fakeredis.aioredis

        broker = TaskBroker("redis://localhost:6379")
        broker._redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        await broker._redis.xgroup_create(
            broker._queue_name, broker._group_name, id="0", mkstream=True
        )

        await broker.cancel("task-789")

        # Verify the status was set in the hash
        data = await broker._redis.hgetall("exo:task:task-789")  # type: ignore[misc]
        assert data["status"] == str(TaskStatus.CANCELLED)


# ---------------------------------------------------------------------------
# Worker._listen_for_cancel
# ---------------------------------------------------------------------------


class TestWorkerListenForCancel:
    @pytest.mark.asyncio
    async def test_cancel_signal_sets_token(self) -> None:
        """When a cancel message is published, the token should be set."""
        import fakeredis.aioredis

        fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
        w = Worker("redis://localhost", worker_id="w1")

        token = CancellationToken()

        # Patch aioredis.from_url to return our shared fakeredis instance
        with patch("exo.distributed.worker.aioredis.from_url", return_value=fake_redis):
            # Start listener in background
            listener_task = asyncio.create_task(w._listen_for_cancel("task-abc", token))

            # Give listener time to subscribe
            await asyncio.sleep(0.05)

            # Publish cancel signal on same fakeredis instance
            await fake_redis.publish("exo:cancel:task-abc", "cancel")  # type: ignore[misc]

            # Wait for listener to pick it up
            await asyncio.wait_for(listener_task, timeout=2.0)

        assert token.cancelled is True

    @pytest.mark.asyncio
    async def test_listener_cancellable(self) -> None:
        """The listener can be cancelled via asyncio task cancellation."""
        w = Worker("redis://localhost", worker_id="w1")
        token = CancellationToken()

        mock_pubsub = AsyncMock()
        mock_pubsub.get_message = AsyncMock(return_value=None)

        mock_redis = AsyncMock()
        # pubsub() is a sync method on Redis, so use MagicMock
        mock_redis.pubsub = MagicMock(return_value=mock_pubsub)

        with patch("exo.distributed.worker.aioredis.from_url", return_value=mock_redis):
            task = asyncio.create_task(w._listen_for_cancel("task-def", token))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert token.cancelled is False


# ---------------------------------------------------------------------------
# Worker._run_agent with cancellation
# ---------------------------------------------------------------------------


class TestWorkerRunAgentCancellation:
    @pytest.mark.asyncio
    async def test_stops_on_cancellation(self) -> None:
        """When token is cancelled, _run_agent breaks and emits StatusEvent."""
        w = Worker("redis://localhost", worker_id="w1")
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-cancel-1",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
            detailed=True,
        )

        from exo.types import TextEvent  # pyright: ignore[reportMissingImports]

        token = CancellationToken()

        events_yielded = 0

        async def _fake_stream_gen(*a: object, **kw: object) -> object:
            nonlocal events_yielded
            for i in range(10):
                events_yielded += 1
                yield TextEvent(text=f"chunk-{i}", agent_name="agent")
                if events_yielded == 2:
                    # Simulate cancel arriving after 2 events
                    token.cancel()

        mock_run = MagicMock()
        mock_run.stream = _fake_stream_gen

        mock_agent = MagicMock()
        mock_agent.name = "agent"

        with patch("exo.runner.run", mock_run):
            await w._run_agent(mock_agent, task, token)

        # Should have stopped early — only the first 2 text events were
        # published (the 3rd iteration sees token.cancelled and breaks).
        # The cancelled StatusEvent is also published.
        published_events = [call.args[1] for call in w._publisher.publish.call_args_list]

        # Text events published before cancellation
        text_events = [ev for ev in published_events if hasattr(ev, "text")]
        assert len(text_events) == 2
        assert text_events[0].text == "chunk-0"
        assert text_events[1].text == "chunk-1"

        # Last event is the cancelled StatusEvent
        from exo.types import StatusEvent  # pyright: ignore[reportMissingImports]

        cancel_event = published_events[-1]
        assert isinstance(cancel_event, StatusEvent)
        assert cancel_event.status == "cancelled"

    @pytest.mark.asyncio
    async def test_no_cancellation_runs_fully(self) -> None:
        """When token is not cancelled, all events are published."""
        w = Worker("redis://localhost", worker_id="w1")
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-no-cancel",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
        )

        from exo.types import TextEvent  # pyright: ignore[reportMissingImports]

        token = CancellationToken()

        async def _fake_stream_gen(*a: object, **kw: object) -> object:
            for i in range(3):
                yield TextEvent(text=f"chunk-{i}", agent_name="agent")

        mock_run = MagicMock()
        mock_run.stream = _fake_stream_gen

        mock_agent = MagicMock()

        with patch("exo.runner.run", mock_run):
            result = await w._run_agent(mock_agent, task, token)

        assert result == "chunk-0chunk-1chunk-2"
        assert w._publisher.publish.call_count == 3


# ---------------------------------------------------------------------------
# Worker._execute_task with cancellation
# ---------------------------------------------------------------------------


class TestWorkerExecuteTaskCancellation:
    @pytest.mark.asyncio
    async def test_cancelled_task_sets_cancelled_status(self) -> None:
        """When the token is cancelled during execution, status becomes CANCELLED."""
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-exec-cancel",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
        )

        async def _fake_run_agent(agent: object, t: TaskPayload, token: CancellationToken) -> str:
            token.cancel()
            return "partial"

        with (
            patch.object(w, "_reconstruct_agent", return_value=MagicMock()),
            patch.object(w, "_run_agent", side_effect=_fake_run_agent),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        # Should set CANCELLED, not COMPLETED
        status_calls = w._store.set_status.call_args_list
        statuses = [c.args[1] for c in status_calls]
        assert TaskStatus.RUNNING in statuses
        assert TaskStatus.CANCELLED in statuses
        assert TaskStatus.COMPLETED not in statuses

        # Should ack the task (remove from queue)
        w._broker.ack.assert_called_once_with("task-exec-cancel")

        # Should NOT increment tasks_processed (cancelled != completed)
        assert w.tasks_processed == 0
        assert w.tasks_failed == 0

    @pytest.mark.asyncio
    async def test_normal_execution_still_completes(self) -> None:
        """Without cancellation, execute_task still completes normally."""
        w = Worker("redis://localhost", worker_id="w1")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()

        task = TaskPayload(
            task_id="task-normal",
            agent_config={"name": "agent", "model": "openai:gpt-4o"},
            input="hello",
        )

        async def _fake_run_agent(agent: object, t: TaskPayload, token: CancellationToken) -> str:
            return "Hello!"

        with (
            patch.object(w, "_reconstruct_agent", return_value=MagicMock()),
            patch.object(w, "_run_agent", side_effect=_fake_run_agent),
            patch.object(w, "_listen_for_cancel", new_callable=AsyncMock),
        ):
            await w._execute_task(task)

        status_calls = w._store.set_status.call_args_list
        statuses = [c.args[1] for c in status_calls]
        assert TaskStatus.RUNNING in statuses
        assert TaskStatus.COMPLETED in statuses
        assert w.tasks_processed == 1
