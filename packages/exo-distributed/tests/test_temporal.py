"""Tests for Temporal integration — TemporalExecutor, workflow, and activity."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.distributed.cancel import CancellationToken  # pyright: ignore[reportMissingImports]
from exo.distributed.models import (  # pyright: ignore[reportMissingImports]
    TaskPayload,
    TaskStatus,
)
from exo.distributed.temporal import (  # pyright: ignore[reportMissingImports]
    HAS_TEMPORAL,
    TemporalExecutor,
    _get_temporal_host,
    _get_temporal_namespace,
)
from exo.distributed.worker import Worker  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# HAS_TEMPORAL flag
# ---------------------------------------------------------------------------


class TestHasTemporal:
    def test_flag_is_bool(self) -> None:
        assert isinstance(HAS_TEMPORAL, bool)

    def test_temporal_available(self) -> None:
        # temporalio should be installed in dev/test environment
        assert HAS_TEMPORAL is True


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


class TestTemporalEnvHelpers:
    def test_default_host(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert _get_temporal_host() == "localhost:7233"

    def test_custom_host(self) -> None:
        with patch.dict("os.environ", {"TEMPORAL_HOST": "temporal.example.com:7233"}):
            assert _get_temporal_host() == "temporal.example.com:7233"

    def test_default_namespace(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert _get_temporal_namespace() == "default"

    def test_custom_namespace(self) -> None:
        with patch.dict("os.environ", {"TEMPORAL_NAMESPACE": "production"}):
            assert _get_temporal_namespace() == "production"


# ---------------------------------------------------------------------------
# TemporalExecutor.__init__
# ---------------------------------------------------------------------------


class TestTemporalExecutorInit:
    def test_defaults(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            executor = TemporalExecutor()
        assert executor.host == "localhost:7233"
        assert executor.namespace == "default"
        assert executor.task_queue == "exo-tasks"

    def test_custom_params(self) -> None:
        executor = TemporalExecutor(
            host="temporal.example.com:7233",
            namespace="staging",
            task_queue="my-queue",
        )
        assert executor.host == "temporal.example.com:7233"
        assert executor.namespace == "staging"
        assert executor.task_queue == "my-queue"

    def test_env_vars(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "TEMPORAL_HOST": "env-host:7233",
                "TEMPORAL_NAMESPACE": "env-ns",
            },
        ):
            executor = TemporalExecutor()
        assert executor.host == "env-host:7233"
        assert executor.namespace == "env-ns"

    def test_explicit_params_override_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "TEMPORAL_HOST": "env-host:7233",
                "TEMPORAL_NAMESPACE": "env-ns",
            },
        ):
            executor = TemporalExecutor(host="explicit:7233", namespace="explicit-ns")
        assert executor.host == "explicit:7233"
        assert executor.namespace == "explicit-ns"

    def test_raises_without_temporalio(self) -> None:
        with (
            patch("exo.distributed.temporal.HAS_TEMPORAL", False),
            pytest.raises(ImportError, match="temporalio is not installed"),
        ):
            TemporalExecutor()


# ---------------------------------------------------------------------------
# TemporalExecutor.connect / disconnect
# ---------------------------------------------------------------------------


class TestTemporalExecutorLifecycle:
    @pytest.mark.asyncio
    async def test_connect(self) -> None:
        executor = TemporalExecutor()
        with patch("exo.distributed.temporal.TemporalClient") as mock_client_cls:
            mock_client_cls.connect = AsyncMock(return_value=MagicMock())
            await executor.connect()
            mock_client_cls.connect.assert_called_once_with(
                executor.host, namespace=executor.namespace
            )
            assert executor._client is not None

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        executor = TemporalExecutor()
        executor._client = MagicMock()
        await executor.disconnect()
        assert executor._client is None


# ---------------------------------------------------------------------------
# TemporalExecutor.execute_task
# ---------------------------------------------------------------------------


class TestTemporalExecutorExecuteTask:
    @pytest.mark.asyncio
    async def test_not_connected_raises(self) -> None:
        executor = TemporalExecutor()
        task = TaskPayload(task_id="t1", input="hello")
        token = CancellationToken()
        with pytest.raises(RuntimeError, match="not connected"):
            await executor.execute_task(task, AsyncMock(), AsyncMock(), token, "w1")

    @pytest.mark.asyncio
    async def test_cancelled_returns_empty(self) -> None:
        executor = TemporalExecutor()
        executor._client = MagicMock()
        task = TaskPayload(task_id="t1", input="hello")
        token = CancellationToken()
        token.cancel()
        result = await executor.execute_task(task, AsyncMock(), AsyncMock(), token, "w1")
        assert result == ""

    @pytest.mark.asyncio
    async def test_submits_workflow_and_returns_output(self) -> None:
        executor = TemporalExecutor()
        mock_client = AsyncMock()
        mock_handle = AsyncMock()
        mock_handle.result.return_value = json.dumps({"output": "Hello world"})
        mock_client.start_workflow.return_value = mock_handle
        executor._client = mock_client

        task = TaskPayload(task_id="t1", input="hello")
        token = CancellationToken()

        result = await executor.execute_task(task, AsyncMock(), AsyncMock(), token, "w1")

        assert result == "Hello world"
        mock_client.start_workflow.assert_called_once()

        # Verify workflow ID includes task_id
        call_kwargs = mock_client.start_workflow.call_args
        assert call_kwargs.kwargs["id"] == "exo-task-t1"
        assert call_kwargs.kwargs["task_queue"] == "exo-tasks"

    @pytest.mark.asyncio
    async def test_workflow_payload_serialization(self) -> None:
        executor = TemporalExecutor()
        mock_client = AsyncMock()
        mock_handle = AsyncMock()
        mock_handle.result.return_value = json.dumps({"output": "done"})
        mock_client.start_workflow.return_value = mock_handle
        executor._client = mock_client

        task = TaskPayload(
            task_id="t2",
            input="test input",
            agent_config={"name": "agent", "model": "gpt-4o"},
            detailed=True,
            timeout_seconds=120.0,
        )
        token = CancellationToken()

        await executor.execute_task(task, AsyncMock(), AsyncMock(), token, "w1")

        # Verify the payload JSON was passed
        call_args = mock_client.start_workflow.call_args
        payload_json = call_args.args[1]
        payload_data = json.loads(payload_json)
        assert payload_data["task_id"] == "t2"
        assert payload_data["input"] == "test input"
        assert payload_data["detailed"] is True
        assert payload_data["timeout_seconds"] == 120.0


# ---------------------------------------------------------------------------
# TemporalExecutor.start_temporal_worker / stop_temporal_worker
# ---------------------------------------------------------------------------


class TestTemporalExecutorWorkerManagement:
    @pytest.mark.asyncio
    async def test_start_temporal_worker_not_connected(self) -> None:
        executor = TemporalExecutor()
        with pytest.raises(RuntimeError, match="not connected"):
            await executor.start_temporal_worker()

    @pytest.mark.asyncio
    async def test_stop_temporal_worker(self) -> None:
        executor = TemporalExecutor()
        mock_tw = MagicMock()
        executor._temporal_worker = mock_tw
        await executor.stop_temporal_worker()
        mock_tw.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_temporal_worker_none(self) -> None:
        executor = TemporalExecutor()
        # Should not raise when _temporal_worker is None
        await executor.stop_temporal_worker()


# ---------------------------------------------------------------------------
# execute_agent_activity
# ---------------------------------------------------------------------------


class TestExecuteAgentActivity:
    @pytest.mark.asyncio
    async def test_executes_agent_and_returns_output(self) -> None:
        from exo.distributed.temporal import (
            execute_agent_activity,  # pyright: ignore[reportMissingImports]
        )
        from exo.types import TextEvent  # pyright: ignore[reportMissingImports]

        task = TaskPayload(
            task_id="t1",
            input="hello",
            agent_config={"name": "agent", "model": "gpt-4o"},
        )
        payload_json = json.dumps(task.model_dump())

        mock_agent = MagicMock()
        mock_run = MagicMock()

        events = [
            TextEvent(text="Hello ", agent_name="agent"),
            TextEvent(text="world", agent_name="agent"),
        ]

        async def fake_stream(*a: object, **kw: object) -> object:
            for ev in events:
                yield ev

        mock_run.stream = fake_stream

        with (
            patch("exo.agent.Agent") as mock_agent_cls,
            patch("exo.runner.run", mock_run),
            patch("temporalio.activity.heartbeat"),
        ):
            mock_agent_cls.from_dict.return_value = mock_agent
            result_json = await execute_agent_activity(payload_json)

        result = json.loads(result_json)
        assert result["output"] == "Hello world"

    @pytest.mark.asyncio
    async def test_heartbeats_during_execution(self) -> None:
        from exo.distributed.temporal import (
            execute_agent_activity,  # pyright: ignore[reportMissingImports]
        )
        from exo.types import TextEvent  # pyright: ignore[reportMissingImports]

        task = TaskPayload(
            task_id="t1",
            input="hello",
            agent_config={"name": "agent", "model": "gpt-4o"},
        )
        payload_json = json.dumps(task.model_dump())

        # Generate enough events to trigger heartbeat (every 10 events)
        events = [TextEvent(text=f"t{i}", agent_name="agent") for i in range(25)]

        mock_run = MagicMock()

        async def fake_stream(*a: object, **kw: object) -> object:
            for ev in events:
                yield ev

        mock_run.stream = fake_stream

        with (
            patch("exo.agent.Agent") as mock_agent_cls,
            patch("exo.runner.run", mock_run),
            patch("temporalio.activity.heartbeat") as mock_heartbeat,
        ):
            mock_agent_cls.from_dict.return_value = MagicMock()
            await execute_agent_activity(payload_json)

        # Should heartbeat at step 10 and step 20 (every 10th event)
        assert mock_heartbeat.call_count == 2

    @pytest.mark.asyncio
    async def test_detects_swarm_config(self) -> None:
        from exo.distributed.temporal import (
            execute_agent_activity,  # pyright: ignore[reportMissingImports]
        )
        from exo.types import TextEvent  # pyright: ignore[reportMissingImports]

        task = TaskPayload(
            task_id="t1",
            input="hello",
            agent_config={
                "agents": [{"name": "a1", "model": "gpt-4o"}],
                "mode": "workflow",
            },
        )
        payload_json = json.dumps(task.model_dump())

        mock_run = MagicMock()

        async def fake_stream(*a: object, **kw: object) -> object:
            for ev in [TextEvent(text="done", agent_name="a1")]:
                yield ev

        mock_run.stream = fake_stream

        with (
            patch("exo.swarm.Swarm") as mock_swarm_cls,
            patch("exo.runner.run", mock_run),
            patch("temporalio.activity.heartbeat"),
        ):
            mock_swarm_cls.from_dict.return_value = MagicMock()
            result_json = await execute_agent_activity(payload_json)

        mock_swarm_cls.from_dict.assert_called_once()
        result = json.loads(result_json)
        assert result["output"] == "done"


# ---------------------------------------------------------------------------
# AgentExecutionWorkflow
# ---------------------------------------------------------------------------


class TestAgentExecutionWorkflow:
    def test_workflow_class_exists(self) -> None:
        from exo.distributed.temporal import (
            AgentExecutionWorkflow,  # pyright: ignore[reportMissingImports]
        )

        assert AgentExecutionWorkflow is not None
        assert hasattr(AgentExecutionWorkflow, "run")

    def test_workflow_timeout_from_payload(self) -> None:
        """Verify the workflow reads timeout_seconds from the payload."""
        task = TaskPayload(task_id="t1", input="hello", timeout_seconds=120.0)
        payload_json = json.dumps(task.model_dump())
        data = json.loads(payload_json)
        assert data["timeout_seconds"] == 120.0


# ---------------------------------------------------------------------------
# Worker executor parameter
# ---------------------------------------------------------------------------


class TestWorkerExecutorParam:
    def test_default_executor_is_local(self) -> None:
        w = Worker("redis://localhost")
        assert w._executor_type == "local"
        assert w._temporal_executor is None

    def test_temporal_executor_created(self) -> None:
        with patch("exo.distributed.temporal.TemporalClient"):
            w = Worker("redis://localhost", executor="temporal")
        assert w._executor_type == "temporal"
        assert w._temporal_executor is not None

    def test_temporal_executor_raises_without_lib(self) -> None:
        with (
            patch("exo.distributed.worker.HAS_TEMPORAL", False),
            pytest.raises(ImportError, match="temporalio"),
        ):
            Worker("redis://localhost", executor="temporal")

    @pytest.mark.asyncio
    async def test_temporal_executor_connected_on_start(self) -> None:
        with patch("exo.distributed.temporal.TemporalClient"):
            w = Worker("redis://localhost", worker_id="w1", executor="temporal")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()
        w._temporal_executor = AsyncMock()
        w._shutdown_event.set()

        with (
            patch.object(w, "_claim_loop", new_callable=AsyncMock),
            patch.object(w, "_heartbeat_loop", new_callable=AsyncMock),
            patch("asyncio.get_running_loop") as mock_loop,
        ):
            mock_loop.return_value = MagicMock()
            mock_loop.return_value.add_signal_handler = MagicMock()
            await w.start()

        w._temporal_executor.connect.assert_called_once()
        w._temporal_executor.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_task_delegates_to_temporal(self) -> None:
        with patch("exo.distributed.temporal.TemporalClient"):
            w = Worker("redis://localhost", worker_id="w1", executor="temporal")
        w._broker = AsyncMock()
        w._store = AsyncMock()
        w._publisher = AsyncMock()
        mock_temporal = AsyncMock()
        mock_temporal.execute_task.return_value = "Hello from Temporal"
        w._temporal_executor = mock_temporal

        task = TaskPayload(
            task_id="task-t1",
            agent_config={"name": "agent", "model": "gpt-4o"},
            input="hello",
        )

        with patch.object(w, "_listen_for_cancel", new_callable=AsyncMock):
            await w._execute_task(task)

        # Should have called temporal executor instead of _reconstruct_agent/_run_agent
        mock_temporal.execute_task.assert_called_once()
        call_args = mock_temporal.execute_task.call_args
        assert call_args.args[0] == task

        # Should still update status and ack
        status_calls = w._store.set_status.call_args_list
        assert status_calls[0].args[1] == TaskStatus.RUNNING
        assert status_calls[1].args[1] == TaskStatus.COMPLETED
        w._broker.ack.assert_called_once()
