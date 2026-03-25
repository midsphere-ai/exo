"""Optional Temporal integration for durable workflow execution.

When ``temporalio`` is installed, provides :class:`TemporalExecutor` as an
alternative execution backend for the distributed worker.  Temporal wraps
agent execution in a durable workflow with heartbeating activities, so tasks
survive worker crashes and can be retried with full state recovery.

Install the optional dependency group::

    pip install exo-distributed[temporal]
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

from exo.distributed.cancel import CancellationToken  # pyright: ignore[reportMissingImports]
from exo.distributed.events import EventPublisher  # pyright: ignore[reportMissingImports]
from exo.distributed.models import TaskPayload  # pyright: ignore[reportMissingImports]
from exo.distributed.store import TaskStore  # pyright: ignore[reportMissingImports]

try:
    from temporalio import activity, workflow  # pyright: ignore[reportMissingImports]
    from temporalio.client import Client as TemporalClient  # pyright: ignore[reportMissingImports]
    from temporalio.worker import Worker as TemporalWorker  # pyright: ignore[reportMissingImports]

    HAS_TEMPORAL = True
except ImportError:
    HAS_TEMPORAL = False
    TemporalClient = None  # type: ignore[assignment,misc]
    TemporalWorker = None  # type: ignore[assignment,misc]

# Default Temporal connection settings
_DEFAULT_TEMPORAL_HOST = "localhost:7233"
_DEFAULT_TEMPORAL_NAMESPACE = "default"
_DEFAULT_TASK_QUEUE = "exo-tasks"


def _get_temporal_host() -> str:
    """Resolve Temporal host from environment or default."""
    return os.environ.get("TEMPORAL_HOST", _DEFAULT_TEMPORAL_HOST)


def _get_temporal_namespace() -> str:
    """Resolve Temporal namespace from environment or default."""
    return os.environ.get("TEMPORAL_NAMESPACE", _DEFAULT_TEMPORAL_NAMESPACE)


if HAS_TEMPORAL:
    from datetime import timedelta

    @activity.defn
    async def execute_agent_activity(payload_json: str) -> str:
        """Temporal activity that executes an agent from a serialized TaskPayload.

        Heartbeats are sent periodically so that Temporal can detect worker
        failures and reassign the activity.  Returns the collected text
        output as a JSON string.
        """
        from exo.agent import Agent  # pyright: ignore[reportMissingImports]
        from exo.runner import run  # pyright: ignore[reportMissingImports]
        from exo.swarm import Swarm  # pyright: ignore[reportMissingImports]
        from exo.types import TextEvent  # pyright: ignore[reportMissingImports]

        payload_data = json.loads(payload_json)
        task = TaskPayload(**payload_data)

        # Reconstruct agent
        agent_config = task.agent_config
        if "agents" in agent_config:
            agent = Swarm.from_dict(agent_config)
        else:
            agent = Agent.from_dict(agent_config)

        # Stream and collect output
        text_parts: list[str] = []
        step = 0
        async for event in run.stream(
            agent,
            task.input,
            messages=None,
            detailed=task.detailed,
        ):
            if isinstance(event, TextEvent):
                text_parts.append(event.text)

            # Heartbeat every few events so Temporal knows we're alive
            step += 1
            if step % 10 == 0:
                activity.heartbeat(f"step:{step}")

        return json.dumps({"output": "".join(text_parts)})

    @workflow.defn
    class AgentExecutionWorkflow:
        """Temporal workflow that wraps agent execution in a durable activity.

        Receives a :class:`TaskPayload` as JSON and delegates to
        :func:`execute_agent_activity` for actual execution.
        """

        @workflow.run
        async def run(self, payload_json: str) -> str:
            """Execute the agent activity with the given task payload."""
            timeout_data = json.loads(payload_json)
            timeout_seconds = timeout_data.get("timeout_seconds", 300.0)

            return await workflow.execute_activity(
                execute_agent_activity,
                payload_json,
                start_to_close_timeout=timedelta(seconds=timeout_seconds),
                heartbeat_timeout=timedelta(seconds=30),
            )


class TemporalExecutor:
    """Alternative execution backend using Temporal for durable workflows.

    Instead of executing agents directly, the worker submits tasks as
    Temporal workflows.  Temporal handles retries, timeouts, and state
    recovery on worker failure.

    Args:
        host: Temporal server address (default: ``TEMPORAL_HOST`` env var or ``localhost:7233``).
        namespace: Temporal namespace (default: ``TEMPORAL_NAMESPACE`` env var or ``default``).
        task_queue: Temporal task queue name (default: ``exo-tasks``).
    """

    def __init__(
        self,
        *,
        host: str | None = None,
        namespace: str | None = None,
        task_queue: str = _DEFAULT_TASK_QUEUE,
    ) -> None:
        if not HAS_TEMPORAL:
            msg = (
                "temporalio is not installed. "
                "Install it with: pip install exo-distributed[temporal]"
            )
            raise ImportError(msg)

        self._host = host or _get_temporal_host()
        self._namespace = namespace or _get_temporal_namespace()
        self._task_queue = task_queue
        self._client: Any | None = None
        self._temporal_worker: Any | None = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def task_queue(self) -> str:
        return self._task_queue

    async def connect(self) -> None:
        """Connect to the Temporal server."""
        logger.debug(
            "TemporalExecutor connecting to %s (namespace=%s)", self._host, self._namespace
        )
        self._client = await TemporalClient.connect(  # pyright: ignore[reportOptionalMemberAccess]
            self._host, namespace=self._namespace
        )
        logger.info("TemporalExecutor connected to %s", self._host)

    async def disconnect(self) -> None:
        """Clean up Temporal resources."""
        self._client = None
        self._temporal_worker = None
        logger.debug("TemporalExecutor disconnected")

    async def execute_task(
        self,
        task: TaskPayload,
        store: TaskStore,
        publisher: EventPublisher,
        token: CancellationToken,
        worker_id: str,
    ) -> str:
        """Submit a task as a Temporal workflow and wait for the result.

        Args:
            task: The task payload to execute.
            store: Task status store for state updates.
            publisher: Event publisher (events published by the activity).
            token: Cancellation token (checked before starting).
            worker_id: ID of the submitting worker.

        Returns:
            The text output from agent execution.
        """
        if self._client is None:
            msg = "TemporalExecutor is not connected. Call connect() first."
            raise RuntimeError(msg)

        if token.cancelled:
            return ""

        # Serialize the task payload to JSON for Temporal
        payload_json = json.dumps(task.model_dump())

        # Start the workflow
        workflow_id = f"exo-task-{task.task_id}"
        logger.info("TemporalExecutor starting workflow %s (task=%s)", workflow_id, task.task_id)
        handle = await self._client.start_workflow(
            AgentExecutionWorkflow.run,
            payload_json,
            id=workflow_id,
            task_queue=self._task_queue,
        )

        # Wait for result
        result_json = await handle.result()
        result_data = json.loads(result_json)
        logger.debug("TemporalExecutor workflow %s completed", workflow_id)
        return result_data.get("output", "")

    async def start_temporal_worker(self) -> None:
        """Start a Temporal worker that processes agent execution workflows.

        This registers the workflow and activity with Temporal and runs
        until stopped.
        """
        if self._client is None:
            msg = "TemporalExecutor is not connected. Call connect() first."
            raise RuntimeError(msg)

        self._temporal_worker = TemporalWorker(  # pyright: ignore[reportOptionalCall]
            self._client,
            task_queue=self._task_queue,
            workflows=[AgentExecutionWorkflow],
            activities=[execute_agent_activity],
        )
        await self._temporal_worker.run()  # pyright: ignore[reportOptionalMemberAccess]

    async def stop_temporal_worker(self) -> None:
        """Request graceful shutdown of the Temporal worker."""
        if self._temporal_worker is not None:
            self._temporal_worker.shutdown()
