"""Distributed task metrics recording helpers.

Records metrics for task submission, completion, failure, and cancellation
using the ``exo.observability.metrics`` infrastructure (both OTel and
in-memory fallback).
"""

from __future__ import annotations

from typing import Any

from exo.observability.metrics import (  # pyright: ignore[reportMissingImports]
    HAS_OTEL,
    _collector,
    _get_meter,
)
from exo.observability.semconv import (  # pyright: ignore[reportMissingImports]
    DIST_QUEUE_NAME,
    DIST_TASK_ID,
    DIST_TASK_STATUS,
    DIST_WORKER_ID,
    METRIC_DIST_QUEUE_DEPTH,
    METRIC_DIST_TASK_DURATION,
    METRIC_DIST_TASK_WAIT_TIME,
    METRIC_DIST_TASKS_CANCELLED,
    METRIC_DIST_TASKS_COMPLETED,
    METRIC_DIST_TASKS_FAILED,
    METRIC_DIST_TASKS_SUBMITTED,
)


def _build_attributes(
    *,
    task_id: str = "",
    worker_id: str = "",
    queue_name: str = "",
    status: str = "",
) -> dict[str, str]:
    """Build attribute dict for distributed task metrics."""
    attrs: dict[str, str] = {}
    if task_id:
        attrs[DIST_TASK_ID] = task_id
    if worker_id:
        attrs[DIST_WORKER_ID] = worker_id
    if queue_name:
        attrs[DIST_QUEUE_NAME] = queue_name
    if status:
        attrs[DIST_TASK_STATUS] = status
    return attrs


def record_task_submitted(
    *,
    task_id: str = "",
    queue_name: str = "",
) -> None:
    """Record that a task was submitted to the distributed queue."""
    attrs = _build_attributes(task_id=task_id, queue_name=queue_name)
    if HAS_OTEL:
        meter = _get_meter()
        meter.create_counter(
            name=METRIC_DIST_TASKS_SUBMITTED,
            unit="1",
            description="Number of distributed tasks submitted",
        ).add(1, attrs)
    else:
        _collector.add_counter(METRIC_DIST_TASKS_SUBMITTED, 1.0, attrs)


def record_task_completed(
    *,
    task_id: str = "",
    worker_id: str = "",
    duration: float = 0.0,
    wait_time: float = 0.0,
) -> None:
    """Record that a task completed successfully.

    Args:
        task_id: The task identifier.
        worker_id: The worker that executed the task.
        duration: Total execution duration in seconds.
        wait_time: Time the task waited in the queue in seconds.
    """
    attrs = _build_attributes(task_id=task_id, worker_id=worker_id)
    if HAS_OTEL:
        meter = _get_meter()
        meter.create_counter(
            name=METRIC_DIST_TASKS_COMPLETED,
            unit="1",
            description="Number of distributed tasks completed",
        ).add(1, attrs)
        if duration > 0:
            meter.create_histogram(
                name=METRIC_DIST_TASK_DURATION,
                unit="s",
                description="Distributed task execution duration",
            ).record(duration, attrs)
        if wait_time > 0:
            meter.create_histogram(
                name=METRIC_DIST_TASK_WAIT_TIME,
                unit="s",
                description="Time task waited in queue before execution",
            ).record(wait_time, attrs)
    else:
        _collector.add_counter(METRIC_DIST_TASKS_COMPLETED, 1.0, attrs)
        if duration > 0:
            _collector.record_histogram(METRIC_DIST_TASK_DURATION, duration, attrs)
        if wait_time > 0:
            _collector.record_histogram(METRIC_DIST_TASK_WAIT_TIME, wait_time, attrs)


def record_task_failed(
    *,
    task_id: str = "",
    worker_id: str = "",
    duration: float = 0.0,
) -> None:
    """Record that a task failed.

    Args:
        task_id: The task identifier.
        worker_id: The worker that attempted the task.
        duration: Execution duration before failure in seconds.
    """
    attrs = _build_attributes(task_id=task_id, worker_id=worker_id)
    if HAS_OTEL:
        meter = _get_meter()
        meter.create_counter(
            name=METRIC_DIST_TASKS_FAILED,
            unit="1",
            description="Number of distributed tasks failed",
        ).add(1, attrs)
        if duration > 0:
            meter.create_histogram(
                name=METRIC_DIST_TASK_DURATION,
                unit="s",
                description="Distributed task execution duration",
            ).record(duration, attrs)
    else:
        _collector.add_counter(METRIC_DIST_TASKS_FAILED, 1.0, attrs)
        if duration > 0:
            _collector.record_histogram(METRIC_DIST_TASK_DURATION, duration, attrs)


def record_task_cancelled(
    *,
    task_id: str = "",
    worker_id: str = "",
) -> None:
    """Record that a task was cancelled."""
    attrs = _build_attributes(task_id=task_id, worker_id=worker_id)
    if HAS_OTEL:
        meter = _get_meter()
        meter.create_counter(
            name=METRIC_DIST_TASKS_CANCELLED,
            unit="1",
            description="Number of distributed tasks cancelled",
        ).add(1, attrs)
    else:
        _collector.add_counter(METRIC_DIST_TASKS_CANCELLED, 1.0, attrs)


def record_queue_depth(
    *,
    depth: int,
    queue_name: str = "",
) -> None:
    """Record the current queue depth."""
    attrs: dict[str, Any] = _build_attributes(queue_name=queue_name)
    if HAS_OTEL:
        meter = _get_meter()
        meter.create_up_down_counter(
            name=METRIC_DIST_QUEUE_DEPTH,
            unit="1",
            description="Current distributed task queue depth",
        ).add(depth, attrs)
    else:
        _collector.set_gauge(METRIC_DIST_QUEUE_DEPTH, float(depth), attrs)
