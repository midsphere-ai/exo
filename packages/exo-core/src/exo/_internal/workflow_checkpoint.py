"""Workflow state checkpointing for resumable workflow execution.

:class:`WorkflowCheckpoint` captures a snapshot of workflow state before
each node so that failed workflows can resume from the last checkpoint.

:class:`WorkflowCheckpointStore` provides an in-memory store for
checkpoints created during a workflow run.

Usage::

    store = WorkflowCheckpointStore()
    cp = WorkflowCheckpoint(
        node_name="agent_b",
        state={"input": "hello", "agent_a": "result_a"},
        completed_nodes=["agent_a"],
        timestamp=1234567890.0,
    )
    store.save(cp)
    assert store.latest() == cp
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkflowCheckpoint:
    """Immutable snapshot of workflow state before a node executes.

    Args:
        node_name: Name of the node about to execute when the checkpoint
            was taken.
        state: Full workflow state dict (agent outputs and metadata).
        completed_nodes: List of node names that have already completed.
        timestamp: Unix timestamp when the checkpoint was created.
    """

    node_name: str
    state: dict[str, Any]
    completed_nodes: list[str]
    timestamp: float = field(default_factory=time.time)


class WorkflowCheckpointStore:
    """In-memory store for workflow checkpoints.

    Stores checkpoints in insertion order and provides access to
    the latest checkpoint for resumption.
    """

    def __init__(self) -> None:
        self._checkpoints: list[WorkflowCheckpoint] = []

    def save(self, checkpoint: WorkflowCheckpoint) -> None:
        """Save a checkpoint to the store.

        Args:
            checkpoint: The checkpoint to save.
        """
        self._checkpoints.append(checkpoint)

    def latest(self) -> WorkflowCheckpoint | None:
        """Return the most recently saved checkpoint, or ``None``.

        Returns:
            The latest checkpoint, or ``None`` if no checkpoints exist.
        """
        return self._checkpoints[-1] if self._checkpoints else None

    def list_all(self) -> list[WorkflowCheckpoint]:
        """Return all saved checkpoints in insertion order.

        Returns:
            List of all checkpoints.
        """
        return list(self._checkpoints)
