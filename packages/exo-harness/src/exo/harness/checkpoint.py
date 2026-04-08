"""Checkpoint persistence via existing MemoryStore backends.

:class:`CheckpointAdapter` serializes :class:`HarnessCheckpoint`
instances to :class:`MemoryItem` objects and stores them in any
``MemoryStore`` backend (SQLite, Postgres, etc.).

Usage::

    from exo.memory.backends.sqlite import SQLiteMemoryStore

    async with SQLiteMemoryStore() as store:
        adapter = CheckpointAdapter(store, "my_harness")
        await adapter.save(checkpoint)
        restored = await adapter.load_latest()
"""

from __future__ import annotations

import json
from typing import Any

from exo.harness.types import HarnessCheckpoint


class CheckpointAdapter:
    """Adapts a ``MemoryStore`` for harness checkpoint persistence.

    Serializes checkpoints as JSON in ``MemoryItem.content`` with
    ``memory_type="harness_checkpoint"``.

    Args:
        store: Any object implementing the ``MemoryStore`` protocol.
        harness_name: Used to scope checkpoints to a specific harness.
    """

    def __init__(self, store: Any, harness_name: str) -> None:
        self._store = store
        self._harness_name = harness_name

    async def save(self, checkpoint: HarnessCheckpoint) -> None:
        """Serialize and persist a checkpoint.

        Args:
            checkpoint: The checkpoint to save.
        """
        from exo.memory.base import (  # pyright: ignore[reportMissingImports]
            MemoryItem,
            MemoryMetadata,
        )

        payload = json.dumps(
            {
                "harness_name": checkpoint.harness_name,
                "session_state": checkpoint.session_state,
                "completed_agents": checkpoint.completed_agents,
                "pending_agent": checkpoint.pending_agent,
                "pending_agents": checkpoint.pending_agents,
                "messages": checkpoint.messages,
                "timestamp": checkpoint.timestamp,
                "metadata": checkpoint.metadata,
            }
        )
        item = MemoryItem(
            content=payload,
            memory_type="harness_checkpoint",
            metadata=MemoryMetadata(agent_id=self._harness_name),
        )
        await self._store.add(item)

    async def load_latest(self) -> HarnessCheckpoint | None:
        """Load the most recent checkpoint for this harness.

        Returns:
            The latest checkpoint, or ``None`` if none exists.
        """
        from exo.memory.base import MemoryMetadata  # pyright: ignore[reportMissingImports]

        items = await self._store.search(
            memory_type="harness_checkpoint",
            metadata=MemoryMetadata(agent_id=self._harness_name),
            limit=1,
        )
        if not items:
            return None
        data: dict[str, Any] = json.loads(items[0].content)
        # Backward compat: migrate old single pending_agent to list
        pending_agents = data.get("pending_agents", [])
        if not pending_agents and data.get("pending_agent"):
            pending_agents = [data["pending_agent"]]
        return HarnessCheckpoint(
            harness_name=data["harness_name"],
            session_state=data["session_state"],
            completed_agents=data["completed_agents"],
            pending_agent=data.get("pending_agent"),
            pending_agents=pending_agents,
            messages=data.get("messages", []),
            timestamp=data.get("timestamp", 0.0),
            metadata=data.get("metadata", {}),
        )
