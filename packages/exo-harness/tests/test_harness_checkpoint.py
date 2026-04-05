"""Tests for harness checkpoint persistence."""

from __future__ import annotations

from typing import Any

from exo.harness.checkpoint import CheckpointAdapter
from exo.harness.types import HarnessCheckpoint

# ---------------------------------------------------------------------------
# In-memory MemoryStore mock
# ---------------------------------------------------------------------------


class _MockMemoryStore:
    """Minimal MemoryStore implementation for testing."""

    def __init__(self) -> None:
        self._items: list[Any] = []

    async def add(self, item: Any) -> None:
        self._items.append(item)

    async def get(self, item_id: str) -> Any:
        for item in self._items:
            if item.id == item_id:
                return item
        return None

    async def search(
        self,
        *,
        query: str = "",
        metadata: Any = None,
        memory_type: str | None = None,
        category: Any = None,
        status: Any = None,
        limit: int = 10,
    ) -> list[Any]:
        results = []
        for item in self._items:
            if memory_type and item.memory_type != memory_type:
                continue
            if metadata and hasattr(metadata, "agent_id") and metadata.agent_id:
                item_agent_id = getattr(item.metadata, "agent_id", None)
                if item_agent_id != metadata.agent_id:
                    continue
            results.append(item)
        return results[-limit:]

    async def clear(self, *, metadata: Any = None) -> int:
        count = len(self._items)
        self._items.clear()
        return count


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCheckpointAdapter:
    async def test_save_and_load(self) -> None:
        store = _MockMemoryStore()
        adapter = CheckpointAdapter(store, "my_harness")

        checkpoint = HarnessCheckpoint(
            harness_name="my_harness",
            session_state={"step": 2, "route": "billing"},
            completed_agents=["classifier", "billing_agent"],
            pending_agent="summary_agent",
        )
        await adapter.save(checkpoint)

        loaded = await adapter.load_latest()

        assert loaded is not None
        assert loaded.harness_name == "my_harness"
        assert loaded.session_state == {"step": 2, "route": "billing"}
        assert loaded.completed_agents == ["classifier", "billing_agent"]
        assert loaded.pending_agent == "summary_agent"

    async def test_load_empty_store(self) -> None:
        store = _MockMemoryStore()
        adapter = CheckpointAdapter(store, "my_harness")

        loaded = await adapter.load_latest()

        assert loaded is None

    async def test_multiple_checkpoints_returns_latest(self) -> None:
        store = _MockMemoryStore()
        adapter = CheckpointAdapter(store, "h")

        cp1 = HarnessCheckpoint(
            harness_name="h",
            session_state={"step": 1},
            completed_agents=["a"],
            timestamp=100.0,
        )
        cp2 = HarnessCheckpoint(
            harness_name="h",
            session_state={"step": 2},
            completed_agents=["a", "b"],
            timestamp=200.0,
        )
        await adapter.save(cp1)
        await adapter.save(cp2)

        loaded = await adapter.load_latest()

        assert loaded is not None
        # search returns last item (most recent)
        assert loaded.session_state["step"] == 2

    async def test_scoped_to_harness_name(self) -> None:
        store = _MockMemoryStore()

        adapter_a = CheckpointAdapter(store, "harness_a")
        adapter_b = CheckpointAdapter(store, "harness_b")

        await adapter_a.save(
            HarnessCheckpoint(
                harness_name="harness_a",
                session_state={"x": 1},
                completed_agents=[],
            )
        )
        await adapter_b.save(
            HarnessCheckpoint(
                harness_name="harness_b",
                session_state={"y": 2},
                completed_agents=[],
            )
        )

        loaded_a = await adapter_a.load_latest()
        loaded_b = await adapter_b.load_latest()

        assert loaded_a is not None
        assert loaded_a.session_state == {"x": 1}
        assert loaded_b is not None
        assert loaded_b.session_state == {"y": 2}

    async def test_checkpoint_preserves_metadata(self) -> None:
        store = _MockMemoryStore()
        adapter = CheckpointAdapter(store, "h")

        cp = HarnessCheckpoint(
            harness_name="h",
            session_state={},
            completed_agents=[],
            metadata={"version": "1.0", "reason": "manual"},
        )
        await adapter.save(cp)

        loaded = await adapter.load_latest()

        assert loaded is not None
        assert loaded.metadata == {"version": "1.0", "reason": "manual"}

    async def test_checkpoint_preserves_messages(self) -> None:
        store = _MockMemoryStore()
        adapter = CheckpointAdapter(store, "h")

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        cp = HarnessCheckpoint(
            harness_name="h",
            session_state={},
            completed_agents=[],
            messages=messages,
        )
        await adapter.save(cp)

        loaded = await adapter.load_latest()

        assert loaded is not None
        assert loaded.messages == messages
