"""Tests for workflow state checkpointing (US-030).

Covers:
- WorkflowCheckpoint creation and immutability
- WorkflowCheckpointStore save/latest/list_all
- Swarm creates checkpoints before each node when enabled
- Swarm.resume() skips completed nodes and finishes the workflow
- Default behavior unchanged when checkpoint=False
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from exo._internal.workflow_checkpoint import (
    WorkflowCheckpoint,
    WorkflowCheckpointStore,
)
from exo.agent import Agent
from exo.swarm import Swarm, SwarmError
from exo.types import AgentOutput, Usage

# ---------------------------------------------------------------------------
# Shared fixtures: mock LLM provider
# ---------------------------------------------------------------------------

_DEFAULT_USAGE = Usage(input_tokens=10, output_tokens=5, total_tokens=15)


def _make_provider(responses: list[AgentOutput]) -> Any:
    """Create a mock provider returning pre-defined AgentOutput values."""
    call_count = 0

    async def complete(messages: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1

        class FakeResponse:
            content = resp.text
            tool_calls = resp.tool_calls
            usage = resp.usage
            reasoning_content = ""
            thought_signatures: list[bytes] = []

        return FakeResponse()

    class Provider:
        pass

    p = Provider()
    p.complete = complete  # type: ignore[attr-defined]
    return p


def _text(text: str) -> AgentOutput:
    """Helper: text-only AgentOutput with default usage."""
    return AgentOutput(text=text, usage=_DEFAULT_USAGE)


# ---------------------------------------------------------------------------
# WorkflowCheckpoint dataclass tests
# ---------------------------------------------------------------------------


class TestWorkflowCheckpoint:
    """WorkflowCheckpoint frozen dataclass."""

    def test_creation(self) -> None:
        cp = WorkflowCheckpoint(
            node_name="agent_b",
            state={"input": "hello", "agent_a": "result_a"},
            completed_nodes=["agent_a"],
            timestamp=1234567890.0,
        )
        assert cp.node_name == "agent_b"
        assert cp.state == {"input": "hello", "agent_a": "result_a"}
        assert cp.completed_nodes == ["agent_a"]
        assert cp.timestamp == 1234567890.0

    def test_frozen(self) -> None:
        cp = WorkflowCheckpoint(
            node_name="a",
            state={},
            completed_nodes=[],
        )
        with pytest.raises(AttributeError):
            cp.node_name = "b"  # type: ignore[misc]

    def test_default_timestamp(self) -> None:
        before = time.time()
        cp = WorkflowCheckpoint(
            node_name="a",
            state={},
            completed_nodes=[],
        )
        after = time.time()
        assert before <= cp.timestamp <= after


# ---------------------------------------------------------------------------
# WorkflowCheckpointStore tests
# ---------------------------------------------------------------------------


class TestWorkflowCheckpointStore:
    """In-memory checkpoint store."""

    def test_empty_store(self) -> None:
        store = WorkflowCheckpointStore()
        assert store.latest() is None
        assert store.list_all() == []

    def test_save_and_latest(self) -> None:
        store = WorkflowCheckpointStore()
        cp1 = WorkflowCheckpoint(node_name="a", state={}, completed_nodes=[])
        cp2 = WorkflowCheckpoint(node_name="b", state={"a": "x"}, completed_nodes=["a"])

        store.save(cp1)
        assert store.latest() == cp1

        store.save(cp2)
        assert store.latest() == cp2

    def test_list_all_preserves_order(self) -> None:
        store = WorkflowCheckpointStore()
        cps = [
            WorkflowCheckpoint(node_name="a", state={}, completed_nodes=[]),
            WorkflowCheckpoint(node_name="b", state={}, completed_nodes=["a"]),
            WorkflowCheckpoint(node_name="c", state={}, completed_nodes=["a", "b"]),
        ]
        for cp in cps:
            store.save(cp)

        assert store.list_all() == cps

    def test_list_all_returns_copy(self) -> None:
        store = WorkflowCheckpointStore()
        cp = WorkflowCheckpoint(node_name="a", state={}, completed_nodes=[])
        store.save(cp)

        result = store.list_all()
        result.clear()
        assert len(store.list_all()) == 1


# ---------------------------------------------------------------------------
# Swarm checkpoint creation tests
# ---------------------------------------------------------------------------


class TestSwarmCheckpointCreation:
    """Swarm creates checkpoints before each node when enabled."""

    async def test_checkpoints_created_for_each_node(self) -> None:
        """Three-agent pipeline creates three checkpoints."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")

        swarm = Swarm(agents=[a, b, c], flow="a >> b >> c")
        store = WorkflowCheckpointStore()
        provider = _make_provider([_text("out-a"), _text("out-b"), _text("out-c")])

        result = await swarm.run("input", provider=provider, checkpoint=store)

        assert result.output == "out-c"
        checkpoints = store.list_all()
        assert len(checkpoints) == 3

        # First checkpoint: before agent "a", no completed nodes
        assert checkpoints[0].node_name == "a"
        assert checkpoints[0].completed_nodes == []

        # Second checkpoint: before agent "b", "a" completed
        assert checkpoints[1].node_name == "b"
        assert checkpoints[1].completed_nodes == ["a"]
        assert checkpoints[1].state.get("a") == "out-a"

        # Third checkpoint: before agent "c", "a" and "b" completed
        assert checkpoints[2].node_name == "c"
        assert checkpoints[2].completed_nodes == ["a", "b"]
        assert checkpoints[2].state.get("b") == "out-b"

    async def test_checkpoint_true_creates_store(self) -> None:
        """checkpoint=True creates an internal store (no error)."""
        a = Agent(name="a")
        swarm = Swarm(agents=[a])
        provider = _make_provider([_text("ok")])

        result = await swarm.run("go", provider=provider, checkpoint=True)
        assert result.output == "ok"

    async def test_no_checkpoints_by_default(self) -> None:
        """Default checkpoint=False does not alter behavior."""
        a = Agent(name="a")
        b = Agent(name="b")

        swarm = Swarm(agents=[a, b], flow="a >> b")
        provider = _make_provider([_text("out-a"), _text("out-b")])

        result = await swarm.run("input", provider=provider)
        assert result.output == "out-b"


# ---------------------------------------------------------------------------
# Swarm.resume() tests
# ---------------------------------------------------------------------------


class TestSwarmResume:
    """Swarm.resume() resumes from latest checkpoint, skipping completed nodes."""

    async def test_resume_skips_completed(self) -> None:
        """Resume from checkpoint after agent 'a' skips 'a' and only runs 'b', 'c'."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")

        swarm = Swarm(agents=[a, b, c], flow="a >> b >> c")

        # Simulate a previous run that completed "a" then failed
        store = WorkflowCheckpointStore()
        store.save(
            WorkflowCheckpoint(
                node_name="b",
                state={"input": "start", "a": "out-a"},
                completed_nodes=["a"],
            )
        )

        # Provider only needs to handle b and c (a is skipped)
        provider = _make_provider([_text("out-b"), _text("out-c")])

        result = await swarm.resume(store, provider=provider)

        assert result.output == "out-c"

    async def test_resume_restores_state(self) -> None:
        """Resumed run has correct current_input from checkpoint state."""
        a = Agent(name="a")
        b = Agent(name="b")

        swarm = Swarm(agents=[a, b], flow="a >> b")

        store = WorkflowCheckpointStore()
        store.save(
            WorkflowCheckpoint(
                node_name="b",
                state={"input": "start", "a": "from-agent-a"},
                completed_nodes=["a"],
            )
        )

        # Only agent b runs, and it receives "from-agent-a" as input
        provider = _make_provider([_text("final")])

        result = await swarm.resume(store, provider=provider)

        assert result.output == "final"

    async def test_resume_empty_store_raises(self) -> None:
        """resume() with empty store raises SwarmError."""
        a = Agent(name="a")
        swarm = Swarm(agents=[a])
        store = WorkflowCheckpointStore()

        with pytest.raises(SwarmError, match="No checkpoints available"):
            await swarm.resume(store, provider=_make_provider([_text("x")]))

    async def test_resume_non_workflow_raises(self) -> None:
        """resume() in handoff mode raises SwarmError."""
        a = Agent(name="a")
        b = Agent(name="b", handoffs=[])

        swarm = Swarm(agents=[a, b], mode="handoff")
        store = WorkflowCheckpointStore()
        store.save(WorkflowCheckpoint(node_name="a", state={}, completed_nodes=[]))

        with pytest.raises(SwarmError, match="only supported in workflow mode"):
            await swarm.resume(store, provider=_make_provider([_text("x")]))

    async def test_resume_with_checkpoint_enabled(self) -> None:
        """Resume can create new checkpoints during the resumed run."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")

        swarm = Swarm(agents=[a, b, c], flow="a >> b >> c")

        # Checkpoint from previous run: "a" completed
        old_store = WorkflowCheckpointStore()
        old_store.save(
            WorkflowCheckpoint(
                node_name="b",
                state={"input": "start", "a": "out-a"},
                completed_nodes=["a"],
            )
        )

        new_store = WorkflowCheckpointStore()
        provider = _make_provider([_text("out-b"), _text("out-c")])

        result = await swarm.resume(
            old_store,
            provider=provider,
            checkpoint=new_store,
        )

        assert result.output == "out-c"
        # New store should have checkpoints for b and c (a was skipped)
        new_cps = new_store.list_all()
        assert len(new_cps) == 2
        assert new_cps[0].node_name == "b"
        assert new_cps[1].node_name == "c"


# ---------------------------------------------------------------------------
# Partial execution / end-to-end test
# ---------------------------------------------------------------------------


class TestCheckpointPartialExecution:
    """Full checkpoint→fail→resume cycle."""

    async def test_checkpoint_and_resume_cycle(self) -> None:
        """Run a workflow that 'fails' mid-way, then resume from checkpoint."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")

        swarm = Swarm(agents=[a, b, c], flow="a >> b >> c")

        # First run: succeeds through all agents, collecting checkpoints
        store = WorkflowCheckpointStore()
        provider1 = _make_provider([_text("out-a"), _text("out-b"), _text("out-c")])

        result1 = await swarm.run("start", provider=provider1, checkpoint=store)
        assert result1.output == "out-c"
        assert len(store.list_all()) == 3

        # Simulate failure after "b" by using the checkpoint at "c"
        # (checkpoint before "c" means a+b completed)
        cp_before_c = store.list_all()[2]
        assert cp_before_c.node_name == "c"
        assert cp_before_c.completed_nodes == ["a", "b"]

        # Create a new store with just this checkpoint for resume
        resume_store = WorkflowCheckpointStore()
        resume_store.save(cp_before_c)

        # Resume only needs provider for "c"
        provider2 = _make_provider([_text("out-c-retry")])
        result2 = await swarm.resume(resume_store, provider=provider2)

        assert result2.output == "out-c-retry"


# ---------------------------------------------------------------------------
# Public API export tests
# ---------------------------------------------------------------------------


class TestCheckpointPublicAPI:
    """Verify checkpoint types are importable from exo."""

    def test_checkpoint_importable(self) -> None:
        from exo import WorkflowCheckpoint as WorkflowCheckpointImport

        assert WorkflowCheckpointImport is WorkflowCheckpoint

    def test_checkpoint_store_importable(self) -> None:
        from exo import WorkflowCheckpointStore as WorkflowCheckpointStoreImport

        assert WorkflowCheckpointStoreImport is WorkflowCheckpointStore
