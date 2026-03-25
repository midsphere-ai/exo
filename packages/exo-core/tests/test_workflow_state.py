"""Tests for WorkflowState and its propagation through Swarm workflow mode."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from exo._internal.branch_node import BranchNode
from exo._internal.loop_node import LoopNode
from exo._internal.workflow_state import WorkflowState
from exo.agent import Agent
from exo.swarm import Swarm
from exo.types import AgentOutput, Usage

# ---------------------------------------------------------------------------
# Fixtures: mock provider
# ---------------------------------------------------------------------------


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

    mock = AsyncMock()
    mock.complete = complete
    return mock


def _output(text: str) -> AgentOutput:
    """Create a simple text-only AgentOutput."""
    return AgentOutput(text=text, usage=Usage(input_tokens=5, output_tokens=5, total_tokens=10))


# ---------------------------------------------------------------------------
# WorkflowState unit tests
# ---------------------------------------------------------------------------


class TestWorkflowStateUnit:
    def test_empty_state(self) -> None:
        """Fresh WorkflowState has empty data."""
        state = WorkflowState()
        assert state.to_dict() == {}
        assert state.data == {}

    def test_initial_data(self) -> None:
        """WorkflowState can be created with initial data."""
        state = WorkflowState({"input": "hello", "score": 42})
        assert state.get("input") == "hello"
        assert state.get("score") == 42

    def test_set_and_get(self) -> None:
        """set() stores values, get() retrieves them."""
        state = WorkflowState()
        state.set("key", "value")
        assert state.get("key") == "value"

    def test_get_default(self) -> None:
        """get() returns default for missing keys."""
        state = WorkflowState()
        assert state.get("missing") is None
        assert state.get("missing", 42) == 42

    def test_to_dict_returns_copy(self) -> None:
        """to_dict() returns a shallow copy, not a reference."""
        state = WorkflowState({"a": 1})
        d = state.to_dict()
        d["a"] = 999
        assert state.get("a") == 1  # original unchanged

    def test_contains(self) -> None:
        """__contains__ checks for key presence."""
        state = WorkflowState({"x": 1})
        assert "x" in state
        assert "y" not in state

    def test_overwrite(self) -> None:
        """set() overwrites existing values."""
        state = WorkflowState({"x": 1})
        state.set("x", 2)
        assert state.get("x") == 2

    def test_repr(self) -> None:
        """__repr__ includes data."""
        state = WorkflowState({"a": 1})
        assert "WorkflowState" in repr(state)
        assert "'a'" in repr(state)

    def test_initial_dict_not_mutated(self) -> None:
        """Modifying WorkflowState does not mutate the initial dict."""
        original = {"key": "val"}
        state = WorkflowState(original)
        state.set("key", "changed")
        assert original["key"] == "val"

    def test_data_property(self) -> None:
        """data property returns the internal dict."""
        state = WorkflowState({"x": 1})
        state.set("y", 2)
        assert state.data == {"x": 1, "y": 2}


# ---------------------------------------------------------------------------
# State propagation through linear workflow
# ---------------------------------------------------------------------------


class TestLinearStatePropagation:
    @pytest.mark.anyio()
    async def test_agent_outputs_stored_in_state(self) -> None:
        """Each agent's output is accessible by downstream agents via state."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")

        swarm = Swarm(agents=[a, b, c], flow="a >> b >> c")

        provider = _make_provider(
            [
                _output("result_a"),
                _output("result_b"),
                _output("result_c"),
            ]
        )

        result = await swarm.run("start", provider=provider)
        assert result.output == "result_c"

    @pytest.mark.anyio()
    async def test_two_agent_chain(self) -> None:
        """Simple two-agent chain works with state tracking."""
        a = Agent(name="a")
        b = Agent(name="b")

        swarm = Swarm(agents=[a, b], flow="a >> b")

        provider = _make_provider(
            [
                _output("from_a"),
                _output("from_b"),
            ]
        )

        result = await swarm.run("hello", provider=provider)
        assert result.output == "from_b"


# ---------------------------------------------------------------------------
# State propagation through branching workflow
# ---------------------------------------------------------------------------


class TestBranchingStatePropagation:
    @pytest.mark.anyio()
    async def test_branch_reads_upstream_agent_output(self) -> None:
        """BranchNode can read prior agent output from workflow state.

        Agent 'scorer' produces output containing the score.
        BranchNode condition checks the upstream agent's name in state.
        """
        scorer = Agent(name="scorer")
        approve = Agent(name="approve")
        review = Agent(name="review")

        # Condition checks the 'scorer' key in state (set by WorkflowState)
        branch = BranchNode(
            name="check",
            condition=lambda state: "high" in state.get("scorer", ""),
            if_true="approve",
            if_false="review",
        )

        swarm = Swarm(
            agents=[scorer, branch, approve, review],
            flow="scorer >> check >> approve",
        )

        # scorer returns text with "high" — branch should route to approve
        provider = _make_provider(
            [
                _output("score: high"),
                _output("approved!"),
            ]
        )

        result = await swarm.run("evaluate this", provider=provider)
        assert result.output == "approved!"

    @pytest.mark.anyio()
    async def test_branch_reads_upstream_false_path(self) -> None:
        """BranchNode routes to if_false when upstream state doesn't match."""
        scorer = Agent(name="scorer")
        approve = Agent(name="approve")
        review = Agent(name="review")

        branch = BranchNode(
            name="check",
            condition=lambda state: "high" in state.get("scorer", ""),
            if_true="approve",
            if_false="review",
        )

        swarm = Swarm(
            agents=[scorer, branch, approve, review],
            flow="scorer >> check >> approve",
        )

        # scorer returns "low" — branch should route to review
        provider = _make_provider(
            [
                _output("score: low"),
                _output("needs review"),
            ]
        )

        result = await swarm.run("evaluate this", provider=provider)
        assert result.output == "needs review"

    @pytest.mark.anyio()
    async def test_branch_with_expression_using_state(self) -> None:
        """BranchNode string expression accesses state from prior agent."""
        first = Agent(name="first")
        yes = Agent(name="yes")
        no = Agent(name="no")

        # Uses callable since expression evaluator works with dict keys
        branch = BranchNode(
            name="decide",
            condition=lambda state: len(state.get("first", "")) > 5,
            if_true="yes",
            if_false="no",
        )

        swarm = Swarm(
            agents=[first, branch, yes, no],
            flow="first >> decide >> yes",
        )

        provider = _make_provider(
            [
                _output("long output text"),
                _output("yes result"),
            ]
        )

        result = await swarm.run("go", provider=provider)
        assert result.output == "yes result"


# ---------------------------------------------------------------------------
# State propagation through looping workflow
# ---------------------------------------------------------------------------


class TestLoopingStatePropagation:
    @pytest.mark.anyio()
    async def test_loop_body_outputs_in_state(self) -> None:
        """Loop body agent outputs are stored in workflow state."""
        worker = Agent(name="worker")
        after = Agent(name="after")
        loop = LoopNode(name="loop", count=3, body="worker")

        swarm = Swarm(agents=[loop, worker, after], flow="loop >> after")

        call_count = 0

        async def counting_complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            resp = _output(f"iter_{call_count}")

            class FakeResponse:
                content = resp.text
                tool_calls = resp.tool_calls
                usage = resp.usage
                reasoning_content = ""
                thought_signatures: list[bytes] = []

            return FakeResponse()

        provider = AsyncMock()
        provider.complete = counting_complete

        result = await swarm.run("start", provider=provider)
        # 3 loop iterations + 1 after = 4 calls
        assert call_count == 4

    @pytest.mark.anyio()
    async def test_loop_condition_reads_workflow_state(self) -> None:
        """Loop condition can read values set by prior agents via workflow state."""
        setup = Agent(name="setup")
        worker = Agent(name="worker")
        loop = LoopNode(
            name="loop",
            condition="'STOP' not in input",
            body="worker",
            max_iterations=10,
        )

        swarm = Swarm(
            agents=[setup, loop, worker],
            flow="setup >> loop",
        )

        call_count = 0

        async def smart_complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                text = "initialized"  # setup
            elif call_count >= 4:
                text = "STOP"
            else:
                text = f"continue_{call_count}"
            resp = _output(text)

            class FakeResponse:
                content = resp.text
                tool_calls = resp.tool_calls
                usage = resp.usage
                reasoning_content = ""
                thought_signatures: list[bytes] = []

            return FakeResponse()

        provider = AsyncMock()
        provider.complete = smart_complete

        result = await swarm.run("go", provider=provider)
        assert "STOP" in result.output

    @pytest.mark.anyio()
    async def test_loop_then_branch_reads_loop_state(self) -> None:
        """Branch after loop can read the loop node's accumulated output."""
        worker = Agent(name="worker")
        yes = Agent(name="yes")
        no = Agent(name="no")
        loop = LoopNode(name="loop", count=2, body="worker")

        branch = BranchNode(
            name="check",
            condition=lambda state: "done" in state.get("loop", ""),
            if_true="yes",
            if_false="no",
        )

        swarm = Swarm(
            agents=[loop, worker, branch, yes, no],
            flow="loop >> check >> yes",
        )

        provider = _make_provider(
            [
                _output("working"),  # worker iter 0
                _output("all done"),  # worker iter 1
                _output("yes result"),  # yes agent
            ]
        )

        result = await swarm.run("start", provider=provider)
        # loop state stores "all done" (last body output),
        # branch checks "done" in state["loop"] → True → yes
        assert result.output == "yes result"


# ---------------------------------------------------------------------------
# Backward compatibility — state is optional
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    @pytest.mark.anyio()
    async def test_workflow_without_special_nodes(self) -> None:
        """Standard workflow without branches/loops works exactly as before."""
        a = Agent(name="a")
        b = Agent(name="b")

        swarm = Swarm(agents=[a, b], flow="a >> b")

        provider = _make_provider(
            [
                _output("from a"),
                _output("from b"),
            ]
        )

        result = await swarm.run("hello", provider=provider)
        assert result.output == "from b"

    @pytest.mark.anyio()
    async def test_branch_with_input_only(self) -> None:
        """Branch using only 'input' key still works (backward compat)."""
        approve = Agent(name="approve")
        review = Agent(name="review")
        branch = BranchNode(
            name="check",
            condition="len(input) > 3",
            if_true="approve",
            if_false="review",
        )

        swarm = Swarm(
            agents=[branch, approve, review],
            flow="check >> approve",
        )

        provider = _make_provider([_output("approved!")])
        result = await swarm.run("hello world", provider=provider)
        assert result.output == "approved!"

    @pytest.mark.anyio()
    async def test_loop_count_unchanged(self) -> None:
        """Count-based loop works identically with WorkflowState."""
        worker = Agent(name="worker")
        loop = LoopNode(name="loop", count=2, body="worker")

        swarm = Swarm(agents=[loop, worker], flow="loop")

        provider = _make_provider(
            [
                _output("iter1"),
                _output("iter2"),
            ]
        )

        result = await swarm.run("start", provider=provider)
        assert result.output == "iter2"

    @pytest.mark.anyio()
    async def test_handoff_mode_unaffected(self) -> None:
        """Handoff mode doesn't use WorkflowState — should still work."""
        a = Agent(name="a")
        b = Agent(name="b")

        swarm = Swarm(agents=[a, b], flow="a >> b", mode="handoff")

        provider = _make_provider([_output("final output")])

        result = await swarm.run("hello", provider=provider)
        assert result.output == "final output"


# ---------------------------------------------------------------------------
# Mixed workflow: linear → branch → loop
# ---------------------------------------------------------------------------


class TestMixedWorkflowState:
    @pytest.mark.anyio()
    async def test_linear_then_branch_state_propagation(self) -> None:
        """State from linear agent is available to downstream branch."""
        analyzer = Agent(name="analyzer")
        fast_path = Agent(name="fast_path")
        slow_path = Agent(name="slow_path")

        branch = BranchNode(
            name="route",
            condition=lambda state: "urgent" in state.get("analyzer", ""),
            if_true="fast_path",
            if_false="slow_path",
        )

        swarm = Swarm(
            agents=[analyzer, branch, fast_path, slow_path],
            flow="analyzer >> route >> fast_path",
        )

        provider = _make_provider(
            [
                _output("classification: urgent"),
                _output("fast result"),
            ]
        )

        result = await swarm.run("process this", provider=provider)
        assert result.output == "fast result"

    @pytest.mark.anyio()
    async def test_multiple_agents_before_branch(self) -> None:
        """Multiple prior agents' outputs available in branch state."""
        step1 = Agent(name="step1")
        step2 = Agent(name="step2")
        yes = Agent(name="yes")
        no = Agent(name="no")

        branch = BranchNode(
            name="decide",
            condition=lambda state: (
                "good" in state.get("step1", "") and "ready" in state.get("step2", "")
            ),
            if_true="yes",
            if_false="no",
        )

        swarm = Swarm(
            agents=[step1, step2, branch, yes, no],
            flow="step1 >> step2 >> decide >> yes",
        )

        provider = _make_provider(
            [
                _output("results: good"),  # step1
                _output("status: ready"),  # step2
                _output("yes!"),  # yes
            ]
        )

        result = await swarm.run("go", provider=provider)
        assert result.output == "yes!"
