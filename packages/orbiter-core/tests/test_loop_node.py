"""Tests for orbiter._internal.loop_node — iteration in workflows."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from orbiter._internal.loop_node import BREAK_SENTINEL, LoopError, LoopNode
from orbiter.agent import Agent
from orbiter.swarm import Swarm, SwarmError
from orbiter.types import AgentOutput, RunResult, Usage

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

        return FakeResponse()

    mock = AsyncMock()
    mock.complete = complete
    return mock


def _output(text: str) -> AgentOutput:
    """Create a simple text-only AgentOutput."""
    return AgentOutput(text=text, usage=Usage(input_tokens=5, output_tokens=5, total_tokens=10))


# ---------------------------------------------------------------------------
# LoopNode construction
# ---------------------------------------------------------------------------


class TestLoopNodeConstruction:
    def test_count_mode(self) -> None:
        """LoopNode with count mode."""
        loop = LoopNode(name="repeat", count=3, body="worker")
        assert loop.name == "repeat"
        assert loop.count == 3
        assert loop.body == ["worker"]
        assert loop.items is None
        assert loop.condition is None
        assert loop.is_loop is True

    def test_items_mode(self) -> None:
        """LoopNode with items mode."""
        loop = LoopNode(name="each", items="tasks", body="processor")
        assert loop.items == "tasks"
        assert loop.count is None
        assert loop.condition is None

    def test_condition_mode(self) -> None:
        """LoopNode with condition mode."""
        loop = LoopNode(name="poll", condition="status != 'done'", body="checker")
        assert loop.condition == "status != 'done'"
        assert loop.count is None
        assert loop.items is None

    def test_body_as_list(self) -> None:
        """LoopNode accepts list of body agents."""
        loop = LoopNode(name="multi", count=2, body=["a", "b"])
        assert loop.body == ["a", "b"]

    def test_empty_name_raises(self) -> None:
        """Empty name raises LoopError."""
        with pytest.raises(LoopError, match="non-empty name"):
            LoopNode(name="", count=1, body="x")

    def test_empty_body_raises(self) -> None:
        """Empty body raises LoopError."""
        with pytest.raises(LoopError, match="at least one body"):
            LoopNode(name="x", count=1, body=[])

    def test_no_mode_raises(self) -> None:
        """No mode specified raises LoopError."""
        with pytest.raises(LoopError, match="exactly one of"):
            LoopNode(name="x", body="worker")

    def test_multiple_modes_raises(self) -> None:
        """Multiple modes raises LoopError."""
        with pytest.raises(LoopError, match="exactly one of"):
            LoopNode(name="x", count=3, items="arr", body="worker")

    def test_negative_count_raises(self) -> None:
        """Negative count raises LoopError."""
        with pytest.raises(LoopError, match="non-negative"):
            LoopNode(name="x", count=-1, body="worker")

    def test_zero_max_iterations_raises(self) -> None:
        """max_iterations < 1 raises LoopError."""
        with pytest.raises(LoopError, match="max_iterations"):
            LoopNode(name="x", count=5, body="worker", max_iterations=0)

    def test_default_max_iterations(self) -> None:
        """Default max_iterations is 100."""
        loop = LoopNode(name="x", count=3, body="w")
        assert loop.max_iterations == 100

    def test_duck_type_markers(self) -> None:
        """LoopNode has is_loop but not is_branch, is_group, is_swarm."""
        loop = LoopNode(name="x", count=1, body="w")
        assert loop.is_loop is True
        assert not getattr(loop, "is_branch", False)
        assert not getattr(loop, "is_group", False)
        assert not getattr(loop, "is_swarm", False)

    def test_describe_count(self) -> None:
        """describe() for count mode."""
        loop = LoopNode(name="repeat", count=5, body="w")
        desc = loop.describe()
        assert desc["type"] == "loop"
        assert desc["mode"] == "count"
        assert desc["count"] == 5
        assert desc["body"] == ["w"]

    def test_describe_items(self) -> None:
        """describe() for items mode."""
        loop = LoopNode(name="each", items="tasks", body="p")
        desc = loop.describe()
        assert desc["mode"] == "items"
        assert desc["items"] == "tasks"

    def test_describe_condition(self) -> None:
        """describe() for condition mode."""
        loop = LoopNode(name="poll", condition="x > 0", body="c")
        desc = loop.describe()
        assert desc["mode"] == "condition"
        assert desc["condition"] == "x > 0"

    def test_repr_count(self) -> None:
        """__repr__ for count mode."""
        loop = LoopNode(name="repeat", count=3, body="w")
        r = repr(loop)
        assert "LoopNode" in r
        assert "count=3" in r

    def test_repr_items(self) -> None:
        """__repr__ for items mode."""
        loop = LoopNode(name="each", items="tasks", body="p")
        assert "items='tasks'" in repr(loop)

    def test_repr_condition(self) -> None:
        """__repr__ for condition mode."""
        loop = LoopNode(name="poll", condition="x > 0", body="c")
        assert "condition='x > 0'" in repr(loop)


# ---------------------------------------------------------------------------
# LoopNode in Swarm workflow — count mode
# ---------------------------------------------------------------------------


class TestLoopNodeCountMode:
    @pytest.mark.anyio()
    async def test_count_loop_executes_n_times(self) -> None:
        """Count loop runs the body agent exactly N times."""
        worker = Agent(name="worker")
        after = Agent(name="after")
        loop = LoopNode(name="loop", count=3, body="worker")

        # Only the loop and after are in the flow; worker is a body agent
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

            return FakeResponse()

        provider = AsyncMock()
        provider.complete = counting_complete

        result = await swarm.run("start", provider=provider)
        # 3 loop iterations + 1 after agent = 4 calls
        assert call_count == 4
        assert result.output == "iter_4"

    @pytest.mark.anyio()
    async def test_count_zero_skips_loop(self) -> None:
        """Count=0 executes body zero times."""
        worker = Agent(name="worker")
        after = Agent(name="after")
        loop = LoopNode(name="loop", count=0, body="worker")

        swarm = Swarm(agents=[loop, worker, after], flow="loop >> after")

        provider = _make_provider([
            _output("after result"),
        ])

        result = await swarm.run("start", provider=provider)
        assert result.output == "after result"

    @pytest.mark.anyio()
    async def test_count_loop_chains_output(self) -> None:
        """Each iteration receives the previous iteration's output as input."""
        worker = Agent(name="worker")
        loop = LoopNode(name="loop", count=3, body="worker")

        # Loop is the only flow node, so no extra execution after
        swarm = Swarm(agents=[loop, worker], flow="loop")

        call_count = 0

        async def counting_complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            resp = _output(f"output_{call_count}")

            class FakeResponse:
                content = resp.text
                tool_calls = resp.tool_calls
                usage = resp.usage

            return FakeResponse()

        provider = AsyncMock()
        provider.complete = counting_complete

        result = await swarm.run("start", provider=provider)
        assert call_count == 3
        assert result.output == "output_3"

    @pytest.mark.anyio()
    async def test_multiple_body_agents(self) -> None:
        """Loop with multiple body agents runs them sequentially each iteration."""
        a = Agent(name="a")
        b = Agent(name="b")
        loop = LoopNode(name="loop", count=2, body=["a", "b"])

        swarm = Swarm(agents=[loop, a, b], flow="loop")

        provider = _make_provider([
            _output("a0"), _output("b0"),  # iteration 0
            _output("a1"), _output("b1"),  # iteration 1
        ])

        result = await swarm.run("start", provider=provider)
        assert result.output == "b1"


# ---------------------------------------------------------------------------
# LoopNode in Swarm workflow — items mode
# ---------------------------------------------------------------------------


class TestLoopNodeItemsMode:
    def test_resolve_iterations_from_state(self) -> None:
        """Items mode resolves iteration count from state array."""
        loop = LoopNode(name="loop", items="tasks", body="worker")
        state = {"input": "start", "tasks": ["task1", "task2", "task3"]}
        n = loop._resolve_iterations(state)
        assert n == 3

    def test_items_missing_key_raises(self) -> None:
        """Missing items key raises LoopError."""
        loop = LoopNode(name="loop", items="tasks", body="worker")
        with pytest.raises(LoopError, match="not found in state"):
            loop._resolve_iterations({"input": "x"})

    def test_items_not_list_raises(self) -> None:
        """Non-list items key raises LoopError."""
        loop = LoopNode(name="loop", items="tasks", body="worker")
        with pytest.raises(LoopError, match="not a list"):
            loop._resolve_iterations({"input": "x", "tasks": "not-a-list"})

    def test_items_tuple_works(self) -> None:
        """Tuple values are accepted for items mode."""
        loop = LoopNode(name="loop", items="data", body="worker")
        state = {"input": "x", "data": ("a", "b")}
        n = loop._resolve_iterations(state)
        assert n == 2


# ---------------------------------------------------------------------------
# LoopNode in Swarm workflow — condition mode
# ---------------------------------------------------------------------------


class TestLoopNodeConditionMode:
    def test_condition_true(self) -> None:
        """Condition evaluating to True allows iteration."""
        loop = LoopNode(name="loop", condition="counter < 3", body="worker")
        assert loop._check_condition({"counter": 0}) is True
        assert loop._check_condition({"counter": 2}) is True

    def test_condition_false(self) -> None:
        """Condition evaluating to False stops iteration."""
        loop = LoopNode(name="loop", condition="counter < 3", body="worker")
        assert loop._check_condition({"counter": 3}) is False
        assert loop._check_condition({"counter": 10}) is False

    def test_invalid_condition_raises(self) -> None:
        """Invalid condition expression raises LoopError."""
        loop = LoopNode(name="loop", condition="import os", body="worker")
        with pytest.raises(LoopError, match="condition evaluation failed"):
            loop._check_condition({})

    @pytest.mark.anyio()
    async def test_condition_loop_in_swarm(self) -> None:
        """Condition loop terminates when condition becomes false.

        The condition checks the 'input' state value. After the body agent
        returns 'STOP', the condition 'STOP not in input' becomes False.
        """
        worker = Agent(name="worker")
        loop = LoopNode(
            name="loop",
            condition="'STOP' not in input",
            body="worker",
            max_iterations=10,
        )

        swarm = Swarm(agents=[loop, worker], flow="loop")

        call_count = 0

        async def counting_complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            text = "STOP" if call_count >= 3 else f"continue_{call_count}"
            resp = _output(text)

            class FakeResponse:
                content = resp.text
                tool_calls = resp.tool_calls
                usage = resp.usage

            return FakeResponse()

        provider = AsyncMock()
        provider.complete = counting_complete

        result = await swarm.run("go", provider=provider)
        assert "STOP" in result.output
        assert call_count == 3


# ---------------------------------------------------------------------------
# LoopNode — break support
# ---------------------------------------------------------------------------


class TestLoopNodeBreak:
    @pytest.mark.anyio()
    async def test_break_terminates_loop_early(self) -> None:
        """Output containing [BREAK] terminates the loop."""
        worker = Agent(name="worker")
        loop = LoopNode(name="loop", count=10, body="worker")

        swarm = Swarm(agents=[loop, worker], flow="loop")

        call_count = 0

        async def break_complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            text = f"result [BREAK]" if call_count >= 2 else f"iter_{call_count}"
            resp = _output(text)

            class FakeResponse:
                content = resp.text
                tool_calls = resp.tool_calls
                usage = resp.usage

            return FakeResponse()

        provider = AsyncMock()
        provider.complete = break_complete

        result = await swarm.run("start", provider=provider)
        assert BREAK_SENTINEL in result.output
        assert call_count == 2  # Should have stopped at iteration 2, not 10

    @pytest.mark.anyio()
    async def test_break_in_first_body_agent_skips_rest(self) -> None:
        """Break in first body agent skips remaining body agents."""
        a = Agent(name="a")
        b = Agent(name="b")
        loop = LoopNode(name="loop", count=5, body=["a", "b"])

        swarm = Swarm(agents=[loop, a, b], flow="loop")

        provider = _make_provider([
            _output("result [BREAK]"),  # agent a returns break immediately
        ])

        result = await swarm.run("start", provider=provider)
        assert BREAK_SENTINEL in result.output


# ---------------------------------------------------------------------------
# LoopNode — max_iterations safety limit
# ---------------------------------------------------------------------------


class TestLoopNodeMaxIterations:
    @pytest.mark.anyio()
    async def test_max_iterations_limits_count(self) -> None:
        """max_iterations caps even count-based loops."""
        worker = Agent(name="worker")
        loop = LoopNode(name="loop", count=1000, body="worker", max_iterations=3)

        swarm = Swarm(agents=[loop, worker], flow="loop")

        call_count = 0

        async def counting_complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            resp = _output(f"iter_{call_count}")

            class FakeResponse:
                content = resp.text
                tool_calls = resp.tool_calls
                usage = resp.usage

            return FakeResponse()

        provider = AsyncMock()
        provider.complete = counting_complete

        result = await swarm.run("start", provider=provider)
        assert call_count == 3  # Capped at max_iterations

    @pytest.mark.anyio()
    async def test_max_iterations_limits_condition(self) -> None:
        """max_iterations caps condition-based loops."""
        worker = Agent(name="worker")
        loop = LoopNode(
            name="loop",
            condition="True",  # always true
            body="worker",
            max_iterations=5,
        )

        swarm = Swarm(agents=[loop, worker], flow="loop")

        call_count = 0

        async def counting_complete(messages: Any, **kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            resp = _output(f"iter_{call_count}")

            class FakeResponse:
                content = resp.text
                tool_calls = resp.tool_calls
                usage = resp.usage

            return FakeResponse()

        provider = AsyncMock()
        provider.complete = counting_complete

        result = await swarm.run("start", provider=provider)
        assert call_count == 5


# ---------------------------------------------------------------------------
# LoopNode — unknown body agent
# ---------------------------------------------------------------------------


class TestLoopNodeErrors:
    @pytest.mark.anyio()
    async def test_unknown_body_agent_raises(self) -> None:
        """Loop referencing unknown body agent raises SwarmError."""
        loop = LoopNode(name="loop", count=1, body="nonexistent")

        dummy = Agent(name="dummy")
        swarm = Swarm(agents=[loop, dummy], flow="loop >> dummy")

        provider = _make_provider([_output("x")])
        with pytest.raises(SwarmError, match="unknown agent"):
            await swarm.run("input", provider=provider)


# ---------------------------------------------------------------------------
# LoopNode.run() — standalone
# ---------------------------------------------------------------------------


class TestLoopNodeRun:
    @pytest.mark.anyio()
    async def test_run_echoes_input(self) -> None:
        """Standalone run() echoes input (loop logic is in Swarm)."""
        loop = LoopNode(name="loop", count=3, body="worker")
        result = await loop.run("test input")
        assert result.output == "test input"


# ---------------------------------------------------------------------------
# Regression: existing workflows unaffected
# ---------------------------------------------------------------------------


class TestLoopNodeRegression:
    @pytest.mark.anyio()
    async def test_workflow_without_loop_unchanged(self) -> None:
        """Standard workflow without loops works exactly as before."""
        a = Agent(name="a")
        b = Agent(name="b")

        swarm = Swarm(agents=[a, b], flow="a >> b")

        provider = _make_provider([
            _output("from a"),
            _output("from b"),
        ])

        result = await swarm.run("hello", provider=provider)
        assert result.output == "from b"

    def test_loop_node_not_treated_as_branch(self) -> None:
        """LoopNode does not have is_branch, is_group, or is_swarm markers."""
        loop = LoopNode(name="loop", count=1, body="w")
        assert not getattr(loop, "is_branch", False)
        assert not getattr(loop, "is_group", False)
        assert not getattr(loop, "is_swarm", False)
        assert loop.is_loop is True
