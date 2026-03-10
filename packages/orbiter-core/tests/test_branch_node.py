"""Tests for orbiter._internal.branch_node — conditional routing in workflows."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from orbiter._internal.branch_node import BranchError, BranchNode
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


class _FakeStreamChunk:
    """Lightweight stream chunk for testing."""

    def __init__(self, delta: str = "") -> None:
        self.delta = delta
        self.tool_call_deltas: list[Any] = []
        self.finish_reason: str | None = None
        self.usage = Usage()


def _make_stream_provider(stream_rounds: list[list[_FakeStreamChunk]]) -> Any:
    """Create a mock provider with stream() returning pre-defined chunks."""
    call_count = 0

    async def stream(messages: Any, **kwargs: Any) -> AsyncIterator[Any]:
        nonlocal call_count
        chunks = stream_rounds[min(call_count, len(stream_rounds) - 1)]
        call_count += 1
        for c in chunks:
            yield c

    mock = AsyncMock()
    mock.stream = stream
    mock.complete = AsyncMock()
    return mock


# ---------------------------------------------------------------------------
# BranchNode construction
# ---------------------------------------------------------------------------


class TestBranchNodeConstruction:
    def test_basic_construction(self) -> None:
        """BranchNode can be created with string condition."""
        branch = BranchNode(
            name="check",
            condition="score > 80",
            if_true="approve",
            if_false="review",
        )

        assert branch.name == "check"
        assert branch.condition == "score > 80"
        assert branch.if_true == "approve"
        assert branch.if_false == "review"
        assert branch.is_branch is True

    def test_callable_condition(self) -> None:
        """BranchNode accepts a callable condition."""
        fn = lambda state: state.get("x", 0) > 5  # noqa: E731

        branch = BranchNode(
            name="check",
            condition=fn,
            if_true="yes",
            if_false="no",
        )

        assert branch.condition is fn
        assert branch.is_branch is True

    def test_empty_name_raises(self) -> None:
        """BranchNode with empty name raises BranchError."""
        with pytest.raises(BranchError, match="non-empty name"):
            BranchNode(name="", condition="True", if_true="a", if_false="b")

    def test_empty_if_true_raises(self) -> None:
        """BranchNode with empty if_true raises BranchError."""
        with pytest.raises(BranchError, match="non-empty if_true"):
            BranchNode(name="b", condition="True", if_true="", if_false="x")

    def test_empty_if_false_raises(self) -> None:
        """BranchNode with empty if_false raises BranchError."""
        with pytest.raises(BranchError, match="non-empty if_false"):
            BranchNode(name="b", condition="True", if_true="x", if_false="")

    def test_describe_string_condition(self) -> None:
        """describe() returns branch metadata with string condition."""
        branch = BranchNode(
            name="check", condition="x > 5", if_true="a", if_false="b"
        )
        desc = branch.describe()

        assert desc["type"] == "branch"
        assert desc["name"] == "check"
        assert desc["condition"] == "x > 5"
        assert desc["if_true"] == "a"
        assert desc["if_false"] == "b"

    def test_describe_callable_condition(self) -> None:
        """describe() represents callable conditions via repr."""
        fn = lambda state: True  # noqa: E731
        branch = BranchNode(name="check", condition=fn, if_true="a", if_false="b")
        desc = branch.describe()

        assert desc["type"] == "branch"
        assert "lambda" in desc["condition"]

    def test_repr_string_condition(self) -> None:
        """__repr__ includes the string condition."""
        branch = BranchNode(
            name="check", condition="x > 5", if_true="a", if_false="b"
        )
        r = repr(branch)
        assert "BranchNode" in r
        assert "check" in r
        assert "x > 5" in r

    def test_repr_callable_condition(self) -> None:
        """__repr__ shows <callable> for callable conditions."""
        branch = BranchNode(
            name="check", condition=lambda s: True, if_true="a", if_false="b"
        )
        r = repr(branch)
        assert "<callable>" in r


# ---------------------------------------------------------------------------
# BranchNode.evaluate() — string conditions
# ---------------------------------------------------------------------------


class TestBranchNodeEvaluateString:
    def test_true_condition(self) -> None:
        """String condition evaluating to True returns if_true agent."""
        branch = BranchNode(
            name="check", condition="score > 80", if_true="approve", if_false="review"
        )
        target = branch.evaluate({"score": 95})
        assert target == "approve"

    def test_false_condition(self) -> None:
        """String condition evaluating to False returns if_false agent."""
        branch = BranchNode(
            name="check", condition="score > 80", if_true="approve", if_false="review"
        )
        target = branch.evaluate({"score": 50})
        assert target == "review"

    def test_equality_condition(self) -> None:
        """String condition with equality check."""
        branch = BranchNode(
            name="check", condition="status == 'active'", if_true="process", if_false="skip"
        )
        assert branch.evaluate({"status": "active"}) == "process"
        assert branch.evaluate({"status": "inactive"}) == "skip"

    def test_boolean_and_condition(self) -> None:
        """String condition with 'and' operator."""
        branch = BranchNode(
            name="check",
            condition="x > 0 and y > 0",
            if_true="proceed",
            if_false="halt",
        )
        assert branch.evaluate({"x": 1, "y": 1}) == "proceed"
        assert branch.evaluate({"x": 1, "y": -1}) == "halt"

    def test_js_style_operators(self) -> None:
        """String condition with JS-style operators (&&, ||)."""
        branch = BranchNode(
            name="check",
            condition="x > 0 && y > 0",
            if_true="yes",
            if_false="no",
        )
        assert branch.evaluate({"x": 1, "y": 1}) == "yes"
        assert branch.evaluate({"x": -1, "y": 1}) == "no"

    def test_input_variable(self) -> None:
        """The 'input' key in state is accessible."""
        branch = BranchNode(
            name="check",
            condition="len(input) > 5",
            if_true="long",
            if_false="short",
        )
        assert branch.evaluate({"input": "hello world"}) == "long"
        assert branch.evaluate({"input": "hi"}) == "short"

    def test_invalid_expression_raises(self) -> None:
        """Invalid expression raises BranchError."""
        branch = BranchNode(
            name="check", condition="import os", if_true="a", if_false="b"
        )
        with pytest.raises(BranchError, match="condition evaluation failed"):
            branch.evaluate({})

    def test_undefined_variable_raises(self) -> None:
        """Undefined variable in expression raises BranchError."""
        branch = BranchNode(
            name="check", condition="missing_var > 5", if_true="a", if_false="b"
        )
        with pytest.raises(BranchError, match="condition evaluation failed"):
            branch.evaluate({})


# ---------------------------------------------------------------------------
# BranchNode.evaluate() — callable conditions
# ---------------------------------------------------------------------------


class TestBranchNodeEvaluateCallable:
    def test_callable_true(self) -> None:
        """Callable returning True routes to if_true."""
        branch = BranchNode(
            name="check",
            condition=lambda state: state.get("score", 0) > 80,
            if_true="approve",
            if_false="review",
        )
        assert branch.evaluate({"score": 95}) == "approve"

    def test_callable_false(self) -> None:
        """Callable returning False routes to if_false."""
        branch = BranchNode(
            name="check",
            condition=lambda state: state.get("score", 0) > 80,
            if_true="approve",
            if_false="review",
        )
        assert branch.evaluate({"score": 50}) == "review"

    def test_callable_receives_state(self) -> None:
        """Callable receives the full state dict."""
        received = {}

        def capture(state: dict[str, Any]) -> bool:
            received.update(state)
            return True

        branch = BranchNode(
            name="check", condition=capture, if_true="a", if_false="b"
        )
        branch.evaluate({"input": "hello", "extra": 42})

        assert received["input"] == "hello"
        assert received["extra"] == 42

    def test_callable_exception_raises_branch_error(self) -> None:
        """Callable raising an exception is wrapped in BranchError."""
        def bad_fn(state: dict[str, Any]) -> bool:
            raise ValueError("something went wrong")

        branch = BranchNode(
            name="check", condition=bad_fn, if_true="a", if_false="b"
        )
        with pytest.raises(BranchError, match="condition raised an error"):
            branch.evaluate({})

    def test_callable_truthy_values(self) -> None:
        """Callable returning truthy non-bool values routes to if_true."""
        branch = BranchNode(
            name="check",
            condition=lambda state: "non-empty string",
            if_true="yes",
            if_false="no",
        )
        assert branch.evaluate({}) == "yes"

    def test_callable_falsy_values(self) -> None:
        """Callable returning falsy non-bool values routes to if_false."""
        branch = BranchNode(
            name="check",
            condition=lambda state: 0,
            if_true="yes",
            if_false="no",
        )
        assert branch.evaluate({}) == "no"


# ---------------------------------------------------------------------------
# BranchNode.run() — async interface
# ---------------------------------------------------------------------------


class TestBranchNodeRun:
    @pytest.mark.anyio()
    async def test_run_returns_target_name(self) -> None:
        """run() returns RunResult with the target agent name as output."""
        branch = BranchNode(
            name="check", condition="len(input) > 3", if_true="long", if_false="short"
        )
        result = await branch.run("hello world")

        assert isinstance(result, RunResult)
        assert result.output == "long"

    @pytest.mark.anyio()
    async def test_run_false_path(self) -> None:
        """run() returns if_false target when condition is false."""
        branch = BranchNode(
            name="check", condition="len(input) > 100", if_true="long", if_false="short"
        )
        result = await branch.run("hi")

        assert result.output == "short"


# ---------------------------------------------------------------------------
# BranchNode in Swarm workflow
# ---------------------------------------------------------------------------


class TestBranchNodeInSwarm:
    @pytest.mark.anyio()
    async def test_branch_routes_to_true_agent(self) -> None:
        """Swarm workflow routes to if_true agent when condition is true."""
        agent_a = Agent(name="a")
        approve = Agent(name="approve")
        review = Agent(name="review")
        branch = BranchNode(
            name="check",
            condition="len(input) > 3",
            if_true="approve",
            if_false="review",
        )

        swarm = Swarm(
            agents=[agent_a, branch, approve, review],
            flow="a >> check >> approve",
        )

        provider = _make_provider([
            _output("long input text"),  # agent a
            _output("approved!"),  # approve
        ])

        result = await swarm.run("hello world", provider=provider)
        assert result.output == "approved!"

    @pytest.mark.anyio()
    async def test_branch_routes_to_false_agent(self) -> None:
        """Swarm workflow routes to if_false agent when condition is false."""
        approve = Agent(name="approve")
        review = Agent(name="review")
        branch = BranchNode(
            name="check",
            condition="len(input) > 100",
            if_true="approve",
            if_false="review",
        )

        swarm = Swarm(
            agents=[branch, approve, review],
            flow="check >> approve",
        )

        provider = _make_provider([
            _output("needs review"),  # review
        ])

        result = await swarm.run("short", provider=provider)
        assert result.output == "needs review"

    @pytest.mark.anyio()
    async def test_branch_with_callable_in_swarm(self) -> None:
        """Swarm workflow works with callable conditions."""
        approve = Agent(name="approve")
        review = Agent(name="review")
        branch = BranchNode(
            name="check",
            condition=lambda state: "urgent" in state.get("input", ""),
            if_true="approve",
            if_false="review",
        )

        swarm = Swarm(
            agents=[branch, approve, review],
            flow="check >> approve",
        )

        # Urgent input → approve
        provider = _make_provider([_output("fast-tracked!")])
        result = await swarm.run("urgent request", provider=provider)
        assert result.output == "fast-tracked!"

    @pytest.mark.anyio()
    async def test_branch_unknown_target_raises(self) -> None:
        """Swarm raises error when branch targets unknown agent."""
        branch = BranchNode(
            name="check",
            condition="True",
            if_true="nonexistent",
            if_false="also_nonexistent",
        )
        dummy = Agent(name="dummy")

        swarm = Swarm(
            agents=[branch, dummy],
            flow="check >> dummy",
        )

        provider = _make_provider([_output("x")])
        with pytest.raises(SwarmError, match="targets unknown agent"):
            await swarm.run("input", provider=provider)

    @pytest.mark.anyio()
    async def test_branch_skips_intermediate_agents(self) -> None:
        """Branch skips agents between it and the target in flow order."""
        step1 = Agent(name="step1")
        step2 = Agent(name="step2")
        step3 = Agent(name="step3")
        branch = BranchNode(
            name="check",
            condition="True",
            if_true="step3",
            if_false="step1",
        )

        swarm = Swarm(
            agents=[branch, step1, step2, step3],
            flow="check >> step1 >> step2 >> step3",
        )

        provider = _make_provider([
            _output("from step3"),  # Only step3 should execute
        ])

        result = await swarm.run("input", provider=provider)
        assert result.output == "from step3"


# ---------------------------------------------------------------------------
# BranchNode in Swarm stream workflow
# ---------------------------------------------------------------------------


class TestBranchNodeStreamWorkflow:
    @pytest.mark.anyio()
    async def test_stream_branch_emits_status(self) -> None:
        """Stream workflow emits StatusEvent for branch routing."""
        from orbiter.runner import run
        from orbiter.types import StatusEvent

        approve = Agent(name="approve")
        review = Agent(name="review")
        branch = BranchNode(
            name="check",
            condition="True",
            if_true="approve",
            if_false="review",
        )

        swarm = Swarm(
            agents=[branch, approve, review],
            flow="check >> approve",
        )

        provider = _make_stream_provider([
            [_FakeStreamChunk(delta="approved!")],
        ])
        events = []
        async for event in run.stream(
            swarm, "test input", provider=provider, detailed=True
        ):
            events.append(event)

        # Should have a status event for the branch routing
        status_events = [e for e in events if isinstance(e, StatusEvent)]
        branch_events = [e for e in status_events if "Branch" in (e.message or "")]
        assert len(branch_events) == 1
        assert "routing to 'approve'" in branch_events[0].message


# ---------------------------------------------------------------------------
# Existing Swarm tests still pass (regression guard)
# ---------------------------------------------------------------------------


class TestBranchNodeRegression:
    @pytest.mark.anyio()
    async def test_workflow_without_branch_unchanged(self) -> None:
        """Standard workflow without branches works exactly as before."""
        a = Agent(name="a")
        b = Agent(name="b")

        swarm = Swarm(agents=[a, b], flow="a >> b")

        provider = _make_provider([
            _output("from a"),
            _output("from b"),
        ])

        result = await swarm.run("hello", provider=provider)
        assert result.output == "from b"

    def test_branch_node_not_treated_as_group(self) -> None:
        """BranchNode does not have is_group or is_swarm markers."""
        branch = BranchNode(
            name="check", condition="True", if_true="a", if_false="b"
        )
        assert not getattr(branch, "is_group", False)
        assert not getattr(branch, "is_swarm", False)
        assert branch.is_branch is True
