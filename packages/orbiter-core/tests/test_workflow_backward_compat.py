"""US-032: Workflow extension backward compatibility tests.

Proves that new workflow features (BranchNode, LoopNode, SwarmNode,
Mermaid visualisation, WorkflowState) work correctly together and
that existing behaviour is unchanged.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from orbiter._internal.branch_node import BranchNode
from orbiter._internal.loop_node import LoopNode
from orbiter._internal.nested import SwarmNode
from orbiter._internal.visualization import to_mermaid
from orbiter.agent import Agent
from orbiter.runner import run
from orbiter.swarm import Swarm
from orbiter.types import AgentOutput, RunResult, ToolCall, Usage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_USAGE = Usage(input_tokens=5, output_tokens=5, total_tokens=10)


def _output(text: str) -> AgentOutput:
    return AgentOutput(text=text, usage=_USAGE)


def _make_provider(responses: list[AgentOutput]) -> Any:
    """Mock provider cycling through pre-defined responses."""
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


def _counting_provider() -> tuple[Any, dict[str, int]]:
    """Provider that counts how many times complete() is called."""
    tracker: dict[str, int] = {"calls": 0}

    async def complete(messages: Any, **kwargs: Any) -> Any:
        tracker["calls"] += 1

        class FakeResponse:
            content = f"call_{tracker['calls']}"
            tool_calls: list[Any] = []
            usage = _USAGE

        return FakeResponse()

    mock = AsyncMock()
    mock.complete = complete
    return mock, tracker


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.strip().splitlines() if ln.strip()]


# ===================================================================
# 1. Existing 'a >> b >> c' DSL still works identically
# ===================================================================


class TestDSLBackwardCompat:
    """The >> DSL produces the same flows as before new node types were added."""

    async def test_linear_two_agents(self) -> None:
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b], flow="a >> b")
        provider = _make_provider([_output("from_a"), _output("from_b")])

        result = await run(swarm, "start", provider=provider)
        assert result.output == "from_b"

    async def test_linear_three_agents(self) -> None:
        a, b, c = Agent(name="a"), Agent(name="b"), Agent(name="c")
        swarm = Swarm(agents=[a, b, c], flow="a >> b >> c")
        provider = _make_provider([_output("a"), _output("b"), _output("c_out")])

        result = await run(swarm, "go", provider=provider)
        assert result.output == "c_out"

    async def test_implicit_flow_order(self) -> None:
        """Without flow DSL, agents run in list order."""
        a, b = Agent(name="a"), Agent(name="b")
        swarm = Swarm(agents=[a, b])
        provider = _make_provider([_output("first"), _output("second")])

        result = await run(swarm, "x", provider=provider)
        assert result.output == "second"

    async def test_parallel_dsl(self) -> None:
        """(b | c) fan-out / fan-in still works."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")
        d = Agent(name="d")
        swarm = Swarm(agents=[a, b, c, d], flow="a >> (b | c) >> d")
        provider, tracker = _counting_provider()

        result = await run(swarm, "go", provider=provider)
        # a + b + c + d = 4 LLM calls
        assert tracker["calls"] == 4

    def test_sync_run_still_works(self) -> None:
        a, b = Agent(name="a"), Agent(name="b")
        swarm = Swarm(agents=[a, b], flow="a >> b")
        provider = _make_provider([_output("sa"), _output("sb")])

        result = run.sync(swarm, "hello", provider=provider)
        assert result.output == "sb"


# ===================================================================
# 2. Existing workflow, handoff, and team modes pass
# ===================================================================


class TestModesBackwardCompat:
    """All three swarm modes continue to work after new node types."""

    async def test_workflow_mode(self) -> None:
        a, b = Agent(name="a"), Agent(name="b")
        swarm = Swarm(agents=[a, b], flow="a >> b")
        provider = _make_provider([_output("wa"), _output("wb")])

        result = await run(swarm, "input", provider=provider)
        assert result.output == "wb"

    async def test_handoff_mode_triggers(self) -> None:
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])
        swarm = Swarm(agents=[a, b], mode="handoff")
        provider = _make_provider([_output("b"), _output("handled")])

        result = await run(swarm, "help", provider=provider)
        assert result.output == "handled"

    async def test_handoff_mode_no_match(self) -> None:
        b = Agent(name="b")
        a = Agent(name="a", handoffs=[b])
        swarm = Swarm(agents=[a, b], mode="handoff")
        provider = _make_provider([_output("I handled it myself")])

        result = await run(swarm, "query", provider=provider)
        assert result.output == "I handled it myself"

    async def test_handoff_chain(self) -> None:
        c = Agent(name="c")
        b = Agent(name="b", handoffs=[c])
        a = Agent(name="a", handoffs=[b])
        swarm = Swarm(agents=[a, b, c], mode="handoff")
        provider = _make_provider([_output("b"), _output("c"), _output("final")])

        result = await run(swarm, "start", provider=provider)
        assert result.output == "final"

    async def test_team_mode_delegation(self) -> None:
        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")

        provider = _make_provider([
            AgentOutput(
                text="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name="delegate_to_worker",
                        arguments='{"task":"do work"}',
                    )
                ],
                usage=_USAGE,
            ),
            _output("worker result"),
            _output("lead synthesis"),
        ])

        result = await run(swarm, "task", provider=provider)
        assert result.output == "lead synthesis"

    async def test_team_mode_no_delegation(self) -> None:
        lead = Agent(name="lead")
        worker = Agent(name="worker")
        swarm = Swarm(agents=[lead, worker], mode="team")
        provider = _make_provider([_output("direct answer")])

        result = await run(swarm, "simple", provider=provider)
        assert result.output == "direct answer"


# ===================================================================
# 3. BranchNode + LoopNode combined flow
# ===================================================================


class TestBranchLoopCombined:
    """Branch and loop nodes working together in a single workflow."""

    async def test_branch_true_then_loop(self) -> None:
        """Branch routes to a loop node, which runs its body N times."""
        branch = BranchNode(
            name="decide",
            condition="len(input) > 3",
            if_true="repeat",
            if_false="skip",
        )
        loop = LoopNode(name="repeat", body="worker", count=2)
        worker = Agent(name="worker")
        skip = Agent(name="skip")
        end = Agent(name="end")

        swarm = Swarm(
            agents=[branch, loop, worker, skip, end],
            flow="decide >> repeat >> end",
        )

        provider, tracker = _counting_provider()
        result = await swarm.run("long input", provider=provider)

        # Loop runs worker 2 times + end = 3 calls
        assert tracker["calls"] == 3

    async def test_branch_false_skips_loop(self) -> None:
        """Branch condition is false → skip loop, go to skip agent."""
        branch = BranchNode(
            name="decide",
            condition="len(input) > 100",
            if_true="repeat",
            if_false="skip",
        )
        loop = LoopNode(name="repeat", body="worker", count=5)
        worker = Agent(name="worker")
        skip = Agent(name="skip")

        swarm = Swarm(
            agents=[branch, loop, worker, skip],
            flow="decide >> repeat",
        )

        provider = _make_provider([_output("skipped")])
        result = await swarm.run("short", provider=provider)
        assert result.output == "skipped"

    async def test_loop_then_branch(self) -> None:
        """Loop runs first, then branch evaluates condition on last output."""
        worker = Agent(name="worker")
        loop = LoopNode(name="repeat", body="worker", count=2)
        branch = BranchNode(
            name="decide",
            condition="len(input) > 3",
            if_true="approve",
            if_false="reject",
        )
        approve = Agent(name="approve")
        reject = Agent(name="reject")

        swarm = Swarm(
            agents=[loop, worker, branch, approve, reject],
            flow="repeat >> decide >> approve",
        )

        provider = _make_provider([
            _output("iter1"),
            _output("long output"),  # loop iter 2
            _output("approved!"),    # approve agent
        ])

        result = await swarm.run("start", provider=provider)
        assert result.output == "approved!"

    async def test_branch_with_callable_then_loop(self) -> None:
        """Branch with callable condition routing to a loop."""
        branch = BranchNode(
            name="check",
            condition=lambda state: state.get("input", "").startswith("repeat"),
            if_true="loop",
            if_false="done",
        )
        loop = LoopNode(name="loop", body="worker", count=3)
        worker = Agent(name="worker")
        done = Agent(name="done")

        swarm = Swarm(
            agents=[branch, loop, worker, done],
            flow="check >> loop",
        )

        provider, tracker = _counting_provider()
        result = await swarm.run("repeat this", provider=provider)
        # 3 loop body iterations
        assert tracker["calls"] == 3


# ===================================================================
# 4. LoopNode with nested SwarmNode
# ===================================================================


class TestLoopWithNestedSwarm:
    """LoopNode body can contain agents, and SwarmNode works within loops."""

    async def test_loop_body_with_nested_swarm_in_flow(self) -> None:
        """Outer: loop >> end, inner swarm used as a regular node."""
        inner_a = Agent(name="inner_a")
        inner_b = Agent(name="inner_b")
        inner_swarm = Swarm(agents=[inner_a, inner_b], flow="inner_a >> inner_b")
        nested = SwarmNode(swarm=inner_swarm, name="pipeline")

        worker = Agent(name="worker")
        loop = LoopNode(name="loop", body="worker", count=2)
        end = Agent(name="end")

        swarm = Swarm(
            agents=[nested, loop, worker, end],
            flow="pipeline >> loop >> end",
        )

        provider, tracker = _counting_provider()
        result = await swarm.run("start", provider=provider)

        # inner_a + inner_b + 2*worker + end = 5
        assert tracker["calls"] == 5

    async def test_swarm_node_before_loop(self) -> None:
        """Nested swarm runs, output feeds into loop."""
        ia = Agent(name="ia")
        inner = Swarm(agents=[ia])
        nested = SwarmNode(swarm=inner, name="pre")

        worker = Agent(name="worker")
        loop = LoopNode(name="loop", body="worker", count=2)

        swarm = Swarm(
            agents=[nested, loop, worker],
            flow="pre >> loop",
        )

        provider, tracker = _counting_provider()
        result = await swarm.run("go", provider=provider)

        # ia + 2*worker = 3
        assert tracker["calls"] == 3

    async def test_swarm_node_after_loop(self) -> None:
        """Loop runs, output feeds into nested swarm."""
        worker = Agent(name="worker")
        loop = LoopNode(name="loop", body="worker", count=2)

        ia = Agent(name="ia")
        ib = Agent(name="ib")
        inner = Swarm(agents=[ia, ib], flow="ia >> ib")
        nested = SwarmNode(swarm=inner, name="post")

        swarm = Swarm(
            agents=[loop, worker, nested],
            flow="loop >> post",
        )

        provider, tracker = _counting_provider()
        result = await swarm.run("go", provider=provider)

        # 2*worker + ia + ib = 4
        assert tracker["calls"] == 4

    async def test_nested_swarm_standalone_in_workflow(self) -> None:
        """SwarmNode works as a regular node in a simple workflow."""
        ia = Agent(name="ia")
        ib = Agent(name="ib")
        inner = Swarm(agents=[ia, ib], flow="ia >> ib")
        nested = SwarmNode(swarm=inner, name="sub")

        start = Agent(name="start")
        end = Agent(name="end")

        swarm = Swarm(
            agents=[start, nested, end],
            flow="start >> sub >> end",
        )

        provider = _make_provider([
            _output("start_out"),
            _output("ia_out"),
            _output("ib_out"),
            _output("end_out"),
        ])

        result = await swarm.run("go", provider=provider)
        assert result.output == "end_out"


# ===================================================================
# 5. Mermaid output for complex topology
# ===================================================================


class TestMermaidComplexTopology:
    """Mermaid generation for combined branch + loop + nested + parallel."""

    def test_branch_loop_mermaid(self) -> None:
        """Branch and loop together produce correct shapes and edges."""
        start = Agent(name="start")
        branch = BranchNode(
            name="decide",
            condition="x > 0",
            if_true="repeat",
            if_false="done",
        )
        loop = LoopNode(name="repeat", body="work", count=3)
        work = Agent(name="work")
        done = Agent(name="done")

        swarm = Swarm(
            agents=[start, branch, loop, work, done],
            flow="start >> decide >> (repeat | done)",
        )
        result = to_mermaid(swarm)
        lines = _lines(result)

        assert lines[0] == "graph TD"
        # Shapes
        assert "decide{decide}" in lines
        assert "repeat{{repeat}}" in lines
        # Branch edges
        assert "decide -->|true| repeat" in lines
        assert "decide -->|false| done" in lines
        # Loop body + loop-back
        assert "repeat -->|body| work" in lines
        assert "work -.->|loop| repeat" in lines

    def test_nested_swarm_mermaid(self) -> None:
        """SwarmNode renders as subroutine shape [[name]]."""
        ia = Agent(name="ia")
        inner = Swarm(agents=[ia])
        nested = SwarmNode(swarm=inner, name="sub")
        start = Agent(name="start")
        end = Agent(name="end")

        swarm = Swarm(
            agents=[start, nested, end],
            flow="start >> sub >> end",
        )
        result = to_mermaid(swarm)
        lines = _lines(result)

        assert "sub[[sub]]" in lines
        assert "start --> sub" in lines
        assert "sub --> end" in lines

    def test_linear_chain_mermaid(self) -> None:
        """Simple a >> b >> c still renders correctly."""
        a, b, c = Agent(name="a"), Agent(name="b"), Agent(name="c")
        swarm = Swarm(agents=[a, b, c], flow="a >> b >> c")
        result = to_mermaid(swarm)
        lines = _lines(result)

        assert "a --> b" in lines
        assert "b --> c" in lines

    def test_parallel_mermaid(self) -> None:
        """Parallel DSL renders fan-out / fan-in edges."""
        a, b, c, d = (
            Agent(name="a"),
            Agent(name="b"),
            Agent(name="c"),
            Agent(name="d"),
        )
        swarm = Swarm(agents=[a, b, c, d], flow="a >> (b | c) >> d")
        result = to_mermaid(swarm)
        lines = _lines(result)

        assert "a --> b" in lines
        assert "a --> c" in lines
        assert "b --> d" in lines
        assert "c --> d" in lines

    def test_loop_multi_body_mermaid(self) -> None:
        """Loop with multiple body agents renders chained body edges."""
        loop = LoopNode(name="lp", body=["s1", "s2"], count=2)
        s1 = Agent(name="s1")
        s2 = Agent(name="s2")
        swarm = Swarm(agents=[loop, s1, s2], flow="lp")
        result = to_mermaid(swarm)
        lines = _lines(result)

        assert "lp -->|body| s1" in lines
        assert "s1 -->|body| s2" in lines
        assert "s2 -.->|loop| lp" in lines

    def test_all_node_types_in_one_flow(self) -> None:
        """Flow with branch, loop, nested swarm, and regular agents."""
        start = Agent(name="start")
        branch = BranchNode(
            name="check",
            condition="True",
            if_true="loop",
            if_false="fallback",
        )
        loop = LoopNode(name="loop", body="worker", count=2)
        worker = Agent(name="worker")
        fallback = Agent(name="fallback")

        ia = Agent(name="ia")
        inner = Swarm(agents=[ia])
        nested = SwarmNode(swarm=inner, name="sub")

        end = Agent(name="end")

        swarm = Swarm(
            agents=[start, branch, loop, worker, fallback, nested, end],
            flow="start >> check >> loop >> sub >> end",
        )
        result = to_mermaid(swarm)
        lines = _lines(result)

        # All shapes present
        assert "check{check}" in lines
        assert "loop{{loop}}" in lines
        assert "sub[[sub]]" in lines
        # Regular nodes
        assert "start[start]" in lines
        assert "end[end]" in lines

    def test_mermaid_header(self) -> None:
        """Mermaid output always starts with 'graph TD'."""
        swarm = Swarm(agents=[Agent(name="x")])
        assert to_mermaid(swarm).startswith("graph TD")

    def test_convenience_method(self) -> None:
        """swarm.to_mermaid() matches module-level to_mermaid()."""
        a, b = Agent(name="a"), Agent(name="b")
        swarm = Swarm(agents=[a, b], flow="a >> b")
        assert swarm.to_mermaid() == to_mermaid(swarm)
