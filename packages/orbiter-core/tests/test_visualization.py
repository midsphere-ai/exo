"""Tests for orbiter._internal.visualization — Mermaid flowchart generation."""

from __future__ import annotations

from orbiter._internal.agent_group import ParallelGroup
from orbiter._internal.branch_node import BranchNode
from orbiter._internal.loop_node import LoopNode
from orbiter._internal.nested import SwarmNode
from orbiter._internal.visualization import to_mermaid
from orbiter.agent import Agent
from orbiter.swarm import Swarm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lines(text: str) -> list[str]:
    """Split into stripped, non-empty lines for assertion convenience."""
    return [ln.strip() for ln in text.strip().splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# Linear chains
# ---------------------------------------------------------------------------


class TestLinearChain:
    def test_single_agent(self) -> None:
        a = Agent(name="a")
        swarm = Swarm(agents=[a])
        result = to_mermaid(swarm)
        lines = _lines(result)

        assert lines[0] == "graph TD"
        assert "a[a]" in lines[1]

    def test_two_agent_chain(self) -> None:
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b], flow="a >> b")
        result = to_mermaid(swarm)
        lines = _lines(result)

        assert lines[0] == "graph TD"
        assert "a[a]" in lines
        assert "b[b]" in lines
        assert "a --> b" in lines

    def test_three_agent_chain(self) -> None:
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")
        swarm = Swarm(agents=[a, b, c], flow="a >> b >> c")
        result = to_mermaid(swarm)
        lines = _lines(result)

        assert "a --> b" in lines
        assert "b --> c" in lines

    def test_linear_chain_no_flow_dsl(self) -> None:
        """Without DSL, agents form a linear chain in list order."""
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b])
        result = to_mermaid(swarm)
        lines = _lines(result)

        assert "a --> b" in lines

    def test_convenience_method(self) -> None:
        """Swarm.to_mermaid() delegates to the module-level function."""
        a = Agent(name="a")
        b = Agent(name="b")
        swarm = Swarm(agents=[a, b], flow="a >> b")
        assert swarm.to_mermaid() == to_mermaid(swarm)


# ---------------------------------------------------------------------------
# Parallel groups (DSL-based)
# ---------------------------------------------------------------------------


class TestParallelDSL:
    def test_parallel_group_in_flow(self) -> None:
        """(b | c) in DSL creates fan-out / fan-in edges."""
        a = Agent(name="a")
        b = Agent(name="b")
        c = Agent(name="c")
        d = Agent(name="d")
        swarm = Swarm(agents=[a, b, c, d], flow="a >> (b | c) >> d")
        result = to_mermaid(swarm)
        lines = _lines(result)

        # Fan-out from a to both b and c
        assert "a --> b" in lines
        assert "a --> c" in lines
        # Fan-in from b and c to d
        assert "b --> d" in lines
        assert "c --> d" in lines


# ---------------------------------------------------------------------------
# BranchNode (diamond shape)
# ---------------------------------------------------------------------------


class TestBranchNode:
    def test_branch_node_diamond_shape(self) -> None:
        a = Agent(name="a")
        branch = BranchNode(
            name="check",
            condition="status == 'ok'",
            if_true="b",
            if_false="c",
        )
        b = Agent(name="b")
        c = Agent(name="c")
        swarm = Swarm(
            agents=[a, branch, b, c],
            flow="a >> check >> (b | c)",
        )
        result = to_mermaid(swarm)
        lines = _lines(result)

        # Diamond shape
        assert "check{check}" in lines
        # Labeled edges
        assert "check -->|true| b" in lines
        assert "check -->|false| c" in lines

    def test_branch_node_out_of_flow_targets(self) -> None:
        """Branch targets not directly connected in DSL still get edges."""
        a = Agent(name="a")
        branch = BranchNode(
            name="route",
            condition="True",
            if_true="fast",
            if_false="slow",
        )
        fast = Agent(name="fast")
        slow = Agent(name="slow")
        d = Agent(name="d")
        swarm = Swarm(
            agents=[a, branch, fast, slow, d],
            flow="a >> route >> d",
        )
        result = to_mermaid(swarm)
        lines = _lines(result)

        # The DSL only has route -> d, but the branch targets
        # fast and slow should appear as extra edges.
        assert "route -->|true| fast" in lines
        assert "route -->|false| slow" in lines


# ---------------------------------------------------------------------------
# LoopNode (hexagon shape)
# ---------------------------------------------------------------------------


class TestLoopNode:
    def test_loop_node_hexagon_shape(self) -> None:
        a = Agent(name="a")
        loop = LoopNode(name="repeat", body="worker", count=3)
        worker = Agent(name="worker")
        b = Agent(name="b")
        swarm = Swarm(
            agents=[a, loop, worker, b],
            flow="a >> repeat >> b",
        )
        result = to_mermaid(swarm)
        lines = _lines(result)

        # Hexagon shape (double braces)
        assert "repeat{{repeat}}" in lines

    def test_loop_body_edges(self) -> None:
        a = Agent(name="a")
        loop = LoopNode(name="repeat", body="worker", count=3)
        worker = Agent(name="worker")
        b = Agent(name="b")
        swarm = Swarm(
            agents=[a, loop, worker, b],
            flow="a >> repeat >> b",
        )
        result = to_mermaid(swarm)
        lines = _lines(result)

        # Body edge from loop to worker
        assert "repeat -->|body| worker" in lines
        # Loop-back edge from worker to loop (dashed)
        assert "worker -.->|loop| repeat" in lines

    def test_loop_multi_body(self) -> None:
        loop = LoopNode(name="lp", body=["step1", "step2"], count=2)
        step1 = Agent(name="step1")
        step2 = Agent(name="step2")
        swarm = Swarm(agents=[loop, step1, step2], flow="lp")
        result = to_mermaid(swarm)
        lines = _lines(result)

        # Chain: lp -> step1 -> step2 -> (loop back to lp)
        assert "lp -->|body| step1" in lines
        assert "step1 -->|body| step2" in lines
        assert "step2 -.->|loop| lp" in lines


# ---------------------------------------------------------------------------
# SwarmNode (subroutine shape)
# ---------------------------------------------------------------------------


class TestSwarmNode:
    def test_swarm_node_subroutine_shape(self) -> None:
        inner_a = Agent(name="inner_a")
        inner_b = Agent(name="inner_b")
        inner_swarm = Swarm(
            agents=[inner_a, inner_b], flow="inner_a >> inner_b"
        )
        nested = SwarmNode(swarm=inner_swarm, name="sub")
        outer = Agent(name="start")
        end = Agent(name="end")
        swarm = Swarm(
            agents=[outer, nested, end],
            flow="start >> sub >> end",
        )
        result = to_mermaid(swarm)
        lines = _lines(result)

        assert "sub[[sub]]" in lines
        assert "start --> sub" in lines
        assert "sub --> end" in lines


# ---------------------------------------------------------------------------
# ParallelGroup (subgraph)
# ---------------------------------------------------------------------------


class TestParallelGroup:
    def test_parallel_group_subgraph(self) -> None:
        w1 = Agent(name="w1")
        w2 = Agent(name="w2")
        group = ParallelGroup(name="workers", agents=[w1, w2])
        start = Agent(name="start")
        finish = Agent(name="finish")
        swarm = Swarm(
            agents=[start, group, finish],
            flow="start >> workers >> finish",
        )
        result = to_mermaid(swarm)
        lines = _lines(result)

        assert "subgraph workers[workers]" in lines
        assert "w1[w1]" in lines
        assert "w2[w2]" in lines
        assert "end" in lines
        assert "start --> workers" in lines
        assert "workers --> finish" in lines


# ---------------------------------------------------------------------------
# Mixed topologies
# ---------------------------------------------------------------------------


class TestMixedTopology:
    def test_branch_and_loop(self) -> None:
        """A flow with both a branch and a loop."""
        start = Agent(name="start")
        branch = BranchNode(
            name="decide",
            condition="x > 0",
            if_true="looper",
            if_false="done",
        )
        loop = LoopNode(name="looper", body="work", count=3)
        work = Agent(name="work")
        done = Agent(name="done")
        swarm = Swarm(
            agents=[start, branch, loop, work, done],
            flow="start >> decide >> (looper | done)",
        )
        result = to_mermaid(swarm)
        lines = _lines(result)

        # Shapes
        assert "decide{decide}" in lines
        assert "looper{{looper}}" in lines
        # Branch edges
        assert "decide -->|true| looper" in lines
        assert "decide -->|false| done" in lines
        # Loop body
        assert "looper -->|body| work" in lines
        assert "work -.->|loop| looper" in lines

    def test_valid_mermaid_header(self) -> None:
        """Output always starts with 'graph TD'."""
        a = Agent(name="x")
        swarm = Swarm(agents=[a])
        assert to_mermaid(swarm).startswith("graph TD")

    def test_special_chars_in_name(self) -> None:
        """Agent names with special characters produce safe IDs."""
        a = Agent(name="my agent")
        b = Agent(name="process-data")
        swarm = Swarm(agents=[a, b], flow="my agent >> process-data")
        result = to_mermaid(swarm)

        # Should contain sanitized IDs
        assert "my_agent" in result
        assert "process_data" in result
        # Should still contain readable labels
        assert "my agent" in result
        assert "process-data" in result
