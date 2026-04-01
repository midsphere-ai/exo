"""Tests for exo._internal.nested — nested swarm support."""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import AsyncMock

import pytest

from exo._internal.nested import NestedSwarmError, RalphNode, SwarmNode
from exo.agent import Agent
from exo.swarm import Swarm
from exo.types import AgentOutput, RalphIterationEvent, RalphStopEvent, TextEvent, Usage

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
# SwarmNode construction
# ---------------------------------------------------------------------------


class TestSwarmNodeConstruction:
    def test_wraps_swarm(self) -> None:
        """SwarmNode wraps a Swarm and exposes name + is_swarm marker."""
        a = Agent(name="a")
        inner = Swarm(agents=[a])
        node = SwarmNode(swarm=inner)

        assert node.is_swarm is True
        assert node.name == inner.name

    def test_custom_name(self) -> None:
        """SwarmNode can override the inner swarm's name."""
        a = Agent(name="a")
        inner = Swarm(agents=[a])
        node = SwarmNode(swarm=inner, name="my_pipeline")

        assert node.name == "my_pipeline"

    def test_rejects_non_swarm(self) -> None:
        """SwarmNode rejects objects that lack flow_order."""
        with pytest.raises(NestedSwarmError, match="requires a Swarm instance"):
            SwarmNode(swarm="not a swarm")

    def test_describe(self) -> None:
        """describe() returns nested swarm metadata."""
        a = Agent(name="a")
        inner = Swarm(agents=[a])
        node = SwarmNode(swarm=inner, name="pipe")

        desc = node.describe()
        assert desc["type"] == "nested_swarm"
        assert desc["name"] == "pipe"
        assert "inner" in desc
        assert desc["inner"]["mode"] == "workflow"

    def test_repr(self) -> None:
        """__repr__ includes node name and inner swarm repr."""
        a = Agent(name="a")
        inner = Swarm(agents=[a])
        node = SwarmNode(swarm=inner, name="pipe")

        r = repr(node)
        assert "SwarmNode" in r
        assert "pipe" in r


# ---------------------------------------------------------------------------
# Two-level nested swarm execution
# ---------------------------------------------------------------------------


class TestNestedSwarmExecution:
    async def test_two_level_nested_workflow(self) -> None:
        """Inner swarm runs as a node in the outer swarm's workflow."""
        # Inner: a >> b  (two agents, pipeline)
        a = Agent(name="a", instructions="Agent A")
        b = Agent(name="b", instructions="Agent B")
        inner = Swarm(agents=[a, b], flow="a >> b")
        inner_node = SwarmNode(swarm=inner, name="inner")

        # Outer: c >> inner >> d
        c = Agent(name="c", instructions="Agent C")
        d = Agent(name="d", instructions="Agent D")
        outer = Swarm(agents=[c, inner_node, d], flow="c >> inner >> d")

        # Provider: c responds, then a responds, then b responds, then d responds
        provider = _make_provider(
            [
                _output("c_out"),  # c
                _output("a_out"),  # inner.a
                _output("b_out"),  # inner.b
                _output("d_out"),  # d
            ]
        )

        result = await outer.run("hello", provider=provider)
        assert result.output == "d_out"

    async def test_inner_swarm_output_chains_to_next(self) -> None:
        """Output of inner swarm becomes input for the next outer agent."""
        a = Agent(name="a", instructions="")
        inner_agent = Agent(name="inner_a", instructions="")
        inner = Swarm(agents=[inner_agent])
        inner_node = SwarmNode(swarm=inner, name="inner")

        outer = Swarm(agents=[a, inner_node], flow="a >> inner")

        provider = _make_provider(
            [
                _output("first_output"),
                _output("final_output"),
            ]
        )

        result = await outer.run("start", provider=provider)
        # inner swarm gets a's output as input, and its output is the final result
        assert result.output == "final_output"

    async def test_standalone_swarm_node(self) -> None:
        """SwarmNode can be run directly (not just via outer Swarm)."""
        a = Agent(name="a", instructions="")
        inner = Swarm(agents=[a])
        node = SwarmNode(swarm=inner, name="standalone")

        provider = _make_provider([_output("direct_output")])
        result = await node.run("hello", provider=provider)
        assert result.output == "direct_output"


# ---------------------------------------------------------------------------
# Context isolation
# ---------------------------------------------------------------------------


class TestContextIsolation:
    async def test_inner_swarm_gets_clean_messages(self) -> None:
        """Inner swarm does NOT receive outer swarm's message history.

        This ensures context isolation — each swarm level maintains
        its own conversation context.
        """
        inner_a = Agent(name="inner_a", instructions="Inner instructions")
        inner = Swarm(agents=[inner_a])
        inner_node = SwarmNode(swarm=inner, name="inner")

        outer_a = Agent(name="outer_a", instructions="Outer instructions")
        outer = Swarm(agents=[outer_a, inner_node], flow="outer_a >> inner")

        # Track what messages each agent's provider.complete receives
        call_messages: list[list[Any]] = []

        async def tracking_complete(messages: Any, **kwargs: Any) -> Any:
            call_messages.append(list(messages))

            class FakeResponse:
                content = "output"
                tool_calls: ClassVar[list[Any]] = []
                usage = Usage(input_tokens=5, output_tokens=5, total_tokens=10)

            return FakeResponse()

        provider = AsyncMock()
        provider.complete = tracking_complete

        await outer.run("hello", provider=provider)

        # Two complete() calls: one for outer_a, one for inner_a
        assert len(call_messages) == 2

        # Inner swarm's messages should NOT contain outer instructions
        inner_msgs = call_messages[1]
        inner_msg_texts = [getattr(m, "content", "") for m in inner_msgs]
        assert "Outer instructions" not in inner_msg_texts

    async def test_inner_state_does_not_leak_to_outer(self) -> None:
        """Each swarm level has independent RunState — no shared mutable state."""
        a = Agent(name="a", instructions="A")
        inner_b = Agent(name="inner_b", instructions="B")
        inner = Swarm(agents=[inner_b])
        inner_node = SwarmNode(swarm=inner, name="inner")

        outer = Swarm(agents=[a, inner_node], flow="a >> inner")

        provider = _make_provider(
            [
                _output("a_result"),
                _output("b_result"),
            ]
        )

        result = await outer.run("start", provider=provider)

        # Outer result contains output from final node (inner swarm)
        assert result.output == "b_result"
        # Usage should be accumulated from both levels
        assert result.usage.total_tokens > 0
        assert result.steps >= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestNestedEdgeCases:
    async def test_inner_swarm_with_multiple_agents(self) -> None:
        """Nested swarm with 3-agent pipeline works correctly."""
        x = Agent(name="x", instructions="")
        y = Agent(name="y", instructions="")
        z = Agent(name="z", instructions="")
        inner = Swarm(agents=[x, y, z], flow="x >> y >> z")
        inner_node = SwarmNode(swarm=inner, name="pipeline")

        outer_a = Agent(name="outer_a", instructions="")
        outer = Swarm(agents=[outer_a, inner_node], flow="outer_a >> pipeline")

        provider = _make_provider(
            [
                _output("outer_out"),
                _output("x_out"),
                _output("y_out"),
                _output("z_out"),
            ]
        )

        result = await outer.run("hello", provider=provider)
        assert result.output == "z_out"

    async def test_inner_handoff_swarm(self) -> None:
        """Nested swarm with handoff mode works as a node."""
        billing = Agent(name="billing", instructions="")
        triage = Agent(
            name="triage",
            instructions="",
            handoffs=[billing],
        )
        inner = Swarm(agents=[triage, billing], mode="handoff")
        inner_node = SwarmNode(swarm=inner, name="support")

        outer_a = Agent(name="entry", instructions="")
        outer = Swarm(agents=[outer_a, inner_node], flow="entry >> support")

        # entry responds, then triage hands off to billing, billing responds
        provider = _make_provider(
            [
                _output("entry_out"),
                _output("billing"),  # triage output matching handoff target
                _output("billing_out"),  # billing's actual response
            ]
        )

        result = await outer.run("help", provider=provider)
        assert result.output == "billing_out"

    async def test_swarm_node_standalone_run(self) -> None:
        """SwarmNode works when run directly, outside any outer Swarm."""
        inner_a = Agent(name="inner_a", instructions="")
        inner = Swarm(agents=[inner_a])
        inner_node = SwarmNode(swarm=inner, name="inner_node")

        provider = _make_provider(
            [
                _output("inner_result"),
            ]
        )

        result = await inner_node.run("hello", provider=provider)
        assert result.output == "inner_result"

    def test_swarm_node_rejects_agent(self) -> None:
        """SwarmNode rejects plain Agent instances (no flow_order)."""
        a = Agent(name="a")
        with pytest.raises(NestedSwarmError, match="requires a Swarm"):
            SwarmNode(swarm=a)


# ---------------------------------------------------------------------------
# SwarmNode.stream()
# ---------------------------------------------------------------------------


class TestSwarmNodeStream:
    def test_has_stream_method(self) -> None:
        """SwarmNode exposes a stream() method."""
        a = Agent(name="a")
        inner = Swarm(agents=[a])
        node = SwarmNode(swarm=inner)
        assert hasattr(node, "stream")

    async def test_stream_delegates_to_inner_swarm(self) -> None:
        """SwarmNode.stream() yields events from the inner swarm."""
        expected_events = [
            TextEvent(text="hello", agent_name="inner_a"),
            TextEvent(text=" world", agent_name="inner_a"),
        ]

        class FakeSwarm:
            flow_order = ["a"]
            name = "fake"

            async def stream(self, input, **kwargs):
                for ev in expected_events:
                    yield ev

        node = SwarmNode(swarm=FakeSwarm(), name="test_node")
        collected = []
        async for event in node.stream("input"):
            collected.append(event)

        assert collected == expected_events


# ---------------------------------------------------------------------------
# RalphNode
# ---------------------------------------------------------------------------


class TestRalphNode:
    def test_is_group_marker(self) -> None:
        """RalphNode has is_group=True for Swarm duck-typing."""

        class FakeRunner:
            pass

        node = RalphNode(runner=FakeRunner(), name="research_loop")
        assert node.is_group is True
        assert node.name == "research_loop"

    def test_default_name(self) -> None:
        """RalphNode defaults to name='ralph'."""

        class FakeRunner:
            pass

        node = RalphNode(runner=FakeRunner())
        assert node.name == "ralph"

    async def test_run_delegates(self) -> None:
        """RalphNode.run() delegates to runner.run() and wraps in RunResult."""
        from dataclasses import dataclass

        @dataclass
        class FakeResult:
            output: str = "done"

        class FakeRunner:
            async def run(self, input):
                return FakeResult(output=f"result:{input}")

        node = RalphNode(runner=FakeRunner(), name="test")
        result = await node.run("hello")
        assert result.output == "result:hello"

    async def test_stream_delegates(self) -> None:
        """RalphNode.stream() delegates to runner.stream() and yields events."""
        expected_events = [
            RalphIterationEvent(iteration=1, status="started", agent_name="test"),
            TextEvent(text="output", agent_name="inner"),
            RalphIterationEvent(iteration=1, status="completed", agent_name="test"),
            RalphStopEvent(
                stop_type="max_iterations",
                reason="done",
                iterations=1,
                agent_name="test",
            ),
        ]

        class FakeRunner:
            async def stream(self, input, *, name="ralph"):
                for ev in expected_events:
                    yield ev

        node = RalphNode(runner=FakeRunner(), name="test")
        collected = []
        async for event in node.stream("hello"):
            collected.append(event)

        assert collected == expected_events

    def test_repr(self) -> None:
        """__repr__ includes node name."""

        class FakeRunner:
            pass

        node = RalphNode(runner=FakeRunner(), name="my_ralph")
        assert "RalphNode" in repr(node)
        assert "my_ralph" in repr(node)
