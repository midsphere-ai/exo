"""Tests for orbiter._internal.handlers — Handler ABC, AgentHandler, ToolHandler, GroupHandler."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from orbiter._internal.handlers import (
    AgentHandler,
    GroupHandler,
    Handler,
    HandlerError,
    SwarmMode,
    ToolHandler,
)
from orbiter.agent import Agent
from orbiter.tool import FunctionTool, ToolError
from orbiter.types import AgentOutput, RunResult, ToolResult

# ---------------------------------------------------------------------------
# Fixtures: mock provider
# ---------------------------------------------------------------------------


def _make_provider(responses: list[AgentOutput]) -> Any:
    """Create a mock provider returning pre-defined AgentOutput values.

    Supports multiple agents by cycling through responses in call order.
    """
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


# ---------------------------------------------------------------------------
# Handler ABC
# ---------------------------------------------------------------------------


class TestHandlerABC:
    def test_handler_is_abstract(self) -> None:
        """Handler cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            Handler()  # type: ignore[abstract]

    async def test_concrete_handler(self) -> None:
        """A concrete Handler subclass can yield results."""

        class EchoHandler(Handler[str, str]):
            async def handle(self, input: str, **kwargs: Any) -> AsyncIterator[str]:
                yield f"echo: {input}"

        handler = EchoHandler()
        results = [r async for r in handler.handle("hello")]

        assert results == ["echo: hello"]

    async def test_handler_multiple_yields(self) -> None:
        """Handler can yield multiple outputs."""

        class SplitHandler(Handler[str, str]):
            async def handle(self, input: str, **kwargs: Any) -> AsyncIterator[str]:
                for word in input.split():
                    yield word

        handler = SplitHandler()
        results = [r async for r in handler.handle("a b c")]

        assert results == ["a", "b", "c"]

    async def test_handler_empty_yield(self) -> None:
        """Handler can yield nothing."""

        class EmptyHandler(Handler[str, str]):
            async def handle(self, input: str, **kwargs: Any) -> AsyncIterator[str]:
                return
                yield

        handler = EmptyHandler()
        results = [r async for r in handler.handle("anything")]

        assert results == []


# ---------------------------------------------------------------------------
# SwarmMode enum
# ---------------------------------------------------------------------------


class TestSwarmMode:
    def test_modes(self) -> None:
        """SwarmMode has workflow, handoff, and team values."""
        assert SwarmMode.WORKFLOW == "workflow"
        assert SwarmMode.HANDOFF == "handoff"
        assert SwarmMode.TEAM == "team"

    def test_mode_from_string(self) -> None:
        """SwarmMode can be created from string value."""
        assert SwarmMode("workflow") == SwarmMode.WORKFLOW


# ---------------------------------------------------------------------------
# AgentHandler — workflow mode
# ---------------------------------------------------------------------------


class TestAgentHandlerWorkflow:
    async def test_single_agent_workflow(self) -> None:
        """Workflow with one agent returns its result."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="Hello!")])

        handler = AgentHandler(
            agents={"bot": agent},
            mode=SwarmMode.WORKFLOW,
            flow_order=["bot"],
            provider=provider,
        )
        results = [r async for r in handler.handle("Hi")]

        assert len(results) == 1
        assert results[0].output == "Hello!"

    async def test_multi_agent_workflow(self) -> None:
        """Workflow runs agents in order, chaining output as input."""
        agent_a = Agent(name="agent_a")
        agent_b = Agent(name="agent_b")
        # agent_a outputs "Step 1", agent_b outputs "Step 2"
        provider = _make_provider(
            [
                AgentOutput(text="Step 1"),
                AgentOutput(text="Step 2"),
            ]
        )

        handler = AgentHandler(
            agents={"agent_a": agent_a, "agent_b": agent_b},
            mode=SwarmMode.WORKFLOW,
            flow_order=["agent_a", "agent_b"],
            provider=provider,
        )
        results = [r async for r in handler.handle("Start")]

        assert len(results) == 2
        assert results[0].output == "Step 1"
        assert results[1].output == "Step 2"

    async def test_three_agent_pipeline(self) -> None:
        """Workflow correctly chains 3 agents."""
        agents = {f"a{i}": Agent(name=f"a{i}") for i in range(3)}
        provider = _make_provider(
            [
                AgentOutput(text="out_0"),
                AgentOutput(text="out_1"),
                AgentOutput(text="out_2"),
            ]
        )

        handler = AgentHandler(
            agents=agents,
            mode=SwarmMode.WORKFLOW,
            flow_order=["a0", "a1", "a2"],
            provider=provider,
        )
        results = [r async for r in handler.handle("input")]

        assert len(results) == 3
        assert [r.output for r in results] == ["out_0", "out_1", "out_2"]

    async def test_workflow_missing_agent_raises(self) -> None:
        """Workflow raises HandlerError for missing agent."""
        handler = AgentHandler(
            agents={},
            mode=SwarmMode.WORKFLOW,
            flow_order=["missing"],
            provider=_make_provider([]),
        )

        with pytest.raises(HandlerError, match="not found"):
            async for _ in handler.handle("test"):
                pass


# ---------------------------------------------------------------------------
# AgentHandler — handoff mode
# ---------------------------------------------------------------------------


class TestAgentHandlerHandoff:
    async def test_no_handoff(self) -> None:
        """Agent without handoff targets terminates after first run."""
        agent = Agent(name="bot")
        provider = _make_provider([AgentOutput(text="Done")])

        handler = AgentHandler(
            agents={"bot": agent},
            mode=SwarmMode.HANDOFF,
            flow_order=["bot"],
            provider=provider,
        )
        results = [r async for r in handler.handle("Hi")]

        assert len(results) == 1
        assert results[0].output == "Done"

    async def test_handoff_chain(self) -> None:
        """Agent A hands off to Agent B by outputting target name."""
        agent_b = Agent(name="billing")
        agent_a = Agent(name="triage", handoffs=[agent_b])

        # triage outputs "billing" (exact match → handoff), billing outputs "Done"
        provider = _make_provider(
            [
                AgentOutput(text="billing"),
                AgentOutput(text="Billing handled."),
            ]
        )

        handler = AgentHandler(
            agents={"triage": agent_a, "billing": agent_b},
            mode=SwarmMode.HANDOFF,
            flow_order=["triage"],
            provider=provider,
        )
        results = [r async for r in handler.handle("Help me")]

        assert len(results) == 2
        assert results[0].output == "billing"
        assert results[1].output == "Billing handled."

    async def test_handoff_max_exceeded(self) -> None:
        """Exceeding max_handoffs raises HandlerError."""
        agent_b = Agent(name="b")
        agent_a = Agent(name="a", handoffs=[agent_b])
        # b hands off back to a
        agent_b_inst = Agent(name="b", handoffs=[agent_a])

        # Alternate: a outputs "b", b outputs "a", a outputs "b", ...
        provider = _make_provider(
            [
                AgentOutput(text="b"),
                AgentOutput(text="a"),
                AgentOutput(text="b"),
                AgentOutput(text="a"),
                AgentOutput(text="b"),
            ]
        )

        handler = AgentHandler(
            agents={"a": agent_a, "b": agent_b_inst},
            mode=SwarmMode.HANDOFF,
            flow_order=["a"],
            provider=provider,
            max_handoffs=3,
        )

        with pytest.raises(HandlerError, match="Max handoffs"):
            async for _ in handler.handle("test"):
                pass

    async def test_handoff_detection_exact_match(self) -> None:
        """Handoff detection requires exact match of output to target name."""
        target = Agent(name="support")
        agent = Agent(name="triage", handoffs=[target])
        # Output contains "support" but isn't exactly "support"
        provider = _make_provider([AgentOutput(text="Contact support please")])

        handler = AgentHandler(
            agents={"triage": agent, "support": target},
            mode=SwarmMode.HANDOFF,
            flow_order=["triage"],
            provider=provider,
        )
        results = [r async for r in handler.handle("help")]

        # No handoff — output doesn't exactly match target name
        assert len(results) == 1

    async def test_handoff_missing_agent_raises(self) -> None:
        """Handoff to missing agent raises HandlerError."""
        handler = AgentHandler(
            agents={},
            mode=SwarmMode.HANDOFF,
            flow_order=["missing"],
            provider=_make_provider([]),
        )

        with pytest.raises(HandlerError, match="not found"):
            async for _ in handler.handle("test"):
                pass


# ---------------------------------------------------------------------------
# AgentHandler — team mode
# ---------------------------------------------------------------------------


class TestAgentHandlerTeam:
    async def test_team_lead_runs(self) -> None:
        """Team mode runs the lead agent."""
        lead = Agent(name="lead")
        worker = Agent(name="worker")
        provider = _make_provider([AgentOutput(text="Lead result")])

        handler = AgentHandler(
            agents={"lead": lead, "worker": worker},
            mode=SwarmMode.TEAM,
            flow_order=["lead", "worker"],
            provider=provider,
        )
        results = [r async for r in handler.handle("coordinate")]

        assert len(results) == 1
        assert results[0].output == "Lead result"

    async def test_team_empty_flow_raises(self) -> None:
        """Team mode with no agents raises HandlerError."""
        handler = AgentHandler(
            agents={},
            mode=SwarmMode.TEAM,
            flow_order=[],
            provider=_make_provider([]),
        )

        with pytest.raises(HandlerError, match="requires at least one agent"):
            async for _ in handler.handle("test"):
                pass

    async def test_team_missing_lead_raises(self) -> None:
        """Team mode with missing lead agent raises HandlerError."""
        handler = AgentHandler(
            agents={"worker": Agent(name="worker")},
            mode=SwarmMode.TEAM,
            flow_order=["missing_lead"],
            provider=_make_provider([]),
        )

        with pytest.raises(HandlerError, match="not found"):
            async for _ in handler.handle("test"):
                pass


# ---------------------------------------------------------------------------
# AgentHandler — stop checks
# ---------------------------------------------------------------------------


class TestStopChecks:
    def test_workflow_stop_last_agent(self) -> None:
        """Workflow stops after the last agent in flow_order."""
        handler = AgentHandler(
            agents={"a": Agent(name="a"), "b": Agent(name="b")},
            flow_order=["a", "b"],
        )

        assert handler._check_workflow_stop("b") is True
        assert handler._check_workflow_stop("a") is False

    def test_workflow_stop_empty_flow(self) -> None:
        """Workflow stop returns True for empty flow_order."""
        handler = AgentHandler(agents={}, flow_order=[])

        assert handler._check_workflow_stop("any") is True

    def test_handoff_stop_no_handoff(self) -> None:
        """Handoff stops when no handoff target is detected."""
        agent = Agent(name="bot")
        handler = AgentHandler(agents={"bot": agent})
        result = RunResult(output="Final answer")

        assert handler._check_handoff_stop(result, agent) is True

    def test_handoff_stop_with_handoff(self) -> None:
        """Handoff continues when handoff target is detected."""
        target = Agent(name="target")
        agent = Agent(name="bot", handoffs=[target])
        handler = AgentHandler(agents={"bot": agent, "target": target})
        result = RunResult(output="target")

        assert handler._check_handoff_stop(result, agent) is False

    def test_team_stop_after_lead(self) -> None:
        """Team stops after the lead agent (first in flow_order)."""
        handler = AgentHandler(
            agents={"lead": Agent(name="lead"), "worker": Agent(name="worker")},
            flow_order=["lead", "worker"],
        )

        assert handler._check_team_stop("lead") is True
        assert handler._check_team_stop("worker") is False


# ---------------------------------------------------------------------------
# AgentHandler — handoff detection
# ---------------------------------------------------------------------------


class TestHandoffDetection:
    def test_detect_handoff_match(self) -> None:
        """Handoff detected when output exactly matches a target name."""
        target = Agent(name="billing")
        agent = Agent(name="triage", handoffs=[target])
        handler = AgentHandler(agents={"triage": agent, "billing": target})
        result = RunResult(output="billing")

        assert handler._detect_handoff(agent, result) == "billing"

    def test_detect_handoff_no_match(self) -> None:
        """No handoff when output doesn't match any target."""
        target = Agent(name="billing")
        agent = Agent(name="triage", handoffs=[target])
        handler = AgentHandler(agents={"triage": agent, "billing": target})
        result = RunResult(output="I can help you directly")

        assert handler._detect_handoff(agent, result) is None

    def test_detect_handoff_no_handoffs(self) -> None:
        """No handoff when agent has no handoff targets."""
        agent = Agent(name="bot")
        handler = AgentHandler(agents={"bot": agent})
        result = RunResult(output="anything")

        assert handler._detect_handoff(agent, result) is None

    def test_detect_handoff_target_not_in_swarm(self) -> None:
        """No handoff when target agent is not registered in the swarm."""
        target = Agent(name="external")
        agent = Agent(name="bot", handoffs=[target])
        # "external" is a handoff target but NOT in the handler's agents dict
        handler = AgentHandler(agents={"bot": agent})
        result = RunResult(output="external")

        assert handler._detect_handoff(agent, result) is None

    def test_detect_handoff_whitespace_stripped(self) -> None:
        """Handoff detection strips whitespace from output."""
        target = Agent(name="billing")
        agent = Agent(name="triage", handoffs=[target])
        handler = AgentHandler(agents={"triage": agent, "billing": target})
        result = RunResult(output="  billing  ")

        assert handler._detect_handoff(agent, result) == "billing"


# ---------------------------------------------------------------------------
# ToolHandler
# ---------------------------------------------------------------------------


def _make_tool(name: str, output: str = "ok") -> FunctionTool:
    """Create a simple FunctionTool that returns a fixed string."""

    async def _fn(**kwargs: Any) -> str:
        return output

    return FunctionTool(_fn, name=name)


def _make_failing_tool(name: str, error_msg: str = "boom") -> FunctionTool:
    """Create a FunctionTool that raises ToolError."""

    async def _fn(**kwargs: Any) -> str:
        raise ToolError(error_msg)

    return FunctionTool(_fn, name=name)


class TestToolHandlerRegistration:
    def test_register_single(self) -> None:
        """Register a single tool."""
        handler = ToolHandler()
        tool = _make_tool("greet")
        handler.register(tool)

        assert "greet" in handler.tools

    def test_register_many(self) -> None:
        """Register multiple tools at once."""
        handler = ToolHandler()
        handler.register_many([_make_tool("a"), _make_tool("b")])

        assert "a" in handler.tools
        assert "b" in handler.tools

    def test_register_duplicate_raises(self) -> None:
        """Registering a duplicate tool name raises HandlerError."""
        handler = ToolHandler()
        handler.register(_make_tool("x"))

        with pytest.raises(HandlerError, match="Duplicate tool"):
            handler.register(_make_tool("x"))

    def test_init_with_tools(self) -> None:
        """Tools can be passed via constructor."""
        tool = _make_tool("init_tool")
        handler = ToolHandler(tools={"init_tool": tool})

        assert "init_tool" in handler.tools


class TestToolHandlerExecution:
    async def test_single_tool_execution(self) -> None:
        """Execute a single tool call."""
        handler = ToolHandler(tools={"greet": _make_tool("greet", "Hello!")})
        input_calls = {"call-1": {"name": "greet", "arguments": {}}}

        results = [r async for r in handler.handle(input_calls)]

        assert len(results) == 1
        assert results[0].tool_call_id == "call-1"
        assert results[0].tool_name == "greet"
        assert results[0].content == "Hello!"
        assert results[0].error is None

    async def test_parallel_tool_execution(self) -> None:
        """Execute multiple tool calls in parallel."""
        handler = ToolHandler(
            tools={
                "add": _make_tool("add", "sum"),
                "mul": _make_tool("mul", "product"),
            }
        )
        input_calls = {
            "c1": {"name": "add", "arguments": {"a": 1, "b": 2}},
            "c2": {"name": "mul", "arguments": {"a": 3, "b": 4}},
        }

        results = [r async for r in handler.handle(input_calls)]

        assert len(results) == 2
        names = {r.tool_name for r in results}
        assert names == {"add", "mul"}

    async def test_unknown_tool_returns_error(self) -> None:
        """Unknown tool produces a ToolResult with an error."""
        handler = ToolHandler()
        input_calls = {"c1": {"name": "nonexistent", "arguments": {}}}

        results = [r async for r in handler.handle(input_calls)]

        assert len(results) == 1
        assert results[0].error is not None
        assert "Unknown tool" in results[0].error

    async def test_tool_error_caught(self) -> None:
        """ToolError from execution is caught and returned in result."""
        handler = ToolHandler(tools={"bad": _make_failing_tool("bad", "it broke")})
        input_calls = {"c1": {"name": "bad", "arguments": {}}}

        results = [r async for r in handler.handle(input_calls)]

        assert len(results) == 1
        assert results[0].error is not None
        assert "it broke" in results[0].error

    async def test_empty_input_yields_nothing(self) -> None:
        """Empty input dict yields no results."""
        handler = ToolHandler()
        results = [r async for r in handler.handle({})]

        assert results == []

    async def test_result_ordering_matches_input(self) -> None:
        """Results are yielded in the same order as input keys."""
        handler = ToolHandler(
            tools={
                "a": _make_tool("a", "out_a"),
                "b": _make_tool("b", "out_b"),
                "c": _make_tool("c", "out_c"),
            }
        )
        input_calls = {
            "id1": {"name": "a", "arguments": {}},
            "id2": {"name": "b", "arguments": {}},
            "id3": {"name": "c", "arguments": {}},
        }

        results = [r async for r in handler.handle(input_calls)]

        assert [r.tool_call_id for r in results] == ["id1", "id2", "id3"]
        assert [r.content for r in results] == ["out_a", "out_b", "out_c"]


class TestToolHandlerAggregate:
    def test_aggregate_success(self) -> None:
        """Aggregate returns content for successful results."""
        handler = ToolHandler()
        results = [
            ToolResult(tool_call_id="c1", tool_name="a", content="ok"),
            ToolResult(tool_call_id="c2", tool_name="b", content="done"),
        ]
        agg = handler.aggregate(results)

        assert agg == {"c1": "ok", "c2": "done"}

    def test_aggregate_with_errors(self) -> None:
        """Aggregate returns error string for failed results."""
        handler = ToolHandler()
        results = [
            ToolResult(tool_call_id="c1", tool_name="a", content="ok"),
            ToolResult(tool_call_id="c2", tool_name="b", error="failed"),
        ]
        agg = handler.aggregate(results)

        assert agg == {"c1": "ok", "c2": "failed"}


# ---------------------------------------------------------------------------
# GroupHandler
# ---------------------------------------------------------------------------


class TestGroupHandlerParallel:
    async def test_parallel_two_agents(self) -> None:
        """Parallel group runs two agents concurrently."""
        agent_a = Agent(name="a")
        agent_b = Agent(name="b")
        provider = _make_provider([AgentOutput(text="result_a"), AgentOutput(text="result_b")])

        handler = GroupHandler(
            agents={"a": agent_a, "b": agent_b},
            provider=provider,
            parallel=True,
        )
        results = [r async for r in handler.handle("input")]

        assert len(results) == 2
        outputs = {r.output for r in results}
        assert "result_a" in outputs or "result_b" in outputs

    async def test_parallel_same_input(self) -> None:
        """All parallel agents receive the same input — both produce results."""
        provider = _make_provider([AgentOutput(text="from_x"), AgentOutput(text="from_y")])
        handler = GroupHandler(
            agents={"x": Agent(name="x"), "y": Agent(name="y")},
            provider=provider,
            parallel=True,
        )
        results = [r async for r in handler.handle("shared_input")]

        # Both agents ran and produced results
        assert len(results) == 2
        outputs = {r.output for r in results}
        assert outputs == {"from_x", "from_y"}

    async def test_parallel_single_agent(self) -> None:
        """Parallel group with one agent works correctly."""
        provider = _make_provider([AgentOutput(text="solo")])
        handler = GroupHandler(
            agents={"only": Agent(name="only")},
            provider=provider,
            parallel=True,
        )
        results = [r async for r in handler.handle("go")]

        assert len(results) == 1
        assert results[0].output == "solo"


class TestGroupHandlerSerial:
    async def test_serial_chaining(self) -> None:
        """Serial group chains output -> input between agents."""
        provider = _make_provider([AgentOutput(text="step1"), AgentOutput(text="step2")])
        handler = GroupHandler(
            agents={"a": Agent(name="a"), "b": Agent(name="b")},
            provider=provider,
            parallel=False,
            dependencies={"b": ["a"]},
        )
        results = [r async for r in handler.handle("start")]

        assert len(results) == 2
        assert results[0].output == "step1"
        assert results[1].output == "step2"

    async def test_serial_no_dependencies(self) -> None:
        """Serial group without dependencies runs in registration order."""
        provider = _make_provider([AgentOutput(text="first"), AgentOutput(text="second")])
        handler = GroupHandler(
            agents={"x": Agent(name="x"), "y": Agent(name="y")},
            provider=provider,
            parallel=False,
        )
        results = [r async for r in handler.handle("go")]

        assert len(results) == 2

    async def test_serial_missing_agent_raises(self) -> None:
        """Serial group with missing agent in agents dict raises HandlerError."""

        class _BadGroupHandler(GroupHandler):
            """Override _resolve_order to return a non-existent agent."""

            def _resolve_order(self) -> list[str]:
                return ["a", "missing"]

        handler = _BadGroupHandler(
            agents={"a": Agent(name="a")},
            provider=_make_provider([AgentOutput(text="a_out")]),
            parallel=False,
        )

        with pytest.raises(HandlerError, match="not found"):
            async for _ in handler.handle("test"):
                pass

    async def test_serial_diamond_dependency(self) -> None:
        """Serial group resolves diamond dependencies correctly."""
        # a -> b, a -> c, b -> d, c -> d
        provider = _make_provider(
            [
                AgentOutput(text="a_out"),
                AgentOutput(text="b_out"),
                AgentOutput(text="c_out"),
                AgentOutput(text="d_out"),
            ]
        )
        handler = GroupHandler(
            agents={
                "a": Agent(name="a"),
                "b": Agent(name="b"),
                "c": Agent(name="c"),
                "d": Agent(name="d"),
            },
            provider=provider,
            parallel=False,
            dependencies={
                "b": ["a"],
                "c": ["a"],
                "d": ["b", "c"],
            },
        )
        results = [r async for r in handler.handle("start")]

        assert len(results) == 4
        # 'a' must come first, 'd' must come last
        assert results[0].output == "a_out"
        assert results[-1].output == "d_out"


class TestGroupHandlerDependencyResolution:
    def test_resolve_no_deps(self) -> None:
        """Resolution with no dependencies returns registration order."""
        handler = GroupHandler(
            agents={"a": Agent(name="a"), "b": Agent(name="b")},
            parallel=False,
        )
        order = handler._resolve_order()

        assert order == ["a", "b"]

    def test_resolve_linear_chain(self) -> None:
        """Resolution handles a linear dependency chain."""
        handler = GroupHandler(
            agents={"c": Agent(name="c"), "b": Agent(name="b"), "a": Agent(name="a")},
            parallel=False,
            dependencies={"b": ["a"], "c": ["b"]},
        )
        order = handler._resolve_order()

        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_resolve_cycle_raises(self) -> None:
        """Cyclic dependencies raise HandlerError."""
        handler = GroupHandler(
            agents={"a": Agent(name="a"), "b": Agent(name="b")},
            parallel=False,
            dependencies={"a": ["b"], "b": ["a"]},
        )

        with pytest.raises(HandlerError, match="cycle"):
            handler._resolve_order()

    def test_resolve_diamond(self) -> None:
        """Diamond dependency graph is resolved correctly."""
        handler = GroupHandler(
            agents={
                "a": Agent(name="a"),
                "b": Agent(name="b"),
                "c": Agent(name="c"),
                "d": Agent(name="d"),
            },
            parallel=False,
            dependencies={"b": ["a"], "c": ["a"], "d": ["b", "c"]},
        )
        order = handler._resolve_order()

        assert order[0] == "a"
        assert order[-1] == "d"
        assert set(order) == {"a", "b", "c", "d"}
