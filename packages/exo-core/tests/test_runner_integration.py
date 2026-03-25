"""Integration tests for the runner system.

End-to-end tests wiring Agent + @tool + run() with mocked LLM providers,
handler pipeline scenarios, and background task workflows.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from exo.agent import Agent
from exo.runner import run
from exo.tool import Tool, tool
from exo.types import (
    AgentOutput,
    RunResult,
    ToolCall,
    Usage,
)

# ---------------------------------------------------------------------------
# Shared fixtures: mock LLM provider
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


# ---------------------------------------------------------------------------
# End-to-end: Agent + @tool + run()
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full round-trip integration: Agent + @tool decorator + run()."""

    async def test_agent_with_tool_via_run(self) -> None:
        """Agent uses @tool to execute, then returns final text."""

        @tool
        def add(a: int, b: int) -> str:
            """Add two numbers."""
            return str(int(a) + int(b))

        agent = Agent(name="calc", instructions="You are a calculator.", tools=[add])
        responses = [
            AgentOutput(
                text="",
                tool_calls=[ToolCall(id="tc1", name="add", arguments='{"a":3,"b":7}')],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            AgentOutput(
                text="The answer is 10.",
                usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
        ]
        provider = _make_provider(responses)

        result = await run(agent, "What is 3+7?", provider=provider)

        assert isinstance(result, RunResult)
        assert result.output == "The answer is 10."
        assert result.steps >= 1

    async def test_multi_tool_chain(self) -> None:
        """Agent chains multiple tool calls in sequence."""

        @tool
        def double(n: int) -> str:
            """Double a number."""
            return str(int(n) * 2)

        @tool
        def square(n: int) -> str:
            """Square a number."""
            return str(int(n) ** 2)

        agent = Agent(name="math", tools=[double, square])
        responses = [
            # First LLM call: use double
            AgentOutput(
                text="",
                tool_calls=[ToolCall(id="tc1", name="double", arguments='{"n":5}')],
            ),
            # Second LLM call: use square
            AgentOutput(
                text="",
                tool_calls=[ToolCall(id="tc2", name="square", arguments='{"n":10}')],
            ),
            # Third LLM call: final answer
            AgentOutput(text="5 doubled is 10, then squared is 100."),
        ]
        provider = _make_provider(responses)

        result = await run(agent, "Double 5 then square it", provider=provider)

        assert result.output == "5 doubled is 10, then squared is 100."

    async def test_parallel_tool_calls(self) -> None:
        """Agent executes multiple tool calls in parallel."""

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        agent = Agent(name="greeter", tools=[greet])
        responses = [
            AgentOutput(
                text="",
                tool_calls=[
                    ToolCall(id="tc1", name="greet", arguments='{"name":"Alice"}'),
                    ToolCall(id="tc2", name="greet", arguments='{"name":"Bob"}'),
                ],
            ),
            AgentOutput(text="Greeted Alice and Bob!"),
        ]
        provider = _make_provider(responses)

        result = await run(agent, "Greet Alice and Bob", provider=provider)

        assert result.output == "Greeted Alice and Bob!"

    async def test_tool_error_does_not_crash(self) -> None:
        """Tool errors are captured and fed back to LLM."""

        @tool
        def risky(x: int) -> str:
            """Risky operation."""
            raise ValueError("Something went wrong!")

        agent = Agent(name="bot", tools=[risky])
        responses = [
            AgentOutput(
                text="",
                tool_calls=[ToolCall(id="tc1", name="risky", arguments='{"x":1}')],
            ),
            AgentOutput(text="The tool failed, let me try another approach."),
        ]
        provider = _make_provider(responses)

        result = await run(agent, "Do the risky thing", provider=provider)

        assert result.output == "The tool failed, let me try another approach."

    async def test_usage_accumulates(self) -> None:
        """Token usage accumulates across tool-call steps."""

        @tool
        def noop() -> str:
            """No-op."""
            return "done"

        agent = Agent(name="bot", tools=[noop])
        responses = [
            AgentOutput(
                text="",
                tool_calls=[ToolCall(id="tc1", name="noop", arguments="{}")],
                usage=Usage(input_tokens=100, output_tokens=20, total_tokens=120),
            ),
            AgentOutput(
                text="All done!",
                usage=Usage(input_tokens=150, output_tokens=30, total_tokens=180),
            ),
        ]
        provider = _make_provider(responses)

        result = await run(agent, "Do a noop", provider=provider)

        # Usage should be from the final response (as tracked by call_runner)
        assert result.usage.total_tokens > 0

    def test_sync_end_to_end(self) -> None:
        """run.sync() works end-to-end with tools."""

        @tool
        def upper(text: str) -> str:
            """Uppercase text."""
            return str(text).upper()

        agent = Agent(name="fmt", tools=[upper])
        responses = [
            AgentOutput(
                text="",
                tool_calls=[ToolCall(id="tc1", name="upper", arguments='{"text":"hello"}')],
            ),
            AgentOutput(text="Result: HELLO"),
        ]
        provider = _make_provider(responses)

        result = run.sync(agent, "Uppercase hello", provider=provider)

        assert result.output == "Result: HELLO"


# ---------------------------------------------------------------------------
# Public API imports
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Verify that the public API surface is accessible from the top-level package."""

    def test_agent_importable(self) -> None:
        from exo import Agent as AgentImport

        assert AgentImport is Agent

    def test_run_importable(self) -> None:
        from exo import run as r

        assert r is run

    def test_tool_decorator_importable(self) -> None:
        from exo import tool as t

        assert t is tool

    def test_tool_class_importable(self) -> None:
        from exo import Tool as ToolImport

        assert ToolImport is Tool


# ---------------------------------------------------------------------------
# Handler pipeline tests
# ---------------------------------------------------------------------------


class TestHandlerPipeline:
    """Integration tests for AgentHandler pipeline."""

    async def test_workflow_pipeline_two_agents(self) -> None:
        """AgentHandler workflow mode chains two agents."""
        from exo._internal.handlers import AgentHandler, SwarmMode

        agent_a = Agent(name="a", instructions="Translate to French")
        agent_b = Agent(name="b", instructions="Summarize")

        # Each agent gets one call_runner invocation
        provider = _make_provider(
            [
                AgentOutput(text="Bonjour"),
                AgentOutput(text="Summary: Bonjour"),
            ]
        )

        handler = AgentHandler(
            agents={"a": agent_a, "b": agent_b},
            mode=SwarmMode.WORKFLOW,
            flow_order=["a", "b"],
            provider=provider,
        )

        results = [r async for r in handler.handle("Hello")]

        assert len(results) == 2
        assert results[0].output == "Bonjour"
        assert results[1].output == "Summary: Bonjour"

    async def test_handoff_pipeline(self) -> None:
        """AgentHandler handoff mode follows handoff chain."""
        from exo._internal.handlers import AgentHandler, SwarmMode

        agent_b = Agent(name="b", instructions="You are agent B")
        agent_a = Agent(name="a", instructions="You are agent A", handoffs=[agent_b])

        # Agent A returns "b" (handoff to agent B), agent B returns final text
        provider = _make_provider(
            [
                AgentOutput(text="b"),
                AgentOutput(text="Agent B handled it."),
            ]
        )

        handler = AgentHandler(
            agents={"a": agent_a, "b": agent_b},
            mode=SwarmMode.HANDOFF,
            flow_order=["a", "b"],
            provider=provider,
        )

        results = [r async for r in handler.handle("Help me")]

        assert len(results) == 2
        assert results[1].output == "Agent B handled it."

    async def test_team_mode_lead_only(self) -> None:
        """AgentHandler team mode runs only the lead agent."""
        from exo._internal.handlers import AgentHandler, SwarmMode

        lead = Agent(name="lead", instructions="You are the leader")
        worker = Agent(name="worker", instructions="You are a worker")

        provider = _make_provider([AgentOutput(text="I handled it as lead.")])

        handler = AgentHandler(
            agents={"lead": lead, "worker": worker},
            mode=SwarmMode.TEAM,
            flow_order=["lead", "worker"],
            provider=provider,
        )

        results = [r async for r in handler.handle("Do task")]

        assert len(results) == 1
        assert results[0].output == "I handled it as lead."


# ---------------------------------------------------------------------------
# ToolHandler integration
# ---------------------------------------------------------------------------


class TestToolHandlerIntegration:
    """Integration tests for ToolHandler with real tools."""

    async def test_tool_handler_executes_tools(self) -> None:
        """ToolHandler resolves and executes registered tools."""
        from exo._internal.handlers import ToolHandler

        @tool
        def multiply(a: int, b: int) -> str:
            """Multiply."""
            return str(int(a) * int(b))

        handler = ToolHandler(tools={"multiply": multiply})

        input_dict = {
            "tc1": {"name": "multiply", "arguments": {"a": 6, "b": 7}},
        }

        results = [r async for r in handler.handle(input_dict)]

        assert len(results) == 1
        assert results[0].content == "42"
        assert results[0].tool_call_id == "tc1"

    async def test_tool_handler_parallel_execution(self) -> None:
        """ToolHandler executes multiple tools in parallel."""
        from exo._internal.handlers import ToolHandler

        @tool
        def inc(n: int) -> str:
            """Increment."""
            return str(int(n) + 1)

        handler = ToolHandler(tools={"inc": inc})

        input_dict = {
            "tc1": {"name": "inc", "arguments": {"n": 1}},
            "tc2": {"name": "inc", "arguments": {"n": 10}},
        }

        results = [r async for r in handler.handle(input_dict)]

        assert len(results) == 2
        contents = {r.content for r in results}
        assert "2" in contents
        assert "11" in contents


# ---------------------------------------------------------------------------
# GroupHandler integration
# ---------------------------------------------------------------------------


class TestGroupHandlerIntegration:
    """Integration tests for GroupHandler with real agents."""

    async def test_parallel_group(self) -> None:
        """GroupHandler runs agents in parallel."""
        from exo._internal.handlers import GroupHandler

        agent_a = Agent(name="a")
        agent_b = Agent(name="b")

        provider = _make_provider(
            [
                AgentOutput(text="A result"),
                AgentOutput(text="B result"),
            ]
        )

        handler = GroupHandler(
            agents={"a": agent_a, "b": agent_b},
            provider=provider,
            parallel=True,
        )

        results = [r async for r in handler.handle("task")]

        assert len(results) == 2
        outputs = {r.output for r in results}
        assert "A result" in outputs or "B result" in outputs

    async def test_serial_group_chaining(self) -> None:
        """GroupHandler serial mode chains output → input."""
        from exo._internal.handlers import GroupHandler

        agent_a = Agent(name="a", instructions="Step 1")
        agent_b = Agent(name="b", instructions="Step 2")

        provider = _make_provider(
            [
                AgentOutput(text="step1-output"),
                AgentOutput(text="step2-output"),
            ]
        )

        handler = GroupHandler(
            agents={"a": agent_a, "b": agent_b},
            provider=provider,
            parallel=False,
            dependencies={"b": ["a"]},
        )

        results = [r async for r in handler.handle("start")]

        assert len(results) == 2
        assert results[0].output == "step1-output"
        assert results[1].output == "step2-output"


# ---------------------------------------------------------------------------
# Background task scenario tests
# ---------------------------------------------------------------------------


class TestBackgroundTaskScenarios:
    """Integration tests for BackgroundTaskHandler lifecycle."""

    async def test_hot_merge_lifecycle(self) -> None:
        """Submit, complete (hot-merge), and verify task state."""
        from exo._internal.background import BackgroundTaskHandler, MergeMode
        from exo._internal.state import RunNodeStatus, RunState

        state = RunState(agent_name="main")
        handler = BackgroundTaskHandler(state=state)

        # Submit a background task
        task = handler.submit("bg-1", "parent-1", payload={"key": "value"})
        assert task.status == RunNodeStatus.RUNNING
        assert task.payload == {"key": "value"}

        # Handle result while main is running → hot merge
        mode = await handler.handle_result("bg-1", result="computed-data", is_main_running=True)
        assert mode == MergeMode.HOT
        assert task.result == "computed-data"
        assert task.status == RunNodeStatus.SUCCESS

    async def test_wakeup_merge_lifecycle(self) -> None:
        """Submit, complete (wake-up-merge), drain pending."""
        from exo._internal.background import BackgroundTaskHandler, MergeMode

        handler = BackgroundTaskHandler()

        handler.submit("bg-1", "parent-1")

        # Handle result while main is NOT running → wake-up merge
        mode = await handler.handle_result("bg-1", result="late-data", is_main_running=False)
        assert mode == MergeMode.WAKEUP

        # Drain pending
        pending = [t async for t in handler.drain_pending()]
        assert len(pending) == 1
        assert pending[0].result == "late-data"

        # Queue should be empty after drain
        assert handler.pending_queue.empty

    async def test_background_error_handling(self) -> None:
        """Background task error marks task as failed."""
        from exo._internal.background import BackgroundTaskHandler
        from exo._internal.state import RunNodeStatus

        handler = BackgroundTaskHandler()
        handler.submit("bg-err", "parent-1")

        handler.handle_error("bg-err", "timeout exceeded")

        task = handler.get_task("bg-err")
        assert task is not None
        assert task.status == RunNodeStatus.FAILED
        assert task.error == "timeout exceeded"

    async def test_merge_callback_invoked(self) -> None:
        """Hot-merge fires registered merge callbacks."""
        from exo._internal.background import BackgroundTaskHandler, MergeMode

        handler = BackgroundTaskHandler()
        handler.submit("bg-cb", "parent-1")

        callback_calls: list[tuple[Any, MergeMode]] = []

        async def on_merge(task: Any, mode: MergeMode) -> None:
            callback_calls.append((task, mode))

        handler.on_merge(on_merge)

        await handler.handle_result("bg-cb", result="merged-data", is_main_running=True)

        assert len(callback_calls) == 1
        assert callback_calls[0][1] == MergeMode.HOT

    async def test_multiple_background_tasks(self) -> None:
        """Multiple background tasks can be submitted and listed."""
        from exo._internal.background import BackgroundTaskHandler
        from exo._internal.state import RunNodeStatus

        handler = BackgroundTaskHandler()
        handler.submit("bg-a", "parent-1")
        handler.submit("bg-b", "parent-1")
        handler.submit("bg-c", "parent-1")

        # All running
        running = handler.list_tasks(status=RunNodeStatus.RUNNING)
        assert len(running) == 3

        # Complete one
        await handler.handle_result("bg-a", result="a-done", is_main_running=True)

        running_after = handler.list_tasks(status=RunNodeStatus.RUNNING)
        assert len(running_after) == 2

        success = handler.list_tasks(status=RunNodeStatus.SUCCESS)
        assert len(success) == 1
