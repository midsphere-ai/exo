"""Tests for Programmatic Tool Calling (PTC).

All tests use mock providers — no real API calls.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from exo.agent import Agent, AgentError
from exo.hooks import HookPoint
from exo.models.types import ModelResponse  # pyright: ignore[reportMissingImports]
from exo.ptc import (
    PTC_TOOL_NAME,
    PTCExecutor,
    PTCTool,
    build_tool_signatures,
    get_ptc_eligible_tools,
    schema_to_python_sig,
)
from exo.runner import run
from exo.tool import Tool, tool
from exo.types import AgentOutput, StreamEvent, TextEvent, ToolCall, ToolCallEvent, ToolResultEvent, Usage

# ---------------------------------------------------------------------------
# Test tools
# ---------------------------------------------------------------------------


@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


@tool
def search(query: str, max_results: int = 5) -> str:
    """Search for information."""
    return json.dumps([{"title": f"Result for {query}", "rank": i} for i in range(max_results)])


@tool
def failing_tool(msg: str) -> str:
    """A tool that always fails."""
    raise ValueError(f"Intentional error: {msg}")


# ---------------------------------------------------------------------------
# Mock provider helpers
# ---------------------------------------------------------------------------


def _mock_provider(
    content: str = "Done!",
    tool_calls: list[ToolCall] | None = None,
) -> AsyncMock:
    resp = ModelResponse(
        id="resp-1",
        model="test-model",
        content=content,
        tool_calls=tool_calls or [],
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    provider = AsyncMock()
    provider.complete = AsyncMock(return_value=resp)
    return provider


def _multi_step_provider(*responses: ModelResponse) -> AsyncMock:
    provider = AsyncMock()
    provider.complete = AsyncMock(side_effect=list(responses))
    return provider


# ---------------------------------------------------------------------------
# Init & registration tests
# ---------------------------------------------------------------------------


class TestPTCInit:
    def test_ptc_false_by_default(self) -> None:
        agent = Agent(name="bot", tools=[greet])
        assert agent.ptc is False
        assert PTC_TOOL_NAME not in agent.tools

    def test_ptc_true_registers_ptc_tool(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        assert agent.ptc is True
        assert PTC_TOOL_NAME in agent.tools
        assert isinstance(agent.tools[PTC_TOOL_NAME], PTCTool)

    def test_ptc_timeout_default(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        assert agent.ptc_timeout == 60

    def test_ptc_timeout_custom(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True, ptc_timeout=120)
        assert agent.ptc_timeout == 120

    def test_ptc_collision_raises(self) -> None:
        @tool(name=PTC_TOOL_NAME)
        def my_tool() -> str:
            """Conflicting name."""
            return "x"

        with pytest.raises(AgentError, match=r"already registered"):
            Agent(name="bot", tools=[my_tool], ptc=True)

    def test_ptc_tool_is_tool_subclass(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        assert isinstance(agent.tools[PTC_TOOL_NAME], Tool)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestPTCSchemas:
    def test_schemas_exclude_user_tools(self) -> None:
        """PTC-eligible tools should NOT appear as individual schemas."""
        agent = Agent(name="bot", tools=[greet, add], ptc=True)
        schemas = agent.get_tool_schemas()
        schema_names = {s["function"]["name"] for s in schemas}

        assert PTC_TOOL_NAME in schema_names
        assert "greet" not in schema_names
        assert "add" not in schema_names
        # retrieve_artifact is always auto-registered and should stay direct
        assert "retrieve_artifact" in schema_names

    def test_schemas_hitl_tools_stay_direct(self) -> None:
        """HITL tools must remain as direct schemas, not PTC-wrapped."""
        agent = Agent(
            name="bot",
            tools=[greet, add],
            hitl_tools=["greet"],
            ptc=True,
        )
        schemas = agent.get_tool_schemas()
        schema_names = {s["function"]["name"] for s in schemas}

        assert PTC_TOOL_NAME in schema_names
        assert "greet" in schema_names  # HITL → stays direct
        assert "add" not in schema_names  # not HITL → PTC-wrapped

    def test_description_lists_tool_signatures(self) -> None:
        agent = Agent(name="bot", tools=[greet, search], ptc=True)
        ptc_tool = agent.tools[PTC_TOOL_NAME]
        desc = ptc_tool.description

        assert "async def greet" in desc
        assert "name: str" in desc
        assert "async def search" in desc
        assert "query: str" in desc
        assert "max_results: int" in desc

    def test_description_excludes_framework_tools(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        desc = agent.tools[PTC_TOOL_NAME].description

        assert "retrieve_artifact" not in desc
        assert PTC_TOOL_NAME not in desc

    def test_description_excludes_hitl_tools(self) -> None:
        agent = Agent(
            name="bot",
            tools=[greet, add],
            hitl_tools=["greet"],
            ptc=True,
        )
        desc = agent.tools[PTC_TOOL_NAME].description

        assert "greet" not in desc  # HITL tools not in PTC namespace
        assert "async def add" in desc

    async def test_dynamic_tool_add_updates_schemas(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)

        # Cache schemas
        schemas1 = agent.get_tool_schemas()
        schema_names1 = {s["function"]["name"] for s in schemas1}
        assert "add" not in schema_names1

        # Add tool dynamically
        await agent.add_tool(add)

        schemas2 = agent.get_tool_schemas()
        schema_names2 = {s["function"]["name"] for s in schemas2}
        # add should still not be in schemas (it's PTC-eligible)
        assert "add" not in schema_names2
        # But should appear in PTC tool description
        desc = agent.tools[PTC_TOOL_NAME].description
        assert "async def add" in desc


# ---------------------------------------------------------------------------
# Signature generation tests
# ---------------------------------------------------------------------------


class TestSignatureGeneration:
    def test_simple_required_param(self) -> None:
        sig = schema_to_python_sig(greet)
        assert sig == "async def greet(name: str) -> str"

    def test_optional_param(self) -> None:
        sig = schema_to_python_sig(search)
        assert "query: str" in sig
        assert "max_results" in sig

    def test_multiple_required(self) -> None:
        sig = schema_to_python_sig(add)
        assert "a: int" in sig
        assert "b: int" in sig

    def test_build_tool_signatures(self) -> None:
        result = build_tool_signatures({"greet": greet, "add": add})
        assert "async def greet" in result
        assert "async def add" in result
        assert '"""Greet someone by name."""' in result
        assert '"""Add two numbers."""' in result

    def test_build_tool_signatures_empty(self) -> None:
        assert build_tool_signatures({}) == "(no tools available)"


# ---------------------------------------------------------------------------
# PTCExecutor unit tests
# ---------------------------------------------------------------------------


class TestPTCExecutor:
    """Test PTCExecutor.run() directly."""

    def _make_agent(self, tools: list[Tool] | None = None) -> Agent:
        return Agent(name="test", tools=tools or [greet, add, search], ptc=True)

    async def test_simple_print(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run('print("hello world")')
        assert result == "hello world"

    async def test_return_value(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run("return 42")
        assert result == "42"

    async def test_print_and_return(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run('print("line1")\nreturn 99')
        assert "line1" in result
        assert "99" in result

    async def test_tool_call(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run('r = await default_api.greet(name="Alice")\nprint(r)')
        assert "Hello, Alice!" in result

    async def test_multiple_tools_sequential(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        code = """\
r1 = await default_api.greet(name="Alice")
r2 = await default_api.add(a=3, b=4)
print(f"{r1} sum={r2}")
"""
        result = await executor.run(code)
        assert "Hello, Alice!" in result
        assert "sum=7" in result

    async def test_parallel_gather(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        code = """\
results = await asyncio.gather(
    default_api.greet(name="A"),
    default_api.greet(name="B"),
    default_api.greet(name="C"),
)
for r in results:
    print(r)
"""
        result = await executor.run(code)
        assert "Hello, A!" in result
        assert "Hello, B!" in result
        assert "Hello, C!" in result

    async def test_loop_with_filter(self) -> None:
        """Core PTC use case: loop + filter to reduce context."""
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        code = """\
data = json.loads(await default_api.search(query="test", max_results=5))
high_rank = [d for d in data if d["rank"] >= 3]
print(json.dumps(high_rank))
"""
        result = await executor.run(code)
        parsed = json.loads(result)
        assert len(parsed) == 2  # rank 3 and 4
        assert all(d["rank"] >= 3 for d in parsed)

    async def test_syntax_error(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run("def foo(")
        assert "SyntaxError" in result

    async def test_runtime_error(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run("x = 1 / 0")
        assert "ZeroDivisionError" in result

    async def test_tool_error_propagates(self) -> None:
        agent = Agent(name="test", tools=[failing_tool], ptc=True)
        executor = PTCExecutor(agent)
        # Error should appear in the output (caught by outer handler)
        result = await executor.run('await default_api.failing_tool(msg="boom")')
        assert "Intentional error: boom" in result

    async def test_tool_error_catchable_in_code(self) -> None:
        """User code can catch tool errors and continue."""
        agent = Agent(name="test", tools=[failing_tool, greet], ptc=True)
        executor = PTCExecutor(agent)
        code = """\
try:
    await default_api.failing_tool(msg="oops")
except Exception as e:
    print(f"caught: {e}")
r = await default_api.greet(name="OK")
print(r)
"""
        result = await executor.run(code)
        assert "caught:" in result
        assert "Hello, OK!" in result

    async def test_timeout(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent, timeout=1)
        result = await executor.run("await asyncio.sleep(10)")
        assert "TimeoutError" in result

    async def test_stdlib_available(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        code = """\
import math as m  # also available without import
print(math.sqrt(16))
print(json.dumps({"a": 1}))
print(re.sub(r"\\d+", "X", "abc123"))
"""
        result = await executor.run(code)
        assert "4.0" in result
        assert '{"a": 1}' in result
        assert "abcX" in result

    async def test_no_output(self) -> None:
        agent = self._make_agent()
        executor = PTCExecutor(agent)
        result = await executor.run("x = 1 + 1")
        assert result == "(no output)"


# ---------------------------------------------------------------------------
# Hook tests
# ---------------------------------------------------------------------------


class TestPTCHooks:
    async def test_hooks_fire_for_inner_calls(self) -> None:
        agent = Agent(name="test", tools=[greet], ptc=True)
        hook_calls: list[tuple[str, str]] = []

        async def capture_hook(**kwargs: Any) -> None:
            hook_calls.append((kwargs.get("tool_name", ""), "called"))

        agent.hook_manager.add(HookPoint.PRE_TOOL_CALL, capture_hook)
        agent.hook_manager.add(HookPoint.POST_TOOL_CALL, capture_hook)

        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="Test")')

        # PRE + POST for the inner greet call
        tool_names = [name for name, _ in hook_calls]
        assert tool_names.count("greet") == 2

    async def test_hooks_correct_arguments(self) -> None:
        agent = Agent(name="test", tools=[greet], ptc=True)
        captured_args: list[dict[str, Any]] = []

        async def capture_pre(**kwargs: Any) -> None:
            if kwargs.get("arguments"):
                captured_args.append(kwargs["arguments"])

        agent.hook_manager.add(HookPoint.PRE_TOOL_CALL, capture_pre)

        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="Alice")')

        assert len(captured_args) == 1
        assert captured_args[0] == {"name": "Alice"}

    async def test_hooks_fire_for_multiple_inner_calls(self) -> None:
        agent = Agent(name="test", tools=[greet, add], ptc=True)
        pre_calls: list[str] = []

        async def capture_pre(**kwargs: Any) -> None:
            pre_calls.append(kwargs.get("tool_name", ""))

        agent.hook_manager.add(HookPoint.PRE_TOOL_CALL, capture_pre)

        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="A")\nawait default_api.add(a=1, b=2)')

        assert "greet" in pre_calls
        assert "add" in pre_calls


# ---------------------------------------------------------------------------
# PTC-eligible tools helper
# ---------------------------------------------------------------------------


class TestPTCEligible:
    def test_excludes_framework_tools(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        eligible = get_ptc_eligible_tools(agent)
        assert "greet" in eligible
        assert PTC_TOOL_NAME not in eligible
        assert "retrieve_artifact" not in eligible

    def test_excludes_hitl_tools(self) -> None:
        agent = Agent(name="bot", tools=[greet, add], hitl_tools=["greet"], ptc=True)
        eligible = get_ptc_eligible_tools(agent)
        assert "greet" not in eligible
        assert "add" in eligible


# ---------------------------------------------------------------------------
# Integration tests: full agent.run()
# ---------------------------------------------------------------------------


class TestPTCIntegration:
    async def test_end_to_end_run(self) -> None:
        """MockProvider returns PTC tool call → agent runs PTC → final text."""
        tc = ToolCall(
            id="tc-1",
            name=PTC_TOOL_NAME,
            arguments=json.dumps(
                {"code": 'r = await default_api.greet(name="World")\nprint(r)'}
            ),
        )
        resp_tool = ModelResponse(
            content="",
            tool_calls=[tc],
            usage=Usage(input_tokens=10, output_tokens=20, total_tokens=30),
        )
        resp_final = ModelResponse(
            content="I greeted World for you!",
            tool_calls=[],
            usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
        )
        provider = _multi_step_provider(resp_tool, resp_final)

        agent = Agent(name="bot", tools=[greet], ptc=True)
        output = await agent.run("Say hi to World", provider=provider)

        assert output.text == "I greeted World for you!"
        assert provider.complete.await_count == 2

    async def test_ptc_tool_result_contains_output(self) -> None:
        """The tool result from PTC tool should contain the code output."""
        tc = ToolCall(
            id="tc-1",
            name=PTC_TOOL_NAME,
            arguments=json.dumps(
                {"code": 'r = await default_api.greet(name="Test")\nprint(r)'}
            ),
        )
        # Track messages sent to provider
        call_count = 0
        captured_msgs: list[Any] = []

        async def fake_complete(messages: Any, **kwargs: Any) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            captured_msgs.append(list(messages))
            if call_count == 1:
                return ModelResponse(
                    content="",
                    tool_calls=[tc],
                    usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
                )
            return ModelResponse(
                content="Done",
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            )

        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=fake_complete)

        agent = Agent(name="bot", tools=[greet], ptc=True)
        await agent.run("test", provider=provider)

        # The second call should have the tool result with "Hello, Test!"
        second_call_msgs = captured_msgs[1]
        tool_results = [m for m in second_call_msgs if getattr(m, "role", "") == "tool"]
        assert len(tool_results) == 1
        assert "Hello, Test!" in tool_results[0].content


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestPTCSerialization:
    def test_to_dict_includes_ptc(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True, ptc_timeout=120)
        data = agent.to_dict()
        assert data["ptc"] is True
        assert data["ptc_timeout"] == 120

    def test_to_dict_ptc_false(self) -> None:
        agent = Agent(name="bot", tools=[greet])
        data = agent.to_dict()
        assert data["ptc"] is False

    def test_ptc_tool_not_in_serialized_tools(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        data = agent.to_dict()
        if "tools" in data:
            tool_names = [t.get("name", t) if isinstance(t, dict) else t for t in data["tools"]]
            assert PTC_TOOL_NAME not in str(tool_names)

    def test_describe_includes_ptc(self) -> None:
        agent = Agent(name="bot", tools=[greet], ptc=True)
        info = agent.describe()
        assert info["ptc"] is True


# ---------------------------------------------------------------------------
# Swarm propagation tests
# ---------------------------------------------------------------------------


class TestSwarmPTC:
    def test_swarm_ptc_propagation(self) -> None:
        from exo.swarm import Swarm

        a1 = Agent(name="a1", tools=[greet])
        a2 = Agent(name="a2", tools=[add])
        assert a1.ptc is False
        assert a2.ptc is False

        Swarm(agents=[a1, a2], ptc=True)
        assert a1.ptc is True
        assert a2.ptc is True

    def test_swarm_ptc_none_no_change(self) -> None:
        from exo.swarm import Swarm

        a1 = Agent(name="a1", tools=[greet], ptc=True)
        a2 = Agent(name="a2", tools=[add])

        Swarm(agents=[a1, a2])  # ptc=None by default
        assert a1.ptc is True  # unchanged
        assert a2.ptc is False  # unchanged


# ---------------------------------------------------------------------------
# PTC transparency tests — synthetic event emission
# ---------------------------------------------------------------------------


class TestPTCTransparency:
    """Verify that PTC emits per-tool ToolCallEvent/ToolResultEvent to the queue."""

    async def test_single_tool_emits_events(self) -> None:
        agent = Agent(name="test", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="Alice")')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        result_events = [e for e in events if isinstance(e, ToolResultEvent)]

        assert len(call_events) == 1
        assert call_events[0].tool_name == "greet"
        assert call_events[0].agent_name == "test"
        assert '"Alice"' in call_events[0].arguments

        assert len(result_events) == 1
        assert result_events[0].tool_name == "greet"
        assert result_events[0].success is True
        assert "Hello, Alice!" in str(result_events[0].result)

    async def test_multiple_tools_emit_ordered_events(self) -> None:
        agent = Agent(name="test", tools=[greet, add], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="A")\nawait default_api.add(a=1, b=2)')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        tool_events = [e for e in events if isinstance(e, (ToolCallEvent, ToolResultEvent))]

        # Expect: call(greet), result(greet), call(add), result(add)
        assert len(tool_events) == 4
        assert tool_events[0].tool_name == "greet"
        assert isinstance(tool_events[0], ToolCallEvent)
        assert tool_events[1].tool_name == "greet"
        assert isinstance(tool_events[1], ToolResultEvent)
        assert tool_events[2].tool_name == "add"
        assert isinstance(tool_events[2], ToolCallEvent)
        assert tool_events[3].tool_name == "add"
        assert isinstance(tool_events[3], ToolResultEvent)

    async def test_error_tool_emits_failure_event(self) -> None:
        agent = Agent(name="test", tools=[failing_tool], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.failing_tool(msg="boom")')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert result_events[0].success is False
        assert "boom" in (result_events[0].error or "")

    async def test_events_have_unique_call_ids(self) -> None:
        agent = Agent(name="test", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="A")\nawait default_api.greet(name="B")')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(call_events) == 2
        assert call_events[0].tool_call_id != call_events[1].tool_call_id

    async def test_call_and_result_share_call_id(self) -> None:
        agent = Agent(name="test", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="X")')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        call_ev = [e for e in events if isinstance(e, ToolCallEvent)][0]
        result_ev = [e for e in events if isinstance(e, ToolResultEvent)][0]
        assert call_ev.tool_call_id == result_ev.tool_call_id

    async def test_result_has_duration(self) -> None:
        agent = Agent(name="test", tools=[greet], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="A")')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert result_events[0].duration_ms >= 0.0

    async def test_no_ptc_tool_name_in_events(self) -> None:
        """No event should reference the internal PTC tool name."""
        agent = Agent(name="test", tools=[greet, add], ptc=True)
        executor = PTCExecutor(agent)
        await executor.run('await default_api.greet(name="A")\nawait default_api.add(a=1, b=2)')

        events: list[Any] = []
        while not agent._event_queue.empty():
            events.append(agent._event_queue.get_nowait())

        for ev in events:
            if hasattr(ev, "tool_name"):
                assert ev.tool_name != PTC_TOOL_NAME


# ---------------------------------------------------------------------------
# Streaming + non-streaming integration helpers
# ---------------------------------------------------------------------------


class _FakeStreamChunk:
    """Minimal stream chunk for PTC integration tests."""

    def __init__(
        self,
        delta: str = "",
        tool_call_deltas: list[Any] | None = None,
        finish_reason: str | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.delta = delta
        self.tool_call_deltas = tool_call_deltas or []
        self.finish_reason = finish_reason
        self.usage = usage or Usage()


class _FakeToolCallDelta:
    """Minimal tool call delta for PTC integration tests."""

    def __init__(
        self,
        index: int = 0,
        id: str | None = None,
        name: str | None = None,
        arguments: str = "",
    ) -> None:
        self.index = index
        self.id = id
        self.name = name
        self.arguments = arguments


def _make_stream_provider(stream_rounds: list[list[_FakeStreamChunk]]) -> Any:
    """Create a mock provider returning pre-defined stream chunks."""
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


def _ptc_tool_call(code: str) -> list[_FakeStreamChunk]:
    """Build a stream round that calls the __exo_ptc__ tool with *code*."""
    return [
        _FakeStreamChunk(
            tool_call_deltas=[
                _FakeToolCallDelta(index=0, id="tc_ptc_1", name=PTC_TOOL_NAME),
            ],
        ),
        _FakeStreamChunk(
            tool_call_deltas=[
                _FakeToolCallDelta(index=0, arguments=json.dumps({"code": code})),
            ],
            finish_reason="tool_calls",
        ),
    ]


# ---------------------------------------------------------------------------
# PTC streaming integration tests
# ---------------------------------------------------------------------------


class TestPTCStreamingIntegration:
    """End-to-end: PTC events flow through run.stream()."""

    async def test_ptc_events_in_stream_detailed(self) -> None:
        """run.stream(detailed=True) yields inner ToolCallEvent + ToolResultEvent."""
        agent = Agent(name="bot", tools=[greet], ptc=True)
        round1 = _ptc_tool_call('await default_api.greet(name="Alice")')
        round2 = [_FakeStreamChunk(delta="Done!")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "hi", provider=provider, detailed=True)]

        call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        result_events = [e for e in events if isinstance(e, ToolResultEvent)]

        assert len(call_events) == 1
        assert call_events[0].tool_name == "greet"
        assert call_events[0].agent_name == "bot"

        assert len(result_events) == 1
        assert result_events[0].tool_name == "greet"
        assert result_events[0].success is True
        assert "Hello, Alice!" in str(result_events[0].result)

    async def test_ptc_tool_call_without_detailed(self) -> None:
        """run.stream(detailed=False) yields ToolCallEvent but NOT ToolResultEvent."""
        agent = Agent(name="bot", tools=[greet], ptc=True)
        round1 = _ptc_tool_call('await default_api.greet(name="Bob")')
        round2 = [_FakeStreamChunk(delta="Done!")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "hi", provider=provider, detailed=False)]

        call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        result_events = [e for e in events if isinstance(e, ToolResultEvent)]

        assert len(call_events) == 1
        assert call_events[0].tool_name == "greet"
        # ToolResultEvent requires detailed=True
        assert len(result_events) == 0

    async def test_ptc_outer_tool_suppressed(self) -> None:
        """No event should reference __exo_ptc__ in the stream."""
        agent = Agent(name="bot", tools=[greet], ptc=True)
        round1 = _ptc_tool_call('await default_api.greet(name="X")')
        round2 = [_FakeStreamChunk(delta="ok")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "hi", provider=provider, detailed=True)]

        for ev in events:
            if hasattr(ev, "tool_name"):
                assert ev.tool_name != PTC_TOOL_NAME

    async def test_ptc_multiple_inner_tools_stream(self) -> None:
        """Multiple PTC inner tool calls emit ordered events in stream."""
        agent = Agent(name="bot", tools=[greet, add], ptc=True)
        code = 'await default_api.greet(name="A")\nawait default_api.add(a=1, b=2)'
        round1 = _ptc_tool_call(code)
        round2 = [_FakeStreamChunk(delta="ok")]
        provider = _make_stream_provider([round1, round2])

        events = [ev async for ev in run.stream(agent, "hi", provider=provider, detailed=True)]

        call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        result_events = [e for e in events if isinstance(e, ToolResultEvent)]

        assert len(call_events) == 2
        assert call_events[0].tool_name == "greet"
        assert call_events[1].tool_name == "add"
        assert len(result_events) == 2
        assert result_events[0].tool_name == "greet"
        assert result_events[1].tool_name == "add"


# ---------------------------------------------------------------------------
# Non-streaming: queue drain prevents stale events
# ---------------------------------------------------------------------------


class TestPTCNonStreamingDrain:
    """Verify that run() (non-streaming) drains PTC events from the queue."""

    async def test_run_drains_ptc_event_queue(self) -> None:
        """After run(), the agent's _event_queue should be empty."""
        agent = Agent(name="bot", tools=[greet], ptc=True)
        provider = _multi_step_provider(
            ModelResponse(
                id="r1",
                model="test",
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name=PTC_TOOL_NAME,
                        arguments=json.dumps({"code": 'await default_api.greet(name="Z")'}),
                    )
                ],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                id="r2",
                model="test",
                content="Done",
                tool_calls=[],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        )

        result = await run(agent, "greet Z", provider=provider)

        assert result.output == "Done"
        assert agent._event_queue.empty(), (
            "_event_queue should be empty after non-streaming run() — "
            "PTC events must be drained to prevent memory leaks"
        )

    async def test_run_then_stream_no_stale_events(self) -> None:
        """Stale PTC events from run() must not leak into a later stream()."""
        agent = Agent(name="bot", tools=[greet], ptc=True)

        # First: non-streaming run
        provider1 = _multi_step_provider(
            ModelResponse(
                id="r1",
                model="test",
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc1",
                        name=PTC_TOOL_NAME,
                        arguments=json.dumps({"code": 'await default_api.greet(name="Old")'}),
                    )
                ],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                id="r2",
                model="test",
                content="done1",
                tool_calls=[],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        )
        await run(agent, "first", provider=provider1)

        # Second: streaming run (text-only, no tool calls)
        provider2 = _make_stream_provider(
            [[_FakeStreamChunk(delta="Hello stream!")]]
        )
        events = [ev async for ev in run.stream(agent, "second", provider=provider2, detailed=True)]

        # No stale ToolCallEvent/ToolResultEvent from the first run should appear
        tool_events = [
            e for e in events if isinstance(e, (ToolCallEvent, ToolResultEvent))
        ]
        assert len(tool_events) == 0, (
            f"Expected no tool events in text-only stream, got {len(tool_events)}"
        )
