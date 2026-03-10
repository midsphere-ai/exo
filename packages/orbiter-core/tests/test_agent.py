"""Tests for orbiter.agent — Agent class init, configuration & run."""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from orbiter.agent import Agent, AgentError
from orbiter.hooks import HookPoint
from orbiter.models.types import ModelError, ModelResponse  # pyright: ignore[reportMissingImports]
from orbiter.tool import FunctionTool, Tool, tool
from orbiter.types import ToolCall, Usage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@tool
def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class ReportOutput(BaseModel):
    title: str
    body: str


# ---------------------------------------------------------------------------
# Agent creation: minimal & full
# ---------------------------------------------------------------------------


class TestAgentCreation:
    def test_minimal(self) -> None:
        """Agent with only a name uses sensible defaults."""
        agent = Agent(name="bot")
        assert agent.name == "bot"
        assert agent.model == "openai:gpt-4o"
        assert agent.provider_name == "openai"
        assert agent.model_name == "gpt-4o"
        assert agent.instructions == ""
        assert agent.tools == {}
        assert agent.handoffs == {}
        assert agent.output_type is None
        assert agent.max_steps == 10
        assert agent.temperature == 1.0
        assert agent.max_tokens is None
        assert agent.memory is None
        assert agent.context is None

    def test_full_config(self) -> None:
        """Agent accepts all configuration parameters."""

        async def my_hook(**data: Any) -> None:
            pass

        agent = Agent(
            name="researcher",
            model="anthropic:claude-sonnet-4-20250514",
            instructions="Research things.",
            tools=[greet, add],
            hooks=[(HookPoint.PRE_LLM_CALL, my_hook)],
            output_type=ReportOutput,
            max_steps=20,
            temperature=0.7,
            max_tokens=4096,
        )
        assert agent.name == "researcher"
        assert agent.provider_name == "anthropic"
        assert agent.model_name == "claude-sonnet-4-20250514"
        assert agent.instructions == "Research things."
        assert len(agent.tools) == 2
        assert agent.output_type is ReportOutput
        assert agent.max_steps == 20
        assert agent.temperature == 0.7
        assert agent.max_tokens == 4096

    def test_name_is_required(self) -> None:
        """Agent() without name raises TypeError."""
        with pytest.raises(TypeError):
            Agent()  # type: ignore[call-arg]

    def test_all_params_keyword_only(self) -> None:
        """Positional arguments are not accepted."""
        with pytest.raises(TypeError):
            Agent("bot")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Model string parsing
# ---------------------------------------------------------------------------


class TestModelParsing:
    def test_provider_colon_model(self) -> None:
        agent = Agent(name="a", model="anthropic:claude-3-opus")
        assert agent.provider_name == "anthropic"
        assert agent.model_name == "claude-3-opus"

    def test_no_colon_defaults_openai(self) -> None:
        agent = Agent(name="a", model="gpt-4o-mini")
        assert agent.provider_name == "openai"
        assert agent.model_name == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_tools_stored_by_name(self) -> None:
        agent = Agent(name="bot", tools=[greet, add])
        assert "greet" in agent.tools
        assert "add" in agent.tools
        assert isinstance(agent.tools["greet"], FunctionTool)

    def test_duplicate_tool_raises(self) -> None:
        dup = FunctionTool(lambda x: x, name="greet")
        with pytest.raises(AgentError, match="Duplicate tool name 'greet'"):
            Agent(name="bot", tools=[greet, dup])

    def test_tool_abc_subclass(self) -> None:
        """Custom Tool subclasses are accepted."""

        class MyTool(Tool):
            name = "custom"
            description = "A custom tool."
            parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

            async def execute(self, **kwargs: Any) -> str:
                return "done"

        agent = Agent(name="bot", tools=[MyTool()])
        assert "custom" in agent.tools

    def test_get_tool_schemas(self) -> None:
        agent = Agent(name="bot", tools=[greet])
        schemas = agent.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "greet"


# ---------------------------------------------------------------------------
# Handoff registration
# ---------------------------------------------------------------------------


class TestHandoffRegistration:
    def test_handoffs_stored_by_name(self) -> None:
        billing = Agent(name="billing")
        support = Agent(name="support")
        triage = Agent(name="triage", handoffs=[billing, support])
        assert "billing" in triage.handoffs
        assert "support" in triage.handoffs

    def test_duplicate_handoff_raises(self) -> None:
        a1 = Agent(name="helper")
        a2 = Agent(name="helper")
        with pytest.raises(AgentError, match="Duplicate handoff agent 'helper'"):
            Agent(name="main", handoffs=[a1, a2])


# ---------------------------------------------------------------------------
# Hook integration
# ---------------------------------------------------------------------------


class TestHookIntegration:
    def test_hook_manager_initialized(self) -> None:
        agent = Agent(name="bot")
        assert not agent.hook_manager.has_hooks(HookPoint.PRE_LLM_CALL)

    async def test_hooks_registered(self) -> None:
        calls: list[str] = []

        async def on_pre(**data: Any) -> None:
            calls.append("pre")

        agent = Agent(
            name="bot",
            hooks=[(HookPoint.PRE_LLM_CALL, on_pre)],
        )
        assert agent.hook_manager.has_hooks(HookPoint.PRE_LLM_CALL)
        await agent.hook_manager.run(HookPoint.PRE_LLM_CALL)
        assert calls == ["pre"]

    async def test_multiple_hooks(self) -> None:
        calls: list[str] = []

        async def hook_a(**data: Any) -> None:
            calls.append("a")

        async def hook_b(**data: Any) -> None:
            calls.append("b")

        agent = Agent(
            name="bot",
            hooks=[
                (HookPoint.START, hook_a),
                (HookPoint.FINISHED, hook_b),
            ],
        )
        await agent.hook_manager.run(HookPoint.START)
        await agent.hook_manager.run(HookPoint.FINISHED)
        assert calls == ["a", "b"]


# ---------------------------------------------------------------------------
# Instructions (str and callable)
# ---------------------------------------------------------------------------


class TestInstructions:
    def test_string_instructions(self) -> None:
        agent = Agent(name="bot", instructions="Be helpful.")
        assert agent.instructions == "Be helpful."

    def test_callable_instructions(self) -> None:
        def make_instructions(agent_name: str) -> str:
            return f"You are {agent_name}."

        agent = Agent(name="bot", instructions=make_instructions)
        assert callable(agent.instructions)
        assert agent.instructions("bot") == "You are bot."  # type: ignore[operator]


# ---------------------------------------------------------------------------
# describe() and __repr__
# ---------------------------------------------------------------------------


class TestDescribeAndRepr:
    def test_describe_minimal(self) -> None:
        agent = Agent(name="bot")
        desc = agent.describe()
        assert desc["name"] == "bot"
        assert desc["model"] == "openai:gpt-4o"
        assert desc["tools"] == []
        assert desc["handoffs"] == []
        assert desc["output_type"] is None

    def test_describe_with_tools_and_handoffs(self) -> None:
        helper = Agent(name="helper")
        agent = Agent(
            name="main",
            tools=[greet],
            handoffs=[helper],
            output_type=ReportOutput,
        )
        desc = agent.describe()
        assert desc["tools"] == ["greet"]
        assert desc["handoffs"] == ["helper"]
        assert desc["output_type"] == "ReportOutput"

    def test_repr_minimal(self) -> None:
        agent = Agent(name="bot")
        r = repr(agent)
        assert "Agent(" in r
        assert "name='bot'" in r
        assert "model='openai:gpt-4o'" in r

    def test_repr_with_tools(self) -> None:
        agent = Agent(name="bot", tools=[greet])
        r = repr(agent)
        assert "tools=['greet']" in r


# ---------------------------------------------------------------------------
# Mock provider helper
# ---------------------------------------------------------------------------


def _mock_provider(
    content: str = "Hello!",
    tool_calls: list[ToolCall] | None = None,
    usage: Usage | None = None,
) -> AsyncMock:
    """Create a mock provider that returns a fixed ModelResponse."""
    resp = ModelResponse(
        id="resp-1",
        model="test-model",
        content=content,
        tool_calls=tool_calls or [],
        usage=usage or Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    provider = AsyncMock()
    provider.complete = AsyncMock(return_value=resp)
    return provider


# ---------------------------------------------------------------------------
# Agent.run() — single-turn execution
# ---------------------------------------------------------------------------


class TestAgentRun:
    async def test_successful_run(self) -> None:
        """Successful single-turn run returns parsed output."""
        provider = _mock_provider(content="Hi there!")
        agent = Agent(name="bot", instructions="Be helpful.")

        output = await agent.run("Hello", provider=provider)

        assert output.text == "Hi there!"
        assert output.usage.input_tokens == 10
        assert output.usage.output_tokens == 5
        provider.complete.assert_awaited_once()

    async def test_run_passes_messages(self) -> None:
        """History messages are included in the LLM call."""
        from orbiter.types import AssistantMessage, UserMessage

        provider = _mock_provider()
        agent = Agent(name="bot")

        history = [
            UserMessage(content="first"),
            AssistantMessage(content="reply"),
        ]
        await agent.run("second", messages=history, provider=provider)

        call_args = provider.complete.call_args
        messages = call_args[0][0]
        # history (2) + new user message (1) = 3 (no system since instructions="")
        assert len(messages) == 3
        assert messages[-1].content == "second"

    async def test_run_with_instructions(self) -> None:
        """System instructions are prepended to message list."""
        provider = _mock_provider()
        agent = Agent(name="bot", instructions="Be concise.")

        await agent.run("Hello", provider=provider)

        call_args = provider.complete.call_args
        messages = call_args[0][0]
        assert messages[0].role == "system"
        assert messages[0].content == "Be concise."

    async def test_run_with_callable_instructions(self) -> None:
        """Callable instructions are resolved before building messages."""
        provider = _mock_provider()
        agent = Agent(
            name="bot",
            instructions=lambda name: f"You are {name}.",
        )

        await agent.run("Hi", provider=provider)

        call_args = provider.complete.call_args
        messages = call_args[0][0]
        assert messages[0].content == "You are bot."

    async def test_run_without_provider_raises(self) -> None:
        """run() without provider raises AgentError."""
        agent = Agent(name="bot")
        with pytest.raises(AgentError, match="requires a provider"):
            await agent.run("Hello")

    async def test_run_with_tool_calls_in_response(self) -> None:
        """Tool calls from the LLM trigger tool execution and re-call."""
        tc = ToolCall(id="tc-1", name="greet", arguments='{"name": "World"}')
        # First call returns tool call, second returns text
        resp_tool = ModelResponse(
            content="",
            tool_calls=[tc],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        resp_text = ModelResponse(
            content="Done!",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=[resp_tool, resp_text])
        agent = Agent(name="bot", tools=[greet])

        output = await agent.run("Hello", provider=provider)

        # Final output should be the text response after tool execution
        assert output.text == "Done!"
        assert provider.complete.await_count == 2

    async def test_run_passes_tool_schemas(self) -> None:
        """Tool schemas are passed to provider.complete()."""
        provider = _mock_provider()
        agent = Agent(name="bot", tools=[greet])

        await agent.run("Hello", provider=provider)

        call_args = provider.complete.call_args
        assert call_args[1]["tools"] is not None
        assert len(call_args[1]["tools"]) == 1


# ---------------------------------------------------------------------------
# Agent.run() — retry logic
# ---------------------------------------------------------------------------


class TestAgentRunRetry:
    async def test_retry_on_transient_error(self) -> None:
        """Transient errors are retried up to max_retries."""
        provider = AsyncMock()
        # Fail twice, succeed on third attempt
        resp = ModelResponse(
            content="ok", usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2)
        )
        provider.complete = AsyncMock(
            side_effect=[RuntimeError("timeout"), RuntimeError("server error"), resp]
        )

        agent = Agent(name="bot")
        output = await agent.run("Hello", provider=provider, max_retries=3)

        assert output.text == "ok"
        assert provider.complete.await_count == 3

    async def test_all_retries_exhausted(self) -> None:
        """AgentError raised when all retries fail."""
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=RuntimeError("server error"))

        agent = Agent(name="bot")
        with pytest.raises(AgentError, match="failed after 2 retries"):
            await agent.run("Hello", provider=provider, max_retries=2)

        assert provider.complete.await_count == 2

    async def test_context_length_error_no_retry(self) -> None:
        """Context-length errors fail immediately without retry."""
        provider = AsyncMock()
        err = ModelError("context window exceeded", model="gpt-4o", code="context_length")
        provider.complete = AsyncMock(side_effect=err)

        agent = Agent(name="bot")
        with pytest.raises(AgentError, match="Context length exceeded"):
            await agent.run("Hello", provider=provider, max_retries=5)

        # Should have called complete exactly once (no retries)
        assert provider.complete.await_count == 1

    async def test_context_length_detected_from_message(self) -> None:
        """Context-length errors detected from error message text."""
        provider = AsyncMock()
        provider.complete = AsyncMock(
            side_effect=RuntimeError("context length exceeded for this model")
        )

        agent = Agent(name="bot")
        with pytest.raises(AgentError, match="Context length exceeded"):
            await agent.run("Hello", provider=provider, max_retries=3)

        assert provider.complete.await_count == 1


# ---------------------------------------------------------------------------
# Agent.run() — hook invocation order
# ---------------------------------------------------------------------------


class TestAgentRunHooks:
    async def test_hook_invocation_order(self) -> None:
        """PRE_LLM_CALL fires before LLM call, POST_LLM_CALL fires after."""
        events: list[str] = []

        async def pre_hook(**data: Any) -> None:
            events.append("pre_llm")

        async def post_hook(**data: Any) -> None:
            events.append("post_llm")

        provider = _mock_provider()
        # Track when complete() is called
        original_complete = provider.complete

        async def tracked_complete(*args: Any, **kwargs: Any) -> ModelResponse:
            events.append("llm_call")
            return await original_complete(*args, **kwargs)

        provider.complete = tracked_complete

        agent = Agent(
            name="bot",
            hooks=[
                (HookPoint.PRE_LLM_CALL, pre_hook),
                (HookPoint.POST_LLM_CALL, post_hook),
            ],
        )

        await agent.run("Hello", provider=provider)

        assert events == ["pre_llm", "llm_call", "post_llm"]

    async def test_pre_hook_receives_agent_and_messages(self) -> None:
        """PRE_LLM_CALL hook receives agent and messages."""
        hook_data: dict[str, Any] = {}

        async def pre_hook(**data: Any) -> None:
            hook_data.update(data)

        provider = _mock_provider()
        agent = Agent(
            name="bot",
            instructions="Hi",
            hooks=[(HookPoint.PRE_LLM_CALL, pre_hook)],
        )

        await agent.run("Hello", provider=provider)

        assert hook_data["agent"] is agent
        assert len(hook_data["messages"]) >= 1

    async def test_post_hook_receives_response(self) -> None:
        """POST_LLM_CALL hook receives the model response."""
        hook_data: dict[str, Any] = {}

        async def post_hook(**data: Any) -> None:
            hook_data.update(data)

        provider = _mock_provider(content="I'm here!")
        agent = Agent(
            name="bot",
            hooks=[(HookPoint.POST_LLM_CALL, post_hook)],
        )

        await agent.run("Hello", provider=provider)

        assert hook_data["response"].content == "I'm here!"

    async def test_hooks_not_called_on_immediate_error(self) -> None:
        """POST_LLM_CALL hook is not called if LLM call raises immediately."""
        events: list[str] = []

        async def post_hook(**data: Any) -> None:
            events.append("post_llm")

        provider = AsyncMock()
        err = ModelError("context length exceeded", code="context_length")
        provider.complete = AsyncMock(side_effect=err)

        agent = Agent(
            name="bot",
            hooks=[(HookPoint.POST_LLM_CALL, post_hook)],
        )

        with pytest.raises(AgentError):
            await agent.run("Hello", provider=provider)

        assert "post_llm" not in events


# ---------------------------------------------------------------------------
# Agent.run() — tool execution loop
# ---------------------------------------------------------------------------


def _multi_step_provider(
    *responses: ModelResponse,
) -> AsyncMock:
    """Create a provider that returns responses sequentially."""
    provider = AsyncMock()
    provider.complete = AsyncMock(side_effect=list(responses))
    return provider


class TestAgentToolLoop:
    async def test_single_tool_call(self) -> None:
        """Single tool call → execute → LLM returns text."""
        tc = ToolCall(id="tc-1", name="greet", arguments='{"name": "Alice"}')
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="I greeted Alice for you!",
                usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
        )
        agent = Agent(name="bot", tools=[greet])

        output = await agent.run("Say hi to Alice", provider=provider)

        assert output.text == "I greeted Alice for you!"
        assert provider.complete.await_count == 2

        # Second call should include the tool result in messages
        second_call_msgs = provider.complete.call_args_list[1][0][0]
        tool_result_msgs = [m for m in second_call_msgs if m.role == "tool"]
        assert len(tool_result_msgs) == 1
        assert "Hello, Alice!" in tool_result_msgs[0].content

    async def test_multi_tool_parallel_call(self) -> None:
        """Multiple tool calls execute in parallel."""
        tc1 = ToolCall(id="tc-1", name="greet", arguments='{"name": "Alice"}')
        tc2 = ToolCall(id="tc-2", name="add", arguments='{"a": 2, "b": 3}')
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc1, tc2],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="Done with both tools.",
                usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
        )
        agent = Agent(name="bot", tools=[greet, add])

        output = await agent.run("Do both tasks", provider=provider)

        assert output.text == "Done with both tools."
        assert provider.complete.await_count == 2

        # Both tool results should be in the second call
        second_call_msgs = provider.complete.call_args_list[1][0][0]
        tool_result_msgs = [m for m in second_call_msgs if m.role == "tool"]
        assert len(tool_result_msgs) == 2

    async def test_tool_error_returned_not_propagated(self) -> None:
        """Tool errors are captured as ToolResult(error=...), not raised."""

        @tool
        def failing_tool(x: str) -> str:
            """A tool that always fails."""
            raise ValueError("boom!")

        tc = ToolCall(id="tc-1", name="failing_tool", arguments='{"x": "hi"}')
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="The tool failed, sorry.",
                usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
        )
        agent = Agent(name="bot", tools=[failing_tool])

        output = await agent.run("Try it", provider=provider)

        assert output.text == "The tool failed, sorry."
        # The error should be in the tool result message
        second_call_msgs = provider.complete.call_args_list[1][0][0]
        tool_result_msgs = [m for m in second_call_msgs if m.role == "tool"]
        assert len(tool_result_msgs) == 1
        assert tool_result_msgs[0].error is not None
        assert "boom" in tool_result_msgs[0].error

    async def test_unknown_tool_returns_error(self) -> None:
        """Calling an unknown tool returns an error ToolResult."""
        tc = ToolCall(id="tc-1", name="nonexistent", arguments="{}")
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="Tool not found.",
                usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
        )
        agent = Agent(name="bot", tools=[greet])

        output = await agent.run("Use nonexistent", provider=provider)

        assert output.text == "Tool not found."
        second_call_msgs = provider.complete.call_args_list[1][0][0]
        tool_result_msgs = [m for m in second_call_msgs if m.role == "tool"]
        assert tool_result_msgs[0].error is not None
        assert "Unknown tool" in tool_result_msgs[0].error

    async def test_max_steps_enforcement(self) -> None:
        """Tool loop stops at max_steps even if LLM keeps returning tool calls."""
        tc = ToolCall(id="tc-1", name="greet", arguments='{"name": "Loop"}')
        # All responses have tool calls — never returns text
        responses = [
            ModelResponse(
                content="",
                tool_calls=[tc],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            )
            for _ in range(5)
        ]
        provider = _multi_step_provider(*responses)
        agent = Agent(name="bot", tools=[greet], max_steps=3)

        output = await agent.run("Loop forever", provider=provider)

        # Should stop after 3 steps, returning last tool-call output
        assert provider.complete.await_count == 3
        assert len(output.tool_calls) == 1

    async def test_tool_hooks_fired(self) -> None:
        """PRE_TOOL_CALL and POST_TOOL_CALL hooks fire for each tool."""
        events: list[str] = []

        async def pre_tool(**data: Any) -> None:
            events.append(f"pre:{data['tool_name']}")

        async def post_tool(**data: Any) -> None:
            events.append(f"post:{data['tool_name']}")

        tc = ToolCall(id="tc-1", name="greet", arguments='{"name": "Bob"}')
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="Done.",
                usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
            ),
        )
        agent = Agent(
            name="bot",
            tools=[greet],
            hooks=[
                (HookPoint.PRE_TOOL_CALL, pre_tool),
                (HookPoint.POST_TOOL_CALL, post_tool),
            ],
        )

        await agent.run("Hello", provider=provider)

        assert events == ["pre:greet", "post:greet"]

    async def test_multi_step_tool_loop(self) -> None:
        """Agent executes multiple rounds of tool calls before final text."""
        tc1 = ToolCall(id="tc-1", name="greet", arguments='{"name": "A"}')
        tc2 = ToolCall(id="tc-2", name="add", arguments='{"a": 1, "b": 2}')
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc1],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="",
                tool_calls=[tc2],
                usage=Usage(input_tokens=15, output_tokens=5, total_tokens=20),
            ),
            ModelResponse(
                content="All done!",
                usage=Usage(input_tokens=20, output_tokens=5, total_tokens=25),
            ),
        )
        agent = Agent(name="bot", tools=[greet, add])

        output = await agent.run("Do everything", provider=provider)

        assert output.text == "All done!"
        assert provider.complete.await_count == 3

    async def test_no_tools_returns_immediately(self) -> None:
        """Agent with no tools returns LLM response without looping."""
        provider = _mock_provider(content="Direct answer.")
        agent = Agent(name="bot")

        output = await agent.run("Hello", provider=provider)

        assert output.text == "Direct answer."
        provider.complete.assert_awaited_once()


# ---------------------------------------------------------------------------
# Agent edge cases
# ---------------------------------------------------------------------------


class TestAgentEdgeCases:
    async def test_retry_during_tool_loop(self) -> None:
        """Retry logic works on the second LLM call (after tool execution)."""
        tc = ToolCall(id="tc-1", name="greet", arguments='{"name": "Eve"}')
        resp_tool = ModelResponse(
            content="",
            tool_calls=[tc],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        resp_text = ModelResponse(
            content="Recovered!",
            usage=Usage(input_tokens=15, output_tokens=5, total_tokens=20),
        )
        provider = AsyncMock()
        # First call succeeds (returns tool call), second fails, third succeeds
        provider.complete = AsyncMock(side_effect=[resp_tool, RuntimeError("transient"), resp_text])
        agent = Agent(name="bot", tools=[greet])

        output = await agent.run("Hello", provider=provider, max_retries=3)

        assert output.text == "Recovered!"
        assert provider.complete.await_count == 3

    async def test_agent_with_handoffs_runs_normally(self) -> None:
        """Agent with handoffs declared still runs normally (no handoff triggered)."""
        helper = Agent(name="helper")
        provider = _mock_provider(content="I handled it myself.")
        agent = Agent(name="triage", handoffs=[helper])

        output = await agent.run("Help me", provider=provider)

        assert output.text == "I handled it myself."
        assert "helper" in agent.handoffs

    async def test_handoffs_do_not_appear_as_tools(self) -> None:
        """Handoff agents are not included in tool schemas."""
        helper = Agent(name="helper")
        agent = Agent(name="triage", tools=[greet], handoffs=[helper])

        schemas = agent.get_tool_schemas()

        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "greet"
        # Handoff should not appear as a tool
        names = [s["function"]["name"] for s in schemas]
        assert "helper" not in names

    async def test_sequential_tool_calls_accumulate_messages(self) -> None:
        """Each tool loop iteration appends assistant + tool result messages."""
        tc1 = ToolCall(id="tc-1", name="add", arguments='{"a": 1, "b": 2}')
        tc2 = ToolCall(id="tc-2", name="add", arguments='{"a": 3, "b": 4}')
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc1],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="",
                tool_calls=[tc2],
                usage=Usage(input_tokens=20, output_tokens=5, total_tokens=25),
            ),
            ModelResponse(
                content="Results: 3 and 7",
                usage=Usage(input_tokens=30, output_tokens=10, total_tokens=40),
            ),
        )
        agent = Agent(name="bot", tools=[add])

        output = await agent.run("Add things", provider=provider)

        assert output.text == "Results: 3 and 7"
        # Third call should have: user + assistant(tc1) + tool(result1) + assistant(tc2) + tool(result2)
        third_call_msgs = provider.complete.call_args_list[2][0][0]
        assistant_msgs = [m for m in third_call_msgs if m.role == "assistant"]
        tool_msgs = [m for m in third_call_msgs if m.role == "tool"]
        assert len(assistant_msgs) == 2
        assert len(tool_msgs) == 2

    async def test_max_steps_one(self) -> None:
        """Agent with max_steps=1 makes exactly one LLM call."""
        tc = ToolCall(id="tc-1", name="greet", arguments='{"name": "X"}')
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        )
        agent = Agent(name="bot", tools=[greet], max_steps=1)

        output = await agent.run("Go", provider=provider)

        assert provider.complete.await_count == 1
        assert len(output.tool_calls) == 1

    async def test_empty_input(self) -> None:
        """Agent handles empty string input gracefully."""
        provider = _mock_provider(content="You said nothing.")
        agent = Agent(name="bot")

        output = await agent.run("", provider=provider)

        assert output.text == "You said nothing."

    async def test_usage_from_final_response(self) -> None:
        """Output usage comes from the final LLM call, not accumulated."""
        tc = ToolCall(id="tc-1", name="greet", arguments='{"name": "Z"}')
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="Done.",
                usage=Usage(input_tokens=50, output_tokens=20, total_tokens=70),
            ),
        )
        agent = Agent(name="bot", tools=[greet])

        output = await agent.run("Go", provider=provider)

        assert output.usage.input_tokens == 50
        assert output.usage.output_tokens == 20

    async def test_tool_with_no_arguments(self) -> None:
        """Tool called with empty arguments dict works."""

        @tool
        def ping() -> str:
            """Ping."""
            return "pong"

        tc = ToolCall(id="tc-1", name="ping", arguments="{}")
        provider = _multi_step_provider(
            ModelResponse(
                content="",
                tool_calls=[tc],
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
            ModelResponse(
                content="Pong received.",
                usage=Usage(input_tokens=20, output_tokens=5, total_tokens=25),
            ),
        )
        agent = Agent(name="bot", tools=[ping])

        output = await agent.run("Ping", provider=provider)

        assert output.text == "Pong received."
        second_call_msgs = provider.complete.call_args_list[1][0][0]
        tool_msgs = [m for m in second_call_msgs if m.role == "tool"]
        assert "pong" in tool_msgs[0].content


# ---------------------------------------------------------------------------
# Thought signature propagation
# ---------------------------------------------------------------------------


class TestThoughtSignaturePropagation:
    async def test_reasoning_content_propagated(self) -> None:
        """reasoning_content from ModelResponse reaches AgentOutput."""
        resp = ModelResponse(
            content="answer",
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            reasoning_content="thinking step by step",
            thought_signatures=[b"\x01\x02"],
        )
        provider = AsyncMock()
        provider.complete = AsyncMock(return_value=resp)
        agent = Agent(name="bot")

        output = await agent.run("Hello", provider=provider)

        assert output.reasoning_content == "thinking step by step"
        assert output.thought_signatures == [b"\x01\x02"]

    async def test_thought_signatures_in_assistant_message(self) -> None:
        """thought_signatures are included in AssistantMessage during tool loop."""
        tc = ToolCall(id="tc-1", name="greet", arguments='{"name": "Alice"}')
        resp_tool = ModelResponse(
            content="",
            tool_calls=[tc],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            reasoning_content="let me think",
            thought_signatures=[b"\xab\xcd"],
        )
        resp_text = ModelResponse(
            content="Done!",
            usage=Usage(input_tokens=20, output_tokens=10, total_tokens=30),
        )
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=[resp_tool, resp_text])
        agent = Agent(name="bot", tools=[greet])

        await agent.run("Hello", provider=provider)

        # Check that the second call has the assistant message with thought_signatures
        second_call_msgs = provider.complete.call_args_list[1][0][0]
        assistant_msgs = [m for m in second_call_msgs if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].thought_signatures == [b"\xab\xcd"]
        assert assistant_msgs[0].reasoning_content == "let me think"
