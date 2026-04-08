"""Tests for the human-in-the-loop tool and HITL enforcement."""

from __future__ import annotations

import asyncio

import pytest

from exo.agent import Agent
from exo.human import ConsoleHandler, HumanInputHandler, HumanInputTool
from exo.tool import Tool, ToolError, tool
from exo.tool_context import ToolContext
from exo.types import ActionModel

# ---------------------------------------------------------------------------
# Mock handler for testing
# ---------------------------------------------------------------------------


class MockHandler(HumanInputHandler):
    """A test handler that returns a predefined response."""

    def __init__(self, response: str = "yes") -> None:
        self.response = response
        self.last_prompt: str | None = None
        self.last_choices: list[str] | None = None
        self.call_count = 0

    async def get_input(self, prompt: str, choices: list[str] | None = None) -> str:
        self.last_prompt = prompt
        self.last_choices = choices
        self.call_count += 1
        return self.response


class SlowHandler(HumanInputHandler):
    """A test handler that delays before responding."""

    def __init__(self, delay: float = 10.0) -> None:
        self._delay = delay

    async def get_input(self, prompt: str, choices: list[str] | None = None) -> str:
        await asyncio.sleep(self._delay)
        return "too late"


# ---------------------------------------------------------------------------
# Tool schema tests
# ---------------------------------------------------------------------------


class TestHumanInputToolSchema:
    def test_tool_name(self) -> None:
        tool = HumanInputTool()
        assert tool.name == "human_input"

    def test_is_tool_subclass(self) -> None:
        tool = HumanInputTool()
        assert isinstance(tool, Tool)

    def test_description_not_empty(self) -> None:
        tool = HumanInputTool()
        assert len(tool.description) > 0

    def test_schema_has_prompt_parameter(self) -> None:
        tool = HumanInputTool()
        assert "prompt" in tool.parameters["properties"]
        assert tool.parameters["properties"]["prompt"]["type"] == "string"

    def test_schema_has_choices_parameter(self) -> None:
        tool = HumanInputTool()
        props = tool.parameters["properties"]
        assert "choices" in props
        assert props["choices"]["type"] == "array"
        assert props["choices"]["items"]["type"] == "string"

    def test_prompt_is_required(self) -> None:
        tool = HumanInputTool()
        assert "prompt" in tool.parameters["required"]

    def test_choices_is_optional(self) -> None:
        tool = HumanInputTool()
        assert "choices" not in tool.parameters["required"]

    def test_to_schema_format(self) -> None:
        tool = HumanInputTool()
        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "human_input"
        assert "parameters" in schema["function"]


# ---------------------------------------------------------------------------
# Handler invocation tests
# ---------------------------------------------------------------------------


class TestHandlerInvocation:
    async def test_handler_receives_prompt(self) -> None:
        handler = MockHandler(response="confirmed")
        tool = HumanInputTool(handler=handler)
        result = await tool.execute(prompt="Do you approve?")
        assert result == "confirmed"
        assert handler.last_prompt == "Do you approve?"

    async def test_handler_receives_choices(self) -> None:
        handler = MockHandler(response="option_a")
        tool = HumanInputTool(handler=handler)
        result = await tool.execute(prompt="Pick one", choices=["option_a", "option_b"])
        assert result == "option_a"
        assert handler.last_choices == ["option_a", "option_b"]

    async def test_handler_called_without_choices(self) -> None:
        handler = MockHandler(response="free text")
        tool = HumanInputTool(handler=handler)
        await tool.execute(prompt="Say something")
        assert handler.last_choices is None

    async def test_handler_call_count(self) -> None:
        handler = MockHandler(response="ok")
        tool = HumanInputTool(handler=handler)
        await tool.execute(prompt="First")
        await tool.execute(prompt="Second")
        assert handler.call_count == 2

    async def test_default_handler_is_console(self) -> None:
        tool = HumanInputTool()
        assert isinstance(tool._handler, ConsoleHandler)

    async def test_custom_handler(self) -> None:
        handler = MockHandler(response="custom")
        tool = HumanInputTool(handler=handler)
        assert tool._handler is handler


# ---------------------------------------------------------------------------
# Timeout behavior tests
# ---------------------------------------------------------------------------


class TestTimeoutBehavior:
    async def test_timeout_raises_tool_error(self) -> None:
        handler = SlowHandler(delay=10.0)
        tool = HumanInputTool(handler=handler, timeout=0.05)
        with pytest.raises(ToolError, match="timed out"):
            await tool.execute(prompt="Will timeout")

    async def test_no_timeout_by_default(self) -> None:
        tool = HumanInputTool()
        assert tool._timeout is None

    async def test_fast_response_within_timeout(self) -> None:
        handler = MockHandler(response="quick")
        tool = HumanInputTool(handler=handler, timeout=5.0)
        result = await tool.execute(prompt="Quick response")
        assert result == "quick"


# ---------------------------------------------------------------------------
# ConsoleHandler tests
# ---------------------------------------------------------------------------


class TestConsoleHandler:
    async def test_console_handler_with_choices_validates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handler = ConsoleHandler()
        # Mock _read_line to return an invalid choice
        monkeypatch.setattr(ConsoleHandler, "_read_line", staticmethod(lambda _: "invalid\n"))
        result = await handler.get_input("Pick one", choices=["a", "b"])
        # Should default to first choice
        assert result == "a"

    async def test_console_handler_with_valid_choice(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = ConsoleHandler()
        monkeypatch.setattr(ConsoleHandler, "_read_line", staticmethod(lambda _: "b\n"))
        result = await handler.get_input("Pick one", choices=["a", "b"])
        assert result == "b"

    async def test_console_handler_freeform(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = ConsoleHandler()
        monkeypatch.setattr(ConsoleHandler, "_read_line", staticmethod(lambda _: "hello world\n"))
        result = await handler.get_input("Say something")
        assert result == "hello world"

    async def test_console_handler_strips_whitespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = ConsoleHandler()
        monkeypatch.setattr(ConsoleHandler, "_read_line", staticmethod(lambda _: "  trimmed  \n"))
        result = await handler.get_input("Say something")
        assert result == "trimmed"


# ---------------------------------------------------------------------------
# Helper tools for HITL enforcement tests
# ---------------------------------------------------------------------------


@tool
def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"


@tool
async def sensitive_greet(name: str, ctx: ToolContext) -> str:
    """Greet someone, but require approval first."""
    await ctx.require_approval(f"About to greet {name}. Approve?")
    return f"Hello, {name}!"


@tool
def add(a: int, b: int) -> str:
    """Add two numbers."""
    return str(a + b)


# ---------------------------------------------------------------------------
# HITL enforcement tests — tool-level require_approval via ToolContext
# ---------------------------------------------------------------------------


class TestRequireApproval:
    """Tests for ToolContext.require_approval() — tool-level HITL gating."""

    def _make_agent(
        self,
        handler: HumanInputHandler | None = None,
    ) -> Agent:
        return Agent(
            name="test_agent",
            tools=[greet, sensitive_greet, add],
            human_input_handler=handler,
        )

    async def test_approved_executes(self) -> None:
        handler = MockHandler(response="yes")
        agent = self._make_agent(handler=handler)
        actions = [
            ActionModel(
                tool_call_id="tc1",
                tool_name="sensitive_greet",
                arguments={"name": "Alice"},
            )
        ]
        results = await agent._execute_tools(actions)
        assert results[0].error is None or results[0].error == ""
        assert "Hello, Alice!" in results[0].content
        assert handler.call_count == 1

    async def test_denied_returns_error(self) -> None:
        handler = MockHandler(response="no")
        agent = self._make_agent(handler=handler)
        actions = [
            ActionModel(
                tool_call_id="tc1",
                tool_name="sensitive_greet",
                arguments={"name": "Alice"},
            )
        ]
        results = await agent._execute_tools(actions)
        assert "denied by human" in results[0].error
        assert handler.call_count == 1

    async def test_non_hitl_tool_unaffected(self) -> None:
        handler = MockHandler(response="no")
        agent = self._make_agent(handler=handler)
        actions = [
            ActionModel(tool_call_id="tc1", tool_name="add", arguments={"a": 1, "b": 2})
        ]
        results = await agent._execute_tools(actions)
        assert results[0].error is None or results[0].error == ""
        assert "3" in results[0].content
        assert handler.call_count == 0

    async def test_no_handler_raises_tool_error(self) -> None:
        agent = self._make_agent(handler=None)
        actions = [
            ActionModel(
                tool_call_id="tc1",
                tool_name="sensitive_greet",
                arguments={"name": "Bob"},
            )
        ]
        results = await agent._execute_tools(actions)
        assert "no human_input_handler" in results[0].error

    async def test_handler_receives_custom_message(self) -> None:
        handler = MockHandler(response="yes")
        agent = self._make_agent(handler=handler)
        actions = [
            ActionModel(
                tool_call_id="tc1",
                tool_name="sensitive_greet",
                arguments={"name": "Eve"},
            )
        ]
        await agent._execute_tools(actions)
        assert "About to greet Eve" in handler.last_prompt

    async def test_mixed_tools_parallel(self) -> None:
        handler = MockHandler(response="yes")
        agent = self._make_agent(handler=handler)
        actions = [
            ActionModel(
                tool_call_id="tc1",
                tool_name="sensitive_greet",
                arguments={"name": "Alice"},
            ),
            ActionModel(tool_call_id="tc2", tool_name="add", arguments={"a": 2, "b": 3}),
        ]
        results = await agent._execute_tools(actions)
        assert "Hello, Alice!" in results[0].content
        assert "5" in results[1].content
        assert handler.call_count == 1

    async def test_to_dict_raises_with_handler(self) -> None:
        handler = MockHandler(response="yes")
        agent = self._make_agent(handler=handler)
        with pytest.raises(ValueError, match="human_input_handler"):
            agent.to_dict()
