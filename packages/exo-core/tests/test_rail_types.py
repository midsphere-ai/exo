"""Tests for exo.rail_types — typed event input models for rails."""

import pytest
from pydantic import ValidationError

from exo.hooks import HookPoint
from exo.rail_types import (
    InvokeInputs,
    ModelCallInputs,
    RailContext,
    ToolCallInputs,
)

# --- InvokeInputs ---


class TestInvokeInputs:
    def test_minimal_creation(self) -> None:
        inputs = InvokeInputs(input="hello")
        assert inputs.input == "hello"
        assert inputs.messages is None
        assert inputs.result is None

    def test_full_creation(self) -> None:
        inputs = InvokeInputs(
            input="hello",
            messages=[{"role": "user", "content": "hi"}],
            result="done",
        )
        assert inputs.input == "hello"
        assert inputs.messages == [{"role": "user", "content": "hi"}]
        assert inputs.result == "done"

    def test_input_required(self) -> None:
        with pytest.raises(ValidationError):
            InvokeInputs()  # type: ignore[call-arg]

    def test_mutable(self) -> None:
        inputs = InvokeInputs(input="hello")
        inputs.input = "updated"
        assert inputs.input == "updated"

    def test_messages_mutable(self) -> None:
        inputs = InvokeInputs(input="hello", messages=[])
        assert inputs.messages is not None
        inputs.messages.append({"role": "user", "content": "new"})
        assert len(inputs.messages) == 1


# --- ModelCallInputs ---


class TestModelCallInputs:
    def test_minimal_creation(self) -> None:
        inputs = ModelCallInputs(messages=[{"role": "user", "content": "hi"}])
        assert len(inputs.messages) == 1
        assert inputs.tools is None
        assert inputs.response is None
        assert inputs.usage is None

    def test_full_creation(self) -> None:
        inputs = ModelCallInputs(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "name": "search"}],
            response={"text": "result"},
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        assert inputs.tools == [{"type": "function", "name": "search"}]
        assert inputs.response == {"text": "result"}
        assert inputs.usage == {"input_tokens": 10, "output_tokens": 5}

    def test_messages_required(self) -> None:
        with pytest.raises(ValidationError):
            ModelCallInputs()  # type: ignore[call-arg]

    def test_mutable(self) -> None:
        inputs = ModelCallInputs(messages=[])
        inputs.messages.append({"role": "system", "content": "be helpful"})
        assert len(inputs.messages) == 1


# --- ToolCallInputs ---


class TestToolCallInputs:
    def test_minimal_creation(self) -> None:
        inputs = ToolCallInputs(tool_name="search")
        assert inputs.tool_name == "search"
        assert inputs.arguments == {}
        assert inputs.result is None
        assert inputs.metadata is None

    def test_full_creation(self) -> None:
        inputs = ToolCallInputs(
            tool_name="search",
            arguments={"query": "exo docs"},
            result="found 3 results",
            metadata={"duration_ms": 42},
        )
        assert inputs.arguments == {"query": "exo docs"}
        assert inputs.result == "found 3 results"
        assert inputs.metadata == {"duration_ms": 42}

    def test_tool_name_required(self) -> None:
        with pytest.raises(ValidationError):
            ToolCallInputs()  # type: ignore[call-arg]

    def test_mutable(self) -> None:
        inputs = ToolCallInputs(tool_name="search")
        inputs.result = "mutated"
        assert inputs.result == "mutated"

    def test_arguments_default_empty_dict(self) -> None:
        a = ToolCallInputs(tool_name="a")
        b = ToolCallInputs(tool_name="b")
        # Ensure default_factory creates distinct dicts
        a.arguments["key"] = "val"
        assert "key" not in b.arguments


# --- RailContext ---


class TestRailContext:
    def test_creation_with_invoke_inputs(self) -> None:
        ctx = RailContext(
            agent="mock_agent",
            event=HookPoint.START,
            inputs=InvokeInputs(input="hello"),
        )
        assert ctx.agent == "mock_agent"
        assert ctx.event == HookPoint.START
        assert isinstance(ctx.inputs, InvokeInputs)
        assert ctx.extra == {}

    def test_creation_with_model_call_inputs(self) -> None:
        ctx = RailContext(
            agent="mock_agent",
            event=HookPoint.PRE_LLM_CALL,
            inputs=ModelCallInputs(messages=[]),
        )
        assert isinstance(ctx.inputs, ModelCallInputs)

    def test_creation_with_tool_call_inputs(self) -> None:
        ctx = RailContext(
            agent="mock_agent",
            event=HookPoint.PRE_TOOL_CALL,
            inputs=ToolCallInputs(tool_name="calc"),
        )
        assert isinstance(ctx.inputs, ToolCallInputs)

    def test_extra_dict_default(self) -> None:
        a = RailContext(
            agent="a",
            event=HookPoint.START,
            inputs=InvokeInputs(input="x"),
        )
        b = RailContext(
            agent="b",
            event=HookPoint.START,
            inputs=InvokeInputs(input="y"),
        )
        a.extra["shared"] = True
        assert "shared" not in b.extra

    def test_extra_dict_passthrough(self) -> None:
        ctx = RailContext(
            agent="mock",
            event=HookPoint.START,
            inputs=InvokeInputs(input="x"),
            extra={"cross_rail_state": 42},
        )
        assert ctx.extra["cross_rail_state"] == 42

    def test_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            RailContext()  # type: ignore[call-arg]

    def test_mutable_extra(self) -> None:
        ctx = RailContext(
            agent="mock",
            event=HookPoint.FINISHED,
            inputs=InvokeInputs(input="x"),
        )
        ctx.extra["key"] = "value"
        assert ctx.extra["key"] == "value"

    def test_all_hook_points_accepted(self) -> None:
        for point in HookPoint:
            ctx = RailContext(
                agent="mock",
                event=point,
                inputs=InvokeInputs(input="x"),
            )
            assert ctx.event == point
