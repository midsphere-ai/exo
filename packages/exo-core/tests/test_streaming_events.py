"""Tests for streaming event types (US-002, US-003)."""

import pytest
from pydantic import ValidationError

from exo.types import (
    ErrorEvent,
    ReasoningEvent,
    StatusEvent,
    StepEvent,
    StreamEvent,
    TextEvent,
    ToolCallDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    Usage,
    UsageEvent,
)


class TestToolResultEvent:
    def test_create_success(self) -> None:
        e = ToolResultEvent(
            tool_name="search",
            tool_call_id="tc_1",
            arguments={"query": "hello"},
            result="found it",
            success=True,
            duration_ms=42.5,
            agent_name="bot",
        )
        assert e.type == "tool_result"
        assert e.tool_name == "search"
        assert e.tool_call_id == "tc_1"
        assert e.arguments == {"query": "hello"}
        assert e.result == "found it"
        assert e.error is None
        assert e.success is True
        assert e.duration_ms == 42.5
        assert e.agent_name == "bot"

    def test_create_failure(self) -> None:
        e = ToolResultEvent(
            tool_name="search",
            tool_call_id="tc_1",
            arguments={"query": "hello"},
            result="",
            error="not found",
            success=False,
            duration_ms=10.0,
            agent_name="bot",
        )
        assert e.success is False
        assert e.error == "not found"
        assert e.result == ""

    def test_defaults(self) -> None:
        e = ToolResultEvent(tool_name="search", tool_call_id="tc_1")
        assert e.arguments == {}
        assert e.result == ""
        assert e.error is None
        assert e.success is True
        assert e.duration_ms == 0.0
        assert e.agent_name == ""

    def test_frozen(self) -> None:
        e = ToolResultEvent(tool_name="search", tool_call_id="tc_1")
        with pytest.raises(ValidationError):
            e.tool_name = "other"  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        e = ToolResultEvent(
            tool_name="search",
            tool_call_id="tc_1",
            arguments={"q": "test"},
            result="ok",
            error=None,
            success=True,
            duration_ms=15.3,
            agent_name="bot",
        )
        data = e.model_dump()
        assert data == {
            "type": "tool_result",
            "tool_name": "search",
            "tool_call_id": "tc_1",
            "arguments": {"q": "test"},
            "result": "ok",
            "error": None,
            "success": True,
            "duration_ms": 15.3,
            "agent_name": "bot",
        }
        restored = ToolResultEvent.model_validate(data)
        assert restored == e

    def test_arguments_default_not_shared(self) -> None:
        a = ToolResultEvent(tool_name="x", tool_call_id="1")
        b = ToolResultEvent(tool_name="y", tool_call_id="2")
        assert a.arguments is not b.arguments

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            ToolResultEvent()  # type: ignore[call-arg]

    def test_type_literal(self) -> None:
        e = ToolResultEvent(tool_name="search", tool_call_id="tc_1")
        assert e.type == "tool_result"


class TestReasoningEvent:
    def test_create(self) -> None:
        e = ReasoningEvent(text="thinking...", agent_name="bot")
        assert e.type == "reasoning"
        assert e.text == "thinking..."
        assert e.agent_name == "bot"

    def test_defaults(self) -> None:
        e = ReasoningEvent(text="hmm")
        assert e.agent_name == ""

    def test_frozen(self) -> None:
        e = ReasoningEvent(text="thinking")
        with pytest.raises(ValidationError):
            e.text = "other"  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        e = ReasoningEvent(text="let me think", agent_name="bot")
        data = e.model_dump()
        assert data == {
            "type": "reasoning",
            "text": "let me think",
            "agent_name": "bot",
        }
        restored = ReasoningEvent.model_validate(data)
        assert restored == e

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            ReasoningEvent()  # type: ignore[call-arg]


class TestErrorEvent:
    def test_create(self) -> None:
        e = ErrorEvent(
            error="something broke",
            error_type="ValueError",
            agent_name="bot",
            step_number=3,
            recoverable=True,
        )
        assert e.type == "error"
        assert e.error == "something broke"
        assert e.error_type == "ValueError"
        assert e.agent_name == "bot"
        assert e.step_number == 3
        assert e.recoverable is True

    def test_defaults(self) -> None:
        e = ErrorEvent(error="fail", error_type="RuntimeError")
        assert e.agent_name == ""
        assert e.step_number is None
        assert e.recoverable is False

    def test_frozen(self) -> None:
        e = ErrorEvent(error="fail", error_type="RuntimeError")
        with pytest.raises(ValidationError):
            e.error = "other"  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        e = ErrorEvent(
            error="timeout",
            error_type="TimeoutError",
            agent_name="bot",
            step_number=2,
            recoverable=True,
        )
        data = e.model_dump()
        assert data == {
            "type": "error",
            "error": "timeout",
            "error_type": "TimeoutError",
            "agent_name": "bot",
            "step_number": 2,
            "recoverable": True,
        }
        restored = ErrorEvent.model_validate(data)
        assert restored == e

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            ErrorEvent()  # type: ignore[call-arg]


class TestStatusEvent:
    def test_create(self) -> None:
        e = StatusEvent(
            status="running",
            agent_name="bot",
            message="Processing step 2",
        )
        assert e.type == "status"
        assert e.status == "running"
        assert e.agent_name == "bot"
        assert e.message == "Processing step 2"

    def test_defaults(self) -> None:
        e = StatusEvent(status="starting")
        assert e.agent_name == ""
        assert e.message == ""

    def test_all_status_values(self) -> None:
        for s in ("starting", "running", "waiting_for_tool", "completed", "cancelled", "error"):
            e = StatusEvent(status=s)  # type: ignore[arg-type]
            assert e.status == s

    def test_invalid_status(self) -> None:
        with pytest.raises(ValidationError):
            StatusEvent(status="invalid")  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        e = StatusEvent(status="starting")
        with pytest.raises(ValidationError):
            e.status = "running"  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        e = StatusEvent(
            status="completed",
            agent_name="bot",
            message="Done",
        )
        data = e.model_dump()
        assert data == {
            "type": "status",
            "status": "completed",
            "agent_name": "bot",
            "message": "Done",
        }
        restored = StatusEvent.model_validate(data)
        assert restored == e


class TestUsageEvent:
    def test_create(self) -> None:
        u = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        e = UsageEvent(
            usage=u,
            agent_name="bot",
            step_number=1,
            model="gpt-4",
        )
        assert e.type == "usage"
        assert e.usage == u
        assert e.agent_name == "bot"
        assert e.step_number == 1
        assert e.model == "gpt-4"

    def test_defaults(self) -> None:
        u = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        e = UsageEvent(usage=u)
        assert e.agent_name == ""
        assert e.step_number == 0
        assert e.model == ""

    def test_frozen(self) -> None:
        u = Usage(input_tokens=10, output_tokens=5, total_tokens=15)
        e = UsageEvent(usage=u)
        with pytest.raises(ValidationError):
            e.model = "other"  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        u = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        e = UsageEvent(
            usage=u,
            agent_name="bot",
            step_number=2,
            model="claude-3",
        )
        data = e.model_dump()
        assert data == {
            "type": "usage",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            },
            "agent_name": "bot",
            "step_number": 2,
            "model": "claude-3",
        }
        restored = UsageEvent.model_validate(data)
        assert restored == e

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            UsageEvent()  # type: ignore[call-arg]


class TestToolCallDeltaEvent:
    def test_create(self) -> None:
        e = ToolCallDeltaEvent(
            index=0,
            tool_call_id="tc1",
            tool_name="search",
            arguments_delta='{"query":',
            agent_name="bot",
        )
        assert e.type == "tool_call_delta"
        assert e.index == 0
        assert e.tool_call_id == "tc1"
        assert e.tool_name == "search"
        assert e.arguments_delta == '{"query":'
        assert e.agent_name == "bot"

    def test_defaults(self) -> None:
        e = ToolCallDeltaEvent()
        assert e.index == 0
        assert e.tool_call_id == ""
        assert e.tool_name == ""
        assert e.arguments_delta == ""
        assert e.agent_name == ""

    def test_frozen(self) -> None:
        e = ToolCallDeltaEvent()
        with pytest.raises(ValidationError):
            e.index = 1  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        e = ToolCallDeltaEvent(
            index=1,
            tool_call_id="tc2",
            tool_name="calc",
            arguments_delta='"hello"}',
            agent_name="bot",
        )
        data = e.model_dump()
        assert data == {
            "type": "tool_call_delta",
            "index": 1,
            "tool_call_id": "tc2",
            "tool_name": "calc",
            "arguments_delta": '"hello"}',
            "agent_name": "bot",
        }
        restored = ToolCallDeltaEvent.model_validate(data)
        assert restored == e


class TestStreamEventUnion:
    def test_tool_result_event_is_stream_event(self) -> None:
        e: StreamEvent = ToolResultEvent(tool_name="x", tool_call_id="tc_1")
        assert isinstance(e, ToolResultEvent)

    def test_all_event_types_in_union(self) -> None:
        events: list[StreamEvent] = [
            TextEvent(text="hi"),
            ToolCallEvent(tool_name="search", tool_call_id="tc_1"),
            ToolCallDeltaEvent(arguments_delta='{"q":'),
            StepEvent(
                step_number=1,
                agent_name="bot",
                status="started",
                started_at=1000.0,
            ),
            ToolResultEvent(
                tool_name="search",
                tool_call_id="tc_1",
                result="ok",
            ),
            ReasoningEvent(text="thinking"),
            ErrorEvent(error="fail", error_type="RuntimeError"),
            StatusEvent(status="starting"),
            UsageEvent(
                usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
            ),
        ]
        expected_types = [
            "text",
            "tool_call",
            "tool_call_delta",
            "step",
            "tool_result",
            "reasoning",
            "error",
            "status",
            "usage",
        ]
        assert [e.type for e in events] == expected_types

    def test_discriminate_by_type_field(self) -> None:
        events: list[StreamEvent] = [
            ToolResultEvent(tool_name="x", tool_call_id="tc_1"),
            TextEvent(text="hello"),
            ToolCallDeltaEvent(index=0, arguments_delta="{}"),
            ReasoningEvent(text="thinking"),
            ErrorEvent(error="oops", error_type="ValueError"),
            StatusEvent(status="running"),
            UsageEvent(
                usage=Usage(input_tokens=1, output_tokens=1, total_tokens=2),
            ),
        ]
        types = [e.type for e in events]
        assert types == [
            "tool_result",
            "text",
            "tool_call_delta",
            "reasoning",
            "error",
            "status",
            "usage",
        ]
