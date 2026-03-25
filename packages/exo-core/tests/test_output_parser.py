"""Tests for exo._internal.output_parser."""

import pytest
from pydantic import BaseModel

from exo._internal.output_parser import (
    OutputParseError,
    parse_response,
    parse_structured_output,
    parse_tool_arguments,
)
from exo.types import (
    ActionModel,
    AgentOutput,
    ToolCall,
    Usage,
)

# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_text_only(self) -> None:
        out = parse_response(content="Hello!", tool_calls=[], usage=Usage())
        assert isinstance(out, AgentOutput)
        assert out.text == "Hello!"
        assert out.tool_calls == []

    def test_tool_calls_only(self) -> None:
        tc = ToolCall(id="tc1", name="search", arguments='{"q":"test"}')
        out = parse_response(content="", tool_calls=[tc], usage=Usage())
        assert out.text == ""
        assert len(out.tool_calls) == 1
        assert out.tool_calls[0].name == "search"

    def test_mixed_text_and_tool_calls(self) -> None:
        tc = ToolCall(id="tc1", name="calc", arguments='{"x":1}')
        out = parse_response(
            content="Let me calculate.",
            tool_calls=[tc],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
        )
        assert out.text == "Let me calculate."
        assert len(out.tool_calls) == 1
        assert out.usage.input_tokens == 10

    def test_empty_response(self) -> None:
        out = parse_response(content="", tool_calls=[], usage=Usage())
        assert out.text == ""
        assert out.tool_calls == []

    def test_usage_preserved(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=50, total_tokens=150)
        out = parse_response(content="ok", tool_calls=[], usage=usage)
        assert out.usage == usage


# ---------------------------------------------------------------------------
# parse_tool_arguments
# ---------------------------------------------------------------------------


class TestParseToolArguments:
    def test_single_tool_call(self) -> None:
        tcs = [ToolCall(id="tc1", name="add", arguments='{"a": 2, "b": 3}')]
        actions = parse_tool_arguments(tcs)
        assert len(actions) == 1
        assert isinstance(actions[0], ActionModel)
        assert actions[0].tool_call_id == "tc1"
        assert actions[0].tool_name == "add"
        assert actions[0].arguments == {"a": 2, "b": 3}

    def test_multiple_tool_calls(self) -> None:
        tcs = [
            ToolCall(id="tc1", name="search", arguments='{"q": "python"}'),
            ToolCall(id="tc2", name="read", arguments='{"url": "https://x.com"}'),
        ]
        actions = parse_tool_arguments(tcs)
        assert len(actions) == 2
        assert actions[0].tool_name == "search"
        assert actions[1].arguments == {"url": "https://x.com"}

    def test_empty_arguments_string(self) -> None:
        tcs = [ToolCall(id="tc1", name="noop", arguments="")]
        actions = parse_tool_arguments(tcs)
        assert actions[0].arguments == {}

    def test_whitespace_arguments(self) -> None:
        tcs = [ToolCall(id="tc1", name="noop", arguments="  ")]
        actions = parse_tool_arguments(tcs)
        assert actions[0].arguments == {}

    def test_invalid_json_raises(self) -> None:
        tcs = [ToolCall(id="tc1", name="bad", arguments="{not json}")]
        with pytest.raises(OutputParseError, match=r"Invalid JSON.*bad"):
            parse_tool_arguments(tcs)

    def test_non_object_json_raises(self) -> None:
        tcs = [ToolCall(id="tc1", name="arr", arguments="[1, 2, 3]")]
        with pytest.raises(OutputParseError, match="must be a JSON object"):
            parse_tool_arguments(tcs)

    def test_string_json_raises(self) -> None:
        tcs = [ToolCall(id="tc1", name="str", arguments='"just a string"')]
        with pytest.raises(OutputParseError, match="must be a JSON object"):
            parse_tool_arguments(tcs)

    def test_empty_list(self) -> None:
        assert parse_tool_arguments([]) == []

    def test_nested_arguments(self) -> None:
        args = '{"filter": {"type": "date", "range": [1, 10]}, "limit": 5}'
        tcs = [ToolCall(id="tc1", name="query", arguments=args)]
        actions = parse_tool_arguments(tcs)
        assert actions[0].arguments["filter"] == {"type": "date", "range": [1, 10]}
        assert actions[0].arguments["limit"] == 5


# ---------------------------------------------------------------------------
# parse_structured_output
# ---------------------------------------------------------------------------


class _Report(BaseModel):
    title: str
    score: float


class TestParseStructuredOutput:
    def test_valid_json_and_schema(self) -> None:
        text = '{"title": "Test Report", "score": 0.95}'
        result = parse_structured_output(text, _Report)
        assert isinstance(result, _Report)
        assert result.title == "Test Report"
        assert result.score == 0.95

    def test_invalid_json(self) -> None:
        with pytest.raises(OutputParseError, match="not valid JSON"):
            parse_structured_output("{bad json}", _Report)

    def test_schema_validation_failure(self) -> None:
        text = '{"title": "ok"}'  # missing required 'score'
        with pytest.raises(OutputParseError, match=r"failed.*Report.*validation"):
            parse_structured_output(text, _Report)

    def test_wrong_type_in_field(self) -> None:
        text = '{"title": "ok", "score": "not a float"}'
        # Pydantic will coerce strings to floats if possible, so use something
        # truly invalid
        text = '{"title": "ok", "score": "abc"}'
        with pytest.raises(OutputParseError, match=r"failed.*Report.*validation"):
            parse_structured_output(text, _Report)

    def test_extra_fields_allowed_by_default(self) -> None:
        text = '{"title": "ok", "score": 1.0, "extra": true}'
        result = parse_structured_output(text, _Report)
        assert result.title == "ok"
        assert result.score == 1.0
