"""Tests for exo._internal.message_builder."""

from exo._internal.message_builder import (
    build_messages,
    extract_last_assistant_tool_calls,
    merge_usage,
    validate_message_order,
)
from exo.types import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolResult,
    UserMessage,
)

# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


class TestBuildMessages:
    def test_instructions_only(self) -> None:
        msgs = build_messages("You are helpful.", [])
        assert len(msgs) == 1
        assert isinstance(msgs[0], SystemMessage)
        assert msgs[0].content == "You are helpful."

    def test_empty_instructions_skipped(self) -> None:
        msgs = build_messages("", [UserMessage(content="hi")])
        assert len(msgs) == 1
        assert isinstance(msgs[0], UserMessage)

    def test_instructions_plus_history(self) -> None:
        history = [
            UserMessage(content="hello"),
            AssistantMessage(content="hi there"),
        ]
        msgs = build_messages("Be nice.", history)
        assert len(msgs) == 3
        assert isinstance(msgs[0], SystemMessage)
        assert isinstance(msgs[1], UserMessage)
        assert isinstance(msgs[2], AssistantMessage)

    def test_with_tool_results(self) -> None:
        history = [
            UserMessage(content="add 2+3"),
            AssistantMessage(
                content="",
                tool_calls=[ToolCall(id="tc1", name="add", arguments='{"a":2,"b":3}')],
            ),
        ]
        results = [ToolResult(tool_call_id="tc1", tool_name="add", content="5")]
        msgs = build_messages("calc", history, tool_results=results)
        assert len(msgs) == 4  # system + user + assistant + tool_result
        assert isinstance(msgs[3], ToolResult)
        assert msgs[3].content == "5"

    def test_no_tool_results_when_none(self) -> None:
        msgs = build_messages("sys", [UserMessage(content="q")], tool_results=None)
        assert len(msgs) == 2

    def test_empty_tool_results_list(self) -> None:
        msgs = build_messages("sys", [UserMessage(content="q")], tool_results=[])
        assert len(msgs) == 2

    def test_history_preserved_in_order(self) -> None:
        history = [
            UserMessage(content="first"),
            AssistantMessage(content="second"),
            UserMessage(content="third"),
            AssistantMessage(content="fourth"),
        ]
        msgs = build_messages("", history)
        contents = [m.content for m in msgs]
        assert contents == ["first", "second", "third", "fourth"]

    def test_multiple_tool_results(self) -> None:
        results = [
            ToolResult(tool_call_id="tc1", tool_name="a", content="r1"),
            ToolResult(tool_call_id="tc2", tool_name="b", content="r2"),
        ]
        msgs = build_messages("", [], tool_results=results)
        assert len(msgs) == 2
        assert all(isinstance(m, ToolResult) for m in msgs)


# ---------------------------------------------------------------------------
# validate_message_order
# ---------------------------------------------------------------------------


class TestValidateMessageOrder:
    def test_no_warnings_for_clean_conversation(self) -> None:
        msgs = [
            SystemMessage(content="sys"),
            UserMessage(content="hi"),
            AssistantMessage(content="hello"),
        ]
        assert validate_message_order(msgs) == []

    def test_no_warnings_with_matched_tool_calls(self) -> None:
        msgs = [
            UserMessage(content="add"),
            AssistantMessage(
                content="",
                tool_calls=[ToolCall(id="tc1", name="add", arguments="")],
            ),
            ToolResult(tool_call_id="tc1", tool_name="add", content="5"),
        ]
        assert validate_message_order(msgs) == []

    def test_dangling_tool_call_detected(self) -> None:
        msgs = [
            UserMessage(content="add"),
            AssistantMessage(
                content="",
                tool_calls=[ToolCall(id="tc1", name="add", arguments="")],
            ),
        ]
        warnings = validate_message_order(msgs)
        assert len(warnings) == 1
        assert "tc1" in warnings[0]

    def test_multiple_dangling_calls(self) -> None:
        msgs = [
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(id="tc1", name="a", arguments=""),
                    ToolCall(id="tc2", name="b", arguments=""),
                ],
            ),
            ToolResult(tool_call_id="tc1", tool_name="a", content="ok"),
        ]
        warnings = validate_message_order(msgs)
        assert len(warnings) == 1
        assert "tc2" in warnings[0]

    def test_empty_messages_no_warnings(self) -> None:
        assert validate_message_order([]) == []


# ---------------------------------------------------------------------------
# extract_last_assistant_tool_calls
# ---------------------------------------------------------------------------


class TestExtractLastAssistantToolCalls:
    def test_returns_tool_call_ids(self) -> None:
        msgs = [
            UserMessage(content="q"),
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(id="tc1", name="a", arguments=""),
                    ToolCall(id="tc2", name="b", arguments=""),
                ],
            ),
        ]
        assert extract_last_assistant_tool_calls(msgs) == ["tc1", "tc2"]

    def test_returns_empty_for_text_only_assistant(self) -> None:
        msgs = [
            UserMessage(content="q"),
            AssistantMessage(content="answer"),
        ]
        assert extract_last_assistant_tool_calls(msgs) == []

    def test_returns_empty_when_no_assistant(self) -> None:
        msgs = [UserMessage(content="q")]
        assert extract_last_assistant_tool_calls(msgs) == []

    def test_returns_empty_for_empty_messages(self) -> None:
        assert extract_last_assistant_tool_calls([]) == []

    def test_stops_at_user_message(self) -> None:
        msgs = [
            AssistantMessage(
                content="",
                tool_calls=[ToolCall(id="old", name="x", arguments="")],
            ),
            ToolResult(tool_call_id="old", tool_name="x", content="ok"),
            UserMessage(content="next question"),
        ]
        # Last message is UserMessage, no assistant after it
        assert extract_last_assistant_tool_calls(msgs) == []


# ---------------------------------------------------------------------------
# merge_usage
# ---------------------------------------------------------------------------


class TestMergeUsage:
    def test_basic_merge(self) -> None:
        total_in, total_out, total = merge_usage(100, 50, 200, 75)
        assert total_in == 300
        assert total_out == 125
        assert total == 425

    def test_merge_from_zero(self) -> None:
        total_in, total_out, total = merge_usage(0, 0, 10, 5)
        assert total_in == 10
        assert total_out == 5
        assert total == 15

    def test_merge_with_zero_new(self) -> None:
        total_in, total_out, total = merge_usage(100, 50, 0, 0)
        assert total_in == 100
        assert total_out == 50
        assert total == 150
