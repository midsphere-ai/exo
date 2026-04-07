"""Tests for tool_result helpers — tool_ok and tool_error."""

import json

from exo.tool_result import tool_error, tool_ok


class TestToolOk:
    def test_returns_valid_json(self) -> None:
        result = tool_ok("done")
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["message"] == "done"

    def test_extra_keys(self) -> None:
        result = tool_ok("created", count=3, ids=["a", "b", "c"])
        parsed = json.loads(result)
        assert parsed["status"] == "ok"
        assert parsed["message"] == "created"
        assert parsed["count"] == 3
        assert parsed["ids"] == ["a", "b", "c"]

    def test_returns_str(self) -> None:
        result = tool_ok("hi")
        assert isinstance(result, str)


class TestToolError:
    def test_returns_valid_json(self) -> None:
        result = tool_error("not found", hint="try again")
        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert parsed["error"] == "not found"
        assert parsed["hint"] == "try again"

    def test_no_context_key_when_empty(self) -> None:
        result = tool_error("fail", hint="retry")
        parsed = json.loads(result)
        assert "context" not in parsed

    def test_context_data_included(self) -> None:
        result = tool_error(
            "Skill 'foo' not found",
            hint="Choose from available skills.",
            available_skills=["bar", "baz"],
        )
        parsed = json.loads(result)
        assert parsed["context"]["available_skills"] == ["bar", "baz"]

    def test_returns_str(self) -> None:
        result = tool_error("oops", hint="fix it")
        assert isinstance(result, str)

    def test_hint_is_required(self) -> None:
        # hint is keyword-only and required
        result = tool_error("err", hint="do X")
        parsed = json.loads(result)
        assert parsed["hint"] == "do X"
