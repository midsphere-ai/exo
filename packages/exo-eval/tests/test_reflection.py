"""Tests for exo.eval.reflection — reflection framework."""

from __future__ import annotations

import json
from typing import Any

import pytest

from exo.eval.reflection import (  # pyright: ignore[reportMissingImports]
    GeneralReflector,
    ReflectionHistory,
    ReflectionLevel,
    ReflectionResult,
    ReflectionType,
    Reflector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_judge(response: dict[str, Any] | str):
    """Return an async callable that returns a JSON string or raw text."""

    async def judge(prompt: str) -> str:
        if isinstance(response, dict):
            return json.dumps(response)
        return response

    return judge


class _ConcreteReflector(Reflector):
    """Minimal concrete subclass for testing the ABC."""

    async def analyze(self, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": f"Analyzed: {context.get('task', 'unknown')}",
            "key_findings": ["finding-1"],
            "root_causes": ["cause-1"],
            "insights": ["insight-1"],
            "suggestions": ["suggestion-1"],
        }


# ===========================================================================
# ReflectionType enum
# ===========================================================================


class TestReflectionType:
    def test_values(self) -> None:
        assert set(ReflectionType) == {
            ReflectionType.SUCCESS,
            ReflectionType.FAILURE,
            ReflectionType.OPTIMIZATION,
            ReflectionType.PATTERN,
            ReflectionType.INSIGHT,
        }

    def test_is_str_enum(self) -> None:
        assert ReflectionType.SUCCESS == "success"
        assert isinstance(ReflectionType.FAILURE, str)


# ===========================================================================
# ReflectionLevel enum
# ===========================================================================


class TestReflectionLevel:
    def test_values(self) -> None:
        assert set(ReflectionLevel) == {
            ReflectionLevel.SHALLOW,
            ReflectionLevel.MEDIUM,
            ReflectionLevel.DEEP,
            ReflectionLevel.META,
        }

    def test_is_str_enum(self) -> None:
        assert ReflectionLevel.DEEP == "deep"


# ===========================================================================
# ReflectionResult
# ===========================================================================


class TestReflectionResult:
    def test_creation(self) -> None:
        r = ReflectionResult(
            reflection_type=ReflectionType.SUCCESS,
            level=ReflectionLevel.SHALLOW,
            summary="All good",
        )
        assert r.summary == "All good"
        assert r.key_findings == []
        assert r.insights == []

    def test_frozen(self) -> None:
        r = ReflectionResult(
            reflection_type=ReflectionType.FAILURE,
            level=ReflectionLevel.DEEP,
            summary="Failed",
        )
        with pytest.raises(AttributeError):
            r.summary = "changed"  # type: ignore[misc]

    def test_with_all_fields(self) -> None:
        r = ReflectionResult(
            reflection_type=ReflectionType.PATTERN,
            level=ReflectionLevel.META,
            summary="Pattern found",
            key_findings=["f1", "f2"],
            root_causes=["c1"],
            insights=["i1"],
            suggestions=["s1", "s2"],
            metadata={"extra": True},
        )
        assert len(r.key_findings) == 2
        assert r.metadata["extra"] is True


# ===========================================================================
# ReflectionHistory
# ===========================================================================


class TestReflectionHistory:
    def _result(self, rtype: ReflectionType = ReflectionType.INSIGHT) -> ReflectionResult:
        return ReflectionResult(
            reflection_type=rtype,
            level=ReflectionLevel.MEDIUM,
            summary="test",
        )

    def test_empty(self) -> None:
        h = ReflectionHistory()
        assert h.total_count == 0
        assert h.get_recent() == []

    def test_add_updates_counters(self) -> None:
        h = ReflectionHistory()
        h.add(self._result(ReflectionType.SUCCESS))
        h.add(self._result(ReflectionType.FAILURE))
        h.add(self._result(ReflectionType.INSIGHT))
        assert h.total_count == 3
        assert h.success_count == 1
        assert h.failure_count == 1

    def test_get_recent(self) -> None:
        h = ReflectionHistory()
        for _i in range(10):
            h.add(self._result())
        recent = h.get_recent(3)
        assert len(recent) == 3

    def test_get_by_type(self) -> None:
        h = ReflectionHistory()
        h.add(self._result(ReflectionType.SUCCESS))
        h.add(self._result(ReflectionType.FAILURE))
        h.add(self._result(ReflectionType.SUCCESS))
        assert len(h.get_by_type(ReflectionType.SUCCESS)) == 2
        assert len(h.get_by_type(ReflectionType.FAILURE)) == 1
        assert len(h.get_by_type(ReflectionType.PATTERN)) == 0

    def test_summarize(self) -> None:
        h = ReflectionHistory()
        h.add(self._result(ReflectionType.SUCCESS))
        h.add(self._result(ReflectionType.FAILURE))
        s = h.summarize()
        assert s["total"] == 2
        assert s["success"] == 1
        assert s["failure"] == 1
        assert s["types"]["success"] == 1
        assert s["types"]["pattern"] == 0


# ===========================================================================
# Reflector ABC
# ===========================================================================


class TestReflectorABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            Reflector()  # type: ignore[abstract]

    def test_concrete_subclass(self) -> None:
        r = _ConcreteReflector(name="test", reflection_type=ReflectionType.PATTERN)
        assert r.name == "test"
        assert r.reflection_type == ReflectionType.PATTERN

    async def test_reflect_pipeline(self) -> None:
        r = _ConcreteReflector()
        result = await r.reflect({"task": "hello"})
        assert isinstance(result, ReflectionResult)
        assert result.summary == "Analyzed: hello"
        assert result.key_findings == ["finding-1"]
        assert result.insights == ["insight-1"]
        assert result.suggestions == ["suggestion-1"]
        assert result.metadata == {"reflector": "reflector"}

    async def test_defaults(self) -> None:
        r = _ConcreteReflector()
        assert r.name == "reflector"
        assert r.reflection_type == ReflectionType.INSIGHT
        assert r.level == ReflectionLevel.MEDIUM


# ===========================================================================
# GeneralReflector
# ===========================================================================


class TestGeneralReflectorInit:
    def test_defaults(self) -> None:
        r = GeneralReflector()
        assert r.name == "general_reflector"
        assert r.level == ReflectionLevel.DEEP
        assert r.reflection_type == ReflectionType.INSIGHT

    def test_custom(self) -> None:
        r = GeneralReflector(
            name="custom",
            reflection_type=ReflectionType.FAILURE,
            level=ReflectionLevel.META,
            system_prompt="Custom prompt",
        )
        assert r.name == "custom"
        assert r.level == ReflectionLevel.META
        assert r._system_prompt == "Custom prompt"


class TestGeneralReflectorNoJudge:
    async def test_no_judge_returns_error(self) -> None:
        r = GeneralReflector()
        result = await r.reflect({})
        assert result.summary == "No judge callable provided"


class TestGeneralReflectorWithJudge:
    async def test_full_reflection(self) -> None:
        response = {
            "summary": "Task completed with issues",
            "key_findings": ["slow execution", "memory spike"],
            "root_causes": ["unoptimized loop"],
            "insights": ["batch processing helps"],
            "suggestions": ["refactor inner loop"],
        }
        r = GeneralReflector(judge=_make_judge(response))
        result = await r.reflect({"input": "test input", "output": "test output"})
        assert result.summary == "Task completed with issues"
        assert len(result.key_findings) == 2
        assert "unoptimized loop" in result.root_causes
        assert "batch processing helps" in result.insights
        assert "refactor inner loop" in result.suggestions

    async def test_with_error_context(self) -> None:
        response = {
            "summary": "Error analysis",
            "key_findings": ["timeout"],
            "root_causes": [],
            "insights": [],
            "suggestions": ["increase timeout"],
        }
        r = GeneralReflector(
            judge=_make_judge(response),
            reflection_type=ReflectionType.FAILURE,
        )
        result = await r.reflect({"error": "ConnectionTimeout", "iteration": 3})
        assert result.reflection_type == ReflectionType.FAILURE
        assert result.summary == "Error analysis"

    async def test_prompt_includes_context(self) -> None:
        """Verify the built prompt includes all context sections."""
        captured: list[str] = []

        async def capture_judge(prompt: str) -> str:
            captured.append(prompt)
            return json.dumps({"summary": "ok"})

        r = GeneralReflector(judge=capture_judge)
        await r.reflect({"input": "IN", "output": "OUT", "error": "ERR", "iteration": 5})
        prompt = captured[0]
        assert "[Input]" in prompt
        assert "[Output]" in prompt
        assert "[Error]" in prompt
        assert "[Iteration] 5" in prompt

    async def test_malformed_json_response(self) -> None:
        r = GeneralReflector(judge=_make_judge("This is not JSON at all"))
        result = await r.reflect({"input": "test"})
        # Falls back to using raw text as summary
        assert "This is not JSON" in result.summary

    async def test_partial_json_response(self) -> None:
        response = {"summary": "partial", "key_findings": ["one"]}
        r = GeneralReflector(judge=_make_judge(response))
        result = await r.reflect({})
        assert result.summary == "partial"
        assert result.key_findings == ["one"]
        # Missing fields default to empty
        assert result.root_causes == []
        assert result.insights == []
        assert result.suggestions == []


class TestGeneralReflectorParseResponse:
    def test_valid_json(self) -> None:
        text = 'Some preamble {"summary": "ok", "insights": ["a"]} trailing'
        result = GeneralReflector._parse_response(text)
        assert result["summary"] == "ok"
        assert result["insights"] == ["a"]

    def test_no_json(self) -> None:
        result = GeneralReflector._parse_response("no json here")
        assert result.get("parse_error") is True
        assert "no json" in result["summary"]

    def test_empty_string(self) -> None:
        result = GeneralReflector._parse_response("")
        assert result.get("parse_error") is True
        assert result["summary"] == ""

    def test_nested_json(self) -> None:
        text = json.dumps(
            {
                "summary": "nested",
                "metadata": {"inner": {"deep": True}},
            }
        )
        result = GeneralReflector._parse_response(text)
        assert result["summary"] == "nested"
        assert result["metadata"]["inner"]["deep"] is True


# ===========================================================================
# Integration
# ===========================================================================


class TestReflectionIntegration:
    async def test_reflect_and_track_history(self) -> None:
        response = {
            "summary": "Iteration went well",
            "key_findings": ["fast"],
            "root_causes": [],
            "insights": ["caching works"],
            "suggestions": ["add more caching"],
        }
        reflector = GeneralReflector(
            judge=_make_judge(response),
            reflection_type=ReflectionType.SUCCESS,
        )
        history = ReflectionHistory()
        result = await reflector.reflect({"input": "task-1"})
        history.add(result)

        assert history.total_count == 1
        assert history.success_count == 1
        assert history.get_recent(1)[0].summary == "Iteration went well"

    async def test_multiple_reflectors_different_types(self) -> None:
        history = ReflectionHistory()

        r1 = GeneralReflector(
            judge=_make_judge({"summary": "success pass"}),
            reflection_type=ReflectionType.SUCCESS,
            name="r1",
        )
        r2 = GeneralReflector(
            judge=_make_judge({"summary": "failure pass"}),
            reflection_type=ReflectionType.FAILURE,
            name="r2",
        )

        for reflector in [r1, r2]:
            result = await reflector.reflect({})
            history.add(result)

        stats = history.summarize()
        assert stats["total"] == 2
        assert stats["types"]["success"] == 1
        assert stats["types"]["failure"] == 1
