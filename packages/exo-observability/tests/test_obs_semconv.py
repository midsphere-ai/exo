"""Tests for exo.observability.semconv — semantic conventions."""

from __future__ import annotations

import re
from collections import Counter

from exo.observability import semconv  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _public_constants() -> dict[str, str]:
    """Return all public uppercase constants from semconv module."""
    return {
        name: getattr(semconv, name)
        for name in dir(semconv)
        if name.isupper() and not name.startswith("_")
    }


# ---------------------------------------------------------------------------
# Basic invariants
# ---------------------------------------------------------------------------


class TestConstantsAreStrings:
    """All exported constants must be plain strings."""

    def test_all_constants_are_strings(self) -> None:
        for name, value in _public_constants().items():
            assert isinstance(value, str), f"{name} should be str, got {type(value)}"

    def test_all_constants_are_nonempty(self) -> None:
        for name, value in _public_constants().items():
            assert value, f"{name} should not be empty"


class TestNoDuplicates:
    """No two constants should share the same value."""

    def test_no_duplicate_values(self) -> None:
        values = list(_public_constants().values())
        counts = Counter(values)
        duplicates = {v: c for v, c in counts.items() if c > 1}
        assert not duplicates, f"Duplicate constant values: {duplicates}"


# ---------------------------------------------------------------------------
# Naming pattern validation
# ---------------------------------------------------------------------------


class TestNamingPatterns:
    """Constants follow dotted namespace conventions."""

    def test_gen_ai_constants_prefixed(self) -> None:
        for name, value in _public_constants().items():
            if name.startswith("GEN_AI_"):
                assert value.startswith("gen_ai."), f"{name}={value!r} should start with 'gen_ai.'"

    def test_agent_constants_prefixed(self) -> None:
        for name, value in _public_constants().items():
            if name.startswith("AGENT_"):
                assert value.startswith("exo.agent."), (
                    f"{name}={value!r} should start with 'exo.agent.'"
                )

    def test_tool_constants_prefixed(self) -> None:
        for name, value in _public_constants().items():
            if name.startswith("TOOL_"):
                assert value.startswith("exo.tool."), (
                    f"{name}={value!r} should start with 'exo.tool.'"
                )

    def test_cost_constants_prefixed(self) -> None:
        for name, value in _public_constants().items():
            if name.startswith("COST_"):
                assert value.startswith("exo.cost."), (
                    f"{name}={value!r} should start with 'exo.cost.'"
                )

    def test_span_prefix_constants_end_with_dot(self) -> None:
        for name, value in _public_constants().items():
            if name.startswith("SPAN_PREFIX_"):
                assert value.endswith("."), f"{name}={value!r} should end with '.'"

    def test_dotted_namespace_format(self) -> None:
        """All non-span-prefix constants use dotted lowercase format."""
        dotted = re.compile(r"^[a-z][a-z0-9_.]+$")
        for name, value in _public_constants().items():
            if name.startswith("SPAN_PREFIX_"):
                continue
            assert dotted.match(value), f"{name}={value!r} doesn't match dotted namespace format"


# ---------------------------------------------------------------------------
# Completeness checks
# ---------------------------------------------------------------------------


class TestCompleteness:
    """All expected convention groups are present."""

    def test_gen_ai_group_present(self) -> None:
        consts = _public_constants()
        gen_ai_keys = [k for k in consts if k.startswith("GEN_AI_")]
        assert len(gen_ai_keys) >= 18, (
            f"Expected at least 18 gen_ai constants, got {len(gen_ai_keys)}"
        )

    def test_agent_group_present(self) -> None:
        consts = _public_constants()
        agent_keys = [k for k in consts if k.startswith("AGENT_")]
        assert len(agent_keys) >= 7, f"Expected at least 7 agent constants, got {len(agent_keys)}"

    def test_tool_group_present(self) -> None:
        consts = _public_constants()
        tool_keys = [k for k in consts if k.startswith("TOOL_")]
        assert len(tool_keys) >= 7, f"Expected at least 7 tool constants, got {len(tool_keys)}"

    def test_cost_group_present(self) -> None:
        consts = _public_constants()
        cost_keys = [k for k in consts if k.startswith("COST_")]
        assert len(cost_keys) >= 3, f"Expected at least 3 cost constants, got {len(cost_keys)}"

    def test_span_prefix_group_present(self) -> None:
        consts = _public_constants()
        prefix_keys = [k for k in consts if k.startswith("SPAN_PREFIX_")]
        assert len(prefix_keys) >= 4, (
            f"Expected at least 4 span prefix constants, got {len(prefix_keys)}"
        )

    def test_task_session_user_present(self) -> None:
        consts = _public_constants()
        assert "TASK_ID" in consts
        assert "TASK_INPUT" in consts
        assert "SESSION_ID" in consts
        assert "USER_ID" in consts
        assert "TRACE_ID" in consts


# ---------------------------------------------------------------------------
# Specific value checks (parity with exo-trace config.py)
# ---------------------------------------------------------------------------


class TestExactValues:
    """Key constants have exact expected values (backward compat)."""

    def test_agent_id(self) -> None:
        assert semconv.AGENT_ID == "exo.agent.id"

    def test_agent_name(self) -> None:
        assert semconv.AGENT_NAME == "exo.agent.name"

    def test_tool_name(self) -> None:
        assert semconv.TOOL_NAME == "exo.tool.name"

    def test_session_id(self) -> None:
        assert semconv.SESSION_ID == "exo.session.id"

    def test_gen_ai_system(self) -> None:
        assert semconv.GEN_AI_SYSTEM == "gen_ai.system"

    def test_gen_ai_usage_input_tokens(self) -> None:
        assert semconv.GEN_AI_USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"

    def test_cost_total_usd(self) -> None:
        assert semconv.COST_TOTAL_USD == "exo.cost.total_usd"

    def test_span_prefix_agent(self) -> None:
        assert semconv.SPAN_PREFIX_AGENT == "agent."

    def test_span_prefix_llm(self) -> None:
        assert semconv.SPAN_PREFIX_LLM == "llm."
