"""Tests for the Operator ABC and TunableSpec."""

from __future__ import annotations

from typing import Any

import pytest

from orbiter.train.operator.base import (  # pyright: ignore[reportMissingImports]
    Operator,
    TunableKind,
    TunableSpec,
)


# ---------------------------------------------------------------------------
# Concrete test implementation
# ---------------------------------------------------------------------------


class _StubOperator(Operator):
    """Minimal concrete operator for testing the ABC."""

    def __init__(self, op_name: str = "stub") -> None:
        self._name = op_name
        self._system_prompt = "You are helpful."
        self._temperature = 0.7

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, **kwargs: Any) -> Any:
        return {"echo": kwargs}

    def get_tunables(self) -> list[TunableSpec]:
        return [
            TunableSpec(
                name="system_prompt",
                kind=TunableKind.PROMPT,
                current_value=self._system_prompt,
            ),
            TunableSpec(
                name="temperature",
                kind=TunableKind.CONTINUOUS,
                current_value=self._temperature,
                constraints={"min": 0.0, "max": 2.0},
            ),
        ]

    def get_state(self) -> dict[str, Any]:
        return {
            "system_prompt": self._system_prompt,
            "temperature": self._temperature,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._system_prompt = state["system_prompt"]
        self._temperature = state["temperature"]


# ---------------------------------------------------------------------------
# TunableKind
# ---------------------------------------------------------------------------


class TestTunableKind:
    def test_values(self) -> None:
        assert TunableKind.PROMPT == "prompt"
        assert TunableKind.CONTINUOUS == "continuous"
        assert TunableKind.DISCRETE == "discrete"
        assert TunableKind.TEXT == "text"

    def test_is_str(self) -> None:
        assert isinstance(TunableKind.PROMPT, str)

    def test_all_members(self) -> None:
        assert len(TunableKind) == 4


# ---------------------------------------------------------------------------
# TunableSpec
# ---------------------------------------------------------------------------


class TestTunableSpec:
    def test_creation(self) -> None:
        spec = TunableSpec(name="lr", kind=TunableKind.CONTINUOUS)
        assert spec.name == "lr"
        assert spec.kind == TunableKind.CONTINUOUS
        assert spec.current_value is None
        assert spec.constraints == {}

    def test_with_value_and_constraints(self) -> None:
        spec = TunableSpec(
            name="model",
            kind=TunableKind.DISCRETE,
            current_value="gpt-4",
            constraints={"choices": ["gpt-4", "gpt-3.5"]},
        )
        assert spec.name == "model"
        assert spec.kind == TunableKind.DISCRETE
        assert spec.current_value == "gpt-4"
        assert spec.constraints == {"choices": ["gpt-4", "gpt-3.5"]}

    def test_frozen(self) -> None:
        spec = TunableSpec(name="x", kind=TunableKind.TEXT)
        with pytest.raises(AttributeError):
            spec.name = "y"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = TunableSpec(name="p", kind=TunableKind.PROMPT, current_value="hello")
        b = TunableSpec(name="p", kind=TunableKind.PROMPT, current_value="hello")
        assert a == b

    def test_prompt_kind(self) -> None:
        spec = TunableSpec(
            name="system_prompt",
            kind=TunableKind.PROMPT,
            current_value="Be concise.",
        )
        assert spec.kind == TunableKind.PROMPT
        assert spec.current_value == "Be concise."

    def test_text_kind(self) -> None:
        spec = TunableSpec(
            name="description",
            kind=TunableKind.TEXT,
            current_value="A search tool",
        )
        assert spec.kind == TunableKind.TEXT


# ---------------------------------------------------------------------------
# Operator ABC — abstract enforcement
# ---------------------------------------------------------------------------


class TestOperatorAbstractEnforcement:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            Operator()  # type: ignore[abstract]

    def test_missing_name_raises(self) -> None:
        class _NoName(Operator):
            async def execute(self, **kwargs: Any) -> Any:
                return None

            def get_tunables(self) -> list[TunableSpec]:
                return []

            def get_state(self) -> dict[str, Any]:
                return {}

            def load_state(self, state: dict[str, Any]) -> None:
                pass

        with pytest.raises(TypeError, match="abstract"):
            _NoName()  # type: ignore[abstract]

    def test_missing_execute_raises(self) -> None:
        class _NoExecute(Operator):
            @property
            def name(self) -> str:
                return "x"

            def get_tunables(self) -> list[TunableSpec]:
                return []

            def get_state(self) -> dict[str, Any]:
                return {}

            def load_state(self, state: dict[str, Any]) -> None:
                pass

        with pytest.raises(TypeError, match="abstract"):
            _NoExecute()  # type: ignore[abstract]

    def test_missing_get_tunables_raises(self) -> None:
        class _NoTunables(Operator):
            @property
            def name(self) -> str:
                return "x"

            async def execute(self, **kwargs: Any) -> Any:
                return None

            def get_state(self) -> dict[str, Any]:
                return {}

            def load_state(self, state: dict[str, Any]) -> None:
                pass

        with pytest.raises(TypeError, match="abstract"):
            _NoTunables()  # type: ignore[abstract]

    def test_missing_get_state_raises(self) -> None:
        class _NoGetState(Operator):
            @property
            def name(self) -> str:
                return "x"

            async def execute(self, **kwargs: Any) -> Any:
                return None

            def get_tunables(self) -> list[TunableSpec]:
                return []

            def load_state(self, state: dict[str, Any]) -> None:
                pass

        with pytest.raises(TypeError, match="abstract"):
            _NoGetState()  # type: ignore[abstract]

    def test_missing_load_state_raises(self) -> None:
        class _NoLoadState(Operator):
            @property
            def name(self) -> str:
                return "x"

            async def execute(self, **kwargs: Any) -> Any:
                return None

            def get_tunables(self) -> list[TunableSpec]:
                return []

            def get_state(self) -> dict[str, Any]:
                return {}

        with pytest.raises(TypeError, match="abstract"):
            _NoLoadState()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Operator — concrete stub behaviour
# ---------------------------------------------------------------------------


class TestOperatorStub:
    def test_name(self) -> None:
        op = _StubOperator("my-llm")
        assert op.name == "my-llm"

    def test_get_tunables(self) -> None:
        op = _StubOperator()
        tunables = op.get_tunables()
        assert len(tunables) == 2
        assert tunables[0].name == "system_prompt"
        assert tunables[0].kind == TunableKind.PROMPT
        assert tunables[0].current_value == "You are helpful."
        assert tunables[1].name == "temperature"
        assert tunables[1].kind == TunableKind.CONTINUOUS
        assert tunables[1].current_value == 0.7
        assert tunables[1].constraints == {"min": 0.0, "max": 2.0}

    async def test_execute(self) -> None:
        op = _StubOperator()
        result = await op.execute(prompt="hello")
        assert result == {"echo": {"prompt": "hello"}}

    def test_get_state(self) -> None:
        op = _StubOperator()
        state = op.get_state()
        assert state == {"system_prompt": "You are helpful.", "temperature": 0.7}

    def test_load_state(self) -> None:
        op = _StubOperator()
        op.load_state({"system_prompt": "Be brief.", "temperature": 0.3})
        assert op.get_state() == {"system_prompt": "Be brief.", "temperature": 0.3}

    def test_state_roundtrip(self) -> None:
        op = _StubOperator()
        original_state = op.get_state()
        op.load_state({"system_prompt": "Changed.", "temperature": 1.0})
        assert op.get_state() != original_state
        op.load_state(original_state)
        assert op.get_state() == original_state

    def test_tunables_reflect_state(self) -> None:
        op = _StubOperator()
        op.load_state({"system_prompt": "New prompt.", "temperature": 0.5})
        tunables = op.get_tunables()
        assert tunables[0].current_value == "New prompt."
        assert tunables[1].current_value == 0.5
