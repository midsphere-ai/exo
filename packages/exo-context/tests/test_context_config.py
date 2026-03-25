"""Tests for ContextConfig and make_config factory."""

import pytest
from pydantic import ValidationError

from exo.context.config import (  # pyright: ignore[reportMissingImports]
    AutomationMode,
    ContextConfig,
    make_config,
)

# ── Defaults ──────────────────────────────────────────────────────────


class TestContextConfigDefaults:
    def test_default_values(self) -> None:
        cfg = ContextConfig()
        assert cfg.mode == AutomationMode.COPILOT
        assert cfg.history_rounds == 20
        assert cfg.summary_threshold == 10
        assert cfg.offload_threshold == 50
        assert cfg.enable_retrieval is False
        assert cfg.neuron_names == ()
        assert cfg.extra == {}
        assert cfg.token_budget_trigger == 0.8

    def test_token_budget_trigger_default(self) -> None:
        cfg = ContextConfig()
        assert cfg.token_budget_trigger == 0.8

    def test_token_budget_trigger_custom(self) -> None:
        cfg = ContextConfig(token_budget_trigger=0.5)
        assert cfg.token_budget_trigger == 0.5

    def test_token_budget_trigger_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            ContextConfig(token_budget_trigger=1.5)

    def test_frozen(self) -> None:
        cfg = ContextConfig()
        with pytest.raises(ValidationError):
            cfg.mode = AutomationMode.PILOT  # type: ignore[misc]

    def test_mode_string_coercion(self) -> None:
        cfg = ContextConfig(mode="navigator")  # type: ignore[arg-type]
        assert cfg.mode == AutomationMode.NAVIGATOR


# ── Validation ────────────────────────────────────────────────────────


class TestContextConfigValidation:
    def test_history_rounds_positive(self) -> None:
        with pytest.raises(ValidationError, match="history_rounds"):
            ContextConfig(history_rounds=0)

    def test_summary_threshold_positive(self) -> None:
        with pytest.raises(ValidationError, match="summary_threshold"):
            ContextConfig(summary_threshold=0)

    def test_offload_threshold_positive(self) -> None:
        with pytest.raises(ValidationError, match="offload_threshold"):
            ContextConfig(offload_threshold=0)

    def test_summary_must_be_lte_offload(self) -> None:
        with pytest.raises(ValueError, match=r"summary_threshold.*must be <= offload_threshold"):
            ContextConfig(summary_threshold=60, offload_threshold=50)

    def test_summary_equals_offload_ok(self) -> None:
        cfg = ContextConfig(summary_threshold=50, offload_threshold=50)
        assert cfg.summary_threshold == cfg.offload_threshold

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValidationError):
            ContextConfig(mode="invalid")  # type: ignore[arg-type]


# ── Neuron names ──────────────────────────────────────────────────────


class TestNeuronNames:
    def test_list_coerced_to_tuple(self) -> None:
        cfg = ContextConfig(neuron_names=["system", "task"])  # type: ignore[arg-type]
        assert cfg.neuron_names == ("system", "task")
        assert isinstance(cfg.neuron_names, tuple)

    def test_tuple_accepted(self) -> None:
        cfg = ContextConfig(neuron_names=("history",))
        assert cfg.neuron_names == ("history",)

    def test_empty_list(self) -> None:
        cfg = ContextConfig(neuron_names=[])  # type: ignore[arg-type]
        assert cfg.neuron_names == ()


# ── Extra metadata ───────────────────────────────────────────────────


class TestExtra:
    def test_extra_dict(self) -> None:
        cfg = ContextConfig(extra={"custom_key": 42})
        assert cfg.extra["custom_key"] == 42

    def test_extra_default_empty(self) -> None:
        cfg = ContextConfig()
        assert cfg.extra == {}


# ── Serialization ────────────────────────────────────────────────────


class TestSerialization:
    def test_model_dump(self) -> None:
        cfg = ContextConfig(mode="pilot", neuron_names=["a", "b"])  # type: ignore[arg-type]
        d = cfg.model_dump()
        assert d["mode"] == "pilot"
        assert d["neuron_names"] == ("a", "b")

    def test_model_dump_json(self) -> None:
        cfg = ContextConfig()
        json_str = cfg.model_dump_json()
        assert '"copilot"' in json_str

    def test_roundtrip(self) -> None:
        original = ContextConfig(
            mode="navigator",
            history_rounds=5,
            enable_retrieval=True,
            neuron_names=["system", "task"],  # type: ignore[arg-type]
        )
        d = original.model_dump()
        restored = ContextConfig(**d)
        assert restored == original


# ── make_config factory ──────────────────────────────────────────────


class TestMakeConfig:
    def test_pilot_defaults(self) -> None:
        cfg = make_config("pilot")
        assert cfg.mode == AutomationMode.PILOT
        assert cfg.history_rounds == 100
        assert cfg.summary_threshold == 100
        assert cfg.offload_threshold == 100
        assert cfg.enable_retrieval is False

    def test_copilot_defaults(self) -> None:
        cfg = make_config("copilot")
        assert cfg.mode == AutomationMode.COPILOT
        assert cfg.history_rounds == 20
        assert cfg.summary_threshold == 10
        assert cfg.offload_threshold == 50

    def test_navigator_defaults(self) -> None:
        cfg = make_config("navigator")
        assert cfg.mode == AutomationMode.NAVIGATOR
        assert cfg.history_rounds == 10
        assert cfg.summary_threshold == 5
        assert cfg.offload_threshold == 20
        assert cfg.enable_retrieval is True

    def test_override_presets(self) -> None:
        cfg = make_config("navigator", history_rounds=50)
        assert cfg.mode == AutomationMode.NAVIGATOR
        assert cfg.history_rounds == 50

    def test_enum_input(self) -> None:
        cfg = make_config(AutomationMode.PILOT)
        assert cfg.mode == AutomationMode.PILOT
