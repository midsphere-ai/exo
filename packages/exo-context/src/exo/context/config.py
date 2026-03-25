"""Context configuration — automation modes, history, summary, retrieval settings."""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field, model_validator


class AutomationMode(StrEnum):
    """Automation level for context management.

    - pilot: Minimal automation, user controls context manually.
    - copilot: Basic automation — summarization, offloading, history windowing.
    - navigator: Full automation — all context features enabled.
    """

    PILOT = "pilot"
    COPILOT = "copilot"
    NAVIGATOR = "navigator"


class ContextConfig(BaseModel, frozen=True):
    """Immutable configuration for the context engine.

    Controls automation level, history windowing, summarization thresholds,
    context offloading, retrieval, and neuron selection.
    """

    mode: AutomationMode = AutomationMode.COPILOT

    # History windowing
    history_rounds: int = Field(default=20, ge=1, description="Max conversation rounds to keep")

    # Summarization
    summary_threshold: int = Field(
        default=10,
        ge=1,
        description="Number of messages before triggering summarization",
    )

    # Context offloading
    offload_threshold: int = Field(
        default=50,
        ge=1,
        description="Number of messages before offloading older context",
    )

    # Retrieval
    enable_retrieval: bool = Field(
        default=False,
        description="Enable RAG retrieval from workspace artifacts",
    )

    # Neuron selection
    neuron_names: tuple[str, ...] = Field(
        default=(),
        description="Names of neurons to include in prompt building",
    )

    # Token budget trigger for early summarization
    token_budget_trigger: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Token fill ratio (0–1) above which summarization triggers early",
    )

    # Extensible metadata
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional configuration for custom processors or neurons",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_neuron_names(cls, data: Any) -> Any:
        """Accept list or tuple for neuron_names, normalise to tuple."""
        if isinstance(data, dict) and "neuron_names" in data:
            val = data["neuron_names"]
            if isinstance(val, list):
                data["neuron_names"] = tuple(val)
        return data

    @model_validator(mode="after")
    def _validate_thresholds(self) -> ContextConfig:
        """Ensure summary_threshold <= offload_threshold."""
        if self.summary_threshold > self.offload_threshold:
            msg = (
                f"summary_threshold ({self.summary_threshold}) "
                f"must be <= offload_threshold ({self.offload_threshold})"
            )
            raise ValueError(msg)
        return self


def make_config(mode: AutomationMode | str = "copilot", **overrides: Any) -> ContextConfig:
    """Factory for creating ContextConfig at a given automation level.

    Preset defaults per mode:
    - pilot:     history_rounds=100, no summary, no offload, no retrieval
    - copilot:   history_rounds=20, summary at 10, offload at 50
    - navigator: history_rounds=10, summary at 5, offload at 20, retrieval enabled
    """
    mode = AutomationMode(mode)
    presets: dict[AutomationMode, dict[str, Any]] = {
        AutomationMode.PILOT: {
            "history_rounds": 100,
            "summary_threshold": 100,
            "offload_threshold": 100,
            "enable_retrieval": False,
        },
        AutomationMode.COPILOT: {
            "history_rounds": 20,
            "summary_threshold": 10,
            "offload_threshold": 50,
            "enable_retrieval": False,
        },
        AutomationMode.NAVIGATOR: {
            "history_rounds": 10,
            "summary_threshold": 5,
            "offload_threshold": 20,
            "enable_retrieval": True,
        },
    }
    defaults = presets[mode]
    defaults.update(overrides)
    config = ContextConfig(mode=mode, **defaults)
    logger.debug(
        "ContextConfig created mode=%s history_rounds=%d", mode.value, config.history_rounds
    )
    return config
