"""Tests for exo.observability.config — ObservabilityConfig + configure()."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exo.observability.config import (  # pyright: ignore[reportMissingImports]
    ObservabilityConfig,
    TraceBackend,
    configure,
    get_config,
    reset,
)


@pytest.fixture(autouse=True)
def _reset_config() -> None:
    """Reset global config state before each test."""
    reset()


# ---------------------------------------------------------------------------
# ObservabilityConfig validation
# ---------------------------------------------------------------------------


class TestObservabilityConfig:
    def test_defaults(self) -> None:
        cfg = ObservabilityConfig()
        assert cfg.log_level == "WARNING"
        assert cfg.log_format == "text"
        assert cfg.trace_enabled is False
        assert cfg.trace_backend == TraceBackend.OTLP
        assert cfg.trace_endpoint is None
        assert cfg.service_name == "exo"
        assert cfg.sample_rate == 1.0
        assert cfg.metrics_enabled is False
        assert cfg.namespace == "exo"

    def test_custom_values(self) -> None:
        cfg = ObservabilityConfig(
            log_level="DEBUG",
            log_format="json",
            trace_enabled=True,
            trace_backend=TraceBackend.CONSOLE,
            trace_endpoint="http://localhost:4318",
            service_name="my-service",
            sample_rate=0.5,
            metrics_enabled=True,
            namespace="custom",
        )
        assert cfg.log_level == "DEBUG"
        assert cfg.log_format == "json"
        assert cfg.trace_enabled is True
        assert cfg.trace_backend == TraceBackend.CONSOLE
        assert cfg.trace_endpoint == "http://localhost:4318"
        assert cfg.service_name == "my-service"
        assert cfg.sample_rate == 0.5
        assert cfg.metrics_enabled is True
        assert cfg.namespace == "custom"

    def test_frozen(self) -> None:
        cfg = ObservabilityConfig()
        with pytest.raises(ValidationError):
            cfg.log_level = "DEBUG"  # type: ignore[misc]

    def test_invalid_log_format(self) -> None:
        with pytest.raises(ValidationError):
            ObservabilityConfig(log_format="yaml")

    def test_sample_rate_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ObservabilityConfig(sample_rate=-0.1)
        with pytest.raises(ValidationError):
            ObservabilityConfig(sample_rate=1.5)

    def test_trace_backend_enum(self) -> None:
        assert TraceBackend.OTLP == "otlp"
        assert TraceBackend.MEMORY == "memory"
        assert TraceBackend.CONSOLE == "console"


# ---------------------------------------------------------------------------
# configure() function
# ---------------------------------------------------------------------------


class TestConfigure:
    def test_basic_configure(self) -> None:
        cfg = configure()
        assert isinstance(cfg, ObservabilityConfig)
        assert cfg.log_level == "WARNING"

    def test_configure_with_kwargs(self) -> None:
        cfg = configure(log_level="DEBUG", log_format="json")
        assert cfg.log_level == "DEBUG"
        assert cfg.log_format == "json"

    def test_configure_with_config_object(self) -> None:
        custom = ObservabilityConfig(log_level="ERROR")
        cfg = configure(config=custom)
        assert cfg is custom
        assert cfg.log_level == "ERROR"

    def test_idempotent(self) -> None:
        cfg1 = configure(log_level="DEBUG")
        cfg2 = configure(log_level="ERROR")  # should be ignored
        assert cfg1 is cfg2
        assert cfg2.log_level == "DEBUG"

    def test_force_reconfigure(self) -> None:
        cfg1 = configure(log_level="DEBUG")
        cfg2 = configure(log_level="ERROR", force=True)
        assert cfg1 is not cfg2
        assert cfg2.log_level == "ERROR"


# ---------------------------------------------------------------------------
# get_config()
# ---------------------------------------------------------------------------


class TestGetConfig:
    def test_returns_default_when_not_configured(self) -> None:
        cfg = get_config()
        assert isinstance(cfg, ObservabilityConfig)
        assert cfg.log_level == "WARNING"

    def test_returns_configured(self) -> None:
        configure(log_level="DEBUG")
        cfg = get_config()
        assert cfg.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_allows_reconfigure(self) -> None:
        configure(log_level="DEBUG")
        reset()
        cfg = configure(log_level="ERROR")
        assert cfg.log_level == "ERROR"
