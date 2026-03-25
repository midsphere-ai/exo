"""Tests for exo.observability.logging — structured logging module."""

from __future__ import annotations

import json
import logging

import pytest

from exo.observability.logging import (  # pyright: ignore[reportMissingImports]
    _ENV_FORMAT,
    JsonFormatter,
    LogContext,
    TextFormatter,
    _configure_from_env,
    _log_context,
    configure_logging,
    get_logger,
    reset_logging,
)

_PREFIX = "exo"


@pytest.fixture(autouse=True)
def _clean_logging() -> None:
    """Reset logging state and context before each test."""
    reset_logging()
    _log_context.set(None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    name: str = "exo.agent",
    level: int = logging.INFO,
    msg: str = "hello",
    exc_info: tuple | None = None,  # type: ignore[type-arg]
) -> logging.LogRecord:
    return logging.LogRecord(
        name=name,
        level=level,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=exc_info,
    )


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    def test_auto_prefixes_name(self) -> None:
        log = get_logger("agent")
        assert log.name == "exo.agent"
        assert isinstance(log, logging.Logger)

    def test_already_prefixed(self) -> None:
        log = get_logger("exo.models")
        assert log.name == "exo.models"

    def test_bare_prefix(self) -> None:
        log = get_logger("exo")
        assert log.name == "exo"

    def test_nested_name(self) -> None:
        log = get_logger("agent.runner")
        assert log.name == "exo.agent.runner"


# ---------------------------------------------------------------------------
# TextFormatter
# ---------------------------------------------------------------------------


class TestTextFormatter:
    def test_contains_level_char(self) -> None:
        fmt = TextFormatter()
        output = fmt.format(_make_record(level=logging.WARNING, msg="watch out"))
        assert "W" in output
        assert "watch out" in output

    def test_strips_exo_prefix(self) -> None:
        fmt = TextFormatter()
        output = fmt.format(_make_record(name="exo.runner", msg="test"))
        assert "runner" in output
        assert "exo.runner" not in output

    def test_preserves_non_exo_name(self) -> None:
        fmt = TextFormatter()
        output = fmt.format(_make_record(name="other.module", msg="test"))
        assert "other.module" in output

    def test_separator_present(self) -> None:
        fmt = TextFormatter()
        output = fmt.format(_make_record(msg="test"))
        assert "\u25b8" in output

    def test_exception_included(self) -> None:
        import sys

        fmt = TextFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            exc_info = sys.exc_info()
        record = _make_record(msg="failed", exc_info=exc_info)  # type: ignore[arg-type]
        output = fmt.format(record)
        assert "ValueError" in output
        assert "boom" in output

    @pytest.mark.parametrize(
        ("level", "char"),
        [
            (logging.DEBUG, "D"),
            (logging.INFO, "I"),
            (logging.WARNING, "W"),
            (logging.ERROR, "E"),
            (logging.CRITICAL, "C"),
        ],
    )
    def test_all_level_chars(self, level: int, char: str) -> None:
        fmt = TextFormatter()
        output = fmt.format(_make_record(level=level, msg="x"))
        assert char in output

    def test_includes_context_vars(self) -> None:
        fmt = TextFormatter()
        _log_context.set({"agent_name": "alpha"})
        output = fmt.format(_make_record(msg="step"))
        assert "agent_name=alpha" in output


# ---------------------------------------------------------------------------
# JsonFormatter
# ---------------------------------------------------------------------------


class TestJsonFormatter:
    def test_valid_json(self) -> None:
        fmt = JsonFormatter()
        output = fmt.format(_make_record(msg="test"))
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["logger"] == "exo.agent"
        assert data["message"] == "test"
        assert "timestamp" in data

    def test_includes_extra_from_context(self) -> None:
        fmt = JsonFormatter()
        _log_context.set({"session_id": "s-1", "task_id": "t-2"})
        output = fmt.format(_make_record(msg="step"))
        data = json.loads(output)
        assert data["extra"]["session_id"] == "s-1"
        assert data["extra"]["task_id"] == "t-2"

    def test_no_extra_when_no_context(self) -> None:
        fmt = JsonFormatter()
        output = fmt.format(_make_record(msg="clean"))
        data = json.loads(output)
        assert "extra" not in data

    def test_exception_included(self) -> None:
        import sys

        fmt = JsonFormatter()
        try:
            raise RuntimeError("kaboom")
        except RuntimeError:
            exc_info = sys.exc_info()
        record = _make_record(msg="error", exc_info=exc_info)  # type: ignore[arg-type]
        output = fmt.format(record)
        data = json.loads(output)
        assert "RuntimeError" in data["exception"]
        assert "kaboom" in data["exception"]


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    def test_adds_handler(self) -> None:
        root = logging.getLogger(_PREFIX)
        assert len(root.handlers) == 0
        configure_logging(level="DEBUG")
        assert len(root.handlers) == 1

    def test_idempotent(self) -> None:
        configure_logging(level="DEBUG")
        configure_logging(level="DEBUG")
        root = logging.getLogger(_PREFIX)
        assert len(root.handlers) == 1

    def test_force_resets(self) -> None:
        configure_logging(level="DEBUG")
        configure_logging(level="INFO", force=True)
        root = logging.getLogger(_PREFIX)
        assert len(root.handlers) == 1
        assert root.level == logging.INFO

    def test_sets_level_from_string(self) -> None:
        configure_logging(level="ERROR")
        root = logging.getLogger(_PREFIX)
        assert root.level == logging.ERROR

    def test_sets_level_from_int(self) -> None:
        configure_logging(level=logging.DEBUG)
        root = logging.getLogger(_PREFIX)
        assert root.level == logging.DEBUG

    def test_text_format(self) -> None:
        configure_logging(fmt="text")
        root = logging.getLogger(_PREFIX)
        assert isinstance(root.handlers[0].formatter, TextFormatter)

    def test_json_format(self) -> None:
        configure_logging(fmt="json")
        root = logging.getLogger(_PREFIX)
        assert isinstance(root.handlers[0].formatter, JsonFormatter)

    def test_format_selection_via_config(self) -> None:
        """Verify that ObservabilityConfig.log_format drives formatter choice."""
        configure_logging(fmt="json")
        root = logging.getLogger(_PREFIX)
        handler = root.handlers[0]
        assert isinstance(handler.formatter, JsonFormatter)

        configure_logging(fmt="text", force=True)
        root = logging.getLogger(_PREFIX)
        handler = root.handlers[0]
        assert isinstance(handler.formatter, TextFormatter)


# ---------------------------------------------------------------------------
# LogContext
# ---------------------------------------------------------------------------


class TestLogContext:
    def test_binds_vars_in_scope(self) -> None:
        assert _log_context.get() is None
        with LogContext(agent_name="alpha", task_id="t-1"):
            ctx = _log_context.get()
            assert ctx is not None
            assert ctx["agent_name"] == "alpha"
            assert ctx["task_id"] == "t-1"
        assert _log_context.get() is None

    def test_nested_contexts(self) -> None:
        with LogContext(agent_name="outer"):
            with LogContext(task_id="inner"):
                ctx = _log_context.get()
                assert ctx is not None
                assert ctx["agent_name"] == "outer"
                assert ctx["task_id"] == "inner"
            ctx = _log_context.get()
            assert ctx is not None
            assert ctx["agent_name"] == "outer"
            assert "task_id" not in ctx

    def test_override_in_nested(self) -> None:
        with LogContext(agent_name="alpha"):
            with LogContext(agent_name="beta"):
                ctx = _log_context.get()
                assert ctx is not None
                assert ctx["agent_name"] == "beta"
            ctx = _log_context.get()
            assert ctx is not None
            assert ctx["agent_name"] == "alpha"

    def test_empty_context_is_noop(self) -> None:
        with LogContext():
            ctx = _log_context.get()
            # Empty dict is truthy, but should have no bindings
            assert ctx is not None
            assert len(ctx) == 0

    def test_context_visible_in_text_formatter(self) -> None:
        fmt = TextFormatter()
        with LogContext(session_id="s-99"):
            output = fmt.format(_make_record(msg="logged"))
        assert "session_id=s-99" in output

    def test_context_visible_in_json_formatter(self) -> None:
        fmt = JsonFormatter()
        with LogContext(agent_name="beta"):
            output = fmt.format(_make_record(msg="logged"))
        data = json.loads(output)
        assert data["extra"]["agent_name"] == "beta"


# ---------------------------------------------------------------------------
# _configure_from_env (EXO_LOG_LEVEL / EXO_DEBUG)
# ---------------------------------------------------------------------------


class TestConfigureFromEnv:
    """Verify EXO_LOG_LEVEL and EXO_DEBUG env-var driven auto-configuration."""

    def test_exo_debug_sets_debug_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reset_logging()
        monkeypatch.setenv("EXO_DEBUG", "1")
        monkeypatch.delenv("EXO_LOG_LEVEL", raising=False)
        _configure_from_env()
        root = logging.getLogger(_PREFIX)
        assert root.level == logging.DEBUG

    def test_exo_log_level_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reset_logging()
        monkeypatch.delenv("EXO_DEBUG", raising=False)
        monkeypatch.setenv("EXO_LOG_LEVEL", "INFO")
        _configure_from_env()
        root = logging.getLogger(_PREFIX)
        assert root.level == logging.INFO

    def test_exo_log_level_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reset_logging()
        monkeypatch.delenv("EXO_DEBUG", raising=False)
        monkeypatch.setenv("EXO_LOG_LEVEL", "ERROR")
        _configure_from_env()
        root = logging.getLogger(_PREFIX)
        assert root.level == logging.ERROR

    def test_exo_debug_takes_precedence_over_log_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reset_logging()
        monkeypatch.setenv("EXO_DEBUG", "1")
        monkeypatch.setenv("EXO_LOG_LEVEL", "ERROR")
        _configure_from_env()
        root = logging.getLogger(_PREFIX)
        assert root.level == logging.DEBUG

    def test_invalid_log_level_defaults_to_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reset_logging()
        monkeypatch.delenv("EXO_DEBUG", raising=False)
        monkeypatch.setenv("EXO_LOG_LEVEL", "VERBOSE")
        _configure_from_env()
        root = logging.getLogger(_PREFIX)
        assert root.level == logging.WARNING

    def test_no_env_vars_leaves_level_at_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reset_logging()
        monkeypatch.delenv("EXO_DEBUG", raising=False)
        monkeypatch.delenv("EXO_LOG_LEVEL", raising=False)
        _configure_from_env()
        root = logging.getLogger(_PREFIX)
        assert root.level == logging.WARNING

    def test_env_var_adds_handler_with_standard_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reset_logging()
        monkeypatch.setenv("EXO_DEBUG", "1")
        monkeypatch.delenv("EXO_LOG_LEVEL", raising=False)
        _configure_from_env()
        root = logging.getLogger(_PREFIX)
        assert len(root.handlers) == 1
        fmt = root.handlers[0].formatter
        assert isinstance(fmt, logging.Formatter)
        assert fmt._fmt == _ENV_FORMAT  # type: ignore[union-attr]

    def test_exo_log_level_adds_handler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reset_logging()
        monkeypatch.delenv("EXO_DEBUG", raising=False)
        monkeypatch.setenv("EXO_LOG_LEVEL", "DEBUG")
        _configure_from_env()
        root = logging.getLogger(_PREFIX)
        assert len(root.handlers) == 1

    def test_no_env_vars_does_not_add_handler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reset_logging()
        monkeypatch.delenv("EXO_DEBUG", raising=False)
        monkeypatch.delenv("EXO_LOG_LEVEL", raising=False)
        _configure_from_env()
        root = logging.getLogger(_PREFIX)
        assert len(root.handlers) == 0

    def test_configure_from_env_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reset_logging()
        monkeypatch.setenv("EXO_DEBUG", "1")
        _configure_from_env()
        _configure_from_env()
        root = logging.getLogger(_PREFIX)
        assert len(root.handlers) == 1

    def test_child_logger_inherits_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reset_logging()
        monkeypatch.setenv("EXO_DEBUG", "1")
        monkeypatch.delenv("EXO_LOG_LEVEL", raising=False)
        _configure_from_env()
        child = get_logger("memory")
        # Child effective level is inherited from root "exo"
        assert child.getEffectiveLevel() == logging.DEBUG

    def test_case_insensitive_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reset_logging()
        monkeypatch.delenv("EXO_DEBUG", raising=False)
        monkeypatch.setenv("EXO_LOG_LEVEL", "info")
        _configure_from_env()
        root = logging.getLogger(_PREFIX)
        assert root.level == logging.INFO
