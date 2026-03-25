"""Tests for exo.log — minimal pretty logging."""

from __future__ import annotations

import logging

import pytest

from exo.log import (
    _PREFIX,
    _Formatter,  # pyright: ignore[reportAttributeAccessIssue]
    configure,  # pyright: ignore[reportAttributeAccessIssue]
    get_logger,  # pyright: ignore[reportAttributeAccessIssue]
)
from exo.observability.logging import reset_logging  # pyright: ignore[reportMissingImports]


class TestGetLogger:
    def test_returns_child_of_exo(self) -> None:
        log = get_logger("mymodule")
        assert log.name == "exo.mymodule"
        assert isinstance(log, logging.Logger)

    def test_already_prefixed(self) -> None:
        log = get_logger("exo.agent")
        assert log.name == "exo.agent"

    def test_bare_prefix(self) -> None:
        log = get_logger("exo")
        assert log.name == "exo"


class TestConfigure:
    def setup_method(self) -> None:
        reset_logging()

    def test_adds_handler(self) -> None:
        root = logging.getLogger(_PREFIX)
        assert len(root.handlers) == 0
        configure(level="DEBUG")
        assert len(root.handlers) == 1

    def test_idempotent(self) -> None:
        configure(level="DEBUG")
        configure(level="DEBUG")
        root = logging.getLogger(_PREFIX)
        assert len(root.handlers) == 1

    def test_force_resets(self) -> None:
        configure(level="DEBUG")
        configure(level="INFO", force=True)
        root = logging.getLogger(_PREFIX)
        assert len(root.handlers) == 1
        assert root.level == logging.INFO

    def test_sets_level_from_string(self) -> None:
        configure(level="ERROR")
        root = logging.getLogger(_PREFIX)
        assert root.level == logging.ERROR

    def test_sets_level_from_int(self) -> None:
        configure(level=logging.DEBUG)
        root = logging.getLogger(_PREFIX)
        assert root.level == logging.DEBUG


class TestFormatter:
    def _make_record(
        self,
        name: str = "exo.agent",
        level: int = logging.INFO,
        msg: str = "hello",
        exc_info: tuple | None = None,  # type: ignore[type-arg]
    ) -> logging.LogRecord:
        record = logging.LogRecord(
            name=name,
            level=level,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=exc_info,
        )
        return record

    def test_output_contains_level_char(self) -> None:
        fmt = _Formatter()
        record = self._make_record(level=logging.WARNING, msg="watch out")
        output = fmt.format(record)
        assert "W" in output
        assert "watch out" in output

    def test_strips_exo_prefix(self) -> None:
        fmt = _Formatter()
        record = self._make_record(name="exo.runner", msg="test")
        output = fmt.format(record)
        assert "runner" in output
        # Should not contain "exo.runner" as the full dotted name
        assert "exo.runner" not in output

    def test_preserves_non_exo_name(self) -> None:
        fmt = _Formatter()
        record = self._make_record(name="other.module", msg="test")
        output = fmt.format(record)
        assert "other.module" in output

    def test_separator_present(self) -> None:
        fmt = _Formatter()
        record = self._make_record(msg="test")
        output = fmt.format(record)
        assert "\u25b8" in output

    def test_exception_included(self) -> None:
        fmt = _Formatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
        record = self._make_record(msg="failed", exc_info=exc_info)  # type: ignore[arg-type]
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
        fmt = _Formatter()
        record = self._make_record(level=level, msg="x")
        output = fmt.format(record)
        assert char in output
