"""Tests for password reset endpoint — ensures token is never logged."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from exo_web.routes.auth import router

# ---------------------------------------------------------------------------
# Minimal test app
# ---------------------------------------------------------------------------

_app = FastAPI()
_app.include_router(router)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMAIL = "user@example.com"
USER_ID = "user-001"


class _FakeRow:
    """Minimal aiosqlite.Row-compatible object."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self) -> Any:
        return self._data.keys()

    def __iter__(self) -> Any:
        return iter(self._data.items())

    def __contains__(self, key: str) -> bool:
        return key in self._data


def _fake_row(data: dict[str, Any] | None) -> _FakeRow | None:
    return _FakeRow(data) if data is not None else None


def _make_get_db(*row_sequence: dict[str, Any] | None):
    """Return a patched get_db that yields mock rows in order across calls."""
    rows = list(row_sequence)
    call_index = 0

    @asynccontextmanager
    async def _mock_get_db():
        nonlocal call_index
        mock_db = AsyncMock()

        async def _execute(sql: str, params: Any = ()) -> AsyncMock:
            nonlocal call_index
            cursor = AsyncMock()
            row = rows[call_index] if call_index < len(rows) else None
            cursor.fetchone = AsyncMock(return_value=_fake_row(row))
            cursor.fetchall = AsyncMock(return_value=[])
            cursor.rowcount = 1
            call_index += 1
            return cursor

        mock_db.execute = _execute
        mock_db.commit = AsyncMock()
        yield mock_db

    return _mock_get_db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPasswordResetTokenNotLogged:
    """Ensure the raw reset token never appears in log output."""

    def test_token_not_in_log_output(self, caplog: Any) -> None:
        """The password reset token must not appear in any log record."""
        user_row = {"id": USER_ID, "email": EMAIL}
        mock_get_db = _make_get_db(
            user_row,  # SELECT user by email
            None,  # INSERT password_resets (fetchone not used)
        )
        with caplog.at_level(logging.DEBUG, logger="exo_web"):
            with patch("exo_web.routes.auth.get_db", mock_get_db):
                client = TestClient(_app, raise_server_exceptions=True)
                resp = client.post("/api/v1/auth/forgot-password", json={"email": EMAIL})

        assert resp.status_code == 200

        import re

        uuid_pattern = re.compile(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
        )

        # The raw token (a UUID4) must never appear in any log record.
        for record in caplog.records:
            assert "Password reset token" not in record.message
            assert not uuid_pattern.search(record.message), (
                f"Raw token found in log record: {record.message}"
            )

    def test_email_sent_log_contains_email(self, caplog: Any) -> None:
        """The debug log line must include the email address."""
        user_row = {"id": USER_ID, "email": EMAIL}
        mock_get_db = _make_get_db(user_row, None)

        with caplog.at_level(logging.DEBUG, logger="exo_web"):
            with patch("exo_web.routes.auth.get_db", mock_get_db):
                client = TestClient(_app, raise_server_exceptions=True)
                client.post("/api/v1/auth/forgot-password", json={"email": EMAIL})

        sent_logs = [r for r in caplog.records if "Password reset email sent" in r.message]
        assert len(sent_logs) == 1
        assert EMAIL in sent_logs[0].message
