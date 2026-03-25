"""Tests for webhook trigger token validation."""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from exo_web.routes.webhooks import router

# ---------------------------------------------------------------------------
# Minimal test app
# ---------------------------------------------------------------------------

_app = FastAPI()
_app.include_router(router)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CORRECT_TOKEN = "correct-token-abc123"
WORKFLOW_ID = "wf-001"
HOOK_ID = "hook-001"


def _make_webhook_row(url_token: str = CORRECT_TOKEN, enabled: bool = True) -> dict[str, Any]:
    return {
        "id": "wh-001",
        "workflow_id": WORKFLOW_ID,
        "hook_id": HOOK_ID,
        "url_token": url_token,
        "enabled": 1 if enabled else 0,
        "request_log_json": "[]",
        "user_id": "user-001",
        "created_at": "2025-01-01 00:00:00",
        "updated_at": "2025-01-01 00:00:00",
    }


def _make_workflow_row() -> dict[str, Any]:
    return {
        "id": WORKFLOW_ID,
        "name": "Test Workflow",
        "nodes_json": json.dumps([{"id": "n1", "type": "start"}]),
        "edges_json": "[]",
        "user_id": "user-001",
        "created_at": "2025-01-01 00:00:00",
        "updated_at": "2025-01-01 00:00:00",
    }


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


class TestWebhookTriggerRequiresToken:
    """Test that the webhook trigger endpoint validates the url_token."""

    def test_missing_token_returns_403(self) -> None:
        """POST without url_token query param returns 403."""
        # get_db returns the webhook row so we reach the validation step
        mock_get_db = _make_get_db(
            _make_webhook_row(),  # webhook lookup
            _make_webhook_row(),  # _append_request_log: SELECT
            None,  # _append_request_log: UPDATE (fetchone not used, but execute called)
        )
        with patch("exo_web.routes.webhooks.get_db", mock_get_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.post(f"/api/v1/webhooks/{WORKFLOW_ID}/{HOOK_ID}", json={})
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Invalid token"

    def test_wrong_token_returns_403(self) -> None:
        """POST with an incorrect url_token query param returns 403."""
        mock_get_db = _make_get_db(
            _make_webhook_row(),  # webhook lookup
            _make_webhook_row(),  # _append_request_log: SELECT
            None,
        )
        with patch("exo_web.routes.webhooks.get_db", mock_get_db):
            client = TestClient(_app, raise_server_exceptions=False)
            resp = client.post(
                f"/api/v1/webhooks/{WORKFLOW_ID}/{HOOK_ID}",
                params={"url_token": "wrong-token"},
                json={},
            )
        assert resp.status_code == 403
        assert resp.json()["detail"] == "Invalid token"

    def test_correct_token_returns_200(self) -> None:
        """POST with the correct url_token triggers the workflow and returns 200."""
        run_id = str(uuid.uuid4())

        mock_get_db = _make_get_db(
            _make_webhook_row(),  # webhook lookup
            _make_workflow_row(),  # workflow lookup
            None,  # INSERT workflow_run
            _make_webhook_row(),  # _append_request_log: SELECT
            None,  # _append_request_log: UPDATE
        )

        mock_task = MagicMock()
        mock_task.add_done_callback = MagicMock()

        with (
            patch("exo_web.routes.webhooks.get_db", mock_get_db),
            patch("exo_web.services.run_queue.can_start_run", AsyncMock(return_value=True)),
            patch("exo_web.engine.execute_workflow", AsyncMock()),
            patch("asyncio.create_task", return_value=mock_task),
            patch("uuid.uuid4", return_value=run_id),
        ):
            # Import here so the patches above take effect inside execute_workflow import
            client = TestClient(_app, raise_server_exceptions=True)
            resp = client.post(
                f"/api/v1/webhooks/{WORKFLOW_ID}/{HOOK_ID}",
                params={"url_token": CORRECT_TOKEN},
                json={"event": "push"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "triggered"
        assert "run_id" in data
