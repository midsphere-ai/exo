"""Tests for AI-assisted agent generation provider resolution."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from orbiter.models.types import ModelResponse
from orbiter_web.routes.agents import _SYSTEM_PROMPT, _call_model_for_generation


def _make_mock_db(provider_row: dict[str, object]) -> AsyncMock:
    db = AsyncMock()
    cursor = AsyncMock()
    cursor.fetchone = AsyncMock(return_value=provider_row)
    db.execute = AsyncMock(return_value=cursor)
    return db


class _RecordingProvider:
    def __init__(self, content: str = '{"agents": []}') -> None:
        self.calls: list[tuple[list[object], dict[str, object]]] = []
        self._content = content

    async def complete(self, messages: list[object], **kwargs: object) -> ModelResponse:
        self.calls.append((list(messages), dict(kwargs)))
        return ModelResponse(content=self._content)


@pytest.mark.asyncio
async def test_generation_uses_shared_runtime_for_keyless_provider() -> None:
    provider_row = {
        "id": "prov-vertex",
        "provider_type": "vertexopenai",
        "encrypted_api_key": None,
        "google_project": "acme-prod",
        "google_location": "us-central1",
    }
    db = _make_mock_db(provider_row)
    provider = _RecordingProvider()

    with (
        patch("orbiter_web.routes.agents.get_db") as mock_get_db,
        patch(
            "orbiter_web.routes.agents._resolve_provider_api_key",
            AsyncMock(return_value=None),
        ) as mock_resolve_key,
        patch(
            "orbiter_web.routes.agents._build_provider_from_row",
            return_value=provider,
        ) as mock_build_provider,
    ):
        mock_get_db.return_value.__aenter__ = AsyncMock(return_value=db)
        mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await _call_model_for_generation(
            "prov-vertex",
            "google/gemini-2.0-flash-001",
            "Design a release triage agent.",
            "user-001",
        )

    assert result["output"] == '{"agents": []}'
    mock_resolve_key.assert_awaited_once_with(db, provider_row)
    mock_build_provider.assert_called_once_with(
        provider_row,
        "google/gemini-2.0-flash-001",
        api_key=None,
    )
    messages, kwargs = provider.calls[0]
    assert messages[0].content == _SYSTEM_PROMPT
    assert messages[1].content == "Design a release triage agent."
    assert kwargs == {"temperature": 0.7, "max_tokens": 2048}


@pytest.mark.asyncio
async def test_generation_uses_shared_runtime_for_keyed_provider() -> None:
    provider_row = {
        "id": "prov-proxy",
        "provider_type": "litellm",
        "encrypted_api_key": "encrypted-proxy-key",
        "base_url": "https://proxy.example.com/v1",
    }
    db = _make_mock_db(provider_row)
    provider = _RecordingProvider('{"agents": [{"name": "planner"}]}')

    with (
        patch("orbiter_web.routes.agents.get_db") as mock_get_db,
        patch(
            "orbiter_web.routes.agents._resolve_provider_api_key",
            AsyncMock(return_value="proxy-key"),
        ) as mock_resolve_key,
        patch(
            "orbiter_web.routes.agents._build_provider_from_row",
            return_value=provider,
        ) as mock_build_provider,
    ):
        mock_get_db.return_value.__aenter__ = AsyncMock(return_value=db)
        mock_get_db.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await _call_model_for_generation(
            "prov-proxy",
            "claude-3-7-sonnet",
            "Design a deployment reviewer agent.",
            "user-001",
        )

    assert result["output"] == '{"agents": [{"name": "planner"}]}'
    mock_resolve_key.assert_awaited_once_with(db, provider_row)
    mock_build_provider.assert_called_once_with(
        provider_row,
        "claude-3-7-sonnet",
        api_key="proxy-key",
    )
