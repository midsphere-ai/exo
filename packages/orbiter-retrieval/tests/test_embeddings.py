"""Tests for Embeddings ABC and OpenAIEmbeddings provider."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from orbiter.retrieval.embeddings import Embeddings
from orbiter.retrieval.openai_embeddings import OpenAIEmbeddings
from orbiter.retrieval.types import RetrievalError


# ---------------------------------------------------------------------------
# Embeddings ABC
# ---------------------------------------------------------------------------


class TestEmbeddingsABC:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            Embeddings()  # type: ignore[abstract]

    def test_concrete_subclass(self) -> None:
        class Dummy(Embeddings):
            async def embed(self, text: str) -> list[float]:
                return [0.0]

            async def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] for _ in texts]

            @property
            def dimension(self) -> int:
                return 1

        d = Dummy()
        assert d.dimension == 1


# ---------------------------------------------------------------------------
# OpenAIEmbeddings
# ---------------------------------------------------------------------------


def _make_response(embeddings: list[list[float]], status: int = 200) -> httpx.Response:
    """Build a mock httpx.Response with embedding data."""
    body = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(embeddings)
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }
    return httpx.Response(
        status_code=status,
        json=body,
        request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
    )


def _make_error_response(status: int, body: str = "error") -> httpx.Response:
    """Build a mock httpx.Response for an error."""
    return httpx.Response(
        status_code=status,
        text=body,
        request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
    )


class TestOpenAIEmbeddings:
    def test_dimension(self) -> None:
        e = OpenAIEmbeddings(api_key="test-key", dimension=768)
        assert e.dimension == 768

    def test_default_dimension(self) -> None:
        e = OpenAIEmbeddings(api_key="test-key")
        assert e.dimension == 1536

    @pytest.mark.asyncio
    async def test_embed_single(self) -> None:
        vec = [0.1, 0.2, 0.3]
        mock_resp = _make_response([vec])

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("orbiter.retrieval.openai_embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(api_key="test-key")
            result = await e.embed("hello")

        assert result == vec
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["input"] == ["hello"]
        assert call_kwargs[1]["json"]["model"] == "text-embedding-3-small"
        assert "Bearer test-key" in call_kwargs[1]["headers"]["Authorization"]

    @pytest.mark.asyncio
    async def test_embed_batch(self) -> None:
        vecs = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        mock_resp = _make_response(vecs)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("orbiter.retrieval.openai_embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(api_key="test-key", dimension=2)
            result = await e.embed_batch(["a", "b", "c"])

        assert result == vecs
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["input"] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self) -> None:
        e = OpenAIEmbeddings(api_key="test-key")
        result = await e.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_preserves_order(self) -> None:
        """API may return embeddings out of order; we sort by index."""
        body = {
            "object": "list",
            "data": [
                {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }
        mock_resp = httpx.Response(
            status_code=200,
            json=body,
            request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("orbiter.retrieval.openai_embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(api_key="test-key", dimension=2)
            result = await e.embed_batch(["first", "second"])

        assert result == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.asyncio
    async def test_embed_http_error(self) -> None:
        mock_resp = _make_error_response(401, "Unauthorized")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("orbiter.retrieval.openai_embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(api_key="bad-key")
            with pytest.raises(RetrievalError, match="401"):
                await e.embed("hello")

    @pytest.mark.asyncio
    async def test_embed_connection_error(self) -> None:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("orbiter.retrieval.openai_embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(api_key="test-key")
            with pytest.raises(RetrievalError, match="request failed"):
                await e.embed("hello")

    @pytest.mark.asyncio
    async def test_custom_base_url(self) -> None:
        vec = [0.1]
        mock_resp = _make_response([vec])

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("orbiter.retrieval.openai_embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(
                api_key="test-key",
                base_url="https://custom.api.com/v1/",
                dimension=1,
            )
            await e.embed("test")

        url = mock_client.post.call_args[0][0]
        assert url == "https://custom.api.com/v1/embeddings"

    @pytest.mark.asyncio
    async def test_custom_model(self) -> None:
        vec = [0.1]
        mock_resp = _make_response([vec])

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("orbiter.retrieval.openai_embeddings.httpx.AsyncClient", return_value=mock_client):
            e = OpenAIEmbeddings(
                api_key="test-key",
                model="text-embedding-3-large",
                dimension=1,
            )
            await e.embed("test")

        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["model"] == "text-embedding-3-large"

    def test_retrieval_error_fields(self) -> None:
        err = RetrievalError("fail", operation="embed", details={"code": 401})
        assert err.operation == "embed"
        assert err.details == {"code": 401}
        assert "fail" in str(err)
