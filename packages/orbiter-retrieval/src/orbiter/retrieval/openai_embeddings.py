"""OpenAI embeddings provider.

Wraps the OpenAI embeddings API to implement the ``Embeddings`` ABC.
Uses ``httpx`` for async HTTP calls so the ``openai`` SDK is not required.
"""

from __future__ import annotations

from typing import Any

import httpx

from .embeddings import Embeddings  # pyright: ignore[reportMissingImports]
from .types import RetrievalError  # pyright: ignore[reportMissingImports]

_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_DIMENSION = 1536
_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIEmbeddings(Embeddings):
    """Embedding provider backed by the OpenAI embeddings API.

    Args:
        api_key: OpenAI API key.
        model: Embedding model name (e.g. ``"text-embedding-3-small"``).
        dimension: Vector dimensionality. Defaults to 1536.
        base_url: API base URL. Defaults to ``https://api.openai.com/v1``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        dimension: int = _DEFAULT_DIMENSION,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._dimension = dimension
        self._base_url = base_url.rstrip("/")

    @property
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string via the OpenAI API.

        Args:
            text: The text to embed.

        Returns:
            A dense vector of length ``dimension``.
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single OpenAI API call.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of dense vectors, one per input text.

        Raises:
            RetrievalError: If the API call fails.
        """
        if not texts:
            return []

        payload: dict[str, Any] = {
            "input": texts,
            "model": self._model,
        }
        if self._dimension:
            payload["dimensions"] = self._dimension

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{self._base_url}/embeddings",
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RetrievalError(
                    f"OpenAI embeddings API error: {exc.response.status_code}",
                    operation="embed",
                    details={"status": exc.response.status_code, "body": exc.response.text},
                ) from exc
            except httpx.HTTPError as exc:
                raise RetrievalError(
                    f"OpenAI embeddings request failed: {exc}",
                    operation="embed",
                ) from exc

        body = resp.json()
        # Sort by index to guarantee order matches input order.
        data = sorted(body["data"], key=lambda d: d["index"])
        return [item["embedding"] for item in data]
