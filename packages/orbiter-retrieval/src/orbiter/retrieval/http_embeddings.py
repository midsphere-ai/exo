"""Generic HTTP embeddings provider.

Sends a POST request to any embedding endpoint and extracts vectors from the
response using configurable field paths. Uses ``httpx`` for async HTTP calls.
"""

from __future__ import annotations

from typing import Any

import httpx

from .embeddings import Embeddings  # pyright: ignore[reportMissingImports]
from .types import RetrievalError  # pyright: ignore[reportMissingImports]


def _get_nested(obj: Any, path: str) -> Any:
    """Traverse a nested dict/list using a dot-separated path.

    Supports integer keys for list indexing (e.g. ``"data.0.embedding"``).
    """
    for key in path.split("."):
        if isinstance(obj, list):
            obj = obj[int(key)]
        else:
            obj = obj[key]
    return obj


class HTTPEmbeddings(Embeddings):
    """Embedding provider that calls any HTTP endpoint.

    Args:
        url: The embedding endpoint URL.
        dimension: Vector dimensionality.
        headers: Optional HTTP headers (e.g. for authentication).
        input_field: Dot path in the request body for the input texts.
            Defaults to ``"input"``.
        output_field: Dot path in the response body to the list of
            embedding objects. Defaults to ``"data"``.
        vector_field: Dot path within each embedding object to the
            vector. Defaults to ``"embedding"``.
        timeout: Request timeout in seconds. Defaults to 60.
    """

    def __init__(
        self,
        *,
        url: str,
        dimension: int,
        headers: dict[str, str] | None = None,
        input_field: str = "input",
        output_field: str = "data",
        vector_field: str = "embedding",
        timeout: float = 60.0,
    ) -> None:
        self._url = url
        self._dimension = dimension
        self._headers = dict(headers) if headers else {}
        self._input_field = input_field
        self._output_field = output_field
        self._vector_field = vector_field
        self._timeout = timeout

    @property
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            A dense vector of length ``dimension``.
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single HTTP call.

        Args:
            texts: A list of texts to embed.

        Returns:
            A list of dense vectors, one per input text.

        Raises:
            RetrievalError: If the HTTP call or response parsing fails.
        """
        if not texts:
            return []

        payload: dict[str, Any] = {self._input_field: texts}

        req_headers = {"Content-Type": "application/json", **self._headers}

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    self._url,
                    json=payload,
                    headers=req_headers,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RetrievalError(
                    f"HTTP embeddings API error: {exc.response.status_code}",
                    operation="embed",
                    details={"status": exc.response.status_code, "body": exc.response.text},
                ) from exc
            except httpx.HTTPError as exc:
                raise RetrievalError(
                    f"HTTP embeddings request failed: {exc}",
                    operation="embed",
                ) from exc

        body = resp.json()
        try:
            items = _get_nested(body, self._output_field)
            return [_get_nested(item, self._vector_field) for item in items]
        except (KeyError, IndexError, TypeError) as exc:
            raise RetrievalError(
                f"Failed to extract embeddings from response: {exc}",
                operation="embed",
                details={"output_field": self._output_field, "vector_field": self._vector_field},
            ) from exc
