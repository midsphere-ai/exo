"""Vertex AI embeddings provider.

Wraps Google Vertex AI's text embedding API to implement the ``Embeddings`` ABC.
Uses ``httpx`` for async HTTP calls so the ``google-cloud-aiplatform`` SDK is
not required.
"""

from __future__ import annotations

from typing import Any

import httpx

from .embeddings import Embeddings  # pyright: ignore[reportMissingImports]
from .types import RetrievalError  # pyright: ignore[reportMissingImports]

_DEFAULT_MODEL = "text-embedding-005"
_DEFAULT_DIMENSION = 768
_DEFAULT_LOCATION = "us-central1"


class VertexEmbeddings(Embeddings):
    """Embedding provider backed by Google Vertex AI.

    Args:
        api_key: API key or access token for authentication.
        project: Google Cloud project ID.
        model: Embedding model name (e.g. ``"text-embedding-005"``).
        dimension: Vector dimensionality. Defaults to 768.
        location: Google Cloud region. Defaults to ``"us-central1"``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        project: str,
        model: str = _DEFAULT_MODEL,
        dimension: int = _DEFAULT_DIMENSION,
        location: str = _DEFAULT_LOCATION,
    ) -> None:
        self._api_key = api_key
        self._project = project
        self._model = model
        self._dimension = dimension
        self._location = location

    @property
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""
        return self._dimension

    def _endpoint_url(self) -> str:
        """Build the Vertex AI prediction endpoint URL."""
        return (
            f"https://{self._location}-aiplatform.googleapis.com/v1/"
            f"projects/{self._project}/locations/{self._location}/"
            f"publishers/google/models/{self._model}:predict"
        )

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string via the Vertex AI API.

        Args:
            text: The text to embed.

        Returns:
            A dense vector of length ``dimension``.
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in a single Vertex AI API call.

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
            "instances": [{"content": t} for t in texts],
            "parameters": {"outputDimensionality": self._dimension},
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    self._endpoint_url(),
                    json=payload,
                    headers=headers,
                    timeout=60.0,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RetrievalError(
                    f"Vertex AI embeddings API error: {exc.response.status_code}",
                    operation="embed",
                    details={"status": exc.response.status_code, "body": exc.response.text},
                ) from exc
            except httpx.HTTPError as exc:
                raise RetrievalError(
                    f"Vertex AI embeddings request failed: {exc}",
                    operation="embed",
                ) from exc

        body = resp.json()
        predictions = body["predictions"]
        return [p["embeddings"]["values"] for p in predictions]
