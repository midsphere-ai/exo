"""Embedding-based vector memory store for semantic search."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from orbiter.memory.base import (  # pyright: ignore[reportMissingImports]
    MemoryCategory,
    MemoryItem,
    MemoryMetadata,
    MemoryStatus,
)

# ---------------------------------------------------------------------------
# Embeddings ABC
# ---------------------------------------------------------------------------


class Embeddings(ABC):
    """Abstract base class for embedding providers.

    Subclasses implement both sync and async embedding generation.
    Each call embeds a single text string into a float vector.
    """

    __slots__ = ()

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector synchronously."""

    @abstractmethod
    async def aembed(self, text: str) -> list[float]:
        """Generate an embedding vector asynchronously."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension for this provider."""


class OpenAIEmbeddings(Embeddings):
    """OpenAI-compatible embedding provider.

    Works with any API that follows the OpenAI embeddings API format
    (OpenAI, Azure OpenAI, vLLM, Ollama with OpenAI compat, etc.).

    Requires the ``openai`` package to be installed.
    """

    __slots__ = ("_client", "_dimension", "_model")

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        import openai  # lazy import

        self._model = model
        self._dimension = dimension
        kwargs: dict[str, Any] = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(
            input=text,
            model=self._model,
            dimensions=self._dimension,
        )
        return list(resp.data[0].embedding)

    async def aembed(self, text: str) -> list[float]:
        # OpenAI sync client used in thread for simplicity
        import asyncio

        return await asyncio.to_thread(self.embed, text)


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# VectorMemoryStore
# ---------------------------------------------------------------------------


class VectorMemoryStore:
    """In-memory vector store backed by an Embeddings provider.

    Stores memory items alongside their embedding vectors and supports
    semantic (cosine similarity) search via the ``search()`` method.

    Implements the MemoryStore protocol.
    """

    __slots__ = ("_embeddings", "_items", "_vectors")

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings
        self._items: dict[str, MemoryItem] = {}
        self._vectors: dict[str, list[float]] = {}

    @property
    def embeddings(self) -> Embeddings:
        """Return the underlying embeddings provider."""
        return self._embeddings

    # -- MemoryStore protocol -------------------------------------------------

    async def add(self, item: MemoryItem) -> None:
        """Persist a memory item and compute its embedding."""
        vec = await self._embeddings.aembed(item.content)
        self._items[item.id] = item
        self._vectors[item.id] = vec

    async def get(self, item_id: str) -> MemoryItem | None:
        """Retrieve a memory item by ID."""
        return self._items.get(item_id)

    async def search(
        self,
        *,
        query: str = "",
        metadata: MemoryMetadata | None = None,
        memory_type: str | None = None,
        category: MemoryCategory | None = None,
        status: MemoryStatus | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """Semantic search: embed query, rank by cosine similarity.

        Metadata, memory_type, category, and status filters are applied as
        post-filters on the candidate set before ranking.
        """
        candidates = list(self._items.values())

        # Apply filters
        if memory_type:
            candidates = [c for c in candidates if c.memory_type == memory_type]
        if category is not None:
            candidates = [c for c in candidates if c.category == category]
        if status:
            candidates = [c for c in candidates if c.status == status]
        if metadata:
            candidates = [c for c in candidates if _matches_metadata(c, metadata)]

        if not candidates:
            return []

        # Semantic ranking
        if query:
            query_vec = await self._embeddings.aembed(query)
            scored = [
                (item, _cosine_similarity(query_vec, self._vectors[item.id]))
                for item in candidates
                if item.id in self._vectors
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [item for item, _ in scored[:limit]]

        # No query — return newest first
        candidates.sort(key=lambda c: c.created_at, reverse=True)
        return candidates[:limit]

    async def clear(
        self,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> int:
        """Remove memory items matching the filter. Returns count."""
        if metadata is None:
            count = len(self._items)
            self._items.clear()
            self._vectors.clear()
            return count

        to_remove = [
            item_id for item_id, item in self._items.items() if _matches_metadata(item, metadata)
        ]
        for item_id in to_remove:
            del self._items[item_id]
            self._vectors.pop(item_id, None)
        return len(to_remove)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        dim = self._embeddings.dimension
        return f"VectorMemoryStore(items={len(self._items)}, dimension={dim})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matches_metadata(item: MemoryItem, metadata: MemoryMetadata) -> bool:
    """Check whether an item's metadata matches the given filter."""
    m = item.metadata
    if metadata.user_id and m.user_id != metadata.user_id:
        return False
    if metadata.session_id and m.session_id != metadata.session_id:
        return False
    if metadata.task_id and m.task_id != metadata.task_id:
        return False
    return not (metadata.agent_id and m.agent_id != metadata.agent_id)
