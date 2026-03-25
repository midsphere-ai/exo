"""Unified SearchManager: query across multiple memory stores in parallel."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from exo.memory.base import (  # pyright: ignore[reportMissingImports]
    MemoryCategory,
    MemoryItem,
    MemoryStore,
)


class SearchManager:
    """Query across multiple :class:`MemoryStore` instances in a single call.

    Results are gathered in parallel, deduplicated by item ID, and sorted by
    ``created_at`` descending (newest first).  A per-store limit prevents any
    single store from dominating the merged result set.
    """

    def __init__(self, stores: Sequence[MemoryStore]) -> None:
        self._stores = list(stores)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        category: MemoryCategory | None = None,
    ) -> list[MemoryItem]:
        """Search all stores and return merged, deduplicated results.

        Each store is queried with *limit* so that no single store can
        return more than *limit* items.  After merging, duplicates (by
        ``item.id``) are removed and the combined list is sorted by
        ``created_at`` descending then trimmed to *limit*.
        """
        if not self._stores:
            return []

        tasks = [
            store.search(query=query, category=category, limit=limit) for store in self._stores
        ]
        results_per_store: list[list[MemoryItem]] = await asyncio.gather(*tasks)

        # Deduplicate by item ID, keeping the first occurrence.
        seen: dict[str, MemoryItem] = {}
        for store_results in results_per_store:
            for item in store_results:
                if item.id not in seen:
                    seen[item.id] = item

        # Sort by created_at descending (newest first).
        merged = sorted(seen.values(), key=lambda m: m.created_at, reverse=True)
        return merged[:limit]

    @property
    def stores(self) -> list[MemoryStore]:
        """Return a copy of the managed store list."""
        return list(self._stores)

    def __repr__(self) -> str:
        return f"SearchManager(stores={len(self._stores)})"
