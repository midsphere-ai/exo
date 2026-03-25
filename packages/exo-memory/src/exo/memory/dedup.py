"""LLM-based semantic deduplication for long-term memory.

Compares new memories against existing ones to decide whether to ADD,
DELETE, MERGE, or SKIP, preventing memory bloat from repeated information.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum

from exo.memory.base import MemoryItem  # pyright: ignore[reportMissingImports]


class UpdateDecision(StrEnum):
    """Decision from deduplication check."""

    ADD = "add"
    DELETE = "delete"
    MERGE = "merge"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class MergeResult:
    """Result of a deduplication check.

    Attributes:
        decision: The deduplication action to take.
        merged_content: New content after merge (only for MERGE).
        delete_ids: IDs of existing memories to delete (for DELETE/MERGE).
    """

    decision: UpdateDecision
    merged_content: str | None = None
    delete_ids: list[str] = field(default_factory=list)


_DEDUP_PROMPT = """\
Compare the NEW memory against EXISTING memories and decide what to do.

NEW MEMORY:
{new_content}

EXISTING MEMORIES:
{existing_content}

Decide one of:
- ADD: The new memory contains distinct information not in existing memories.
- SKIP: The new memory is a duplicate (same information already exists).
- MERGE: The new memory overlaps with existing memories; combine into one.
- DELETE: Existing memories are outdated/superseded by the new memory.

Respond with ONLY a JSON object:
{{"decision": "add|skip|merge|delete", "merged_content": "combined text if merge, null otherwise", "delete_ids": ["id1"] }}
Only include delete_ids for DELETE or MERGE decisions."""


class MemUpdateChecker:
    """LLM-based semantic deduplication for long-term memory.

    When a checker callable is provided, uses LLM to compare new memories
    against existing ones for semantic deduplication. Falls back to exact
    content matching when no checker is available.

    Attributes:
        top_k: Maximum number of existing memories to compare against.
    """

    __slots__ = ("_checker", "_top_k")

    def __init__(
        self,
        *,
        checker: Callable[[str], Awaitable[str]] | None = None,
        top_k: int = 5,
    ) -> None:
        self._checker = checker
        self._top_k = top_k

    async def check(
        self,
        new_item: MemoryItem,
        existing: list[MemoryItem],
    ) -> MergeResult:
        """Check if new_item should be added, merged, skipped, or replaces existing.

        Args:
            new_item: The new memory to check.
            existing: List of existing memories to compare against.

        Returns:
            MergeResult with the deduplication decision.
        """
        if not existing:
            return MergeResult(decision=UpdateDecision.ADD)

        if self._checker is None:
            return self._exact_match(new_item, existing)

        return await self._llm_check(new_item, existing)

    def _exact_match(
        self,
        new_item: MemoryItem,
        existing: list[MemoryItem],
    ) -> MergeResult:
        """Fallback: exact content match (matches existing LongTermMemory behavior)."""
        for item in existing:
            if item.content == new_item.content and item.memory_type == new_item.memory_type:
                return MergeResult(decision=UpdateDecision.SKIP)
        return MergeResult(decision=UpdateDecision.ADD)

    async def _llm_check(
        self,
        new_item: MemoryItem,
        existing: list[MemoryItem],
    ) -> MergeResult:
        """Use LLM to compare new memory against existing memories."""
        candidates = existing[: self._top_k]

        existing_lines: list[str] = []
        for item in candidates:
            existing_lines.append(f"[ID={item.id}] [{item.memory_type}]: {item.content}")

        prompt = _DEDUP_PROMPT.format(
            new_content=f"[{new_item.memory_type}]: {new_item.content}",
            existing_content="\n".join(existing_lines),
        )

        assert self._checker is not None  # guarded by caller
        response = await self._checker(prompt)
        return _parse_llm_response(response)

    def __repr__(self) -> str:
        mode = "llm" if self._checker else "exact"
        return f"MemUpdateChecker(mode={mode!r}, top_k={self._top_k})"


def _parse_llm_response(response: str) -> MergeResult:
    """Parse LLM JSON response into a MergeResult."""
    text = response.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # If we can't parse, default to ADD (safe fallback)
        return MergeResult(decision=UpdateDecision.ADD)

    decision_str = data.get("decision", "add")
    try:
        decision = UpdateDecision(decision_str.lower())
    except ValueError:
        decision = UpdateDecision.ADD

    merged_content = data.get("merged_content")
    delete_ids = data.get("delete_ids", [])
    if not isinstance(delete_ids, list):
        delete_ids = []

    return MergeResult(
        decision=decision,
        merged_content=merged_content,
        delete_ids=[str(did) for did in delete_ids],
    )
