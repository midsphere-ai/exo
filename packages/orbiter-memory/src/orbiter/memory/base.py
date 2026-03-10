"""Memory store protocol and typed memory item hierarchy.

Provides pluggable memory storage with scoped metadata and status lifecycle.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class MemoryError(Exception):
    """Base exception for memory operations."""


# ---------------------------------------------------------------------------
# Status lifecycle
# ---------------------------------------------------------------------------


class MemoryStatus(StrEnum):
    """Memory item status lifecycle: DRAFT -> ACCEPTED -> DISCARD."""

    DRAFT = "draft"
    ACCEPTED = "accepted"
    DISCARD = "discard"


class MemoryCategory(StrEnum):
    """Taxonomy for classifying memory knowledge types.

    Orthogonal to ``memory_type`` (which tracks conversation roles like
    human/ai/tool/system). Categories classify *what kind of knowledge*
    a memory item represents.
    """

    USER_PROFILE = "user_profile"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    VARIABLE = "variable"
    SUMMARY = "summary"
    CONVERSATION = "conversation"


_VALID_TRANSITIONS: dict[MemoryStatus, set[MemoryStatus]] = {
    MemoryStatus.DRAFT: {MemoryStatus.ACCEPTED, MemoryStatus.DISCARD},
    MemoryStatus.ACCEPTED: {MemoryStatus.DISCARD},
    MemoryStatus.DISCARD: set(),
}


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class MemoryMetadata(BaseModel):
    """Scoping metadata for memory items.

    All fields are optional — set the ones relevant to your use case.
    """

    model_config = {"frozen": True}

    user_id: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Memory item hierarchy
# ---------------------------------------------------------------------------


class MemoryItem(BaseModel):
    """Base class for all memory entries.

    Attributes:
        id: Unique identifier (auto-generated UUID).
        content: The stored content.
        memory_type: Discriminator for subtype dispatch.
        category: Optional knowledge taxonomy (orthogonal to memory_type).
        status: Current lifecycle status.
        metadata: Scoping information.
        created_at: ISO-8601 creation timestamp.
        updated_at: ISO-8601 last-update timestamp.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    content: str
    memory_type: str
    category: MemoryCategory | None = None
    status: MemoryStatus = MemoryStatus.ACCEPTED
    metadata: MemoryMetadata = Field(default_factory=MemoryMetadata)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def transition(self, new_status: MemoryStatus) -> None:
        """Transition to a new status.

        Raises:
            MemoryError: If the transition is invalid.
        """
        allowed = _VALID_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            msg = f"Cannot transition from {self.status!r} to {new_status!r}"
            raise MemoryError(msg)
        self.status = new_status
        self.updated_at = datetime.now(UTC).isoformat()


class SystemMemory(MemoryItem):
    """System/initialization memory (e.g. system prompts)."""

    memory_type: str = "system"


class HumanMemory(MemoryItem):
    """User/human message memory."""

    memory_type: str = "human"


class AIMemory(MemoryItem):
    """AI/assistant response memory."""

    memory_type: str = "ai"
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class ToolMemory(MemoryItem):
    """Tool execution result memory."""

    memory_type: str = "tool"
    tool_call_id: str = ""
    tool_name: str = ""
    is_error: bool = False


# ---------------------------------------------------------------------------
# MemoryStore protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class MemoryStore(Protocol):
    """Protocol for pluggable memory backends.

    Implementations must support add, get, search, and clear operations.
    All methods are async to support both in-memory and remote backends.
    """

    async def add(self, item: MemoryItem) -> None:
        """Persist a memory item."""
        ...

    async def get(self, item_id: str) -> MemoryItem | None:
        """Retrieve a memory item by ID. Returns None if not found."""
        ...

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
        """Search memory items with optional filters."""
        ...

    async def clear(
        self,
        *,
        metadata: MemoryMetadata | None = None,
    ) -> int:
        """Remove memory items matching the filter. Returns count removed."""
        ...
