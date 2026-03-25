"""Context state inspector API endpoints.

Provides endpoints for inspecting hierarchical context state during agent
execution.  Context state mirrors Exo's ContextState parent-child
hierarchy and tracks fork/merge events with token deltas.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from exo_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/v1/context-state", tags=["context-state"])


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


def _empty_tree(task_id: str = "root") -> dict[str, Any]:
    """Return a minimal empty context state tree."""
    return {
        "task_id": task_id,
        "parent_task_id": None,
        "local_entries": {},
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "token_delta": None,
        "fork_event": None,
        "merge_event": None,
        "children": [],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/conversation/{conversation_id}")
async def get_context_state(
    conversation_id: str,
    step: int | None = None,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return the context state tree for a conversation.

    The ``step`` query param selects a specific step snapshot.  When omitted
    the latest accumulated state is returned.

    Once agent runtime integration is complete this will return live state
    from the running Context object.  For now it returns a placeholder tree
    so the frontend inspector can be exercised.
    """
    return {
        "conversation_id": conversation_id,
        "step": step,
        "tree": _empty_tree(),
    }
