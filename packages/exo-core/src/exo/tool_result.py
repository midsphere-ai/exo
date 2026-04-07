"""Standardized tool result helpers for agent-facing tools.

Every tool result the LLM sees should be parseable and consistent.
Status messages use ``tool_ok`` / ``tool_error`` which return pre-serialized
JSON strings.  Raw content returns (file text, artifact content) should be
returned as plain strings — only wrap status/error signals with these helpers.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["tool_error", "tool_ok"]


def tool_ok(message: str, **extra: Any) -> str:
    """Return a structured success result as a JSON string.

    Args:
        message: Human-readable description of what happened.
        **extra: Additional key-value pairs to include in the result.
    """
    return json.dumps({"status": "ok", "message": message, **extra})


def tool_error(error: str, *, hint: str, **context: Any) -> str:
    """Return a structured error result as a JSON string.

    The ``hint`` is designed for the LLM — it should be actionable and
    reference specific tool names, parameter values, or alternative actions
    the agent should try next.

    Args:
        error: What went wrong.
        hint: Exactly what the LLM should do next to recover.
        **context: Additional metadata (available items, valid ranges, etc.)
            included under a ``"context"`` key so the LLM can self-correct.
    """
    payload: dict[str, Any] = {"status": "error", "error": error, "hint": hint}
    if context:
        payload["context"] = context
    return json.dumps(payload)
