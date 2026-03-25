"""Structured LLM execution logging with token breakdown and context analysis."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]

logger = get_logger("prompt")

# ---------------------------------------------------------------------------
# Role-based token counting
# ---------------------------------------------------------------------------

# Default character-to-token ratio (approximate).
DEFAULT_CHAR_TOKEN_RATIO: float = 4.0


def estimate_tokens(text: str, ratio: float = DEFAULT_CHAR_TOKEN_RATIO) -> int:
    """Estimate token count from character length.

    Uses a simple chars/ratio heuristic. Callers can plug in a precise
    tokeniser by overriding *ratio* or replacing this function.
    """
    if not text:
        return 0
    return max(1, int(len(text) / ratio))


# ---------------------------------------------------------------------------
# Token breakdown
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TokenBreakdown:
    """Per-role token counts and context window analysis."""

    system: int = 0
    user: int = 0
    assistant: int = 0
    tool: int = 0
    other: int = 0

    @property
    def total(self) -> int:
        return self.system + self.user + self.assistant + self.tool + self.other

    def percentages(self, context_window: int) -> dict[str, float]:
        """Return role->percentage mapping relative to *context_window*."""
        if context_window <= 0:
            return {r: 0.0 for r in ("system", "user", "assistant", "tool", "other", "free")}
        total = self.total
        free = max(0, context_window - total)
        return {
            "system": self.system / context_window * 100,
            "user": self.user / context_window * 100,
            "assistant": self.assistant / context_window * 100,
            "tool": self.tool / context_window * 100,
            "other": self.other / context_window * 100,
            "free": free / context_window * 100,
        }


def compute_token_breakdown(
    messages: Sequence[dict[str, Any]],
    *,
    ratio: float = DEFAULT_CHAR_TOKEN_RATIO,
) -> TokenBreakdown:
    """Compute a ``TokenBreakdown`` from a list of message dicts.

    Each message is expected to have ``role`` and ``content`` keys.
    Multi-modal content (a ``list`` of content items) is handled by
    summing the text items and counting images/tool_use as a fixed
    estimate.
    """
    counts: dict[str, int] = {"system": 0, "user": 0, "assistant": 0, "tool": 0, "other": 0}

    for msg in messages:
        role = msg.get("role", "other")
        bucket = role if role in counts else "other"
        content = msg.get("content", "")
        tokens = _content_tokens(content, ratio=ratio)

        # Include tool_calls serialisation cost for assistant messages.
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            tokens += _tool_calls_tokens(tool_calls, ratio=ratio)

        counts[bucket] += tokens

    return TokenBreakdown(**counts)


def _content_tokens(content: Any, *, ratio: float) -> int:
    """Estimate tokens for a single ``content`` field.

    Handles plain strings and multi-modal content lists.
    """
    if content is None:
        return 0
    if isinstance(content, str):
        return estimate_tokens(content, ratio=ratio)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "text":
                    total += estimate_tokens(str(item.get("text", "")), ratio=ratio)
                elif item_type == "image_url":
                    # Fixed cost estimate for images (base64 payloads vary).
                    total += 85
                elif item_type == "tool_use":
                    total += estimate_tokens(str(item.get("tool_use", "")), ratio=ratio)
                else:
                    total += estimate_tokens(str(item), ratio=ratio)
            else:
                total += estimate_tokens(str(item), ratio=ratio)
        return total
    return estimate_tokens(str(content), ratio=ratio)


def _tool_calls_tokens(tool_calls: Any, *, ratio: float) -> int:
    """Estimate token cost for tool_calls."""
    total = 0
    if not isinstance(tool_calls, list):
        return estimate_tokens(str(tool_calls), ratio=ratio)
    for tc in tool_calls:
        if isinstance(tc, dict):
            fn = tc.get("function", {})
            total += estimate_tokens(str(fn.get("name", "")), ratio=ratio)
            total += estimate_tokens(str(fn.get("arguments", "")), ratio=ratio)
        else:
            total += estimate_tokens(str(tc), ratio=ratio)
    return total


# ---------------------------------------------------------------------------
# Execution log entry
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExecutionLogEntry:
    """Structured record of a single LLM execution."""

    agent_name: str = ""
    model_name: str = ""
    message_count: int = 0
    tool_names: list[str] = field(default_factory=list)
    breakdown: TokenBreakdown = field(default_factory=TokenBreakdown)
    context_window: int = 0
    duration_s: float = 0.0

    def format_summary(self) -> str:
        """Return a human-readable multi-line summary string."""
        lines: list[str] = []
        lines.append(f"Agent: {self.agent_name}  Model: {self.model_name}")
        lines.append(f"Messages: {self.message_count}  Duration: {self.duration_s:.3f}s")

        bd = self.breakdown
        if self.context_window > 0:
            pcts = bd.percentages(self.context_window)
            used_k = bd.total / 1024
            limit_k = self.context_window / 1024
            lines.append(
                f"Tokens: {used_k:.1f}k / {limit_k:.1f}k ({bd.total / self.context_window * 100:.1f}%)"
            )
            lines.append(
                f"  system={bd.system} ({pcts['system']:.1f}%)  "
                f"user={bd.user} ({pcts['user']:.1f}%)  "
                f"assistant={bd.assistant} ({pcts['assistant']:.1f}%)"
            )
            lines.append(
                f"  tool={bd.tool} ({pcts['tool']:.1f}%)  "
                f"other={bd.other} ({pcts['other']:.1f}%)  "
                f"free={max(0, self.context_window - bd.total)} ({pcts['free']:.1f}%)"
            )
        else:
            lines.append(
                f"Tokens: {bd.total} (system={bd.system} user={bd.user} assistant={bd.assistant} tool={bd.tool})"
            )

        if self.tool_names:
            lines.append(f"Tools: {', '.join(self.tool_names)}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PromptLogger — structured logger
# ---------------------------------------------------------------------------


class PromptLogger:
    """Structured LLM execution logger.

    Accepts message dicts (OpenAI-style), computes token breakdown,
    and writes structured log entries.
    """

    __slots__ = ("_logger", "_ratio")

    def __init__(
        self,
        *,
        log: logging.Logger | None = None,
        ratio: float = DEFAULT_CHAR_TOKEN_RATIO,
    ) -> None:
        self._logger = log or logger
        self._ratio = ratio

    def log_execution(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        agent_name: str = "",
        model_name: str = "",
        context_window: int = 0,
        tool_names: Sequence[str] | None = None,
        duration_s: float = 0.0,
        level: int = logging.INFO,
    ) -> ExecutionLogEntry:
        """Compute token breakdown and log a structured execution entry.

        Returns the ``ExecutionLogEntry`` for programmatic access.
        """
        breakdown = compute_token_breakdown(messages, ratio=self._ratio)
        entry = ExecutionLogEntry(
            agent_name=agent_name,
            model_name=model_name,
            message_count=len(messages),
            tool_names=list(tool_names) if tool_names else [],
            breakdown=breakdown,
            context_window=context_window,
            duration_s=duration_s,
        )
        self._logger.log(level, "LLM Execution\n%s", entry.format_summary())
        return entry
