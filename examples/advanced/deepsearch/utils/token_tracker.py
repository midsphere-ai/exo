"""Token usage tracking across the research process."""
from __future__ import annotations
import logging

logger = logging.getLogger("deepsearch")


class TokenTracker:
    def __init__(self, budget: int = 1_000_000) -> None:
        self.budget = budget
        self._usages: list[dict] = []

    def track_usage(self, tool: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        total = prompt_tokens + completion_tokens
        self._usages.append({
            "tool": tool,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
        })

    @property
    def total_tokens(self) -> int:
        return sum(u["total_tokens"] for u in self._usages)

    @property
    def prompt_tokens(self) -> int:
        return sum(u["prompt_tokens"] for u in self._usages)

    @property
    def completion_tokens(self) -> int:
        return sum(u["completion_tokens"] for u in self._usages)

    def get_breakdown(self) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        for u in self._usages:
            breakdown[u["tool"]] = breakdown.get(u["tool"], 0) + u["total_tokens"]
        return breakdown

    def print_summary(self) -> None:
        logger.info(
            "Token Usage: %d/%d (%.1f%%) | Breakdown: %s",
            self.total_tokens, self.budget,
            self.total_tokens / self.budget * 100 if self.budget else 0,
            self.get_breakdown(),
        )
