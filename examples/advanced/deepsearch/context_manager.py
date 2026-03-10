"""Context management for DeepSearch using Orbiter's context system.

Integrates Orbiter's Context, ProcessorPipeline, and built-in processors
to manage the two unbounded growth vectors in the research loop:
1. Knowledge messages (all_knowledge → compose_msgs)
2. Diary context (diary_context → system prompt)
"""

from __future__ import annotations

import logging
from typing import Any

from orbiter.context.context import Context  # pyright: ignore[reportMissingImports]
from orbiter.context.config import ContextConfig, make_config  # pyright: ignore[reportMissingImports]
from orbiter.context.processor import (  # pyright: ignore[reportMissingImports]
    ContextProcessor,
    ProcessorPipeline,
    MessageOffloader,
    RoundWindowProcessor,
)

from .types import KnowledgeItem

logger = logging.getLogger("deepsearch")


# ---------------------------------------------------------------------------
# Custom processors for DeepSearch-specific patterns
# ---------------------------------------------------------------------------


class KnowledgeWindowProcessor(ContextProcessor):
    """Window knowledge items to the most recent N entries.

    In DeepSearch, knowledge grows with every search/visit/reflect step.
    Older knowledge items are less relevant as the research narrows.
    This processor keeps only the most recent ``max_items`` knowledge entries
    while always preserving items tagged as ``type="qa"`` (direct Q&A from
    the LLM evaluator, which contain critical reasoning).

    Fires on ``"pre_llm_call"``.
    """

    __slots__ = ("_event", "_max_items", "_name")

    def __init__(self, *, max_items: int = 30, name: str = "knowledge_window") -> None:
        super().__init__("pre_llm_call", name=name)
        self._max_items = max_items

    @property
    def max_items(self) -> int:
        return self._max_items

    async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
        knowledge: list[dict[str, Any]] | None = ctx.state.get("knowledge")
        if not knowledge or len(knowledge) <= self._max_items:
            return

        # Partition: QA items are always kept, others are windowed
        qa_items = [k for k in knowledge if k.get("type") == "qa"]
        other_items = [k for k in knowledge if k.get("type") != "qa"]

        # Keep the most recent non-QA items that fit within budget
        budget = max(0, self._max_items - len(qa_items))
        kept_other = other_items[-budget:] if budget > 0 else []
        trimmed = len(other_items) - len(kept_other)

        result = qa_items + kept_other
        ctx.state.set("knowledge", result)

        if trimmed > 0:
            logger.debug(
                "KnowledgeWindow: trimmed %d old knowledge items (kept %d QA + %d recent)",
                trimmed, len(qa_items), len(kept_other),
            )


class DiaryCompressor(ContextProcessor):
    """Compress old diary entries into a summary when diary exceeds a threshold.

    The diary_context list grows every step. This processor keeps the last
    ``keep_recent`` entries verbatim and compresses older entries into a
    single summary line.

    Fires on ``"pre_llm_call"``.
    """

    __slots__ = ("_event", "_max_entries", "_keep_recent", "_name")

    def __init__(
        self,
        *,
        max_entries: int = 10,
        keep_recent: int = 5,
        name: str = "diary_compressor",
    ) -> None:
        super().__init__("pre_llm_call", name=name)
        self._max_entries = max_entries
        self._keep_recent = keep_recent

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def keep_recent(self) -> int:
        return self._keep_recent

    async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
        diary: list[str] | None = ctx.state.get("diary")
        if not diary or len(diary) <= self._max_entries:
            return

        # Keep the most recent entries verbatim
        old_entries = diary[: -self._keep_recent]
        recent_entries = diary[-self._keep_recent :]

        # Compress old entries into a summary
        actions = []
        for entry in old_entries:
            # Extract the key action from each diary entry
            first_line = entry.split("\n")[0].strip()
            if first_line:
                actions.append(first_line)

        summary = (
            f"[Earlier steps summary ({len(old_entries)} steps compressed): "
            + "; ".join(actions)
            + "]"
        )

        result = [summary, *recent_entries]
        ctx.state.set("diary", result)

        logger.debug(
            "DiaryCompressor: compressed %d old entries into summary, kept %d recent",
            len(old_entries), len(recent_entries),
        )


class KnowledgeContentTrimmer(ContextProcessor):
    """Trim individual knowledge item answers that are too long.

    URL content knowledge items can contain up to 2000 chars of page content.
    When context is getting large, this processor trims them to a shorter
    limit to free up tokens.

    Fires on ``"pre_llm_call"``.
    """

    __slots__ = ("_event", "_max_answer_len", "_name")

    def __init__(
        self,
        *,
        max_answer_len: int = 500,
        name: str = "knowledge_trimmer",
    ) -> None:
        super().__init__("pre_llm_call", name=name)
        self._max_answer_len = max_answer_len

    @property
    def max_answer_len(self) -> int:
        return self._max_answer_len

    async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
        knowledge: list[dict[str, Any]] | None = ctx.state.get("knowledge")
        if not knowledge:
            return

        total_tokens = ctx.state.get("total_tokens", 0)
        budget = ctx.state.get("token_budget", 1_000_000)
        usage_pct = total_tokens / budget if budget else 0

        # Only trim when we're using >50% of budget
        if usage_pct < 0.5:
            return

        trimmed_count = 0
        for k in knowledge:
            answer = k.get("answer", "")
            if k.get("type") in ("url", "side-info") and len(answer) > self._max_answer_len:
                k["answer"] = answer[: self._max_answer_len] + "..."
                trimmed_count += 1

        if trimmed_count:
            ctx.state.set("knowledge", knowledge)
            logger.debug(
                "KnowledgeTrimmer: trimmed %d long answers (budget at %.0f%%)",
                trimmed_count, usage_pct * 100,
            )


# ---------------------------------------------------------------------------
# DeepSearch Context Manager
# ---------------------------------------------------------------------------


class DeepSearchContextManager:
    """Manages context for the DeepSearch research loop.

    Wraps Orbiter's Context and ProcessorPipeline to provide bounded
    context growth during research.

    Args:
        token_budget: Total token budget for the research session.
        max_knowledge_items: Maximum knowledge items to keep in context.
        max_diary_entries: Maximum diary entries before compression.
        keep_recent_diary: Number of recent diary entries to keep verbatim.
        max_knowledge_answer_len: Max chars per knowledge answer when trimming.
    """

    def __init__(
        self,
        *,
        token_budget: int = 1_000_000,
        max_knowledge_items: int = 30,
        max_diary_entries: int = 10,
        keep_recent_diary: int = 5,
        max_knowledge_answer_len: int = 500,
    ) -> None:
        config = make_config(
            "copilot",
            history_rounds=20,
            summary_threshold=max_diary_entries,
        )
        self._ctx = Context("deepsearch", config=config)
        self._pipeline = ProcessorPipeline()

        # Register processors in execution order
        self._pipeline.register(
            KnowledgeWindowProcessor(max_items=max_knowledge_items)
        )
        self._pipeline.register(
            DiaryCompressor(
                max_entries=max_diary_entries,
                keep_recent=keep_recent_diary,
            )
        )
        self._pipeline.register(
            KnowledgeContentTrimmer(max_answer_len=max_knowledge_answer_len)
        )

        # Initialize state
        self._ctx.state.set("knowledge", [])
        self._ctx.state.set("diary", [])
        self._ctx.state.set("token_budget", token_budget)
        self._ctx.state.set("total_tokens", 0)

    @property
    def ctx(self) -> Context:
        """The underlying Orbiter Context."""
        return self._ctx

    @property
    def pipeline(self) -> ProcessorPipeline:
        """The processor pipeline."""
        return self._pipeline

    def update_token_usage(self, total_tokens: int) -> None:
        """Update the current token usage for budget-aware processors."""
        self._ctx.state.set("total_tokens", total_tokens)

    def add_knowledge(self, items: list[KnowledgeItem]) -> None:
        """Add new knowledge items to the context state."""
        current: list[dict[str, Any]] = self._ctx.state.get("knowledge") or []
        for item in items:
            current.append(item.model_dump())
        self._ctx.state.set("knowledge", current)

    def add_diary_entry(self, entry: str) -> None:
        """Add a diary entry to the context state."""
        diary: list[str] = self._ctx.state.get("diary") or []
        diary.append(entry)
        self._ctx.state.set("diary", diary)

    def reset_diary(self) -> None:
        """Clear the diary (e.g., after a failed answer resets the step count)."""
        self._ctx.state.set("diary", [])

    def get_knowledge(self) -> list[KnowledgeItem]:
        """Get the current (possibly windowed) knowledge items."""
        raw: list[dict[str, Any]] = self._ctx.state.get("knowledge") or []
        return [KnowledgeItem.model_validate(k) for k in raw]

    def get_diary(self) -> list[str]:
        """Get the current (possibly compressed) diary entries."""
        return list(self._ctx.state.get("diary") or [])

    async def pre_llm_call(self) -> None:
        """Fire all pre_llm_call processors to manage context size.

        Call this before each LLM call in the research loop.
        """
        await self._pipeline.fire("pre_llm_call", self._ctx)

    def snapshot(self, step: int) -> None:
        """Create a checkpoint of the current context state."""
        self._ctx.snapshot(metadata={"step": step})
