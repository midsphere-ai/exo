"""ContextProcessor — event-driven context processing pipeline.

Processors intervene at specific points in the LLM execution cycle to
dynamically transform context.  Each processor declares which event type
it handles (e.g., ``"pre_llm_call"``, ``"post_tool_call"``).

The :class:`ProcessorPipeline` collects processors and fires them by event.
Processors are called sequentially in registration order for a given event.

Built-in processors:
- SummarizeProcessor  — ``pre_llm_call``: marks context for summarization
  when history exceeds the configured threshold.
- ToolResultOffloader — ``post_tool_call``: offloads large tool results to
  workspace when they exceed a size threshold.
- MessageOffloader    — ``pre_llm_call``: replaces oversized messages with
  ``[[OFFLOAD: handle=<id>]]`` markers to keep context within budget.
"""

from __future__ import annotations

import contextlib
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from orbiter.context.context import Context  # pyright: ignore[reportMissingImports]


class ProcessorError(Exception):
    """Raised for processor pipeline errors."""


# ── ABC ──────────────────────────────────────────────────────────────


class ContextProcessor(ABC):
    """Abstract base for context processors.

    Each processor handles a single event type (e.g., ``"pre_llm_call"``).
    The :meth:`process` method receives the context and an arbitrary payload
    dict containing event-specific data.

    Parameters
    ----------
    event:
        The event type this processor handles.
    name:
        Human-readable name for debugging.  Defaults to the class name.
    """

    __slots__ = ("_event", "_name")

    def __init__(self, event: str, *, name: str | None = None) -> None:
        if not event:
            msg = "event must be a non-empty string"
            raise ProcessorError(msg)
        self._event = event
        self._name = name or type(self).__name__

    @property
    def event(self) -> str:
        """The event type this processor handles."""
        return self._event

    @property
    def name(self) -> str:
        """Human-readable processor name."""
        return self._name

    @abstractmethod
    async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
        """Process the event with context and payload.

        Implementations may mutate ``ctx.state`` to transform context.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(event={self._event!r}, name={self._name!r})"


# ── Pipeline ─────────────────────────────────────────────────────────


class ProcessorPipeline:
    """Registers and fires context processors by event type.

    Processors are called sequentially in registration order for each
    event.  Errors from processors propagate immediately.

    Usage::

        pipeline = ProcessorPipeline()
        pipeline.register(SummarizeProcessor())
        pipeline.register(ToolResultOffloader(max_size=5000))
        await pipeline.fire("pre_llm_call", ctx, {"messages": [...]})
    """

    __slots__ = ("_processors",)

    def __init__(self) -> None:
        self._processors: defaultdict[str, list[ContextProcessor]] = defaultdict(list)

    def register(self, processor: ContextProcessor) -> ProcessorPipeline:
        """Register a processor for its declared event type.

        Returns ``self`` for method chaining.
        """
        self._processors[processor.event].append(processor)
        return self

    def unregister(self, processor: ContextProcessor) -> None:
        """Remove a processor.  Silently does nothing if not registered."""
        procs = self._processors.get(processor.event)
        if procs is not None:
            with contextlib.suppress(ValueError):
                procs.remove(processor)

    async def fire(self, event: str, ctx: Context, payload: dict[str, Any] | None = None) -> None:
        """Fire all processors registered for *event* in order.

        Parameters
        ----------
        event:
            The event type to fire.
        ctx:
            The context passed to each processor.
        payload:
            Optional event-specific data dict.  Defaults to ``{}``.
        """
        data = payload if payload is not None else {}
        for proc in self._processors.get(event, []):
            await proc.process(ctx, data)

    def has_processors(self, event: str) -> bool:
        """Check whether any processors are registered for *event*."""
        return len(self._processors.get(event, [])) > 0

    def list_processors(self, event: str | None = None) -> list[ContextProcessor]:
        """List processors, optionally filtered by event."""
        if event is not None:
            return list(self._processors.get(event, []))
        result: list[ContextProcessor] = []
        for procs in self._processors.values():
            result.extend(procs)
        return result

    def clear(self) -> None:
        """Remove all processors."""
        self._processors.clear()

    def __len__(self) -> int:
        """Total number of registered processors across all events."""
        return sum(len(procs) for procs in self._processors.values())

    def __repr__(self) -> str:
        events = list(self._processors.keys())
        total = len(self)
        return f"ProcessorPipeline(events={events}, total={total})"


# ── Built-in processors ─────────────────────────────────────────────


class SummarizeProcessor(ContextProcessor):
    """Marks context for summarization when history exceeds a threshold.

    Fires on ``"pre_llm_call"``.  Checks the ``history`` list in
    ``ctx.state`` against ``ctx.config.summary_threshold``.  When
    exceeded, sets ``needs_summary=True`` in state and stores the
    excess messages under ``summary_candidates``.

    Parameters
    ----------
    name:
        Processor name.  Default ``"summarize"``.
    """

    def __init__(self, *, name: str = "summarize") -> None:
        super().__init__("pre_llm_call", name=name)

    async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
        history: list[dict[str, Any]] | None = ctx.state.get("history")
        if not history:
            return

        threshold = ctx.config.summary_threshold
        if len(history) <= threshold:
            return

        # Mark for summarization and store candidates
        ctx.state.set("needs_summary", True)
        # Candidates are the oldest messages beyond the threshold
        excess_count = len(history) - threshold
        ctx.state.set("summary_candidates", history[:excess_count])


class ToolResultOffloader(ContextProcessor):
    """Offloads large tool results to workspace.

    Fires on ``"post_tool_call"``.  When a tool result's content exceeds
    ``max_size`` characters, replaces it with a reference placeholder
    and stores the full content under ``offloaded_results`` in state.

    Parameters
    ----------
    max_size:
        Maximum character length before offloading.  Default 5000.
    name:
        Processor name.  Default ``"tool_result_offloader"``.
    """

    __slots__ = ("_event", "_max_size", "_name")

    def __init__(self, *, max_size: int = 5000, name: str = "tool_result_offloader") -> None:
        super().__init__("post_tool_call", name=name)
        self._max_size = max_size

    @property
    def max_size(self) -> int:
        """Maximum content size before offloading."""
        return self._max_size

    async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
        tool_result = payload.get("tool_result")
        if tool_result is None:
            return

        content = str(tool_result)
        if len(content) <= self._max_size:
            return

        # Store full content in offloaded results
        offloaded: list[dict[str, Any]] = ctx.state.get("offloaded_results") or []
        tool_name = payload.get("tool_name", "unknown")
        tool_call_id = payload.get("tool_call_id", "unknown")

        offloaded.append(
            {
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "content": content,
                "size": len(content),
            }
        )
        ctx.state.set("offloaded_results", offloaded)

        # Replace tool result content with a reference
        truncated = content[: self._max_size // 2]
        reference = (
            f"{truncated}...\n\n"
            f"[Result truncated — full content offloaded to workspace "
            f"({len(content)} chars)]"
        )
        payload["tool_result"] = reference


class MessageOffloader(ContextProcessor):
    """Replaces oversized messages with ``[[OFFLOAD: handle=<id>]]`` markers.

    Fires on ``"pre_llm_call"``.  Scans ``ctx.state['history']`` and replaces
    any user, assistant, or tool message whose content exceeds
    ``max_message_size`` characters with a short marker.  The original content
    is stored in ``ctx.state['offloaded_messages']`` keyed by handle ID so it
    can be recovered later (e.g. via a reload tool).

    System messages are never offloaded.

    Parameters
    ----------
    max_message_size:
        Maximum character length per message before offloading.  Default 10000.
    name:
        Processor name.  Default ``"message_offloader"``.
    """

    __slots__ = ("_event", "_max_message_size", "_name")

    def __init__(
        self, *, max_message_size: int = 10000, name: str = "message_offloader"
    ) -> None:
        super().__init__("pre_llm_call", name=name)
        self._max_message_size = max_message_size

    @property
    def max_message_size(self) -> int:
        """Maximum message content size before offloading."""
        return self._max_message_size

    async def process(self, ctx: Context, payload: dict[str, Any]) -> None:
        history: list[dict[str, Any]] | None = ctx.state.get("history")
        if not history:
            return

        offloaded: dict[str, str] = ctx.state.get("offloaded_messages") or {}

        for msg in history:
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role == "system":
                continue

            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if content is None:
                continue
            content_str = str(content)
            if len(content_str) <= self._max_message_size:
                continue

            handle_id = uuid.uuid4().hex[:12]
            offloaded[handle_id] = content_str
            marker = f"[[OFFLOAD: handle={handle_id}]]"
            if isinstance(msg, dict):
                msg["content"] = marker
            else:
                msg.content = marker  # type: ignore[attr-defined]

        ctx.state.set("offloaded_messages", offloaded)
