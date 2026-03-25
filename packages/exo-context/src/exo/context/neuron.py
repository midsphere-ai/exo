"""Neuron — modular prompt composition components.

A Neuron is a composable unit that produces a prompt fragment from context.
Neurons are prioritised: lower priority numbers execute first and appear
earlier in the assembled prompt.

Core neurons:
- SystemNeuron    (priority 100) — date, time, platform info
- TaskNeuron      (priority 1)   — task ID, input, plan
- HistoryNeuron   (priority 10)  — conversation history with windowing

Extended neurons:
- TodoNeuron      (priority 2)   — todo/checklist items
- KnowledgeNeuron (priority 20)  — knowledge base snippets
- WorkspaceNeuron (priority 30)  — workspace artifact summaries
- SkillNeuron     (priority 40)  — available skill descriptions
- FactNeuron      (priority 50)  — extracted facts
- EntityNeuron    (priority 60)  — named entities
"""

from __future__ import annotations

import logging
import platform
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

from exo.context.context import Context  # pyright: ignore[reportMissingImports]
from exo.registry import Registry  # pyright: ignore[reportMissingImports]

# ── Registry ─────────────────────────────────────────────────────────

neuron_registry: Registry[Neuron] = Registry("neuron_registry")

# ── ABC ──────────────────────────────────────────────────────────────


class Neuron(ABC):
    """Abstract base for prompt neurons.

    Subclasses implement :meth:`format` to produce a prompt fragment
    from the given context.  The :attr:`priority` controls ordering
    when multiple neurons are composed: lower values appear first.

    Parameters
    ----------
    name:
        Human-readable name for registry and debugging.
    priority:
        Ordering priority (lower = earlier in prompt). Default 50.
    """

    __slots__ = ("_name", "_priority")

    def __init__(self, name: str, *, priority: int = 50) -> None:
        self._name = name
        self._priority = priority

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    @abstractmethod
    async def format(self, ctx: Context, **kwargs: Any) -> str:
        """Produce a prompt fragment from *ctx*.

        Returns an empty string to signal "nothing to contribute".
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self._name!r}, priority={self._priority})"


# ── Built-in neurons ────────────────────────────────────────────────


class SystemNeuron(Neuron):
    """Provides dynamic system variables: date, time, platform.

    Priority 100 (low) — appended near the end of system prompts.
    """

    def __init__(self, name: str = "system", *, priority: int = 100) -> None:
        super().__init__(name, priority=priority)

    async def format(self, ctx: Context, **kwargs: Any) -> str:
        now = datetime.now(tz=UTC)
        lines = [
            "<system_info>",
            f"Current date: {now.strftime('%Y-%m-%d')}",
            f"Current time: {now.strftime('%H:%M:%S UTC')}",
            f"Platform: {platform.system()} {platform.release()}",
            "</system_info>",
        ]
        return "\n".join(lines)


class TaskNeuron(Neuron):
    """Provides task context: task ID, input, output, subtask plan.

    Reads from ``ctx.state``:
    - ``task_input``  — the current task input text
    - ``task_output`` — any partial output so far
    - ``subtasks``    — list of subtask descriptions (plan)

    Priority 1 (high) — appears first in the prompt.
    """

    def __init__(self, name: str = "task", *, priority: int = 1) -> None:
        super().__init__(name, priority=priority)

    async def format(self, ctx: Context, **kwargs: Any) -> str:
        parts: list[str] = ["<task_info>", f"Task ID: {ctx.task_id}"]

        task_input = ctx.state.get("task_input")
        if task_input:
            parts.append(f"Input: {task_input}")

        task_output = ctx.state.get("task_output")
        if task_output:
            parts.append(f"Output: {task_output}")

        subtasks: list[str] | None = ctx.state.get("subtasks")
        if subtasks:
            parts.append("Plan:")
            for i, step in enumerate(subtasks, 1):
                parts.append(f"  <step{i}>{step}</step{i}>")

        parts.append("</task_info>")
        return "\n".join(parts)


class HistoryNeuron(Neuron):
    """Provides windowed conversation history.

    Reads ``history`` from ``ctx.state`` — expected to be a list of
    message dicts (``[{"role": ..., "content": ...}, ...]``).

    Uses ``ctx.config.history_rounds`` to limit the number of rounds
    included.

    Priority 10 — appears early in the prompt, after task info.
    """

    def __init__(self, name: str = "history", *, priority: int = 10) -> None:
        super().__init__(name, priority=priority)

    async def format(self, ctx: Context, **kwargs: Any) -> str:
        history: list[dict[str, Any]] | None = ctx.state.get("history")
        if not history:
            return ""

        # Window to last N rounds (each round = user + assistant = 2 messages)
        max_messages = ctx.config.history_rounds * 2
        windowed = history[-max_messages:]

        lines = ["<conversation_history>"]
        for msg in windowed:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"[{role}]: {content}")
        lines.append("</conversation_history>")
        return "\n".join(lines)


# ── Extended neurons ────────────────────────────────────────────────


class TodoNeuron(Neuron):
    """Provides todo/checklist items for task planning.

    Reads ``todos`` from ``ctx.state`` — expected to be a list of
    dicts with ``item`` (str) and optional ``done`` (bool) keys.

    Priority 2 — appears right after task info.
    """

    def __init__(self, name: str = "todo", *, priority: int = 2) -> None:
        super().__init__(name, priority=priority)

    async def format(self, ctx: Context, **kwargs: Any) -> str:
        todos: list[dict[str, Any]] | None = ctx.state.get("todos")
        if not todos:
            return ""
        lines = ["<todo_list>"]
        for item in todos:
            text = item.get("item", "")
            done = item.get("done", False)
            marker = "[x]" if done else "[ ]"
            lines.append(f"  {marker} {text}")
        lines.append("</todo_list>")
        return "\n".join(lines)


class KnowledgeNeuron(Neuron):
    """Provides knowledge base snippets relevant to the current task.

    Reads ``knowledge_items`` from ``ctx.state`` — expected to be a list
    of dicts with ``source`` (str) and ``content`` (str) keys.

    Priority 20 — appears after history, before workspace.
    """

    def __init__(self, name: str = "knowledge", *, priority: int = 20) -> None:
        super().__init__(name, priority=priority)

    async def format(self, ctx: Context, **kwargs: Any) -> str:
        items: list[dict[str, str]] | None = ctx.state.get("knowledge_items")
        if not items:
            return ""
        lines = ["<knowledge>"]
        for item in items:
            source = item.get("source", "unknown")
            content = item.get("content", "")
            lines.append(f"  [{source}]: {content}")
        lines.append("</knowledge>")
        return "\n".join(lines)


class WorkspaceNeuron(Neuron):
    """Provides workspace artifact summaries.

    Reads ``workspace_artifacts`` from ``ctx.state`` — expected to be
    a list of dicts with ``name`` (str) and optional ``type`` and
    ``size`` keys.

    Priority 30 — middle priority.
    """

    def __init__(self, name: str = "workspace", *, priority: int = 30) -> None:
        super().__init__(name, priority=priority)

    async def format(self, ctx: Context, **kwargs: Any) -> str:
        artifacts: list[dict[str, Any]] | None = ctx.state.get("workspace_artifacts")
        if not artifacts:
            return ""
        lines = ["<workspace>"]
        for art in artifacts:
            name = art.get("name", "unnamed")
            art_type = art.get("type", "file")
            size = art.get("size")
            size_info = f" ({size} bytes)" if size is not None else ""
            lines.append(f"  {name} [{art_type}]{size_info}")
        lines.append("</workspace>")
        return "\n".join(lines)


class SkillNeuron(Neuron):
    """Provides available skill descriptions.

    Reads ``skills`` from ``ctx.state`` — expected to be a list of
    dicts with ``name`` (str) and ``description`` (str) keys, plus
    optional ``active`` (bool).

    Priority 40 — mid-range.
    """

    def __init__(self, name: str = "skill", *, priority: int = 40) -> None:
        super().__init__(name, priority=priority)

    async def format(self, ctx: Context, **kwargs: Any) -> str:
        skills: list[dict[str, Any]] | None = ctx.state.get("skills")
        if not skills:
            return ""
        lines = ["<available_skills>"]
        for skill in skills:
            sname = skill.get("name", "unnamed")
            desc = skill.get("description", "")
            active = skill.get("active", True)
            status = "" if active else " [inactive]"
            lines.append(f"  - {sname}: {desc}{status}")
        lines.append("</available_skills>")
        return "\n".join(lines)


class FactNeuron(Neuron):
    """Provides extracted facts relevant to the current task.

    Reads ``facts`` from ``ctx.state`` — expected to be a list of
    strings representing established facts.

    Priority 50 — lower priority.
    """

    def __init__(self, name: str = "fact", *, priority: int = 50) -> None:
        super().__init__(name, priority=priority)

    async def format(self, ctx: Context, **kwargs: Any) -> str:
        facts: list[str] | None = ctx.state.get("facts")
        if not facts:
            return ""
        lines = ["<facts>"]
        for fact in facts:
            lines.append(f"  - {fact}")
        lines.append("</facts>")
        return "\n".join(lines)


class EntityNeuron(Neuron):
    """Provides named entities extracted from conversations.

    Reads ``entities`` from ``ctx.state`` — expected to be a list
    of dicts with ``name`` (str) and ``type`` (str) keys.

    Priority 60 — low priority, supplementary context.
    """

    def __init__(self, name: str = "entity", *, priority: int = 60) -> None:
        super().__init__(name, priority=priority)

    async def format(self, ctx: Context, **kwargs: Any) -> str:
        entities: list[dict[str, str]] | None = ctx.state.get("entities")
        if not entities:
            return ""
        lines = ["<entities>"]
        for ent in entities:
            ename = ent.get("name", "unknown")
            etype = ent.get("type", "unknown")
            lines.append(f"  {ename} ({etype})")
        lines.append("</entities>")
        return "\n".join(lines)


# ── Register built-ins ──────────────────────────────────────────────

neuron_registry.register("system", SystemNeuron())
neuron_registry.register("task", TaskNeuron())
neuron_registry.register("history", HistoryNeuron())
neuron_registry.register("todo", TodoNeuron())
neuron_registry.register("knowledge", KnowledgeNeuron())
neuron_registry.register("workspace", WorkspaceNeuron())
neuron_registry.register("skill", SkillNeuron())
neuron_registry.register("fact", FactNeuron())
neuron_registry.register("entity", EntityNeuron())
