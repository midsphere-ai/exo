"""Context tools — planning, knowledge, and file tools for agent self-management.

Provides ``@tool``-decorated functions that let agents manipulate their own
context during execution: manage a todo checklist, search knowledge artifacts,
and read files from a working directory.

All tools operate on a :class:`Context` instance passed via a ``ctx`` keyword
argument at call time.  The ``ctx`` parameter is excluded from the generated
JSON schema so that LLMs only see the user-facing parameters.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from exo.tool import Tool, ToolError, _extract_description, _generate_schema

logger = logging.getLogger(__name__)

# ── Helper: build a tool that receives ctx at call time ─────────────


class _ContextTool(Tool):
    """A tool whose ``execute`` injects a ``ctx`` from bound state.

    The ``ctx`` parameter is NOT part of the JSON schema — it is injected
    by the caller (runner / agent) at execution time.
    """

    __slots__ = ("_ctx", "_fn", "_is_context_tool", "description", "name", "parameters")

    def __init__(
        self,
        fn: Any,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self._fn = fn
        self.name = name or fn.__name__
        self.description = description or _extract_description(fn)
        # Generate schema but strip out 'ctx' — it's injected, not user-visible
        schema = _generate_schema(fn)
        props = schema.get("properties", {})
        props.pop("ctx", None)
        req = schema.get("required", [])
        schema["required"] = [r for r in req if r != "ctx"]
        if not schema["required"]:
            del schema["required"]
        schema["properties"] = props
        self.parameters = schema
        self._ctx: Any = None
        self._is_context_tool: bool = True

    def bind(self, ctx: Any) -> _ContextTool:
        """Bind a :class:`Context` instance for subsequent calls."""
        self._ctx = ctx
        return self

    async def execute(self, **kwargs: Any) -> str | dict[str, Any]:
        if self._ctx is None:
            logger.error("tool %r executed without bound context", self.name)
            raise ToolError(f"Tool '{self.name}' requires a bound context (call .bind(ctx) first)")
        logger.debug("executing context tool %r", self.name)
        return await self._fn(ctx=self._ctx, **kwargs)


# ── Planning tool ───────────────────────────────────────────────────


async def _add_todo(ctx: Any, item: str) -> str:
    """Add a todo item to the planning checklist.

    Args:
        item: The todo item text to add.
    """
    todos: list[dict[str, Any]] = ctx.state.get("todos", [])
    todos = list(todos)  # defensive copy
    todos.append({"item": item, "done": False})
    ctx.state.set("todos", todos)
    return f"Added todo: {item}"


async def _complete_todo(ctx: Any, index: int) -> str:
    """Mark a todo item as completed by its index (0-based).

    Args:
        index: The 0-based index of the todo item to mark done.
    """
    todos: list[dict[str, Any]] = ctx.state.get("todos", [])
    if not todos:
        return "No todos found."
    todos = [dict(t) for t in todos]  # defensive copy
    if index < 0 or index >= len(todos):
        return f"Invalid index {index}. Have {len(todos)} todos."
    todos[index]["done"] = True
    ctx.state.set("todos", todos)
    return f"Marked todo #{index} as done: {todos[index]['item']}"


async def _get_todo(ctx: Any) -> str:
    """Retrieve the current todo checklist.

    Returns the full checklist in markdown format.
    """
    todos: list[dict[str, Any]] = ctx.state.get("todos", [])
    if not todos:
        return "No todos."
    lines: list[str] = []
    for i, t in enumerate(todos):
        mark = "x" if t.get("done") else " "
        lines.append(f"{i}. [{mark}] {t.get('item', '')}")
    return "\n".join(lines)


planning_tool_add = _ContextTool(
    _add_todo, name="add_todo", description="Add a todo item to the planning checklist."
)
planning_tool_complete = _ContextTool(
    _complete_todo, name="complete_todo", description="Mark a todo item as completed by index."
)
planning_tool_get = _ContextTool(
    _get_todo, name="get_todo", description="Retrieve the current todo checklist."
)


def get_planning_tools() -> list[Tool]:
    """Return fresh planning tool instances (safe for per-agent binding)."""
    return [
        _ContextTool(
            _add_todo, name="add_todo", description="Add a todo item to the planning checklist."
        ),
        _ContextTool(
            _complete_todo,
            name="complete_todo",
            description="Mark a todo item as completed by index.",
        ),
        _ContextTool(
            _get_todo, name="get_todo", description="Retrieve the current todo checklist."
        ),
    ]


# ── Knowledge tool ──────────────────────────────────────────────────


async def _get_knowledge(ctx: Any, name: str) -> str:
    """Retrieve a knowledge artifact by name.

    Args:
        name: The artifact name to retrieve.
    """
    workspace = ctx.state.get("workspace")
    if workspace is None:
        return "No workspace attached to context."
    content = workspace.read(name)
    if content is None:
        return f"Artifact '{name}' not found."
    return content


async def _grep_knowledge(ctx: Any, name: str, pattern: str) -> str:
    """Search a knowledge artifact for lines matching a regex pattern.

    Args:
        name: The artifact name to search in.
        pattern: A regex pattern to match against each line.
    """
    workspace = ctx.state.get("workspace")
    if workspace is None:
        return "No workspace attached to context."
    content = workspace.read(name)
    if content is None:
        return f"Artifact '{name}' not found."
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex pattern: {e}"
    lines = content.splitlines()
    matches: list[str] = []
    for i, line in enumerate(lines):
        if compiled.search(line):
            matches.append(f"{i + 1}: {line}")
    if not matches:
        return f"No matches for pattern '{pattern}' in '{name}'."
    return "\n".join(matches)


async def _search_knowledge(ctx: Any, query: str, top_k: int = 5) -> str:
    """Search across all knowledge artifacts using keyword matching.

    Args:
        query: The search query string.
        top_k: Maximum number of results to return.
    """
    knowledge_store = ctx.state.get("knowledge_store")
    if knowledge_store is None:
        return "No knowledge store attached to context."
    results = knowledge_store.search(query, top_k=top_k)
    if not results:
        return f"No results for query '{query}'."
    lines: list[str] = []
    for r in results:
        lines.append(
            f"[{r.chunk.artifact_name}#{r.chunk.index}] (score={r.score:.2f})\n{r.chunk.content[:200]}"
        )
    return "\n---\n".join(lines)


knowledge_tool_get = _ContextTool(
    _get_knowledge, name="get_knowledge", description="Retrieve a knowledge artifact by name."
)
knowledge_tool_grep = _ContextTool(
    _grep_knowledge,
    name="grep_knowledge",
    description="Search a knowledge artifact for regex matches.",
)
knowledge_tool_search = _ContextTool(
    _search_knowledge, name="search_knowledge", description="Search across all knowledge artifacts."
)


def get_knowledge_tools() -> list[Tool]:
    """Return fresh knowledge tool instances (safe for per-agent binding)."""
    return [
        _ContextTool(
            _get_knowledge,
            name="get_knowledge",
            description="Retrieve a knowledge artifact by name.",
        ),
        _ContextTool(
            _grep_knowledge,
            name="grep_knowledge",
            description="Search a knowledge artifact for regex matches.",
        ),
        _ContextTool(
            _search_knowledge,
            name="search_knowledge",
            description="Search across all knowledge artifacts.",
        ),
    ]


# ── File tool ───────────────────────────────────────────────────────


async def _read_file(ctx: Any, path: str) -> str:
    """Read a file from the working directory.

    Args:
        path: Relative path to the file within the working directory.
    """
    working_dir = ctx.state.get("working_dir")
    if working_dir is None:
        return "No working directory set in context."
    base = Path(working_dir).resolve()
    target = (base / path).resolve()
    # Prevent path traversal outside working directory
    if not target.is_relative_to(base):
        logger.warning("path traversal blocked: %r resolved outside working dir %s", path, base)
        return f"Access denied: '{path}' is outside the working directory."
    if not target.is_file():
        return f"File not found: '{path}'."
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Cannot read '{path}' as text."
    except OSError as e:
        return f"Error reading '{path}': {e}"


file_tool_read = _ContextTool(
    _read_file, name="read_file", description="Read a file from the working directory."
)


def get_file_tools() -> list[Tool]:
    """Return fresh file tool instances (safe for per-agent binding)."""
    return [
        _ContextTool(
            _read_file, name="read_file", description="Read a file from the working directory."
        ),
    ]


# ── All context tools ──────────────────────────────────────────────


def get_context_tools() -> list[Tool]:
    """Return all context tools (planning + knowledge + file)."""
    return get_planning_tools() + get_knowledge_tools() + get_file_tools()
