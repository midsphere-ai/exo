"""Context tools — planning, knowledge, and file tools for agent self-management.

Provides ``@tool``-decorated functions that let agents manipulate their own
context during execution: manage a todo checklist, search knowledge artifacts,
and read files from a working directory.

All tools operate on a :class:`Context` instance passed via a ``ctx`` keyword
argument at call time.  The ``ctx`` parameter is excluded from the generated
JSON schema so that LLMs only see the user-facing parameters.
"""

from __future__ import annotations

import glob as _glob_mod
import re
from pathlib import Path
from typing import Any

from orbiter.tool import Tool, ToolError, _extract_description, _generate_schema

# ── Helper: build a tool that receives ctx at call time ─────────────


class _ContextTool(Tool):
    """A tool whose ``execute`` injects a ``ctx`` from bound state.

    The ``ctx`` parameter is NOT part of the JSON schema — it is injected
    by the caller (runner / agent) at execution time.
    """

    __slots__ = ("_ctx", "_fn", "description", "name", "parameters")

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

    def bind(self, ctx: Any) -> _ContextTool:
        """Bind a :class:`Context` instance for subsequent calls."""
        self._ctx = ctx
        return self

    async def execute(self, **kwargs: Any) -> str | dict[str, Any]:
        if self._ctx is None:
            raise ToolError(f"Tool '{self.name}' requires a bound context (call .bind(ctx) first)")
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
    """Return all planning tools."""
    return [planning_tool_add, planning_tool_complete, planning_tool_get]


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
    """Return all knowledge tools."""
    return [knowledge_tool_get, knowledge_tool_grep, knowledge_tool_search]


# ── File tool ───────────────────────────────────────────────────────


def _resolve_safe_path(ctx: Any, path: str) -> tuple[Path, Path] | str:
    """Resolve *path* relative to the working directory and validate safety.

    Returns ``(base, target)`` on success or an error message string.
    """
    working_dir = ctx.state.get("working_dir")
    if working_dir is None:
        return "No working directory set in context."
    base = Path(working_dir).resolve()
    target = (base / path).resolve()
    if not str(target).startswith(str(base)):
        return f"Access denied: '{path}' is outside the working directory."
    return base, target


async def _read_file(ctx: Any, path: str) -> str:
    """Read a file from the working directory.

    Args:
        path: Relative path to the file within the working directory.
    """
    resolved = _resolve_safe_path(ctx, path)
    if isinstance(resolved, str):
        return resolved
    _base, target = resolved
    if not target.is_file():
        return f"File not found: '{path}'."
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Cannot read '{path}' as text."
    except OSError as e:
        return f"Error reading '{path}': {e}"


async def _write_file(ctx: Any, path: str, content: str) -> str:
    """Write content to a file in the working directory.

    Creates parent directories as needed.  Overwrites existing files.

    Args:
        path: Relative path to the file within the working directory.
        content: The text content to write.
    """
    resolved = _resolve_safe_path(ctx, path)
    if isinstance(resolved, str):
        return resolved
    _base, target = resolved
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to '{path}'."
    except OSError as e:
        return f"Error writing '{path}': {e}"


async def _edit_file(ctx: Any, path: str, old_text: str, new_text: str) -> str:
    """Find and replace text in a file.

    Replaces the **first** occurrence of *old_text* with *new_text*.

    Args:
        path: Relative path to the file within the working directory.
        old_text: The exact text to find.
        new_text: The replacement text.
    """
    resolved = _resolve_safe_path(ctx, path)
    if isinstance(resolved, str):
        return resolved
    _base, target = resolved
    if not target.is_file():
        return f"File not found: '{path}'."
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Cannot read '{path}' as text."
    except OSError as e:
        return f"Error reading '{path}': {e}"
    if old_text not in text:
        return f"Text not found in '{path}'."
    updated = text.replace(old_text, new_text, 1)
    try:
        target.write_text(updated, encoding="utf-8")
        return f"Replaced text in '{path}'."
    except OSError as e:
        return f"Error writing '{path}': {e}"


async def _glob_files(ctx: Any, pattern: str) -> str:
    """Search for files matching a glob pattern in the working directory.

    Args:
        pattern: A glob pattern (e.g. ``**/*.py``).
    """
    working_dir = ctx.state.get("working_dir")
    if working_dir is None:
        return "No working directory set in context."
    base = Path(working_dir).resolve()
    matches: list[str] = []
    for hit in sorted(_glob_mod.glob(str(base / pattern), recursive=True)):
        hit_path = Path(hit).resolve()
        if not str(hit_path).startswith(str(base)):
            continue
        try:
            matches.append(str(hit_path.relative_to(base)))
        except ValueError:
            continue
    if not matches:
        return f"No files matched pattern '{pattern}'."
    return "\n".join(matches)


async def _grep_files(ctx: Any, pattern: str, path: str = ".") -> str:
    """Search file contents for lines matching a regex pattern.

    Args:
        pattern: A regex pattern to match against each line.
        path: Relative path to a file or directory to search (default ``"."``).
    """
    resolved = _resolve_safe_path(ctx, path)
    if isinstance(resolved, str):
        return resolved
    base, target = resolved
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex pattern: {e}"
    results: list[str] = []

    def _search_file(fpath: Path) -> None:
        try:
            text = fpath.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return
        rel = str(fpath.relative_to(base))
        for i, line in enumerate(text.splitlines(), 1):
            if compiled.search(line):
                results.append(f"{rel}:{i}: {line}")

    if target.is_file():
        _search_file(target)
    elif target.is_dir():
        for fpath in sorted(target.rglob("*")):
            if fpath.is_file():
                _search_file(fpath)
    else:
        return f"Path not found: '{path}'."
    if not results:
        return f"No matches for pattern '{pattern}'."
    return "\n".join(results)


file_tool_read = _ContextTool(
    _read_file, name="read_file", description="Read a file from the working directory."
)
file_tool_write = _ContextTool(
    _write_file, name="write_file", description="Write content to a file in the working directory."
)
file_tool_edit = _ContextTool(
    _edit_file, name="edit_file", description="Find and replace text in a file."
)
file_tool_glob = _ContextTool(
    _glob_files, name="glob_files", description="Search for files matching a glob pattern."
)
file_tool_grep = _ContextTool(
    _grep_files, name="grep_files", description="Search file contents for regex matches."
)


def get_file_tools() -> list[Tool]:
    """Return all file tools."""
    return [file_tool_read, file_tool_write, file_tool_edit, file_tool_glob, file_tool_grep]


# ── Reload tool ────────────────────────────────────────────────────


async def _reload_offloaded(ctx: Any, handle: str) -> str:
    """Reload offloaded message content by its handle ID.

    When messages are offloaded to save context space, they are replaced with
    ``[[OFFLOAD: handle=<id>]]`` markers.  Use this tool to retrieve the
    original content.

    Args:
        handle: The handle ID from an ``[[OFFLOAD: handle=...]]`` marker.
    """
    offloaded: dict[str, str] | None = ctx.state.get("offloaded_messages")
    if not offloaded:
        return "No offloaded messages found."
    content = offloaded.get(handle)
    if content is None:
        return f"Offload handle '{handle}' not found."
    return content


reload_tool = _ContextTool(
    _reload_offloaded,
    name="reload_offloaded",
    description="Reload offloaded message content by handle ID.",
)


def get_reload_tools() -> list[Tool]:
    """Return all reload tools."""
    return [reload_tool]


# ── All context tools ──────────────────────────────────────────────


def get_context_tools() -> list[Tool]:
    """Return all context tools (planning + knowledge + file + reload)."""
    return get_planning_tools() + get_knowledge_tools() + get_file_tools() + get_reload_tools()
