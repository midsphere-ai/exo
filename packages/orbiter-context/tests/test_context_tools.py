"""Tests for context tools: planning, knowledge, and file tools."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from orbiter.context._internal.knowledge import (  # pyright: ignore[reportMissingImports]
    KnowledgeStore,
)
from orbiter.context.context import Context  # pyright: ignore[reportMissingImports]
from orbiter.context.tools import (  # pyright: ignore[reportMissingImports]
    _ContextTool,
    file_tool_read,
    get_context_tools,
    get_file_tools,
    get_knowledge_tools,
    get_planning_tools,
    get_reload_tools,
    knowledge_tool_get,
    knowledge_tool_grep,
    knowledge_tool_search,
    planning_tool_add,
    planning_tool_complete,
    planning_tool_get,
    reload_tool,
)
from orbiter.context.workspace import Workspace  # pyright: ignore[reportMissingImports]
from orbiter.tool import ToolError

# ── Helpers ──────────────────────────────────────────────────────────


def _ctx(state: dict[str, Any] | None = None) -> Context:
    """Create a minimal context with optional initial state."""
    ctx = Context("test-task")
    if state:
        ctx.state.update(state)
    return ctx


def _bind(tool: _ContextTool, ctx: Context) -> _ContextTool:
    """Create a fresh copy-like binding (bind mutates in place)."""
    tool.bind(ctx)
    return tool


# ═══════════════════════════════════════════════════════════════════
# Planning tools
# ═══════════════════════════════════════════════════════════════════


class TestPlanningToolSchema:
    """Planning tools have correct schemas (no ctx param)."""

    def test_add_todo_schema(self) -> None:
        schema = planning_tool_add.parameters
        assert "item" in schema["properties"]
        assert "ctx" not in schema["properties"]
        assert "item" in schema.get("required", [])

    def test_get_todo_schema(self) -> None:
        schema = planning_tool_get.parameters
        assert "ctx" not in schema.get("properties", {})
        assert "required" not in schema  # no required params

    def test_complete_todo_schema(self) -> None:
        schema = planning_tool_complete.parameters
        assert "index" in schema["properties"]
        assert "ctx" not in schema["properties"]

    def test_tool_names(self) -> None:
        assert planning_tool_add.name == "add_todo"
        assert planning_tool_get.name == "get_todo"
        assert planning_tool_complete.name == "complete_todo"


class TestAddTodo:
    """add_todo mutates context state."""

    async def test_add_first_todo(self) -> None:
        ctx = _ctx()
        _bind(planning_tool_add, ctx)
        result = await planning_tool_add.execute(item="Write tests")
        assert "Write tests" in result
        todos = ctx.state.get("todos")
        assert len(todos) == 1
        assert todos[0]["item"] == "Write tests"
        assert todos[0]["done"] is False

    async def test_add_multiple_todos(self) -> None:
        ctx = _ctx()
        _bind(planning_tool_add, ctx)
        await planning_tool_add.execute(item="First")
        await planning_tool_add.execute(item="Second")
        todos = ctx.state.get("todos")
        assert len(todos) == 2
        assert todos[0]["item"] == "First"
        assert todos[1]["item"] == "Second"


class TestCompleteTodo:
    """complete_todo marks items as done."""

    async def test_complete_existing(self) -> None:
        ctx = _ctx({"todos": [{"item": "Task A", "done": False}]})
        _bind(planning_tool_complete, ctx)
        result = await planning_tool_complete.execute(index=0)
        assert "done" in result
        assert ctx.state.get("todos")[0]["done"] is True

    async def test_complete_invalid_index(self) -> None:
        ctx = _ctx({"todos": [{"item": "Task A", "done": False}]})
        _bind(planning_tool_complete, ctx)
        result = await planning_tool_complete.execute(index=5)
        assert "Invalid index" in result

    async def test_complete_empty_list(self) -> None:
        ctx = _ctx()
        _bind(planning_tool_complete, ctx)
        result = await planning_tool_complete.execute(index=0)
        assert "No todos" in result


class TestGetTodo:
    """get_todo returns formatted checklist."""

    async def test_empty_todos(self) -> None:
        ctx = _ctx()
        _bind(planning_tool_get, ctx)
        result = await planning_tool_get.execute()
        assert result == "No todos."

    async def test_with_todos(self) -> None:
        ctx = _ctx(
            {
                "todos": [
                    {"item": "Task A", "done": False},
                    {"item": "Task B", "done": True},
                ]
            }
        )
        _bind(planning_tool_get, ctx)
        result = await planning_tool_get.execute()
        assert "[ ] Task A" in result
        assert "[x] Task B" in result


# ═══════════════════════════════════════════════════════════════════
# Knowledge tools
# ═══════════════════════════════════════════════════════════════════


class TestKnowledgeToolSchema:
    """Knowledge tools have correct schemas."""

    def test_get_knowledge_schema(self) -> None:
        schema = knowledge_tool_get.parameters
        assert "name" in schema["properties"]
        assert "ctx" not in schema["properties"]

    def test_grep_knowledge_schema(self) -> None:
        schema = knowledge_tool_grep.parameters
        assert "name" in schema["properties"]
        assert "pattern" in schema["properties"]
        assert "ctx" not in schema["properties"]

    def test_search_knowledge_schema(self) -> None:
        schema = knowledge_tool_search.parameters
        assert "query" in schema["properties"]
        assert "ctx" not in schema["properties"]


class TestGetKnowledge:
    """get_knowledge retrieves artifacts from workspace."""

    async def test_no_workspace(self) -> None:
        ctx = _ctx()
        _bind(knowledge_tool_get, ctx)
        result = await knowledge_tool_get.execute(name="doc")
        assert "No workspace" in result

    async def test_artifact_not_found(self) -> None:
        ws = Workspace("test-ws")
        ctx = _ctx({"workspace": ws})
        _bind(knowledge_tool_get, ctx)
        result = await knowledge_tool_get.execute(name="missing")
        assert "not found" in result

    async def test_artifact_found(self) -> None:
        ws = Workspace("test-ws")
        await ws.write("doc", "Hello world content")
        ctx = _ctx({"workspace": ws})
        _bind(knowledge_tool_get, ctx)
        result = await knowledge_tool_get.execute(name="doc")
        assert result == "Hello world content"


class TestGrepKnowledge:
    """grep_knowledge searches artifact lines by regex."""

    async def test_no_workspace(self) -> None:
        ctx = _ctx()
        _bind(knowledge_tool_grep, ctx)
        result = await knowledge_tool_grep.execute(name="doc", pattern="test")
        assert "No workspace" in result

    async def test_artifact_not_found(self) -> None:
        ws = Workspace("test-ws")
        ctx = _ctx({"workspace": ws})
        _bind(knowledge_tool_grep, ctx)
        result = await knowledge_tool_grep.execute(name="missing", pattern="test")
        assert "not found" in result

    async def test_matching_lines(self) -> None:
        ws = Workspace("test-ws")
        await ws.write("doc", "line one\nline two test\nline three\nline four test")
        ctx = _ctx({"workspace": ws})
        _bind(knowledge_tool_grep, ctx)
        result = await knowledge_tool_grep.execute(name="doc", pattern="test")
        assert "2: line two test" in result
        assert "4: line four test" in result
        assert "line one" not in result

    async def test_no_matches(self) -> None:
        ws = Workspace("test-ws")
        await ws.write("doc", "hello\nworld")
        ctx = _ctx({"workspace": ws})
        _bind(knowledge_tool_grep, ctx)
        result = await knowledge_tool_grep.execute(name="doc", pattern="xyz")
        assert "No matches" in result

    async def test_invalid_regex(self) -> None:
        ws = Workspace("test-ws")
        await ws.write("doc", "hello")
        ctx = _ctx({"workspace": ws})
        _bind(knowledge_tool_grep, ctx)
        result = await knowledge_tool_grep.execute(name="doc", pattern="[invalid")
        assert "Invalid regex" in result


class TestSearchKnowledge:
    """search_knowledge uses KnowledgeStore for keyword search."""

    async def test_no_knowledge_store(self) -> None:
        ctx = _ctx()
        _bind(knowledge_tool_search, ctx)
        result = await knowledge_tool_search.execute(query="test")
        assert "No knowledge store" in result

    async def test_no_results(self) -> None:
        ks = KnowledgeStore()
        ks.add("doc", "hello world")
        ctx = _ctx({"knowledge_store": ks})
        _bind(knowledge_tool_search, ctx)
        result = await knowledge_tool_search.execute(query="xyznotfound")
        assert "No results" in result

    async def test_with_results(self) -> None:
        ks = KnowledgeStore()
        ks.add("doc", "Python programming language guide")
        ctx = _ctx({"knowledge_store": ks})
        _bind(knowledge_tool_search, ctx)
        result = await knowledge_tool_search.execute(query="python")
        assert "doc#" in result
        assert "score=" in result


# ═══════════════════════════════════════════════════════════════════
# File tools
# ═══════════════════════════════════════════════════════════════════


class TestFileToolSchema:
    """File tool has correct schema."""

    def test_read_file_schema(self) -> None:
        schema = file_tool_read.parameters
        assert "path" in schema["properties"]
        assert "ctx" not in schema["properties"]
        assert "path" in schema.get("required", [])


class TestReadFile:
    """read_file reads from working directory."""

    async def test_no_working_dir(self) -> None:
        ctx = _ctx()
        _bind(file_tool_read, ctx)
        result = await file_tool_read.execute(path="test.txt")
        assert "No working directory" in result

    async def test_file_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _ctx({"working_dir": tmpdir})
            _bind(file_tool_read, ctx)
            result = await file_tool_read.execute(path="nonexistent.txt")
            assert "not found" in result

    async def test_read_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "hello.txt"
            p.write_text("Hello from file!", encoding="utf-8")
            ctx = _ctx({"working_dir": tmpdir})
            _bind(file_tool_read, ctx)
            result = await file_tool_read.execute(path="hello.txt")
            assert result == "Hello from file!"

    async def test_path_traversal_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = _ctx({"working_dir": tmpdir})
            _bind(file_tool_read, ctx)
            result = await file_tool_read.execute(path="../../../etc/passwd")
            assert "Access denied" in result

    async def test_subdirectory_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = Path(tmpdir) / "subdir"
            sub.mkdir()
            f = sub / "data.txt"
            f.write_text("nested content", encoding="utf-8")
            ctx = _ctx({"working_dir": tmpdir})
            _bind(file_tool_read, ctx)
            result = await file_tool_read.execute(path="subdir/data.txt")
            assert result == "nested content"


# ═══════════════════════════════════════════════════════════════════
# Context tool binding
# ═══════════════════════════════════════════════════════════════════


class TestContextToolBinding:
    """_ContextTool binding and error handling."""

    async def test_unbound_raises(self) -> None:
        # Create a fresh tool instance to ensure it's unbound
        async def dummy(ctx: Any) -> str:
            return "ok"

        t = _ContextTool(dummy, name="dummy")
        with pytest.raises(ToolError, match="requires a bound context"):
            await t.execute()

    def test_bind_returns_self(self) -> None:
        ctx = _ctx()
        result = planning_tool_add.bind(ctx)
        assert result is planning_tool_add

    def test_to_schema(self) -> None:
        schema = planning_tool_add.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "add_todo"
        assert "ctx" not in schema["function"]["parameters"].get("properties", {})


# ═══════════════════════════════════════════════════════════════════
# Reload tools
# ═══════════════════════════════════════════════════════════════════


class TestReloadToolSchema:
    """Reload tool has correct schema (no ctx param)."""

    def test_reload_offloaded_schema(self) -> None:
        schema = reload_tool.parameters
        assert "handle" in schema["properties"]
        assert "ctx" not in schema["properties"]
        assert "handle" in schema.get("required", [])

    def test_tool_name(self) -> None:
        assert reload_tool.name == "reload_offloaded"


class TestReloadOffloaded:
    """reload_offloaded retrieves offloaded message content."""

    async def test_no_offloaded_messages(self) -> None:
        ctx = _ctx()
        _bind(reload_tool, ctx)
        result = await reload_tool.execute(handle="abc123")
        assert "No offloaded messages" in result

    async def test_empty_offloaded_dict(self) -> None:
        ctx = _ctx({"offloaded_messages": {}})
        _bind(reload_tool, ctx)
        result = await reload_tool.execute(handle="abc123")
        assert "No offloaded messages" in result

    async def test_handle_found(self) -> None:
        original = "This is the original long message content that was offloaded."
        ctx = _ctx({"offloaded_messages": {"abc123def456": original}})
        _bind(reload_tool, ctx)
        result = await reload_tool.execute(handle="abc123def456")
        assert result == original

    async def test_handle_not_found(self) -> None:
        ctx = _ctx({"offloaded_messages": {"abc123def456": "some content"}})
        _bind(reload_tool, ctx)
        result = await reload_tool.execute(handle="unknown_handle")
        assert "not found" in result
        assert "unknown_handle" in result

    async def test_multiple_handles(self) -> None:
        ctx = _ctx(
            {
                "offloaded_messages": {
                    "handle_aaa": "first message",
                    "handle_bbb": "second message",
                }
            }
        )
        _bind(reload_tool, ctx)
        result_a = await reload_tool.execute(handle="handle_aaa")
        assert result_a == "first message"
        result_b = await reload_tool.execute(handle="handle_bbb")
        assert result_b == "second message"


# ═══════════════════════════════════════════════════════════════════
# Factory functions
# ═══════════════════════════════════════════════════════════════════


class TestFactoryFunctions:
    """get_*_tools() return the expected tool lists."""

    def test_get_planning_tools(self) -> None:
        tools = get_planning_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"add_todo", "complete_todo", "get_todo"}

    def test_get_knowledge_tools(self) -> None:
        tools = get_knowledge_tools()
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert names == {"get_knowledge", "grep_knowledge", "search_knowledge"}

    def test_get_file_tools(self) -> None:
        tools = get_file_tools()
        assert len(tools) == 1
        assert tools[0].name == "read_file"

    def test_get_reload_tools(self) -> None:
        tools = get_reload_tools()
        assert len(tools) == 1
        assert tools[0].name == "reload_offloaded"

    def test_get_context_tools(self) -> None:
        tools = get_context_tools()
        assert len(tools) == 8
        names = {t.name for t in tools}
        assert "add_todo" in names
        assert "get_knowledge" in names
        assert "read_file" in names
        assert "reload_offloaded" in names
