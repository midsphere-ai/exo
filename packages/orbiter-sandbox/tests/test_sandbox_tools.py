"""Tests for built-in sandbox tools (FilesystemTool, TerminalTool, ShellTool, CodeTool)."""

from __future__ import annotations

from pathlib import Path

import pytest

from orbiter.sandbox.tools import (  # pyright: ignore[reportMissingImports]
    _DANGEROUS_COMMANDS,
    _DEFAULT_ALLOWED_COMMANDS,
    _DEFAULT_BLOCKED_NAMES,
    CodeTool,
    FilesystemTool,
    ShellTool,
    TerminalTool,
    code_tool,
    shell_tool,
)
from orbiter.tool import ToolError

# ---------------------------------------------------------------------------
# FilesystemTool — path validation
# ---------------------------------------------------------------------------


class TestFilesystemPathValidation:
    def test_no_restrictions(self, tmp_path: Path) -> None:
        tool = FilesystemTool()
        result = tool._validate_path(str(tmp_path))
        assert result == tmp_path.resolve()

    def test_allowed_directory_accepted(self, tmp_path: Path) -> None:
        child = tmp_path / "sub" / "file.txt"
        tool = FilesystemTool(allowed_directories=[str(tmp_path)])
        result = tool._validate_path(str(child))
        assert result == child.resolve()

    def test_outside_allowed_rejected(self, tmp_path: Path) -> None:
        tool = FilesystemTool(allowed_directories=[str(tmp_path / "allowed")])
        with pytest.raises(ToolError, match="outside allowed directories"):
            tool._validate_path("/etc/passwd")

    def test_multiple_allowed_directories(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        tool = FilesystemTool(allowed_directories=[str(dir_a), str(dir_b)])
        assert tool._validate_path(str(dir_a / "f.txt")) == (dir_a / "f.txt").resolve()
        assert tool._validate_path(str(dir_b / "g.txt")) == (dir_b / "g.txt").resolve()

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        allowed = tmp_path / "workspace"
        allowed.mkdir()
        tool = FilesystemTool(allowed_directories=[str(allowed)])
        with pytest.raises(ToolError, match="outside allowed directories"):
            tool._validate_path(str(allowed / ".." / "secret.txt"))


# ---------------------------------------------------------------------------
# FilesystemTool — read
# ---------------------------------------------------------------------------


class TestFilesystemRead:
    async def test_read_file(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("hello world", encoding="utf-8")
        tool = FilesystemTool(allowed_directories=[str(tmp_path)])
        result = await tool.execute(action="read", path=str(f))
        assert result == "hello world"

    async def test_read_missing_file(self, tmp_path: Path) -> None:
        tool = FilesystemTool(allowed_directories=[str(tmp_path)])
        with pytest.raises(ToolError, match="File not found"):
            await tool.execute(action="read", path=str(tmp_path / "nope.txt"))


# ---------------------------------------------------------------------------
# FilesystemTool — write
# ---------------------------------------------------------------------------


class TestFilesystemWrite:
    async def test_write_file(self, tmp_path: Path) -> None:
        f = tmp_path / "out.txt"
        tool = FilesystemTool(allowed_directories=[str(tmp_path)])
        result = await tool.execute(action="write", path=str(f), content="data")
        assert "4 chars" in result
        assert f.read_text(encoding="utf-8") == "data"

    async def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        f = tmp_path / "a" / "b" / "c.txt"
        tool = FilesystemTool(allowed_directories=[str(tmp_path)])
        await tool.execute(action="write", path=str(f), content="nested")
        assert f.read_text(encoding="utf-8") == "nested"

    async def test_write_missing_content(self, tmp_path: Path) -> None:
        tool = FilesystemTool(allowed_directories=[str(tmp_path)])
        with pytest.raises(ToolError, match=r"content.*required"):
            await tool.execute(action="write", path=str(tmp_path / "x.txt"))


# ---------------------------------------------------------------------------
# FilesystemTool — list
# ---------------------------------------------------------------------------


class TestFilesystemList:
    async def test_list_directory(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "b_dir").mkdir()
        tool = FilesystemTool(allowed_directories=[str(tmp_path)])
        result = await tool.execute(action="list", path=str(tmp_path))
        assert isinstance(result, dict)
        names = [e["name"] for e in result["entries"]]
        assert "a.txt" in names
        assert "b_dir" in names
        # Check types
        by_name = {e["name"]: e for e in result["entries"]}
        assert by_name["a.txt"]["type"] == "file"
        assert by_name["b_dir"]["type"] == "dir"

    async def test_list_missing_directory(self, tmp_path: Path) -> None:
        tool = FilesystemTool(allowed_directories=[str(tmp_path)])
        with pytest.raises(ToolError, match="Directory not found"):
            await tool.execute(action="list", path=str(tmp_path / "nope"))

    async def test_list_file_not_dir(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("x", encoding="utf-8")
        tool = FilesystemTool(allowed_directories=[str(tmp_path)])
        with pytest.raises(ToolError, match="Not a directory"):
            await tool.execute(action="list", path=str(f))


# ---------------------------------------------------------------------------
# FilesystemTool — unknown action
# ---------------------------------------------------------------------------


class TestFilesystemUnknownAction:
    async def test_unknown_action(self, tmp_path: Path) -> None:
        tool = FilesystemTool()
        with pytest.raises(ToolError, match="Unknown filesystem action"):
            await tool.execute(action="delete", path="/tmp/x")


# ---------------------------------------------------------------------------
# FilesystemTool — schema
# ---------------------------------------------------------------------------


class TestFilesystemSchema:
    def test_schema(self) -> None:
        tool = FilesystemTool()
        schema = tool.to_schema()
        assert schema["function"]["name"] == "filesystem"
        assert "action" in schema["function"]["parameters"]["properties"]
        assert "path" in schema["function"]["parameters"]["properties"]


# ---------------------------------------------------------------------------
# TerminalTool — command filtering
# ---------------------------------------------------------------------------


class TestTerminalCommandFiltering:
    def test_dangerous_command_blocked(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="blocked by sandbox policy"):
            tool._check_command("rm -rf /")

    def test_safe_command_allowed(self) -> None:
        tool = TerminalTool()
        tool._check_command("echo hello")  # should not raise

    def test_full_path_command_stripped(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="blocked by sandbox policy"):
            tool._check_command("/usr/bin/rm -rf /")

    def test_case_insensitive_blocking(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="blocked by sandbox policy"):
            tool._check_command("RM -rf /")

    def test_custom_blacklist(self) -> None:
        tool = TerminalTool(blacklist=frozenset({"curl"}))
        with pytest.raises(ToolError, match="blocked by sandbox policy"):
            tool._check_command("curl http://evil.com")
        # rm should be allowed with custom blacklist
        tool._check_command("rm -rf /")

    def test_empty_command_rejected(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="Empty command"):
            tool._check_command("")

    def test_whitespace_only_command(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="Empty command"):
            tool._check_command("   ")

    def test_default_blacklist_contents(self) -> None:
        assert "rm" in _DANGEROUS_COMMANDS
        assert "shutdown" in _DANGEROUS_COMMANDS
        assert "kill" in _DANGEROUS_COMMANDS


# ---------------------------------------------------------------------------
# TerminalTool — execution
# ---------------------------------------------------------------------------


class TestTerminalExecution:
    async def test_echo_command(self) -> None:
        tool = TerminalTool()
        result = await tool.execute(command="echo hello")
        assert isinstance(result, dict)
        assert result["exit_code"] == 0
        assert "hello" in result["stdout"]
        assert "platform" in result

    async def test_command_stderr(self) -> None:
        tool = TerminalTool()
        result = await tool.execute(command="echo error >&2")
        assert "error" in result["stderr"]

    async def test_nonzero_exit(self) -> None:
        tool = TerminalTool()
        result = await tool.execute(command="exit 42")
        assert result["exit_code"] == 42

    async def test_empty_command(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="Empty command"):
            await tool.execute(command="")

    async def test_blocked_command(self) -> None:
        tool = TerminalTool()
        with pytest.raises(ToolError, match="blocked by sandbox policy"):
            await tool.execute(command="rm -rf /tmp/test")

    async def test_timeout(self) -> None:
        tool = TerminalTool(timeout=0.1)
        with pytest.raises(ToolError, match="timed out"):
            await tool.execute(command="sleep 10")


# ---------------------------------------------------------------------------
# TerminalTool — platform detection
# ---------------------------------------------------------------------------


class TestTerminalPlatform:
    def test_platform_property(self) -> None:
        import sys

        tool = TerminalTool()
        assert tool.platform == sys.platform


# ---------------------------------------------------------------------------
# TerminalTool — schema
# ---------------------------------------------------------------------------


class TestTerminalSchema:
    def test_schema(self) -> None:
        tool = TerminalTool()
        schema = tool.to_schema()
        assert schema["function"]["name"] == "terminal"
        assert "command" in schema["function"]["parameters"]["properties"]


# ---------------------------------------------------------------------------
# ShellTool — command validation
# ---------------------------------------------------------------------------


class TestShellCommandValidation:
    def test_allowed_command_passes(self) -> None:
        tool = ShellTool()
        parts = tool._validate_command("ls -la /tmp")
        assert parts[0] == "ls"

    def test_disallowed_command_rejected(self) -> None:
        tool = ShellTool()
        with pytest.raises(ToolError, match="not in the allowed list"):
            tool._validate_command("rm -rf /")

    def test_empty_command_rejected(self) -> None:
        tool = ShellTool()
        with pytest.raises(ToolError, match="Empty command"):
            tool._validate_command("")

    def test_whitespace_only_rejected(self) -> None:
        tool = ShellTool()
        with pytest.raises(ToolError, match="Empty command"):
            tool._validate_command("   ")

    def test_full_path_stripped(self) -> None:
        tool = ShellTool()
        parts = tool._validate_command("/usr/bin/ls -la")
        assert parts[0] == "/usr/bin/ls"

    def test_full_path_disallowed(self) -> None:
        tool = ShellTool()
        with pytest.raises(ToolError, match="not in the allowed list"):
            tool._validate_command("/usr/bin/rm -rf /")

    def test_custom_allowlist(self) -> None:
        tool = ShellTool(allowed_commands=["curl", "wget"])
        tool._validate_command("curl http://example.com")  # should pass
        with pytest.raises(ToolError, match="not in the allowed list"):
            tool._validate_command("ls -la")

    def test_default_allowlist_contents(self) -> None:
        expected = {"ls", "cat", "grep", "find", "echo", "wc", "sort", "head", "tail", "diff"}
        assert set(_DEFAULT_ALLOWED_COMMANDS) == expected

    def test_quoted_arguments_parsed(self) -> None:
        tool = ShellTool()
        parts = tool._validate_command('grep "hello world" file.txt')
        assert parts == ["grep", "hello world", "file.txt"]


# ---------------------------------------------------------------------------
# ShellTool — execution
# ---------------------------------------------------------------------------


class TestShellExecution:
    async def test_echo_command(self) -> None:
        tool = ShellTool()
        result = await tool.execute(command="echo hello")
        assert isinstance(result, dict)
        assert result["exit_code"] == 0
        assert "hello" in result["stdout"]

    async def test_ls_command(self, tmp_path: Path) -> None:
        (tmp_path / "test.txt").write_text("x", encoding="utf-8")
        tool = ShellTool()
        result = await tool.execute(command=f"ls {tmp_path}")
        assert result["exit_code"] == 0
        assert "test.txt" in result["stdout"]

    async def test_cat_command(self, tmp_path: Path) -> None:
        f = tmp_path / "data.txt"
        f.write_text("file contents here", encoding="utf-8")
        tool = ShellTool()
        result = await tool.execute(command=f"cat {f}")
        assert result["exit_code"] == 0
        assert "file contents here" in result["stdout"]

    async def test_disallowed_command_blocked(self) -> None:
        tool = ShellTool()
        with pytest.raises(ToolError, match="not in the allowed list"):
            await tool.execute(command="rm -rf /tmp/test")

    async def test_empty_command(self) -> None:
        tool = ShellTool()
        with pytest.raises(ToolError, match="Empty command"):
            await tool.execute(command="")

    async def test_timeout(self) -> None:
        tool = ShellTool(allowed_commands=["sleep"], timeout=0.1)
        with pytest.raises(ToolError, match="timed out"):
            await tool.execute(command="sleep 10")

    async def test_stderr_captured(self) -> None:
        tool = ShellTool()
        result = await tool.execute(command="ls /nonexistent_path_xyz_123")
        assert result["exit_code"] != 0
        assert result["stderr"]

    async def test_output_truncation(self, tmp_path: Path) -> None:
        # Create a file with content that produces >10000 chars of output
        big = tmp_path / "big.txt"
        big.write_text("x" * 20000, encoding="utf-8")
        tool = ShellTool()
        result = await tool.execute(command=f"cat {big}")
        assert "truncated" in result["stdout"]
        assert len(result["stdout"]) < 20000


# ---------------------------------------------------------------------------
# ShellTool — schema
# ---------------------------------------------------------------------------


class TestShellSchema:
    def test_schema(self) -> None:
        tool = ShellTool()
        schema = tool.to_schema()
        assert schema["function"]["name"] == "shell"
        assert "command" in schema["function"]["parameters"]["properties"]


# ---------------------------------------------------------------------------
# shell_tool factory
# ---------------------------------------------------------------------------


class TestShellToolFactory:
    def test_factory_default(self) -> None:
        tool = shell_tool()
        assert isinstance(tool, ShellTool)
        assert tool._allowed == _DEFAULT_ALLOWED_COMMANDS

    def test_factory_custom_allowlist(self) -> None:
        tool = shell_tool(allowed_commands=["python", "node"])
        assert tool._allowed == ["python", "node"]

    def test_factory_custom_timeout(self) -> None:
        tool = shell_tool(timeout=60.0)
        assert tool._timeout == 60.0


# ---------------------------------------------------------------------------
# CodeTool — safe code execution
# ---------------------------------------------------------------------------


class TestCodeSafeExecution:
    async def test_simple_print(self) -> None:
        tool = CodeTool()
        result = await tool.execute(code="print('hello world')")
        assert "hello world" in result

    async def test_arithmetic(self) -> None:
        tool = CodeTool()
        result = await tool.execute(code="print(2 + 3)")
        assert "5" in result

    async def test_multiline_code(self) -> None:
        tool = CodeTool()
        result = await tool.execute(code="x = 10\ny = 20\nprint(x + y)")
        assert "30" in result

    async def test_no_output(self) -> None:
        tool = CodeTool()
        result = await tool.execute(code="x = 42")
        assert result == "(no output)"

    async def test_loop(self) -> None:
        tool = CodeTool()
        result = await tool.execute(code="for i in range(3):\n    print(i)")
        assert "0" in result
        assert "1" in result
        assert "2" in result


# ---------------------------------------------------------------------------
# CodeTool — blocked code
# ---------------------------------------------------------------------------


class TestCodeBlockedExecution:
    async def test_import_blocked(self) -> None:
        tool = CodeTool()
        result = await tool.execute(code="import os")
        assert "Error" in result

    async def test_open_blocked(self) -> None:
        tool = CodeTool()
        result = await tool.execute(code="open('/etc/passwd')")
        assert "Error" in result

    async def test_dunder_import_blocked(self) -> None:
        tool = CodeTool()
        result = await tool.execute(code="__import__('os')")
        assert "Error" in result

    async def test_custom_blocked_names(self) -> None:
        tool = CodeTool(blocked_names=frozenset({"print"}))
        result = await tool.execute(code="print('hello')")
        assert "Error" in result

    async def test_default_blocked_names_contents(self) -> None:
        assert "__import__" in _DEFAULT_BLOCKED_NAMES
        assert "open" in _DEFAULT_BLOCKED_NAMES
        assert "os" in _DEFAULT_BLOCKED_NAMES
        assert "sys" in _DEFAULT_BLOCKED_NAMES


# ---------------------------------------------------------------------------
# CodeTool — error handling
# ---------------------------------------------------------------------------


class TestCodeErrorHandling:
    async def test_empty_code(self) -> None:
        tool = CodeTool()
        with pytest.raises(ToolError, match="Empty code"):
            await tool.execute(code="")

    async def test_whitespace_only(self) -> None:
        tool = CodeTool()
        with pytest.raises(ToolError, match="Empty code"):
            await tool.execute(code="   ")

    async def test_syntax_error(self) -> None:
        tool = CodeTool()
        result = await tool.execute(code="def f(")
        assert "Error" in result

    async def test_runtime_error(self) -> None:
        tool = CodeTool()
        result = await tool.execute(code="1 / 0")
        assert "Error" in result

    async def test_timeout(self) -> None:
        tool = CodeTool(timeout=0.1)
        with pytest.raises(ToolError, match="timed out"):
            await tool.execute(code="while True:\n    pass")


# ---------------------------------------------------------------------------
# CodeTool — sandbox delegation
# ---------------------------------------------------------------------------


class TestCodeSandboxDelegation:
    async def test_delegates_to_sandbox(self) -> None:
        from orbiter.sandbox.base import LocalSandbox  # pyright: ignore[reportMissingImports]

        sandbox = LocalSandbox()
        await sandbox.start()
        tool = CodeTool(sandbox=sandbox)
        result = await tool.execute(code="print('hello')")
        assert isinstance(result, dict)
        assert result["tool"] == "code"
        assert result["arguments"] == {"code": "print('hello')"}
        await sandbox.cleanup()

    async def test_sandbox_not_running_error(self) -> None:
        from orbiter.sandbox.base import LocalSandbox  # pyright: ignore[reportMissingImports]

        sandbox = LocalSandbox()
        # Don't start — should fail
        tool = CodeTool(sandbox=sandbox)
        with pytest.raises(ToolError, match="Sandbox execution failed"):
            await tool.execute(code="print('hello')")


# ---------------------------------------------------------------------------
# CodeTool — schema
# ---------------------------------------------------------------------------


class TestCodeSchema:
    def test_schema(self) -> None:
        tool = CodeTool()
        schema = tool.to_schema()
        assert schema["function"]["name"] == "code"
        assert "code" in schema["function"]["parameters"]["properties"]


# ---------------------------------------------------------------------------
# code_tool factory
# ---------------------------------------------------------------------------


class TestCodeToolFactory:
    def test_factory_default(self) -> None:
        tool = code_tool()
        assert isinstance(tool, CodeTool)
        assert tool._sandbox is None
        assert tool._blocked == _DEFAULT_BLOCKED_NAMES

    def test_factory_with_sandbox(self) -> None:
        from orbiter.sandbox.base import LocalSandbox  # pyright: ignore[reportMissingImports]

        sandbox = LocalSandbox()
        tool = code_tool(sandbox=sandbox)
        assert tool._sandbox is sandbox

    def test_factory_custom_blocked(self) -> None:
        blocked = frozenset({"print", "eval"})
        tool = code_tool(blocked_names=blocked)
        assert tool._blocked == blocked

    def test_factory_custom_timeout(self) -> None:
        tool = code_tool(timeout=30.0)
        assert tool._timeout == 30.0
