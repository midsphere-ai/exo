"""Tests for MCP execution — retry logic, env var substitution, config loading."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from exo.mcp.client import MCPClientError  # pyright: ignore[reportMissingImports]
from exo.mcp.execution import (  # pyright: ignore[reportMissingImports]
    MCPExecutionError,
    _substitute_recursive,
    call_tool_with_retry,
    load_mcp_client,
    load_mcp_config,
    substitute_env_vars,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_client(
    results: list[Any] | None = None,
    errors: Sequence[Exception | None] | None = None,
) -> AsyncMock:
    """Create a mock MCPClient.

    ``results`` are returned in order.  ``errors`` are raised in order
    (``None`` means no error for that call).
    """
    client = AsyncMock()
    client.server_names = ["test-server"]

    if results and not errors:
        client.call_tool = AsyncMock(side_effect=results)
    elif errors:
        side_effects: list[Any] = []
        for i, err in enumerate(errors):
            if err is not None:
                side_effects.append(err)
            elif results and i < len(results):
                side_effects.append(results[i])
            else:
                side_effects.append(MagicMock(content=[]))
        client.call_tool = AsyncMock(side_effect=side_effects)
    else:
        client.call_tool = AsyncMock(return_value=MagicMock(content=[]))

    return client


def _mock_connection(
    results: list[Any] | None = None,
    errors: list[Exception | None] | None = None,
) -> AsyncMock:
    """Mock connection (no server_names attribute)."""
    conn = AsyncMock(spec=[])
    conn.name = "test-conn"

    if results and not errors:
        conn.call_tool = AsyncMock(side_effect=results)
    elif errors:
        side_effects: list[Any] = []
        for i, err in enumerate(errors):
            if err is not None:
                side_effects.append(err)
            elif results and i < len(results):
                side_effects.append(results[i])
            else:
                side_effects.append(MagicMock(content=[]))
        conn.call_tool = AsyncMock(side_effect=side_effects)
    else:
        conn.call_tool = AsyncMock(return_value=MagicMock(content=[]))

    return conn


# ===========================================================================
# call_tool_with_retry
# ===========================================================================


class TestRetrySuccess:
    """Tests for successful tool calls."""

    async def test_call_succeeds_first_try(self) -> None:
        result = MagicMock(content=["ok"])
        client = _mock_client(results=[result])
        got = await call_tool_with_retry(client, "s", "tool")
        assert got is result

    async def test_call_with_arguments(self) -> None:
        result = MagicMock(content=["ok"])
        client = _mock_client(results=[result])
        await call_tool_with_retry(client, "s", "tool", {"a": 1})
        client.call_tool.assert_awaited_once_with("s", "tool", {"a": 1})

    async def test_succeeds_after_retry(self) -> None:
        result = MagicMock(content=["ok"])
        client = _mock_client(errors=[RuntimeError("transient"), None], results=[None, result])
        got = await call_tool_with_retry(client, "s", "tool", max_retries=1, backoff_base=0.0)
        assert got is result

    async def test_connection_dispatch(self) -> None:
        """Connection (no server_names) dispatches call_tool(tool, args)."""
        result = MagicMock(content=["ok"])
        conn = _mock_connection(results=[result])
        got = await call_tool_with_retry(conn, "ignored", "tool", {"x": 1})
        assert got is result
        conn.call_tool.assert_awaited_once_with("tool", {"x": 1})


class TestRetryFailure:
    """Tests for exhausted retries."""

    async def test_all_retries_exhausted(self) -> None:
        client = _mock_client(errors=[RuntimeError("fail")] * 3)
        with pytest.raises(MCPExecutionError, match="failed after 3 attempts"):
            await call_tool_with_retry(client, "s", "tool", max_retries=2, backoff_base=0.0)

    async def test_no_retries(self) -> None:
        client = _mock_client(errors=[RuntimeError("fail")])
        with pytest.raises(MCPExecutionError, match="failed after 1 attempts"):
            await call_tool_with_retry(client, "s", "tool", max_retries=0, backoff_base=0.0)

    async def test_mcp_client_error_not_retried(self) -> None:
        """MCPClientError is not retryable — raised immediately."""
        client = _mock_client(errors=[MCPClientError("not found")])
        with pytest.raises(MCPClientError, match="not found"):
            await call_tool_with_retry(client, "s", "tool", max_retries=3, backoff_base=0.0)
        assert client.call_tool.await_count == 1


class TestRetryTimeout:
    """Tests for per-attempt timeout."""

    async def test_timeout_triggers_retry(self) -> None:
        result = MagicMock(content=["ok"])

        async def _slow_then_fast(*args: Any, **kwargs: Any) -> Any:
            if _slow_then_fast.call_count == 0:  # type: ignore[attr-defined]
                _slow_then_fast.call_count += 1  # type: ignore[attr-defined]
                await asyncio.sleep(10)
            return result

        _slow_then_fast.call_count = 0  # type: ignore[attr-defined]
        import asyncio

        client = AsyncMock()
        client.server_names = ["s"]
        client.call_tool = _slow_then_fast
        got = await call_tool_with_retry(
            client, "s", "tool", max_retries=1, timeout=0.01, backoff_base=0.0
        )
        assert got is result

    async def test_timeout_exhausts_retries(self) -> None:
        import asyncio

        async def _always_slow(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(10)

        client = AsyncMock()
        client.server_names = ["s"]
        client.call_tool = _always_slow
        with pytest.raises(MCPExecutionError, match="failed after 2 attempts"):
            await call_tool_with_retry(
                client, "s", "tool", max_retries=1, timeout=0.01, backoff_base=0.0
            )


class TestRetryBackoff:
    """Test backoff timing."""

    async def test_custom_backoff_base(self) -> None:
        """Verify backoff_base scales the delay (0.0 = instant)."""
        client = _mock_client(errors=[RuntimeError("fail")] * 3)
        with pytest.raises(MCPExecutionError):
            await call_tool_with_retry(client, "s", "tool", max_retries=2, backoff_base=0.0)
        assert client.call_tool.await_count == 3


# ===========================================================================
# substitute_env_vars
# ===========================================================================


class TestSubstituteEnvVars:
    """Tests for env var substitution."""

    def test_simple_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_VAR", "hello")
        assert substitute_env_vars("${MY_VAR}") == "hello"

    def test_multiple_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("A", "1")
        monkeypatch.setenv("B", "2")
        assert substitute_env_vars("${A}-${B}") == "1-2"

    def test_unset_var_becomes_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DOES_NOT_EXIST", raising=False)
        assert substitute_env_vars("prefix-${DOES_NOT_EXIST}-suffix") == "prefix--suffix"

    def test_no_vars(self) -> None:
        assert substitute_env_vars("no vars here") == "no vars here"

    def test_empty_string(self) -> None:
        assert substitute_env_vars("") == ""

    def test_nested_braces_ignored(self) -> None:
        assert substitute_env_vars("${") == "${"


class TestSubstituteRecursive:
    """Tests for recursive substitution in dicts/lists."""

    def test_dict_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("X", "val")
        result = _substitute_recursive({"key": "${X}", "num": 42})
        assert result == {"key": "val", "num": 42}

    def test_nested_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("V", "found")
        result = _substitute_recursive(["${V}", ["inner-${V}"]])
        assert result == ["found", ["inner-found"]]

    def test_passthrough_non_string(self) -> None:
        assert _substitute_recursive(42) == 42
        assert _substitute_recursive(None) is None


# ===========================================================================
# load_mcp_config / load_mcp_client
# ===========================================================================


class TestLoadMCPConfig:
    """Tests for config file loading."""

    def test_load_stdio_server(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PY", "/usr/bin/python")
        cfg_file = tmp_path / "mcp.json"
        cfg_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "my-server": {
                            "transport": "stdio",
                            "command": "${PY}",
                            "args": ["-m", "server"],
                        }
                    }
                }
            )
        )
        configs = load_mcp_config(cfg_file)
        assert len(configs) == 1
        assert configs[0].name == "my-server"
        assert configs[0].command == "/usr/bin/python"
        assert configs[0].args == ["-m", "server"]

    def test_load_multiple_servers(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "mcp.json"
        cfg_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "a": {"transport": "stdio", "command": "cmd-a"},
                        "b": {"transport": "sse", "url": "http://localhost:8080"},
                    }
                }
            )
        )
        configs = load_mcp_config(cfg_file)
        assert len(configs) == 2
        names = {c.name for c in configs}
        assert names == {"a", "b"}

    def test_env_substitution_in_nested(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("API_KEY", "secret123")
        cfg_file = tmp_path / "mcp.json"
        cfg_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "api": {
                            "transport": "sse",
                            "url": "http://host",
                            "headers": {"Authorization": "Bearer ${API_KEY}"},
                        }
                    }
                }
            )
        )
        configs = load_mcp_config(cfg_file)
        assert configs[0].headers == {"Authorization": "Bearer secret123"}

    def test_missing_file(self) -> None:
        with pytest.raises(MCPExecutionError, match="not found"):
            load_mcp_config("/nonexistent/mcp.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "mcp.json"
        cfg_file.write_text("not json")
        with pytest.raises(MCPExecutionError, match="Failed to parse"):
            load_mcp_config(cfg_file)

    def test_invalid_servers_type(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "mcp.json"
        cfg_file.write_text(json.dumps({"mcpServers": "not-a-dict"}))
        with pytest.raises(MCPExecutionError, match="Expected 'mcpServers'"):
            load_mcp_config(cfg_file)

    def test_empty_servers(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "mcp.json"
        cfg_file.write_text(json.dumps({"mcpServers": {}}))
        configs = load_mcp_config(cfg_file)
        assert configs == []

    def test_defaults(self, tmp_path: Path) -> None:
        """Default transport is stdio, default timeout is 30."""
        cfg_file = tmp_path / "mcp.json"
        cfg_file.write_text(json.dumps({"mcpServers": {"s": {"command": "cmd"}}}))
        configs = load_mcp_config(cfg_file)
        assert configs[0].transport.value == "stdio"
        assert configs[0].timeout == 30.0


class TestLoadMCPClient:
    """Tests for the convenience client loader."""

    def test_creates_client_with_servers(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "mcp.json"
        cfg_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "s1": {"transport": "stdio", "command": "cmd1"},
                        "s2": {"transport": "stdio", "command": "cmd2"},
                    }
                }
            )
        )
        client = load_mcp_client(cfg_file)
        assert set(client.server_names) == {"s1", "s2"}


# ===========================================================================
# Integration: retry + config
# ===========================================================================


class TestIntegration:
    """Integration tests for the execution module."""

    async def test_retry_with_mock_client_end_to_end(self) -> None:
        """Full lifecycle: create client mock, call with retry, succeed on 2nd try."""
        result = MagicMock(content=["success"])
        client = _mock_client(errors=[RuntimeError("transient"), None], results=[None, result])
        got = await call_tool_with_retry(
            client,
            "server",
            "tool_a",
            {"input": "test"},
            max_retries=2,
            backoff_base=0.0,
        )
        assert got is result
        assert client.call_tool.await_count == 2

    def test_config_to_client_end_to_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Load config with env vars, create client, verify server configs."""
        monkeypatch.setenv("SERVER_CMD", "/usr/local/bin/serve")
        monkeypatch.setenv("SERVER_URL", "http://api.example.com")
        cfg_file = tmp_path / "mcp.json"
        cfg_file.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "local": {
                            "transport": "stdio",
                            "command": "${SERVER_CMD}",
                            "args": ["--port", "8080"],
                        },
                        "remote": {
                            "transport": "sse",
                            "url": "${SERVER_URL}/mcp",
                        },
                    }
                }
            )
        )
        client = load_mcp_client(cfg_file)
        assert set(client.server_names) == {"local", "remote"}

    async def test_execution_error_is_mcp_client_error(self) -> None:
        """MCPExecutionError is a subclass of MCPClientError."""
        assert issubclass(MCPExecutionError, MCPClientError)
