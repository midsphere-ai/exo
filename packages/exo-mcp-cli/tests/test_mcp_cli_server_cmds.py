"""Tests for exo_mcp_cli.commands.server — server management CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from exo_mcp_cli.main import app
from exo_mcp_cli.vault import Vault

runner = CliRunner()


# ---------------------------------------------------------------------------
# server list
# ---------------------------------------------------------------------------


class TestServerList:
    def test_no_config_shows_no_servers(self, tmp_path: Path) -> None:
        """When the config file does not exist, shows 'No servers configured'."""
        cfg = tmp_path / "mcp.json"
        # The file does not exist — find_config raises MCPConfigError for
        # an explicit path that is missing.  However resolve_config catches
        # that and exits 1.  We need a file that is valid but has no servers.
        cfg.write_text('{"mcpServers": {}}')
        result = runner.invoke(app, ["--config", str(cfg), "server", "list"])
        assert result.exit_code == 0
        assert "No servers configured" in result.output

    def test_list_shows_server_names(self, tmp_path: Path) -> None:
        cfg = tmp_path / "mcp.json"
        cfg.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "my-db": {
                            "transport": "stdio",
                            "command": "python",
                            "args": ["-m", "db_server"],
                        },
                        "web-search": {
                            "transport": "sse",
                            "url": "http://localhost:8080/sse",
                        },
                    }
                }
            )
        )
        result = runner.invoke(app, ["--config", str(cfg), "server", "list"])
        assert result.exit_code == 0
        assert "my-db" in result.output
        assert "web-search" in result.output

    def test_missing_explicit_config_exits_1(self, tmp_path: Path) -> None:
        """If --config points to a nonexistent file, exit 1 with error."""
        cfg = tmp_path / "nope.json"
        result = runner.invoke(app, ["--config", str(cfg), "server", "list"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "Error" in result.output


# ---------------------------------------------------------------------------
# server add
# ---------------------------------------------------------------------------


class TestServerAdd:
    def test_add_creates_config(self, tmp_path: Path) -> None:
        cfg = tmp_path / "mcp.json"
        result = runner.invoke(
            app,
            [
                "--config",
                str(cfg),
                "server",
                "add",
                "myserver",
                "--transport",
                "stdio",
                "--command",
                "python",
                "--arg",
                "-m",
                "--arg",
                "my_server",
            ],
        )
        assert result.exit_code == 0
        assert "myserver" in result.output

        # Verify file contents
        data = json.loads(cfg.read_text())
        assert "myserver" in data["mcpServers"]
        entry = data["mcpServers"]["myserver"]
        assert entry["transport"] == "stdio"
        assert entry["command"] == "python"
        assert entry["args"] == ["-m", "my_server"]

    def test_add_with_header_auto_vaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--header values are auto-encrypted and stored as ${vault:...} refs."""
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-pass")
        cfg = tmp_path / "mcp.json"
        test_vault = Vault(vault_path=tmp_path / "vault.enc")

        with patch("exo_mcp_cli.main.Vault", return_value=test_vault):
            result = runner.invoke(
                app,
                [
                    "--config",
                    str(cfg),
                    "server",
                    "add",
                    "api-srv",
                    "--transport",
                    "sse",
                    "--url",
                    "http://localhost:9000/sse",
                    "--header",
                    "Authorization=Bearer sk-secret123",
                ],
            )

        assert result.exit_code == 0
        data = json.loads(cfg.read_text())
        headers = data["mcpServers"]["api-srv"].get("headers", {})
        # The raw value must NOT appear — only a vault reference
        assert "sk-secret123" not in json.dumps(data)
        assert "${vault:" in headers.get("Authorization", "")

        # Verify the vault actually stored the value
        assert test_vault.get("api-srv_header_Authorization") == "Bearer sk-secret123"

    def test_add_stdio_requires_command(self, tmp_path: Path) -> None:
        """stdio transport without --command exits 1."""
        cfg = tmp_path / "mcp.json"
        result = runner.invoke(
            app,
            [
                "--config",
                str(cfg),
                "server",
                "add",
                "bad",
                "--transport",
                "stdio",
            ],
        )
        assert result.exit_code == 1
        assert "command" in result.output.lower()

    def test_add_sse_requires_url(self, tmp_path: Path) -> None:
        """sse transport without --url exits 1."""
        cfg = tmp_path / "mcp.json"
        result = runner.invoke(
            app,
            [
                "--config",
                str(cfg),
                "server",
                "add",
                "bad",
                "--transport",
                "sse",
            ],
        )
        assert result.exit_code == 1
        assert "url" in result.output.lower()

    def test_add_with_env_auto_vaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--env values are auto-encrypted and stored as ${vault:...} refs."""
        monkeypatch.setenv("EXO_MCP_VAULT_KEY", "test-pass")
        cfg = tmp_path / "mcp.json"
        test_vault = Vault(vault_path=tmp_path / "vault.enc")

        with patch("exo_mcp_cli.main.Vault", return_value=test_vault):
            result = runner.invoke(
                app,
                [
                    "--config",
                    str(cfg),
                    "server",
                    "add",
                    "env-srv",
                    "--transport",
                    "stdio",
                    "--command",
                    "node",
                    "--env",
                    "API_KEY=sk-secret456",
                ],
            )

        assert result.exit_code == 0
        data = json.loads(cfg.read_text())
        env_block = data["mcpServers"]["env-srv"].get("env", {})
        assert "sk-secret456" not in json.dumps(data)
        assert "${vault:" in env_block.get("API_KEY", "")
        assert test_vault.get("env-srv_env_API_KEY") == "sk-secret456"

    def test_add_replaces_existing_server(self, tmp_path: Path) -> None:
        """Adding a server with an existing name replaces it."""
        cfg = tmp_path / "mcp.json"
        cfg.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "demo": {"transport": "stdio", "command": "old-cmd"},
                    }
                }
            )
        )
        result = runner.invoke(
            app,
            [
                "--config",
                str(cfg),
                "server",
                "add",
                "demo",
                "--transport",
                "stdio",
                "--command",
                "new-cmd",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(cfg.read_text())
        assert data["mcpServers"]["demo"]["command"] == "new-cmd"


# ---------------------------------------------------------------------------
# server remove
# ---------------------------------------------------------------------------


class TestServerRemove:
    def test_remove_existing_server(self, tmp_path: Path) -> None:
        cfg = tmp_path / "mcp.json"
        cfg.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "deleteme": {"transport": "stdio", "command": "echo"},
                        "keepme": {"transport": "stdio", "command": "echo"},
                    }
                }
            )
        )
        result = runner.invoke(
            app,
            [
                "--config",
                str(cfg),
                "server",
                "remove",
                "deleteme",
            ],
        )
        assert result.exit_code == 0
        assert "deleteme" in result.output
        assert "removed" in result.output.lower()

        data = json.loads(cfg.read_text())
        assert "deleteme" not in data["mcpServers"]
        assert "keepme" in data["mcpServers"]

    def test_remove_nonexistent_exits_1(self, tmp_path: Path) -> None:
        cfg = tmp_path / "mcp.json"
        cfg.write_text('{"mcpServers": {"real": {"transport": "stdio", "command": "echo"}}}')
        result = runner.invoke(
            app,
            [
                "--config",
                str(cfg),
                "server",
                "remove",
                "ghost",
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_remove_no_config_exits_1(self, tmp_path: Path) -> None:
        """If the config file does not exist at the explicit path, exit 1."""
        cfg = tmp_path / "nope.json"
        result = runner.invoke(
            app,
            [
                "--config",
                str(cfg),
                "server",
                "remove",
                "anything",
            ],
        )
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# server test (just check --help works; real connectivity can't be tested)
# ---------------------------------------------------------------------------


class TestServerTestHelp:
    def test_server_test_help_shows_options(self) -> None:
        result = runner.invoke(app, ["server", "test", "--help"])
        assert result.exit_code == 0
        assert "server name" in result.output.lower() or "SERVER" in result.output
