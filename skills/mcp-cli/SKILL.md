---
name: exo:mcp-cli
description: "Use when building or using the exo-mcp standalone CLI for MCP server interaction — exo-mcp commands, mcp.json config, encrypted vault, ServerEntry, connect_to_server, Vault, credential management, server add/remove/test, tool list/call, resource list/read, prompt list/get, auth set/list/remove, ${vault:NAME} references, auto-vault headers, PyInstaller binary. Triggers on: exo-mcp, mcp-cli, mcp cli, exo-mcp-cli, mcp.json, vault, credential vault, server add, tool call, resource read, prompt get, auth set, EXO_MCP_VAULT_KEY, ${vault:, connect_to_server, ServerEntry, standalone mcp, mcp binary."
---

# Exo MCP CLI — Standalone MCP Server Interaction Tool

## When To Use This Skill

Use this skill when the developer needs to:
- Use the `exo-mcp` CLI to interact with MCP servers (list/call tools, read resources, get prompts)
- Configure MCP servers via `mcp.json` with the `exo-mcp server add` command
- Store API keys and auth tokens securely using the encrypted vault
- Understand the `${vault:NAME}` credential reference system
- Connect to MCP servers over stdio, SSE, streamable HTTP, or WebSocket transports
- Extend or modify the `exo-mcp-cli` package code
- Build a standalone binary of `exo-mcp` via PyInstaller

## Decision Guide

1. **Need to interact with an MCP server from CLI?** → Use `exo-mcp tool list/call`, `resource list/read`, or `prompt list/get`
2. **Need to configure a server connection?** → Use `exo-mcp server add NAME --transport TYPE`
3. **Need to store API keys securely?** → Use `exo-mcp auth set NAME VALUE` or `--header` on `server add` (auto-vaults)
4. **Need to pass secrets in mcp.json?** → Use `${vault:NAME}` references (resolved at connection time, never stored in plain text)
5. **Need environment-variable-based secrets?** → Use `${ENV_VAR}` syntax in mcp.json (existing MCP convention)
6. **Need to test if a server is reachable?** → Use `exo-mcp server test NAME` (connects, pings, reports latency)
7. **Need JSON output for scripting?** → Add `--json` to `tool list`, `resource list`, `prompt list`, or `--raw` to `tool call`
8. **Need a standalone binary?** → `cd packages/exo-mcp-cli && pyinstaller exo_mcp.spec --clean`
9. **Need to use programmatically (not CLI)?** → Import from `exo_mcp_cli.config`, `exo_mcp_cli.connection`, `exo_mcp_cli.vault`

## Reference

### Package Structure

```
packages/exo-mcp-cli/
├── pyproject.toml              # Entry point: exo-mcp = "exo_mcp_cli.main:app"
├── exo_mcp.spec                # PyInstaller spec for standalone binary
└── src/exo_mcp_cli/
    ├── main.py                 # Typer app, global opts, subcommand registration
    ├── config.py               # mcp.json loading, ServerEntry, env var substitution
    ├── connection.py           # Async context manager for MCP sessions
    ├── vault.py                # Fernet-encrypted credential vault (PBKDF2)
    ├── output.py               # Rich formatting (tables, JSON, errors)
    └── commands/
        ├── server.py           # server list/add/remove/test
        ├── tool.py             # tool list/call
        ├── resource.py         # resource list/read
        ├── prompt.py           # prompt list/get
        └── auth.py             # auth set/list/remove
```

**Dependencies (zero exo deps):** `mcp>=1.0`, `typer>=0.12`, `rich>=13.0`, `cryptography>=42.0`

### CLI Command Reference

```bash
# Global options
exo-mcp [--config PATH] [--verbose]

# Server management
exo-mcp server list                                    # List configured servers
exo-mcp server add NAME --transport stdio --command CMD [--arg ARG...]  # Add stdio server
exo-mcp server add NAME --transport sse --url URL [--header K=V...]    # Add SSE server (auto-vaults headers)
exo-mcp server remove NAME                             # Remove a server
exo-mcp server test NAME                               # Ping test + latency report

# Tool operations
exo-mcp tool list SERVER [--json]                      # List tools with schemas
exo-mcp tool call SERVER TOOL --arg key=value          # Call with key=value args
exo-mcp tool call SERVER TOOL --json '{"k": "v"}'     # Call with JSON args
exo-mcp tool call SERVER TOOL --inject k=v             # Injected args (auto-filled, LLM doesn't specify)
exo-mcp tool call SERVER TOOL --raw                    # Output raw JSON result

# Resource operations
exo-mcp resource list SERVER [--json]                  # List resources
exo-mcp resource read SERVER URI [--output FILE]       # Read a resource

# Prompt operations
exo-mcp prompt list SERVER [--json]                    # List prompts
exo-mcp prompt get SERVER NAME [--arg k=v] [--json]   # Get a prompt

# Credential vault
exo-mcp auth set NAME VALUE                            # Store a secret
exo-mcp auth list                                      # List secret names (not values)
exo-mcp auth remove NAME                               # Remove a secret
```

### Config File Format (mcp.json)

Config file search order: explicit `--config` → `./mcp.json` → `~/.exo-mcp/mcp.json`

```json
{
  "mcpServers": {
    "local-server": {
      "transport": "stdio",
      "command": "python",
      "args": ["-m", "my_mcp_server"],
      "env": {"API_KEY": "${vault:my-api-key}"}
    },
    "remote-server": {
      "transport": "sse",
      "url": "https://api.example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${vault:remote-token}"
      },
      "timeout": 60.0
    }
  }
}
```

**Supported transports:** `stdio`, `sse`, `streamable_http`, `websocket`

**Variable substitution (resolved at connection time):**
- `${ENV_VAR}` — replaced with environment variable value
- `${vault:NAME}` — replaced with decrypted vault secret

### ServerEntry Dataclass

```python
from exo_mcp_cli.config import ServerEntry

entry = ServerEntry(
    name="my-server",           # Server identifier
    transport="stdio",          # stdio | sse | streamable_http | websocket
    command="python",           # Executable (stdio only)
    args=["-m", "server"],      # Command args (stdio only)
    env={"KEY": "val"},         # Environment vars (stdio only)
    cwd="/path",                # Working directory (stdio only)
    url="https://...",          # Server URL (sse/http/ws only)
    headers={"Auth": "..."},    # HTTP headers (sse/http only)
    timeout=30.0,               # Connection timeout seconds
)
entry.validate()                # Raises MCPConfigError if invalid
entry.to_dict()                 # Serialize for mcp.json
```

### Connection Manager

```python
from exo_mcp_cli.connection import connect_to_server
from exo_mcp_cli.config import ServerEntry
from exo_mcp_cli.vault import Vault

entry = ServerEntry(name="test", transport="stdio", command="python", args=["-m", "server"])
vault = Vault()  # Optional, for ${vault:...} resolution

async with connect_to_server(entry, vault) as session:
    # session is a mcp.ClientSession — all MCP operations available
    tools = await session.list_tools()
    result = await session.call_tool("tool_name", {"arg": "value"})
    resources = await session.list_resources()
    prompts = await session.list_prompts()
    await session.send_ping()
```

### Encrypted Vault

```python
from exo_mcp_cli.vault import Vault

vault = Vault()  # Default: ~/.exo-mcp/credentials.vault
# Or custom path:
vault = Vault(vault_path=Path("/custom/vault.enc"))

# Passphrase: EXO_MCP_VAULT_KEY env var, or interactive prompt

vault.set("api-key", "sk-abc123")      # Store
vault.get("api-key")                    # → "sk-abc123"
vault.has("api-key")                    # → True
vault.list_names()                      # → ["api-key"]
vault.remove("api-key")                # → True
vault.resolve("Bearer ${vault:api-key}")  # → "Bearer sk-abc123"
```

**Vault file:** `~/.exo-mcp/credentials.vault` — Fernet-encrypted JSON, keyed with PBKDF2-HMAC-SHA256 (480k iterations)

**Vault file layout:** `<16-byte salt><Fernet ciphertext>`

### Auto-Vault on Server Add

When `server add` receives `--header` or `--env` values, they are automatically stored in the encrypted vault. The mcp.json file only contains `${vault:...}` references:

```bash
exo-mcp server add my-api --transport sse --url https://api.example.com \
    --header "Authorization=Bearer sk-secret123"
```

Results in mcp.json:
```json
{
  "mcpServers": {
    "my-api": {
      "transport": "sse",
      "url": "https://api.example.com",
      "headers": {
        "Authorization": "${vault:my-api_header_Authorization}"
      }
    }
  }
}
```

The raw value `sk-secret123` is stored only in the encrypted vault file.

### Injected Arguments (`--inject` / `-i` and `EXO_MCP_TOOL_INJECT`)

The CLI equivalent of `injected_tool_args` from the Exo agent runtime. These arguments are merged into every tool call without being explicitly specified each time — useful for API keys, trace IDs, user context, etc.

**Precedence (lowest → highest):** `EXO_MCP_TOOL_INJECT` → `--inject` → `--json` → `--arg`

```bash
# 1. Environment variable (set once, always active — JSON object)
export EXO_MCP_TOOL_INJECT='{"api_key": "sk-123", "user_id": "u42"}'
exo-mcp tool call my-server search --arg query=hello
# → calls with {"query": "hello", "api_key": "sk-123", "user_id": "u42"}

# 2. CLI flag (per-call)
exo-mcp tool call my-server search --arg query=hello -i trace_id=abc -i user_id=u42

# 3. Both (flag overrides env for overlapping keys)
export EXO_MCP_TOOL_INJECT='{"user_id": "env-default"}'
exo-mcp tool call my-server search --arg query=hello -i user_id=override
# → user_id="override" (flag wins)

# 4. --arg overrides everything
exo-mcp tool call my-server search -i key=injected --arg key=explicit
# → key="explicit" (--arg wins)
```

**Malformed `EXO_MCP_TOOL_INJECT`** (not valid JSON) is silently ignored.

## Patterns

### Pattern 1: Quick Tool Execution

```bash
# Set up vault passphrase
export EXO_MCP_VAULT_KEY="my-secure-passphrase"

# Add a server
exo-mcp server add my-server --transport stdio --command python --arg -m --arg my_mcp_server

# List and call tools
exo-mcp tool list my-server
exo-mcp tool call my-server search --arg query="hello world"
```

### Pattern 2: Remote Server with Auth

```bash
export EXO_MCP_VAULT_KEY="my-passphrase"

# Add with auto-vaulted credentials
exo-mcp server add openai-mcp --transport sse \
    --url https://mcp.openai.com/v1 \
    --header "Authorization=Bearer sk-proj-xxx"

# Verify connection
exo-mcp server test openai-mcp

# Use
exo-mcp tool list openai-mcp --json
exo-mcp tool call openai-mcp generate --json '{"prompt": "hello"}'
```

### Pattern 3: Scripting with JSON Output

```bash
# Get tool schemas for automation
tools=$(exo-mcp tool list my-server --json)

# Call tool and capture raw result
result=$(exo-mcp tool call my-server process --arg input=data --raw)

# Parse with jq
echo "$result" | jq '.content[0].text'
```

### Pattern 4: Programmatic Usage (Python)

```python
import asyncio
from exo_mcp_cli.config import ServerEntry
from exo_mcp_cli.connection import connect_to_server
from exo_mcp_cli.vault import Vault

async def main():
    entry = ServerEntry(
        name="local",
        transport="stdio",
        command="python",
        args=["-m", "my_server"],
    )
    vault = Vault()

    async with connect_to_server(entry, vault) as session:
        result = await session.list_tools()
        for tool in result.tools:
            print(f"{tool.name}: {tool.description}")

        output = await session.call_tool("greet", {"name": "World"})
        for item in output.content:
            print(item.text)

asyncio.run(main())
```

### Pattern 5: Agent Tool Offloading with Inject

```bash
# Set up once — agent bootstrap
export EXO_MCP_TOOL_INJECT='{"user_id": "u42", "session_token": "tok_abc"}'
export EXO_MCP_VAULT_KEY="my-passphrase"

# Agent calls tools via bash — injected args auto-merged
exo-mcp tool call my-server search --arg query="python docs"
# → MCP server receives: {"query": "python docs", "user_id": "u42", "session_token": "tok_abc"}

exo-mcp tool call my-server write_file --json '{"path": "/tmp/out.txt", "content": "hello"}'
# → MCP server receives: {"path": "/tmp/out.txt", "content": "hello", "user_id": "u42", "session_token": "tok_abc"}
```

### Pattern 6: Building Standalone Binary

```bash
cd packages/exo-mcp-cli

# Install PyInstaller (one-time)
pip install pyinstaller

# Build single-file binary
pyinstaller exo_mcp.spec --clean

# Result: dist/exo-mcp (standalone, no Python needed)
./dist/exo-mcp --help
```

## Gotchas

- **Vault passphrase is required** for any operation that touches credentials. Set `EXO_MCP_VAULT_KEY` env var for non-interactive use (CI/CD, scripts).
- **mcp.json is NOT the same as Claude Desktop's config** — same format (`mcpServers` key), but exo-mcp stores its own config independently.
- **`${vault:...}` resolution happens at connection time**, not at config load. This means `server list` shows the raw `${vault:...}` references, not the decrypted values.
- **`${ENV_VAR}` substitution happens at config load** (before vault resolution). Unset vars become empty strings.
- **Wrong vault passphrase** raises `VaultError` — the vault doesn't store the passphrase itself, so there's no way to check correctness until decryption fails.
- **Auto-vault key naming** follows `{server}_{type}_{key}` convention (e.g., `my-api_header_Authorization`). You can reference these directly in other configs.
- **Transports have different requirements**: stdio needs `--command`, sse/streamable_http/websocket need `--url`. Validation happens at `server add` and at connection time.
- **The `exo-mcp` binary is fully independent** — it does NOT depend on exo-core, exo-mcp, or any other workspace package. It uses the `mcp` library directly.
- **`--arg` in `server add` is for command arguments** (stdio process args), while `--arg` in `tool call` is for tool parameters (key=value pairs). Different semantics, same flag name.
- **Server test** does three things: connect, ping, list tools — to verify full server functionality, not just TCP reachability.
- **`--inject` vs `--arg`:** `--inject` has lower precedence than `--json` and `--arg`. Use `--inject` for baseline values the LLM shouldn't override; use `--arg` for explicit per-call values.
- **`EXO_MCP_TOOL_INJECT` must be valid JSON** — malformed values are silently ignored (no error, no args injected). Use an object: `'{"k": "v"}'`.
- **`EXO_MCP_TOOL_INJECT` vs `EXO_MCP_VAULT_KEY`:** Different env vars for different purposes. INJECT provides tool arguments; VAULT_KEY unlocks the credential store.
- **File paths in tests** must use `--config` pointing to a `tmp_path` mcp.json, and vault must use `Vault(vault_path=tmp_path / "v.enc")` to avoid touching the real vault.
