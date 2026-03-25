# exo.mcp.execution

MCP execution utilities -- retry logic, config loading, and environment variable substitution.

```python
from exo.mcp.execution import (
    MCPExecutionError,
    call_tool_with_retry,
    load_mcp_client,
    load_mcp_config,
    substitute_env_vars,
)
```

---

## MCPExecutionError

```python
class MCPExecutionError(MCPClientError)
```

Error raised during MCP tool execution with retries. Inherits from `MCPClientError`.

---

## substitute_env_vars

```python
def substitute_env_vars(value: str) -> str
```

Replace `${VAR_NAME}` placeholders with environment variable values. Unset variables are replaced with empty strings.

| Name | Type | Default | Description |
|---|---|---|---|
| `value` | `str` | *(required)* | String potentially containing `${VAR_NAME}` placeholders |

**Returns:** `str` -- String with placeholders replaced by environment variable values.

### Example

```python
import os
os.environ["API_KEY"] = "secret123"

result = substitute_env_vars("Bearer ${API_KEY}")
# result == "Bearer secret123"

result = substitute_env_vars("${UNSET_VAR}")
# result == ""
```

---

## load_mcp_config

```python
def load_mcp_config(path: str | Path) -> list[MCPServerConfig]
```

Load MCP server configurations from an `mcp.json` file. Environment variables in `${VAR}` format are substituted in all string values. The file must contain a JSON object with a `"mcpServers"` key mapping server names to configuration objects.

| Name | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| Path` | *(required)* | Path to the mcp.json file |

**Returns:** `list[MCPServerConfig]` -- List of server configurations.

**Raises:** `MCPExecutionError` -- If the file cannot be read, parsed, or has an invalid structure.

### Expected mcp.json format

```json
{
    "mcpServers": {
        "my-server": {
            "transport": "stdio",
            "command": "${PYTHON_PATH}",
            "args": ["-m", "my_module"]
        },
        "remote-server": {
            "transport": "sse",
            "url": "https://example.com/mcp"
        }
    }
}
```

### Example

```python
configs = load_mcp_config("mcp.json")
for cfg in configs:
    print(f"Server: {cfg.name}, Transport: {cfg.transport}")
```

---

## load_mcp_client

```python
def load_mcp_client(path: str | Path) -> MCPClient
```

Create an `MCPClient` from an `mcp.json` config file. Convenience wrapper around `load_mcp_config`.

| Name | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| Path` | *(required)* | Path to the mcp.json file |

**Returns:** `MCPClient` -- Client with all servers from the config registered (not yet connected).

### Example

```python
client = load_mcp_client("mcp.json")

async with client:
    tools = await client.list_tools("my-server")
```

---

## call_tool_with_retry

```python
async def call_tool_with_retry(
    client: Any,
    server_name: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    max_retries: int = 3,
    timeout: float | None = None,
    backoff_base: float = 1.0,
) -> Any
```

Call an MCP tool with retry logic and optional timeout. Retries on transient failures with exponential backoff (`backoff_base * 2^attempt`). Timeout wraps each individual attempt, not the total call. `MCPClientError` exceptions are re-raised immediately without retry.

| Name | Type | Default | Description |
|---|---|---|---|
| `client` | `Any` | *(required)* | An MCPClient or MCPServerConnection with `call_tool()` |
| `server_name` | `str` | *(required)* | Server name (used for MCPClient routing; ignored for connections) |
| `tool_name` | `str` | *(required)* | Name of the tool to call |
| `arguments` | `dict[str, Any] \| None` | `None` | Tool arguments |
| `max_retries` | `int` | `3` | Maximum number of retry attempts (0 = no retries, just one attempt) |
| `timeout` | `float \| None` | `None` | Per-attempt timeout in seconds. None = no timeout |
| `backoff_base` | `float` | `1.0` | Base delay in seconds for exponential backoff |

**Returns:** `CallToolResult` from the MCP server.

**Raises:** `MCPExecutionError` -- After all retries are exhausted or on non-retryable errors.

### Example

```python
result = await call_tool_with_retry(
    client, "my-server", "search",
    {"query": "hello"},
    max_retries=2,
    timeout=10.0,
)
```
