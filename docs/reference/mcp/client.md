# exo.mcp.client

MCP client with multiple transport support and server instance caching.

```python
from exo.mcp.client import (
    MCPClient,
    MCPClientError,
    MCPServerConfig,
    MCPServerConnection,
    MCPTransport,
)
```

---

## MCPTransport

```python
class MCPTransport(StrEnum)
```

Transport types for MCP server connections.

| Value | Description |
|---|---|
| `STDIO = "stdio"` | Subprocess stdio transport |
| `SSE = "sse"` | Server-Sent Events over HTTP |
| `STREAMABLE_HTTP = "streamable_http"` | Streamable HTTP transport |

---

## MCPClientError

```python
class MCPClientError(Exception)
```

Error raised by MCP client operations.

---

## MCPServerConfig

```python
class MCPServerConfig(
    name: str,
    transport: MCPTransport | str = MCPTransport.STDIO,
    *,
    command: str | None = None,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    sse_read_timeout: float = 300.0,
    cache_tools: bool = False,
    session_timeout: float | None = 120.0,
)
```

Configuration for an MCP server connection.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | Human-readable server name |
| `transport` | `MCPTransport \| str` | `MCPTransport.STDIO` | Transport type (stdio, sse, streamable_http) |
| `command` | `str \| None` | `None` | Executable for stdio transport |
| `args` | `list[str] \| None` | `None` | Command-line arguments for stdio transport |
| `env` | `dict[str, str] \| None` | `None` | Environment variables for stdio transport |
| `cwd` | `str \| None` | `None` | Working directory for stdio transport |
| `url` | `str \| None` | `None` | Server URL for sse/streamable_http transports |
| `headers` | `dict[str, str] \| None` | `None` | HTTP headers for sse/streamable_http transports |
| `timeout` | `float` | `30.0` | Connection timeout in seconds |
| `sse_read_timeout` | `float` | `300.0` | SSE read timeout in seconds |
| `cache_tools` | `bool` | `False` | Whether to cache the tools list |
| `session_timeout` | `float \| None` | `120.0` | ClientSession read timeout in seconds |

### Methods

#### validate

```python
def validate(self) -> None
```

Validate config fields for the chosen transport. Raises `MCPClientError` if the config is invalid (e.g. stdio without `command`, sse without `url`).

### Example

```python
config = MCPServerConfig(
    name="my-server",
    transport="stdio",
    command="python",
    args=["-m", "my_mcp_server"],
)
config.validate()  # raises if invalid
```

---

## MCPServerConnection

```python
class MCPServerConnection(config: MCPServerConfig)
```

A live connection to an MCP server. Manages the async context stack (transport + session) and provides `list_tools()` and `call_tool()` methods.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `config` | `MCPServerConfig` | *(required)* | Configuration for the server connection |

### Properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | The server name from the config |
| `config` | `MCPServerConfig` | The underlying config object |
| `session` | `ClientSession \| None` | The MCP client session (None if not connected) |
| `init_result` | `InitializeResult \| None` | The initialization result from the server |
| `is_connected` | `bool` | Whether the session is active |

### Methods

#### connect

```python
async def connect(self) -> None
```

Open transport and initialize the MCP session. No-op if already connected.

**Raises:** `MCPClientError` -- If connection fails.

#### list_tools

```python
async def list_tools(self) -> list[MCPTool]
```

List available tools from the server. Uses cache if `cache_tools` is enabled and the cache is clean.

**Returns:** List of MCP tool definitions.

**Raises:** `MCPClientError` -- If the server is not connected.

#### invalidate_tools_cache

```python
def invalidate_tools_cache(self) -> None
```

Mark the tools cache as dirty so the next `list_tools()` re-fetches.

#### call_tool

```python
async def call_tool(
    self,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> CallToolResult
```

Invoke a tool on the server.

| Name | Type | Default | Description |
|---|---|---|---|
| `tool_name` | `str` | *(required)* | Name of the tool to call |
| `arguments` | `dict[str, Any] \| None` | `None` | Tool arguments |

**Returns:** `CallToolResult` from the MCP server.

**Raises:** `MCPClientError` -- If the server is not connected.

#### cleanup

```python
async def cleanup(self) -> None
```

Close the transport and session.

### Async context manager

`MCPServerConnection` supports `async with`:

```python
async with MCPServerConnection(config) as conn:
    tools = await conn.list_tools()
```

---

## MCPClient

```python
class MCPClient()
```

High-level MCP client managing multiple server connections. Provides server instance caching/reuse with session isolation. Servers are identified by name and cached after first connection.

### Methods

#### add_server

```python
def add_server(self, config: MCPServerConfig) -> MCPClient
```

Register a server configuration. Returns self for method chaining.

#### remove_server

```python
def remove_server(self, name: str) -> None
```

Remove a server configuration (does not disconnect an active connection).

#### connect

```python
async def connect(self, name: str) -> MCPServerConnection
```

Connect to a specific server. Re-uses a cached connection if alive.

**Raises:** `MCPClientError` -- If no server is registered with that name.

#### connect_all

```python
async def connect_all(self) -> None
```

Connect to all registered servers.

#### disconnect

```python
async def disconnect(self, name: str) -> None
```

Disconnect a specific server and remove the cached connection.

#### disconnect_all

```python
async def disconnect_all(self) -> None
```

Disconnect all servers.

#### list_tools

```python
async def list_tools(self, name: str) -> list[MCPTool]
```

List tools from a specific server (connects if needed).

#### call_tool

```python
async def call_tool(
    self,
    server_name: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> CallToolResult
```

Call a tool on a specific server (connects if needed).

| Name | Type | Default | Description |
|---|---|---|---|
| `server_name` | `str` | *(required)* | Name of the registered server |
| `tool_name` | `str` | *(required)* | Name of the tool to call |
| `arguments` | `dict[str, Any] \| None` | `None` | Tool arguments |

#### get_connection

```python
def get_connection(self, name: str) -> MCPServerConnection | None
```

Get a cached connection by server name, or None.

### Properties

| Property | Type | Description |
|---|---|---|
| `server_names` | `list[str]` | Names of all registered servers |

### Async context manager

`MCPClient` supports `async with`, which calls `connect_all()` on entry and `disconnect_all()` on exit:

```python
client = MCPClient()
client.add_server(MCPServerConfig(
    name="my-server",
    transport="stdio",
    command="python",
    args=["-m", "my_mcp_server"],
))

async with client:
    tools = await client.list_tools("my-server")
    result = await client.call_tool("my-server", "tool_name", {"arg": "val"})
```
