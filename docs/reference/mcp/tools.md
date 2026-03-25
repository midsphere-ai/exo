# exo.mcp.tools

MCP tool schema extraction, conversion to Exo Tool format, filtering, and namespacing.

```python
from exo.mcp.tools import (
    MCPToolError,
    MCPToolFilter,
    MCPToolWrapper,
    convert_mcp_tools,
    extract_schema,
    load_tools_from_client,
    load_tools_from_connection,
    namespace_tool_name,
    parse_namespaced_name,
)
```

---

## MCPToolError

```python
class MCPToolError(ToolError)
```

Error raised during MCP tool operations. Inherits from `exo.tool.ToolError`.

---

## MCPToolFilter

```python
class MCPToolFilter(
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
)
```

Filter for including/excluding MCP tools by name.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `include` | `list[str] \| None` | `None` | Whitelist of tool names (if non-empty, only these are included) |
| `exclude` | `list[str] \| None` | `None` | Blacklist of tool names (always excluded, takes priority over include) |

### Methods

#### accepts

```python
def accepts(self, name: str) -> bool
```

Check if a tool name passes the filter.

#### apply

```python
def apply(self, tools: list[MCPTool]) -> list[MCPTool]
```

Filter a list of MCP tools, returning only accepted ones.

### Example

```python
f = MCPToolFilter(include=["search", "read"], exclude=["read"])
f.accepts("search")  # True
f.accepts("read")    # False (exclude takes priority)
f.accepts("write")   # False (not in include list)
```

---

## namespace_tool_name

```python
def namespace_tool_name(
    tool_name: str,
    server_name: str,
    *,
    namespace: str = "mcp",
) -> str
```

Create a namespaced tool name: `{namespace}__{server}__{tool}`. Non-alphanumeric characters in server and tool names are replaced by underscores.

| Name | Type | Default | Description |
|---|---|---|---|
| `tool_name` | `str` | *(required)* | Original tool name |
| `server_name` | `str` | *(required)* | MCP server name |
| `namespace` | `str` | `"mcp"` | Namespace prefix |

**Returns:** `str` -- Namespaced tool name.

```python
namespace_tool_name("search", "my-server")
# "mcp__my_server__search"
```

---

## parse_namespaced_name

```python
def parse_namespaced_name(namespaced: str) -> tuple[str, str, str]
```

Parse a namespaced tool name back into `(namespace, server_name, tool_name)`.

| Name | Type | Default | Description |
|---|---|---|---|
| `namespaced` | `str` | *(required)* | A name like `"mcp__server__tool"` |

**Returns:** `tuple[str, str, str]` -- Tuple of (namespace, server_name, tool_name).

**Raises:** `MCPToolError` -- If the name doesn't match the expected `namespace__server__tool` format.

```python
ns, server, tool = parse_namespaced_name("mcp__my_server__search")
# ns="mcp", server="my_server", tool="search"
```

---

## extract_schema

```python
def extract_schema(mcp_tool: MCPTool) -> dict[str, Any]
```

Extract the JSON Schema parameters from an MCP tool definition.

| Name | Type | Default | Description |
|---|---|---|---|
| `mcp_tool` | `MCPTool` | *(required)* | An MCP tool definition |

**Returns:** `dict[str, Any]` -- JSON Schema parameters dict (`type: "object"`, `properties`, etc.). Returns a fallback empty schema if no `inputSchema` is present.

---

## MCPToolWrapper

```python
class MCPToolWrapper(
    mcp_tool: MCPTool,
    server_name: str,
    call_fn: Any,
    *,
    namespace: str = "mcp",
)
```

An Exo `Tool` that wraps an MCP tool for execution. The tool delegates execution to an MCP server connection's `call_tool` method. Schema is extracted from the MCP tool definition.

Inherits from `exo.tool.Tool`.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `mcp_tool` | `MCPTool` | *(required)* | The MCP tool definition |
| `server_name` | `str` | *(required)* | Name of the MCP server providing this tool |
| `call_fn` | `Any` | *(required)* | Async callable that invokes the tool on the server |
| `namespace` | `str` | `"mcp"` | Namespace prefix for the tool name |

### Properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | Namespaced tool name (e.g. `mcp__server__tool`) |
| `description` | `str` | Tool description from the MCP tool definition |
| `parameters` | `dict[str, Any]` | JSON Schema extracted from the MCP tool |
| `original_name` | `str` | The original (un-namespaced) tool name from the MCP server |
| `server_name` | `str` | The MCP server providing this tool |

### Methods

#### execute

```python
async def execute(self, **kwargs: Any) -> str | dict[str, Any]
```

Execute the MCP tool via the server connection.

**Returns:** `str` -- String result from the MCP tool.

**Raises:** `MCPToolError` -- If execution fails or the result indicates an error.

---

## convert_mcp_tools

```python
def convert_mcp_tools(
    mcp_tools: list[MCPTool],
    server_name: str,
    call_fn: Any,
    *,
    namespace: str = "mcp",
    tool_filter: MCPToolFilter | None = None,
) -> list[MCPToolWrapper]
```

Convert a list of MCP tools to Exo `MCPToolWrapper` instances.

| Name | Type | Default | Description |
|---|---|---|---|
| `mcp_tools` | `list[MCPTool]` | *(required)* | MCP tool definitions from `list_tools()` |
| `server_name` | `str` | *(required)* | Name of the MCP server |
| `call_fn` | `Any` | *(required)* | Async callable `(tool_name, arguments) -> CallToolResult` |
| `namespace` | `str` | `"mcp"` | Namespace prefix for tool names |
| `tool_filter` | `MCPToolFilter \| None` | `None` | Optional filter for including/excluding tools |

**Returns:** `list[MCPToolWrapper]`

---

## load_tools_from_connection

```python
async def load_tools_from_connection(
    connection: Any,
    *,
    namespace: str = "mcp",
    tool_filter: MCPToolFilter | None = None,
) -> list[MCPToolWrapper]
```

Load and convert tools from a live MCP server connection.

| Name | Type | Default | Description |
|---|---|---|---|
| `connection` | `Any` | *(required)* | An MCPServerConnection (duck-typed: needs `name`, `list_tools()`, `call_tool()`) |
| `namespace` | `str` | `"mcp"` | Namespace prefix for tool names |
| `tool_filter` | `MCPToolFilter \| None` | `None` | Optional filter |

**Returns:** `list[MCPToolWrapper]` -- Tool wrappers ready for agent use.

**Raises:** `MCPToolError` -- If the connection fails to list tools.

---

## load_tools_from_client

```python
async def load_tools_from_client(
    client: Any,
    *,
    namespace: str = "mcp",
    tool_filter: MCPToolFilter | None = None,
) -> list[MCPToolWrapper]
```

Load and convert tools from all servers in an MCP client.

| Name | Type | Default | Description |
|---|---|---|---|
| `client` | `Any` | *(required)* | An MCPClient (duck-typed: needs `server_names`, `connect(name)`) |
| `namespace` | `str` | `"mcp"` | Namespace prefix for tool names |
| `tool_filter` | `MCPToolFilter \| None` | `None` | Optional filter |

**Returns:** `list[MCPToolWrapper]` -- Tool wrappers from all servers.

**Raises:** `MCPToolError` -- If tool loading fails for any server.

### Example

```python
client = load_mcp_client("mcp.json")
async with client:
    tools = await load_tools_from_client(
        client,
        tool_filter=MCPToolFilter(exclude=["dangerous_tool"]),
    )
    for tool in tools:
        print(f"{tool.name}: {tool.description}")
```
