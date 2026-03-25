# MCP (Model Context Protocol)

The `exo-mcp` package provides client-side MCP integration, allowing agents to discover and use tools from MCP servers. It supports stdio, SSE, and streamable HTTP transports, multi-server management, tool namespace mapping, and automatic conversion of MCP tools into Exo `Tool` instances.

## Basic Usage

```python
from exo.mcp import MCPClient, MCPServerConfig, MCPTransport

# Configure an MCP server
config = MCPServerConfig(
    name="filesystem",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    transport=MCPTransport.STDIO,
)

# Create client and connect
client = MCPClient()
client.add_server(config)
await client.connect_all()

# List available tools
tools = await client.list_tools("filesystem")

# Call a tool
result = await client.call_tool("filesystem", "read_file", {"path": "/tmp/data.txt"})

# Clean up
await client.cleanup()
```

## Server Configuration

`MCPServerConfig` defines how to connect to an MCP server:

```python
from exo.mcp import MCPServerConfig, MCPTransport

# Stdio transport (spawns a subprocess)
stdio_config = MCPServerConfig(
    name="my-server",
    command="python",
    args=["-m", "my_mcp_server"],
    transport=MCPTransport.STDIO,
    env={"API_KEY": "sk-..."},  # optional environment variables
)

# SSE transport (connects to HTTP endpoint)
sse_config = MCPServerConfig(
    name="remote-server",
    url="http://localhost:8080/sse",
    transport=MCPTransport.SSE,
)

# Streamable HTTP transport
http_config = MCPServerConfig(
    name="http-server",
    url="http://localhost:8080/mcp",
    transport=MCPTransport.STREAMABLE_HTTP,
)
```

## Transport Types

| Transport | Value | Description |
|-----------|-------|-------------|
| `STDIO` | `"stdio"` | Spawns a subprocess, communicates via stdin/stdout |
| `SSE` | `"sse"` | Server-Sent Events over HTTP |
| `STREAMABLE_HTTP` | `"streamable_http"` | Bidirectional HTTP streaming |

## MCPClient Multi-Server Management

The `MCPClient` manages connections to multiple MCP servers:

```python
client = MCPClient()

# Add multiple servers
client.add_server(filesystem_config)
client.add_server(database_config)
client.add_server(search_config)

# Connect to all at once
await client.connect_all()

# Or connect to specific servers
await client.connect("filesystem")

# List tools from a specific server
fs_tools = await client.list_tools("filesystem")

# Call a tool on a specific server
result = await client.call_tool("database", "query", {"sql": "SELECT 1"})

# Cleanup all connections
await client.cleanup()
```

## Converting MCP Tools to Exo Tools

Use `MCPToolWrapper` to convert MCP tools into Exo `Tool` instances that agents can use:

```python
from exo.mcp import load_tools_from_client

# Convert all MCP tools to Exo Tools
exo_tools = await load_tools_from_client(client)

# Pass directly to an agent
from exo.agent import Agent

agent = Agent(
    name="assistant",
    model="openai:gpt-4o",
    tools=exo_tools,
)
```

### Namespace Mapping

When loading tools from multiple servers, tool names are namespaced to avoid collisions:

```python
from exo.mcp.tools import namespace_tool_name, parse_namespaced_name

# Namespace: "server_name__tool_name"
namespaced = namespace_tool_name("filesystem", "read_file")
print(namespaced)  # "filesystem__read_file"

# Parse back
server, tool = parse_namespaced_name("filesystem__read_file")
print(server, tool)  # "filesystem", "read_file"
```

### Tool Filtering

Use `MCPToolFilter` to include or exclude specific tools:

```python
from exo.mcp import MCPToolFilter

# Only load specific tools
filter = MCPToolFilter(include=["read_file", "write_file"])
tools = await load_tools_from_connection(connection, filter=filter)

# Exclude tools
filter = MCPToolFilter(exclude=["dangerous_tool"])
tools = await load_tools_from_connection(connection, filter=filter)
```

## Loading from Config File

Load MCP server configurations from a `mcp.json` file:

```python
from exo.mcp import load_mcp_config, load_mcp_client

# mcp.json format:
# {
#   "mcpServers": {
#     "filesystem": {
#       "command": "npx",
#       "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
#       "transport": "stdio"
#     }
#   }
# }

configs = load_mcp_config("mcp.json")
client = await load_mcp_client("mcp.json")
tools = await load_tools_from_client(client)
```

Environment variables in the config are automatically substituted using `substitute_env_vars()`.

## Creating MCP Servers

Use the `@mcp_server` decorator to expose an Exo class as an MCP server:

```python
from exo.mcp import mcp_server

@mcp_server(name="calculator", description="Math operations")
class CalculatorServer:
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
```

The decorator uses FastMCP to convert the class methods into MCP tools and create a server.

## Retry Logic

Tool calls include automatic retry with exponential backoff:

```python
from exo.mcp.execution import call_tool_with_retry

result = await call_tool_with_retry(
    connection,
    tool_name="flaky_api",
    arguments={"query": "test"},
    max_retries=3,
    base_delay=1.0,  # seconds
)
```

## Advanced Patterns

### Dynamic Server Discovery

Add servers at runtime based on project configuration:

```python
import json

with open(".exo.yaml") as f:
    config = yaml.safe_load(f)

client = MCPClient()
for name, server_cfg in config.get("mcp_servers", {}).items():
    client.add_server(MCPServerConfig(name=name, **server_cfg))

await client.connect_all()
```

### Tool Schema Extraction

Inspect tool schemas before passing them to agents:

```python
from exo.mcp.tools import extract_schema

tools = await client.list_tools("filesystem")
for tool in tools:
    schema = extract_schema(tool)
    print(f"{tool.name}: {json.dumps(schema, indent=2)}")
```

### Mixing MCP and Native Tools

Combine MCP tools with native Exo function tools:

```python
from exo.tool import tool
from exo.mcp import load_tools_from_client

@tool
def local_calculator(expression: str) -> str:
    """Evaluate a math expression locally."""
    return str(eval(expression))

mcp_tools = await load_tools_from_client(client)
all_tools = mcp_tools + [local_calculator]

agent = Agent(name="assistant", tools=all_tools)
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `MCPClient` | `exo.mcp` | Multi-server MCP client with caching |
| `MCPServerConfig` | `exo.mcp` | Server connection configuration |
| `MCPTransport` | `exo.mcp` | Enum: `STDIO`, `SSE`, `STREAMABLE_HTTP` |
| `MCPServerConnection` | `exo.mcp.client` | Single server connection lifecycle |
| `MCPToolWrapper` | `exo.mcp.tools` | Wraps MCP tool as Exo `Tool` |
| `MCPToolFilter` | `exo.mcp.tools` | Include/exclude filter for tool loading |
| `mcp_server` | `exo.mcp` | Decorator to create MCP servers from classes |
| `load_tools_from_client` | `exo.mcp.tools` | Convert all client tools to Exo `Tool`s |
| `load_tools_from_connection` | `exo.mcp.tools` | Convert one connection's tools |
| `load_mcp_config` | `exo.mcp.execution` | Load `MCPServerConfig`s from `mcp.json` |
| `load_mcp_client` | `exo.mcp.execution` | Load and connect an `MCPClient` from config |
| `call_tool_with_retry` | `exo.mcp.execution` | Tool call with exponential backoff retry |
