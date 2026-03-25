# exo.mcp

Model Context Protocol client, server, and tool integration for Exo.

## Installation

```bash
pip install exo-mcp
```

## Module path

```python
import exo.mcp
```

## Package exports

| Export | Module | Description |
|---|---|---|
| `MCPClient` | `exo.mcp.client` | High-level client managing multiple MCP server connections |
| `MCPClientError` | `exo.mcp.client` | Error raised by MCP client operations |
| `MCPServerConfig` | `exo.mcp.client` | Configuration for an MCP server connection |
| `MCPServerConnection` | `exo.mcp.client` | A live connection to an MCP server |
| `MCPTransport` | `exo.mcp.client` | Enum of transport types (stdio, sse, streamable_http) |
| `MCPExecutionError` | `exo.mcp.execution` | Error raised during MCP tool execution with retries |
| `load_mcp_client` | `exo.mcp.execution` | Create an MCPClient from an mcp.json config file |
| `load_mcp_config` | `exo.mcp.execution` | Load MCP server configurations from an mcp.json file |
| `substitute_env_vars` | `exo.mcp.execution` | Replace `${VAR_NAME}` placeholders with environment variable values |
| `MCPServerError` | `exo.mcp.server` | Error raised by MCP server operations |
| `MCPServerRegistry` | `exo.mcp.server` | Singleton registry for @mcp_server-decorated classes |
| `mcp_server` | `exo.mcp.server` | Class decorator that converts a Python class into an MCP server |
| `MCPToolError` | `exo.mcp.tools` | Error raised during MCP tool operations |
| `MCPToolFilter` | `exo.mcp.tools` | Filter for including/excluding MCP tools by name |
| `MCPToolWrapper` | `exo.mcp.tools` | An Exo Tool wrapping an MCP tool for execution |
| `convert_mcp_tools` | `exo.mcp.tools` | Convert a list of MCP tools to Exo MCPToolWrapper instances |
| `extract_schema` | `exo.mcp.tools` | Extract JSON Schema parameters from an MCP tool |
| `namespace_tool_name` | `exo.mcp.tools` | Create a namespaced tool name: `{namespace}__{server}__{tool}` |
| `parse_namespaced_name` | `exo.mcp.tools` | Parse a namespaced tool name back into (namespace, server, tool) |

## Submodules

- [exo.mcp.client](client.md) -- Client classes and transport configuration
- [exo.mcp.execution](execution.md) -- Config loading, env var substitution, retry logic
- [exo.mcp.server](server.md) -- `@mcp_server` decorator and server registry
- [exo.mcp.tools](tools.md) -- Tool wrappers, filters, schema extraction, namespacing
