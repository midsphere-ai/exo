# exo.mcp.server

`@mcp_server()` class decorator and `MCPServerRegistry` for exposing tools as MCP servers.

```python
from exo.mcp.server import (
    MCPServerError,
    MCPServerRegistry,
    mcp_server,
    server_registry,
)
```

---

## MCPServerError

```python
class MCPServerError(Exception)
```

Error raised by MCP server operations.

---

## mcp_server

```python
def mcp_server(
    name: str | None = None,
    *,
    transport: str = "stdio",
) -> Any
```

Class decorator that converts a Python class into an MCP server. Public methods (non-underscored, excluding `run`/`stop`) are registered as MCP tools via FastMCP.

After decoration the class gains:

- `_mcp` -- the `FastMCP` instance
- `_tool_names` -- list of registered tool names
- `run(**kwargs)` -- start the server (`transport` kwarg overrides default)
- `stop()` -- placeholder for graceful shutdown

The class is also registered in the module-level `server_registry`.

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str \| None` | `None` | Server name. Defaults to the class name |
| `transport` | `str` | `"stdio"` | Default transport mode (`"stdio"` or `"sse"`) |

**Returns:** The decorated class.

### Example

```python
from exo.mcp.server import mcp_server

@mcp_server(name="calculator")
class Calculator:
    """A simple calculator server."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

# Create and run
calc = Calculator()
print(calc._tool_names)  # ['add', 'multiply']
calc.run()  # starts the MCP server via stdio
```

### Injected methods

#### run

```python
def run(self, *, transport: str = <default_transport>, **kwargs) -> None
```

Run the MCP server.

| Name | Type | Default | Description |
|---|---|---|---|
| `transport` | `str` | Decorator's `transport` value | `"stdio"` or `"sse"` |
| `**kwargs` | `Any` | | Passed to `FastMCP.run()` |

**Raises:** `MCPServerError` -- If the MCP server is not initialized.

#### stop

```python
def stop(self) -> None
```

Stop the MCP server (placeholder for graceful shutdown).

---

## MCPServerRegistry

```python
class MCPServerRegistry()
```

Singleton registry for `@mcp_server`-decorated classes. Stores class references and lazily-created singleton instances.

### Methods

#### register

```python
def register(self, name: str, cls: type) -> None
```

Register a server class by name.

#### get_class

```python
def get_class(self, name: str) -> type
```

Get a registered server class.

**Raises:** `MCPServerError` -- If the name is not registered.

#### get_instance

```python
def get_instance(self, name: str, *args: Any, **kwargs: Any) -> Any
```

Get or create a singleton instance of a registered server. Subsequent calls with the same name return the cached instance.

**Raises:** `MCPServerError` -- If the name is not registered.

#### has

```python
def has(self, name: str) -> bool
```

Check if a server name is registered.

#### clear

```python
def clear(self) -> None
```

Remove all registrations and instances.

### Properties

| Property | Type | Description |
|---|---|---|
| `names` | `list[str]` | All registered server names |

---

## server_registry

```python
server_registry = MCPServerRegistry()
```

Module-level global registry instance. All `@mcp_server`-decorated classes are automatically registered here.

### Example

```python
from exo.mcp.server import server_registry

# After decorating classes with @mcp_server
print(server_registry.names)        # ['calculator']
calc = server_registry.get_instance("calculator")
```
