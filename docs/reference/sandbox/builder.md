# exo.sandbox.builder

Fluent builder for constructing `Sandbox` instances with lazy evaluation.

```python
from exo.sandbox.builder import SandboxBuilder
```

---

## SandboxBuilder

```python
class SandboxBuilder(sandbox_class: type[Sandbox] | None = None)
```

Fluent builder for creating `Sandbox` instances. Supports method chaining and lazy evaluation -- the sandbox is automatically built on the first lifecycle or tool API call (any attribute not part of the builder itself).

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `sandbox_class` | `type[Sandbox] \| None` | `None` | Sandbox implementation class. Defaults to `LocalSandbox` |

### Fluent setter methods

All setters return `self` for chaining.

#### with_sandbox_id

```python
def with_sandbox_id(self, sandbox_id: str) -> SandboxBuilder
```

Set the sandbox ID.

#### with_workspace

```python
def with_workspace(self, workspace: list[str]) -> SandboxBuilder
```

Set allowed workspace directories.

#### with_mcp_config

```python
def with_mcp_config(self, mcp_config: dict[str, Any]) -> SandboxBuilder
```

Set MCP server configuration.

#### with_agents

```python
def with_agents(self, agents: dict[str, Any]) -> SandboxBuilder
```

Set agent configurations.

#### with_timeout

```python
def with_timeout(self, timeout: float) -> SandboxBuilder
```

Set execution timeout in seconds.

#### with_sandbox_class

```python
def with_sandbox_class(self, cls: type[Sandbox]) -> SandboxBuilder
```

Override the sandbox implementation class.

#### with_extra

```python
def with_extra(self, **kwargs: Any) -> SandboxBuilder
```

Pass additional keyword arguments to the sandbox constructor.

### Build and lifecycle methods

#### build

```python
def build(self) -> Sandbox
```

Construct and return the `Sandbox` instance. Returns the cached instance on subsequent calls. Call `reset()` first to reuse the builder.

#### reset

```python
def reset(self) -> SandboxBuilder
```

Clear the built instance so the builder can be reused. Returns self for chaining.

### Lazy evaluation

Any attribute access that is not a builder method (`with_*`, `build`, `reset`) triggers `build()` and is forwarded to the resulting `Sandbox`. This means you can call `await builder.start()` directly without calling `build()` first.

### Example

```python
from exo.sandbox import SandboxBuilder

sb = (
    SandboxBuilder()
    .with_workspace(["/tmp/ws"])
    .with_timeout(60.0)
    .with_mcp_config({"server": "local"})
)

# Not built yet -- lazy until first use:
await sb.start()   # triggers build(), then start()
result = await sb.run_tool("my_tool", {"arg": "value"})
await sb.cleanup()
```

### Chaining with a custom sandbox class

```python
from exo.sandbox import SandboxBuilder, KubernetesSandbox

sb = (
    SandboxBuilder(KubernetesSandbox)
    .with_workspace(["/data"])
    .with_timeout(120.0)
    .with_extra(namespace="production", image="python:3.12")
)

await sb.start()  # creates Kubernetes pod
```
