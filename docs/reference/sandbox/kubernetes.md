# exo.sandbox.kubernetes

Kubernetes-based sandbox for remote agent execution.

```python
from exo.sandbox.kubernetes import KubernetesSandbox
```

**Requires:** `pip install exo-sandbox[kubernetes]`

---

## KubernetesSandbox

```python
class KubernetesSandbox(Sandbox)(
    *,
    sandbox_id: str | None = None,
    workspace: list[str] | None = None,
    mcp_config: dict[str, Any] | None = None,
    agents: dict[str, Any] | None = None,
    timeout: float = 60.0,
    namespace: str | None = None,
    image: str | None = None,
)
```

Sandbox that manages a Kubernetes pod for isolated execution. Inherits from `Sandbox`.

Pod lifecycle: `start` creates the pod and service, waits for readiness; `stop` deletes the pod and service; `cleanup` ensures all resources are removed.

### Constructor parameters

All parameters from `Sandbox` are inherited, plus:

| Name | Type | Default | Description |
|---|---|---|---|
| `sandbox_id` | `str \| None` | `None` | Unique identifier (auto-generated if omitted) |
| `workspace` | `list[str] \| None` | `None` | Allowed workspace directories |
| `mcp_config` | `dict[str, Any] \| None` | `None` | MCP server configuration |
| `agents` | `dict[str, Any] \| None` | `None` | Agent configurations |
| `timeout` | `float` | `60.0` | Execution timeout in seconds |
| `namespace` | `str \| None` | `None` | Kubernetes namespace. Falls back to `EXO_K8S_NAMESPACE` env var, then `"default"` |
| `image` | `str \| None` | `None` | Container image. Falls back to `EXO_K8S_IMAGE` env var, then `"python:3.11-slim"` |

### Properties

| Property | Type | Description |
|---|---|---|
| `namespace` | `str` | The Kubernetes namespace |
| `image` | `str` | The container image |
| `pod_name` | `str \| None` | Name of the created pod (None before start) |
| `cluster_ip` | `str \| None` | Cluster IP of the created service (None before start) |

### Methods

#### start

```python
async def start(self) -> None
```

Create the pod and service, wait for readiness. Transitions to `RUNNING` status. The pod is named `exo-{sandbox_id}` and the service `exo-svc-{sandbox_id}`.

**Raises:** `SandboxError` -- If pod creation or readiness polling fails.

#### stop

```python
async def stop(self) -> None
```

Delete the pod and service (sandbox can be restarted). Transitions to `IDLE` status.

#### cleanup

```python
async def cleanup(self) -> None
```

Release all Kubernetes resources permanently. Transitions to `CLOSED` status.

#### run_tool

```python
async def run_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any
```

Execute a tool within the Kubernetes sandbox.

**Returns:** Dict with `tool`, `arguments`, `sandbox_id`, `pod`, `cluster_ip`, and `status` keys.

**Raises:** `SandboxError` -- If the sandbox is not in `RUNNING` status.

#### describe

```python
def describe(self) -> dict[str, Any]
```

Return a dict describing the sandbox state, including Kubernetes-specific fields (`namespace`, `image`, `pod_name`, `service_name`, `cluster_ip`).

### Async context manager

```python
async with KubernetesSandbox(namespace="dev", image="python:3.12") as sb:
    result = await sb.run_tool("my_tool", {"arg": "value"})
# Pod and service automatically cleaned up
```

### Environment variables

| Variable | Description | Default |
|---|---|---|
| `EXO_K8S_NAMESPACE` | Kubernetes namespace | `"default"` |
| `EXO_K8S_IMAGE` | Container image | `"python:3.11-slim"` |
| `KUBECONFIG` | Path to kubeconfig file | *(auto-detect)* |

### Example

```python
from exo.sandbox import KubernetesSandbox

sandbox = KubernetesSandbox(
    namespace="agents",
    image="python:3.12-slim",
    workspace=["/data"],
    timeout=120.0,
)

await sandbox.start()
print(sandbox.pod_name)     # "exo-a1b2c3d4e5f6"
print(sandbox.cluster_ip)   # "10.96.0.42"

result = await sandbox.run_tool("process", {"data": "input"})

await sandbox.cleanup()
```
