# exo.a2a.server

A2A server -- FastAPI-based agent serving with agent card discovery.

```python
from exo.a2a.server import (
    A2AServer,
    A2AServerError,
    AgentExecutor,
    InMemoryTaskStore,
    TaskStore,
)
```

---

## A2AServerError

```python
class A2AServerError(ExoError)
```

Raised for A2A server-level errors. Inherits from `exo.types.ExoError`.

---

## TaskStore

```python
@runtime_checkable
class TaskStore(Protocol)
```

Minimal storage interface for A2A task state.

### Methods

```python
async def get(self, task_id: str) -> dict[str, Any] | None: ...
async def save(self, task_id: str, data: dict[str, Any]) -> None: ...
async def delete(self, task_id: str) -> None: ...
```

---

## InMemoryTaskStore

```python
class InMemoryTaskStore()
```

Simple in-memory task store for development and testing.

### Methods

#### get

```python
async def get(self, task_id: str) -> dict[str, Any] | None
```

Retrieve task data by ID. Returns `None` if not found.

#### save

```python
async def save(self, task_id: str, data: dict[str, Any]) -> None
```

Save task data.

#### delete

```python
async def delete(self, task_id: str) -> None
```

Delete task data. No-op if not found.

---

## AgentExecutor

```python
class AgentExecutor(
    agent: Any,
    *,
    streaming: bool = False,
)
```

Wraps an agent for A2A task execution. Accepts any object with a `run(input, ...)` async method and a `name` attribute (i.e. an Exo `Agent`).

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `agent` | `Any` | *(required)* | Agent object with `run()` async method and `name` attribute |
| `streaming` | `bool` | `False` | Whether to enable streaming |

### Properties

| Property | Type | Description |
|---|---|---|
| `agent_name` | `str` | Name of the wrapped agent |

### Methods

#### execute

```python
async def execute(self, text: str, *, provider: Any = None) -> str
```

Run the wrapped agent and return text output.

| Name | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | *(required)* | Input text for the agent |
| `provider` | `Any` | `None` | Optional LLM provider |

**Returns:** `str` -- Agent text output.

---

## A2AServer

```python
class A2AServer(
    executor: AgentExecutor,
    config: ServingConfig | None = None,
    *,
    task_store: TaskStore | None = None,
    provider: Any = None,
)
```

FastAPI-based A2A server with agent card discovery.

Exposes:
- `GET /.well-known/agent-card` -- agent card JSON
- `POST /` -- task execution (send text, get response)
- `GET /tasks/{task_id}` -- retrieve task state
- `POST /stream` -- streaming task execution (when streaming is enabled in config)

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `executor` | `AgentExecutor` | *(required)* | An `AgentExecutor` wrapping the agent to serve |
| `config` | `ServingConfig \| None` | `None` | Server configuration (host, port, skills, etc.). Defaults to `ServingConfig()` |
| `task_store` | `TaskStore \| None` | `None` | Task state storage. Defaults to `InMemoryTaskStore` |
| `provider` | `Any` | `None` | Optional LLM provider passed through to the agent |

### Properties

| Property | Type | Description |
|---|---|---|
| `agent_card` | `AgentCard` | The constructed agent card |
| `task_store` | `TaskStore` | The task store instance |

### Methods

#### build_app

```python
def build_app(self) -> Any
```

Create and return the FastAPI application. Registers all routes.

**Returns:** `FastAPI` application instance.

**Raises:** `A2AServerError` -- If `fastapi` is not installed.

### API endpoints

#### GET /.well-known/agent-card

Returns the agent card as JSON.

**Response:** `AgentCard` as JSON.

#### POST /

Execute a task.

**Request body:**
```json
{
    "text": "What is Python?",
    "task_id": "optional-id"
}
```

**Response (success):**
```json
{
    "task_id": "...",
    "status": {"state": "completed", "reason": ""},
    "artifact": {"task_id": "...", "text": "...", "last_chunk": true}
}
```

**Response (error, 500):**
```json
{
    "task_id": "...",
    "status": {"state": "failed", "reason": "error message"}
}
```

#### GET /tasks/{task_id}

Retrieve task state.

#### POST /stream (when streaming enabled)

Stream task execution as NDJSON. Each line is a `TaskStatusUpdateEvent` or `TaskArtifactUpdateEvent`.

### Example

```python
from exo.a2a import A2AServer, AgentExecutor, ServingConfig, AgentSkill

executor = AgentExecutor(my_agent, streaming=True)

config = ServingConfig(
    host="0.0.0.0",
    port=8000,
    streaming=True,
    skills=(
        AgentSkill(id="qa", name="Q&A", description="Answer questions"),
    ),
)

server = A2AServer(executor, config)
app = server.build_app()

# Run with uvicorn:
# uvicorn my_module:app --host 0.0.0.0 --port 8000
```
