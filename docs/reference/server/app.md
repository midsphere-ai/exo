# exo_server.app

FastAPI application factory with `/chat` endpoint. Supports both synchronous request/response and streaming SSE.

```python
from exo_server.app import ChatRequest, ChatResponse, create_app, register_agent
```

---

## ChatRequest

```python
class ChatRequest(BaseModel)
```

Request body for the `/chat` endpoint.

| Field | Type | Default | Description |
|---|---|---|---|
| `message` | `str` | *(required)* | The user's input message |
| `agent_name` | `str \| None` | `None` | Name of the agent to invoke (uses default if omitted) |
| `stream` | `bool` | `False` | Whether to stream the response via SSE |

---

## ChatResponse

```python
class ChatResponse(BaseModel)
```

Non-streaming response from the `/chat` endpoint.

| Field | Type | Default | Description |
|---|---|---|---|
| `output` | `str` | `""` | The agent's text response |
| `agent_name` | `str` | `""` | Name of the agent that produced the response |
| `steps` | `int` | `0` | Number of LLM call steps taken |
| `usage` | `dict[str, int]` | `{}` | Token usage statistics (input_tokens, output_tokens, total_tokens) |

---

## InjectRequest

```python
class InjectRequest(BaseModel)
```

Request body for the `/inject` endpoint.

| Field | Type | Default | Description |
|---|---|---|---|
| `message` | `str` | *(required)* | The message to inject into the running agent's context |
| `agent_name` | `str \| None` | `None` | Name of the agent to inject into (uses default if omitted) |

---

## register_agent

```python
def register_agent(app: FastAPI, agent: Any, *, default: bool = False) -> None
```

Register an agent with the FastAPI app. The first registered agent automatically becomes the default.

| Name | Type | Default | Description |
|---|---|---|---|
| `app` | `FastAPI` | *(required)* | The FastAPI application instance |
| `agent` | `Any` | *(required)* | An `Agent` (or `Swarm`) instance with a `name` attribute |
| `default` | `bool` | `False` | If `True`, set this agent as the default for requests that do not specify `agent_name` |

### Example

```python
from exo_server import create_app, register_agent

app = create_app()
register_agent(app, helper_agent, default=True)
register_agent(app, coder_agent)
```

---

## create_app

```python
def create_app() -> FastAPI
```

Create a configured FastAPI application with the following routers:

- `/chat` (POST) -- Run an agent synchronously or stream via SSE
- `/inject` (POST) -- Inject a message into a running agent's context
- `/agents` -- Agent management and workspace (from `agent_router`)
- `/sessions` -- Session CRUD (from `session_router`)
- `/ws/chat` -- WebSocket streaming (from `stream_router`)
- `/stream` -- SSE streaming (from `stream_router`)

**Returns:** `FastAPI` -- Configured application instance.

### `/chat` endpoint

**POST** `/chat`

When `stream=False` (default), runs the agent and returns a `ChatResponse`. When `stream=True`, returns `text/event-stream` with SSE events.

SSE event format:
```
data: {"type": "text", "text": "..."}
data: {"type": "tool_call", "tool_name": "...", "tool_call_id": "..."}
data: [DONE]
```

**Error responses:**

| Status | Condition |
|---|---|
| 400 | No `agent_name` specified and no default agent |
| 404 | Named agent not found |
| 500 | Agent execution failed |
| 503 | No agents registered |

### `/inject` endpoint

**POST** `/inject`

Inject a message into a running agent's context. The message is added as a `UserMessage` before the agent's next LLM call.

**Request body:** `InjectRequest`

**Response:** `{"status": "injected"}`

**Error responses:**

| Status | Condition |
|---|---|
| 400 | No `agent_name` specified and no default agent |
| 404 | Named agent not found |
| 503 | No agents registered |

### Example

```python
from exo_server import create_app, register_agent

app = create_app()
register_agent(app, my_agent, default=True)

# Run with uvicorn
# uvicorn my_app:app --host 0.0.0.0 --port 8000
```

```bash
# Non-streaming request
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Python?"}'

# Streaming request
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain decorators", "stream": true}'

# Specify agent
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "agent_name": "coder"}'

# Inject a message into a running agent
curl -X POST http://localhost:8000/inject \
  -H "Content-Type: application/json" \
  -d '{"message": "Also check security implications"}'
```
