# Server

The `exo-server` package provides an HTTP server for exposing agents as web services. Built on FastAPI, it supports REST endpoints for chat, agent management, session persistence, and real-time streaming via Server-Sent Events (SSE) and WebSocket.

## Basic Usage

```python
from exo_server import create_app, register_agent
from exo.agent import Agent

# Create an agent
agent = Agent(
    name="assistant",
    model="openai:gpt-4o",
    instructions="You are a helpful assistant.",
)

# Register it with the server
register_agent("assistant", agent)

# Create and run the FastAPI app
app = create_app()

# Run with uvicorn
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Creating the App

The `create_app()` factory builds a FastAPI application with all routes:

```python
from exo_server import create_app

app = create_app()
# Includes:
#   POST /chat          - Chat endpoint
#   GET/POST /agents/*  - Agent management
#   GET/POST /sessions/* - Session management
#   GET /stream         - SSE streaming
#   WS /ws/chat         - WebSocket chat
```

## Registering Agents

Register agents before creating the app:

```python
from exo_server import register_agent
from exo.agent import Agent

# Register multiple agents
register_agent("assistant", Agent(name="assistant", model="openai:gpt-4o"))
register_agent("coder", Agent(name="coder", model="openai:gpt-4o", instructions="You write code."))
register_agent("researcher", Agent(name="researcher", model="anthropic:claude-3-5-sonnet"))
```

## Chat Endpoint

The main interaction endpoint:

```
POST /chat
```

### Request

```python
from exo_server.app import ChatRequest

# Request body
request = ChatRequest(
    agent_name="assistant",
    input="What is Python?",
    session_id="session-123",  # optional: for conversation continuity
    stream=False,              # optional: enable SSE streaming
)
```

```json
{
    "agent_name": "assistant",
    "input": "What is Python?",
    "session_id": "session-123",
    "stream": false
}
```

### Response

```json
{
    "output": "Python is a high-level programming language...",
    "agent_name": "assistant",
    "session_id": "session-123"
}
```

### Streaming Response

When `stream=true`, the endpoint returns Server-Sent Events:

```python
import httpx

async with httpx.AsyncClient() as client:
    async with client.stream("POST", "http://localhost:8000/chat", json={
        "agent_name": "assistant",
        "input": "Explain decorators",
        "stream": True,
    }) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                print(line[6:], end="", flush=True)
```

## Message Injection

Inject a message into a running agent's context. The message is picked up before the agent's next LLM call, allowing you to steer or augment a running agent without restarting it.

```
POST /inject
```

### Request

```json
{
    "message": "Also check the official Python docs",
    "agent_name": "researcher"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | `str` | *(required)* | The message to inject |
| `agent_name` | `str \| None` | `None` | Target agent (uses default if omitted) |

### Response

```json
{
    "status": "injected"
}
```

### Example

```bash
# Inject into the default agent
curl -X POST http://localhost:8000/inject \
  -H "Content-Type: application/json" \
  -d '{"message": "Use metric units instead"}'

# Inject into a specific agent
curl -X POST http://localhost:8000/inject \
  -H "Content-Type: application/json" \
  -d '{"message": "Focus on security aspects", "agent_name": "researcher"}'
```

### Error Responses

| Status | Condition |
|--------|-----------|
| 400 | No `agent_name` specified and no default agent |
| 404 | Named agent not found |
| 503 | No agents registered |

## Agent Management

### List Agents

```
GET /agents
```

Returns a list of registered agents with metadata:

```json
[
    {
        "name": "assistant",
        "model": "openai:gpt-4o",
        "description": "A helpful assistant"
    }
]
```

### Get Agent Details

```
GET /agents/{name}
```

Returns detailed information about a specific agent:

```json
{
    "name": "assistant",
    "model": "openai:gpt-4o",
    "description": "A helpful assistant",
    "tools": ["search", "calculator"],
    "max_steps": 10
}
```

### Workspace Files

```
GET /agents/{name}/workspace
GET /agents/{name}/workspace/{path}
```

Access workspace files for an agent.

## Session Management

Sessions maintain conversation state across multiple requests.

### Create Session

```
POST /sessions
```

```json
{
    "agent_name": "assistant",
    "metadata": {"user_id": "u-123"}
}
```

### Get Session

```
GET /sessions/{session_id}
```

Returns session details including message history:

```json
{
    "id": "session-123",
    "agent_name": "assistant",
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ],
    "created_at": "2025-01-15T10:30:00Z"
}
```

### List Sessions

```
GET /sessions
```

### Delete Session

```
DELETE /sessions/{session_id}
```

## WebSocket Streaming

For real-time bidirectional communication:

```
WS /ws/chat
```

```python
import websockets
import json

async with websockets.connect("ws://localhost:8000/ws/chat") as ws:
    # Send a message
    await ws.send(json.dumps({
        "agent_name": "assistant",
        "input": "Hello!",
        "session_id": "session-123",
    }))

    # Receive streaming response
    async for message in ws:
        data = json.loads(message)
        if data.get("type") == "text":
            print(data["content"], end="")
        elif data.get("type") == "done":
            break
```

## SSE Streaming

The dedicated SSE endpoint:

```
GET /stream?agent_name=assistant&input=Hello&session_id=s-1
```

```python
import httpx

async with httpx.AsyncClient() as client:
    async with client.stream("GET", "http://localhost:8000/stream", params={
        "agent_name": "assistant",
        "input": "Tell me a story",
    }) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                print(line[6:], end="")
```

## Advanced Patterns

### Custom Middleware

Add authentication or logging middleware:

```python
from fastapi import Request
from exo_server import create_app

app = create_app()

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return JSONResponse(status_code=401, content={"error": "Missing API key"})
    response = await call_next(request)
    return response
```

### Multiple Agent Deployment

Deploy multiple agents on a single server:

```python
from exo_server import create_app, register_agent

# Register specialized agents
register_agent("general", general_agent)
register_agent("coder", coding_agent)
register_agent("analyst", data_agent)

app = create_app()
# All agents accessible via /chat with different agent_name values
```

### Health Check

The app can be extended with health check endpoints:

```python
app = create_app()

@app.get("/health")
async def health():
    return {"status": "ok", "agents": len(registered_agents)}
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `create_app` | `exo_server` | FastAPI app factory |
| `register_agent` | `exo_server` | Register an agent with the server |
| `ChatRequest` | `exo_server.app` | Chat request model: `agent_name`, `input`, `session_id`, `stream` |
| `ChatResponse` | `exo_server.app` | Chat response model: `output`, `agent_name`, `session_id` |
| `agent_router` | `exo_server.agents` | Router for `/agents` endpoints |
| `session_router` | `exo_server.sessions` | Router for `/sessions` endpoints |
| `stream_router` | `exo_server.streaming` | Router for `/stream` and `/ws/chat` endpoints |
