# exo_server.streaming

WebSocket and SSE streaming for Exo Server. Provides real-time streaming of agent output via WebSocket and Server-Sent Events (SSE) as a fallback for non-WebSocket clients.

```python
from exo_server.streaming import stream_router
```

---

## stream_router

```python
stream_router = APIRouter()
```

FastAPI router providing streaming endpoints. Must be included in a FastAPI app that has agents registered via `register_agent`.

---

## Protocols

### WebSocket protocol

**Endpoint:** `ws://host:port/ws/chat`

**Client sends:**

```json
{"message": "What is Python?", "agent_name": "helper"}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `message` | `str` | Yes | The user's input message |
| `agent_name` | `str` | No | Agent to invoke (uses default if omitted) |

**Server sends (text event):**

```json
{"type": "text", "text": "Python is a programming language."}
```

**Server sends (tool call event):**

```json
{"type": "tool_call", "tool_name": "web_search", "tool_call_id": "call_abc123"}
```

**Server sends (completion):**

```json
{"type": "done"}
```

**Server sends (error):**

```json
{"type": "error", "error": "Agent not found"}
```

### SSE protocol

**Endpoint:** `GET /stream?message=...&agent_name=...`

Returns `text/event-stream` with the same JSON payloads as WebSocket, formatted as SSE:

```
data: {"type": "text", "text": "Python is..."}

data: {"type": "tool_call", "tool_name": "search", "tool_call_id": "call_1"}

data: [DONE]
```

---

## Endpoints

### WebSocket /ws/chat

```python
@stream_router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None
```

WebSocket endpoint for real-time agent streaming. Accepts one JSON message from the client, streams back events, and sends a final `{"type": "done"}` message before closing.

**Connection flow:**

1. Client connects to `/ws/chat`
2. Server accepts the connection
3. Client sends a JSON message with `message` and optional `agent_name`
4. Server streams back events (text, tool_call)
5. Server sends `{"type": "done"}`
6. Server closes the connection

**Error conditions:**

| Condition | Behavior |
|---|---|
| Empty message | Sends `{"type": "error", "error": "Empty message"}` and closes |
| No agents registered | Sends `{"type": "error", "error": "No agents registered"}` and closes |
| Agent not found | Sends `{"type": "error", "error": "Agent 'name' not found"}` and closes |
| Client disconnects | Silently returns |

### GET /stream

```python
@stream_router.get("/stream")
async def sse_stream(
    req: Request,
    message: str = Query(..., description="The user message"),
    agent_name: str | None = Query(None, description="Agent to invoke"),
) -> StreamingResponse
```

SSE fallback endpoint for non-WebSocket clients. Returns `text/event-stream` with JSON payloads matching the WebSocket protocol.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `message` | `str` (query) | Yes | The user message |
| `agent_name` | `str` (query) | No | Agent to invoke (uses default if omitted) |

**Returns:** `StreamingResponse` with `media_type="text/event-stream"`.

If the agent is not found, returns an error event followed by `[DONE]`.

---

## Examples

### WebSocket client (Python)

```python
import asyncio
import json
import websockets

async def stream_chat():
    async with websockets.connect("ws://localhost:8000/ws/chat") as ws:
        # Send request
        await ws.send(json.dumps({
            "message": "Explain Python decorators",
            "agent_name": "helper",
        }))

        # Receive streaming events
        while True:
            data = json.loads(await ws.recv())
            if data["type"] == "done":
                break
            elif data["type"] == "text":
                print(data["text"], end="")
            elif data["type"] == "tool_call":
                print(f"\n[Tool: {data['tool_name']}]")
            elif data["type"] == "error":
                print(f"Error: {data['error']}")
                break
        print()

asyncio.run(stream_chat())
```

### SSE client (curl)

```bash
# Stream agent output via SSE
curl -N "http://localhost:8000/stream?message=What+is+Python%3F&agent_name=helper"
```

### SSE client (JavaScript)

```javascript
const source = new EventSource("/stream?message=Hello&agent_name=helper");

source.onmessage = (event) => {
    if (event.data === "[DONE]") {
        source.close();
        return;
    }
    const payload = JSON.parse(event.data);
    if (payload.type === "text") {
        document.getElementById("output").textContent += payload.text;
    } else if (payload.type === "error") {
        console.error(payload.error);
        source.close();
    }
};
```
