# WebSocket Protocol Reference

Exo Web uses WebSocket connections for all real-time features. This document covers the multiplexed WebSocket endpoint and the dedicated per-feature WebSocket endpoints.

## Endpoints

| Endpoint | Purpose | Multiplexed |
|----------|---------|-------------|
| `/api/v1/ws` | Main multiplexed connection (system, notifications, logs, execution, sandbox) | Yes |
| `/api/v1/playground/{agent_id}/chat` | Agent chat streaming | No (dedicated) |
| `/api/v1/playground/compare/chat` | Multi-model comparison chat | No (dedicated) |
| `/api/v1/logs/stream` | Dedicated log streaming with filters | No (dedicated) |
| `/api/v1/workflow-runs/{workflow_id}/runs/{run_id}/stream` | Workflow run events | No (dedicated) |
| `/api/v1/workflow-runs/{workflow_id}/runs/{run_id}/debug` | Workflow run debug events | No (dedicated) |
| `/api/v1/crews/{crew_id}/runs/{run_id}/stream` | Crew run events | No (dedicated) |
| `/api/v1/runs/{run_id}/stream` | Agent run events | No (dedicated) |

---

## Connection and Authentication

All WebSocket endpoints authenticate via the `exo_session` cookie. The browser sends this cookie automatically with the WebSocket upgrade request.

### Connection Flow

1. Client initiates WebSocket upgrade (cookie sent automatically)
2. Server extracts `exo_session` cookie and validates against the `sessions` table
3. If valid and not expired: server calls `websocket.accept()`
4. If invalid or missing: server closes with code `4001` and reason `"Unauthorized"`

### URL Construction (Client)

```typescript
const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
const url = `${proto}//${window.location.host}/api/v1/ws`;
const ws = new WebSocket(url);
```

### Successful Connection (Main Endpoint)

After accepting the connection, the server:

1. Registers the connection in the WebSocket manager
2. Auto-subscribes the client to the `system` channel
3. Sends a `connected` message with available channels:

```json
{
  "channel": "system",
  "type": "connected",
  "payload": {
    "user_id": "user-uuid",
    "channels": ["chat", "execution", "logs", "sandbox", "notifications", "system"]
  }
}
```

---

## Message Envelope Format

All messages on the multiplexed `/api/v1/ws` endpoint use a standardized JSON envelope:

```json
{
  "channel": "chat" | "execution" | "logs" | "sandbox" | "notifications" | "system",
  "type": "<message-type>",
  "payload": { ... }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `channel` | string | One of the 6 valid channels |
| `type` | string | Message type identifier, specific to the channel |
| `payload` | object | Message-specific data |

**Note:** Dedicated WebSocket endpoints (chat, logs stream, etc.) use their own message formats documented below — they do **not** use the channel envelope.

---

## Subscribe / Unsubscribe Flow

Clients subscribe to channels to receive messages on the multiplexed endpoint.

### Subscribe

Client sends:

```json
{ "type": "subscribe", "channel": "notifications" }
```

Server responds:

```json
{
  "channel": "system",
  "type": "subscribed",
  "payload": { "channel": "notifications" }
}
```

### Unsubscribe

Client sends:

```json
{ "type": "unsubscribe", "channel": "notifications" }
```

Server responds:

```json
{
  "channel": "system",
  "type": "unsubscribed",
  "payload": { "channel": "notifications" }
}
```

### Client-Side API

```typescript
import { exoSocket } from "./utils/websocket";

// Subscribe — returns an unsubscribe function
const unsubscribe = exoSocket.subscribe("notifications", (message) => {
  console.log("Received:", message.type, message.payload);
});

// Unsubscribe later
unsubscribe();
```

The client automatically connects on first subscription and re-subscribes to all channels after reconnection.

---

## Heartbeat Protocol

The server sends periodic pings to detect stale connections.

| Parameter | Value |
|-----------|-------|
| Ping interval | 30 seconds |
| Pong timeout | 10 seconds |
| Timeout action | Close with code `1001`, reason `"heartbeat timeout"` |

### Flow

1. **Server sends ping** (every 30 seconds):

```json
{ "channel": "system", "type": "ping", "payload": {} }
```

2. **Client responds with pong** (immediately):

```json
{ "type": "pong" }
```

Note: The pong message is a bare JSON object without the channel envelope.

3. If no pong is received within 10 seconds, the server closes the connection.

---

## Reconnection Strategy

The TypeScript client (`ExoSocket`) implements exponential backoff:

| Parameter | Value |
|-----------|-------|
| Initial delay | 1 second |
| Backoff factor | 2x |
| Maximum delay | 30 seconds |

### Backoff Timeline

| Attempt | Delay |
|---------|-------|
| 1 | 1 second |
| 2 | 2 seconds |
| 3 | 4 seconds |
| 4 | 8 seconds |
| 5 | 16 seconds |
| 6+ | 30 seconds (capped) |

### Reconnection Behavior

- Reconnection is triggered automatically on any unintentional close
- On reconnect, the client re-subscribes to all active channels
- Messages sent while disconnected are queued and flushed on reconnect
- Calling `exoSocket.close()` sets an intentional close flag and disables reconnection

### Connection States

```typescript
type ConnectionState = "connecting" | "connected" | "disconnected" | "reconnecting";
```

Monitor state changes:

```typescript
exoSocket.onConnectionChange((state) => {
  console.log("Connection state:", state);
});
```

---

## Channels and Message Types

### System Channel

Auto-subscribed on connection. Handles connection lifecycle and control messages.

#### Server → Client

| Type | Payload | Description |
|------|---------|-------------|
| `connected` | `{ user_id, channels }` | Sent after connection accepted |
| `ping` | `{}` | Heartbeat ping (expect pong response) |
| `subscribed` | `{ channel }` | Subscription confirmation |
| `unsubscribed` | `{ channel }` | Unsubscription confirmation |
| `error` | `{ message }` | Error response |

#### Client → Server

| Type | Fields | Description |
|------|--------|-------------|
| `subscribe` | `{ type, channel }` | Subscribe to a channel |
| `unsubscribe` | `{ type, channel }` | Unsubscribe from a channel |
| `pong` | `{ type }` | Heartbeat response |

### Notifications Channel

Push notifications for user events (approvals, alerts, budget warnings, etc.).

#### Server → Client

| Type | Payload | Description |
|------|---------|-------------|
| `notification_created` | `{ id, notification_type, title, message, entity_type, entity_id, created_at }` | New notification |

Example:

```json
{
  "channel": "notifications",
  "type": "notification_created",
  "payload": {
    "id": "notif-uuid",
    "notification_type": "approval",
    "title": "Action Required",
    "message": "Your workflow needs approval",
    "entity_type": "workflow",
    "entity_id": "workflow-123",
    "created_at": "2024-01-15T10:30:00Z"
  }
}
```

### Logs Channel

Log entries broadcast on the multiplexed connection.

#### Server → Client

| Type | Payload | Description |
|------|---------|-------------|
| `log` | `{ id, timestamp, level, source, agent_id, message, metadata_json }` | New log entry |

Log levels: `debug`, `info`, `warn`, `error`

Log sources: `agent`, `tool`, `model`, `system`

### Execution Channel

Reserved for agent and workflow execution events. Used by the server to broadcast execution state changes to subscribed clients.

### Sandbox Channel

Reserved for sandbox code execution events. Sandbox results in the chat context are delivered through the chat WebSocket (see below).

---

## Dedicated WebSocket Endpoints

### Chat: `/api/v1/playground/{agent_id}/chat`

Streaming chat with an agent. Uses its own message format (no channel envelope).

#### Connection

1. Client connects with session cookie
2. Server validates session and loads agent configuration
3. If agent has no model: sends `{"type": "error", "message": "Agent has no model configured"}` and closes
4. If no provider: sends `{"type": "error", "message": "No {provider} provider configured"}` and closes
5. If no API key: sends `{"type": "error", "message": "No API key configured for provider"}` and closes

#### Client → Server

| Type | Fields | Description |
|------|--------|-------------|
| (content message) | `{ content }` | Send a chat message |
| `load_conversation` | `{ type, conversation_id }` | Load conversation history |
| `takeover` | `{ type }` | Pause streaming and enter takeover mode |
| `stop` | `{ type }` | Stop generation in takeover mode |
| `resume` | `{ type }` | Resume after takeover |
| `inject` | `{ type, content }` | Inject a message during takeover |

#### Server → Client

| Type | Fields | Description |
|------|--------|-------------|
| `error` | `{ type, message }` | Error response |
| `conversation_created` | `{ type, conversation_id }` | New conversation started |
| `conversation_loaded` | `{ type, conversation_id, messages }` | History loaded |
| `message_saved` | `{ type, message_id, role }` | Message persisted to database |
| `token` | `{ type, content }` | Streamed token from LLM |
| `tool_call` | `{ type, name, arguments, result, duration_ms }` | Tool execution trace |
| `sandbox_result` | `{ type, success, stdout, stderr, error, generated_files, execution_time_ms }` | Code execution result |
| `done` | `{ type, usage, model, finish_reason, latency_ms }` | Stream complete |
| `takeover_ack` | `{ type, partial_content }` | Takeover initiated |
| `takeover_event` | `{ type, action, content, timestamp }` | Takeover state change |
| `takeover_stopped` | `{ type, partial_content }` | Takeover ended |

#### Example: Basic Chat Session

```javascript
const proto = location.protocol === "https:" ? "wss:" : "ws:";
const ws = new WebSocket(`${proto}//${location.host}/api/v1/playground/${agentId}/chat`);

ws.onopen = () => {
  // Send a message
  ws.send(JSON.stringify({ content: "What is the weather today?" }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);

  switch (msg.type) {
    case "conversation_created":
      console.log("Conversation:", msg.conversation_id);
      break;

    case "token":
      // Append streamed text to the UI
      process.stdout.write(msg.content);
      break;

    case "tool_call":
      console.log(`Tool: ${msg.name} (${msg.duration_ms}ms)`);
      break;

    case "sandbox_result":
      if (msg.success) {
        console.log("Output:", msg.stdout);
      } else {
        console.error("Error:", msg.error || msg.stderr);
      }
      break;

    case "done":
      console.log(`\nDone: ${msg.model}, ${msg.usage.completion_tokens} tokens, ${msg.latency_ms}ms`);
      break;

    case "error":
      console.error("Error:", msg.message);
      break;
  }
};
```

#### Example: Loading a Previous Conversation

```javascript
ws.onopen = () => {
  // Load an existing conversation
  ws.send(JSON.stringify({
    type: "load_conversation",
    conversation_id: "conv-uuid-here"
  }));
};

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "conversation_loaded") {
    // msg.messages is an array of { id, role, content } objects
    for (const m of msg.messages) {
      console.log(`${m.role}: ${m.content}`);
    }
  }
};
```

#### Example: Takeover (Human-in-the-Loop)

```javascript
// While the agent is streaming, send takeover to pause
ws.send(JSON.stringify({ type: "takeover" }));
// Server responds with takeover_ack { partial_content: "..." }

// Inject additional context
ws.send(JSON.stringify({ type: "inject", content: "Actually, focus on New York weather" }));
// Server responds with takeover_event { action: "injected", content: "..." }

// Resume generation
ws.send(JSON.stringify({ type: "resume" }));
// Server responds with takeover_event { action: "resumed" }

// Or stop completely
ws.send(JSON.stringify({ type: "stop" }));
// Server responds with takeover_stopped { partial_content: "..." }
```

### Logs Stream: `/api/v1/logs/stream`

Dedicated endpoint for real-time log streaming with server-side filtering.

#### Client → Server

Send a filter object to narrow the stream. Unset fields mean "no filter":

```json
{ "level": "error", "source": "agent", "agent_id": "agent-uuid" }
```

#### Server → Client

Log entries matching the current filter:

```json
{ "type": "log", "payload": { "id": "...", "timestamp": "...", "level": "error", ... } }
```

### Workflow Run Stream: `/api/v1/workflow-runs/{workflow_id}/runs/{run_id}/stream`

Streams execution events for a specific workflow run.

### Workflow Run Debug: `/api/v1/workflow-runs/{workflow_id}/runs/{run_id}/debug`

Streams debug-level execution events for a workflow run, including node-level state transitions.

### Crew Run Stream: `/api/v1/crews/{crew_id}/runs/{run_id}/stream`

Streams execution events for a crew (multi-agent) run.

### Agent Run Stream: `/api/v1/runs/{run_id}/stream`

Streams execution events for a standalone agent run.

---

## Error Messages and Disconnect Reasons

### WebSocket Close Codes

| Code | Meaning | When Used |
|------|---------|-----------|
| `1000` | Normal closure | Client intentionally closed |
| `1001` | Going away | Heartbeat timeout |
| `4001` | Unauthorized | Invalid or missing session cookie |
| `4004` | Not found | Resource not found (e.g., agent ID) |

### Error Messages (System Channel)

Sent on the multiplexed endpoint when an invalid message is received:

```json
{ "channel": "system", "type": "error", "payload": { "message": "..." } }
```

| Error | Cause |
|-------|-------|
| `"Invalid JSON"` | Message could not be parsed as JSON |
| `"Invalid channel: xyz"` | Subscribe request with unknown channel name |
| `"Unknown message type: abc"` | Unrecognized message type |

### Error Messages (Chat Endpoint)

Sent on the dedicated chat WebSocket:

```json
{ "type": "error", "message": "..." }
```

| Error | Cause |
|-------|-------|
| `"Agent has no model configured"` | Agent missing `model_provider` or `model_name` |
| `"No {provider} provider configured"` | No matching provider for the agent's model |
| `"No API key configured for provider"` | Provider has no decryptable API key |
| `"Invalid JSON"` | Message could not be parsed |
| `"Empty message"` | Content field is empty |
| `"Stream error: ..."` | Error during LLM streaming |
| `"No active stream to take over"` | Takeover sent when not streaming |
| `"Not in takeover mode"` | Stop/resume/inject sent without takeover |

---

## Server-Side API

### Broadcasting to Users

```python
from exo_web.websocket import manager

# Send to all of a user's connections subscribed to a channel
sent_count = await manager.broadcast_to_user(
    user_id="user-123",
    channel="notifications",
    message={
        "type": "notification_created",
        "id": "notif-uuid",
        "notification_type": "alert",
        "title": "Alert",
        "message": "Something happened",
    },
)
```

### Sending to a Specific Connection

```python
await manager.send_to_connection(
    ws=websocket,
    channel="logs",
    msg_type="log",
    payload={"level": "info", "message": "Processing complete"},
)
```

### Authentication Helper

```python
from exo_web.websocket import get_ws_user

user = await get_ws_user(websocket)
if user is None:
    await websocket.close(code=4001, reason="Unauthorized")
    return
# user is a dict with: id, email, created_at
```

---

## Implementation Files

| File | Description |
|------|-------------|
| `src/exo_web/websocket.py` | Multiplexed WebSocket manager, heartbeat, message dispatcher |
| `src/exo_web/routes/playground.py` | Chat WebSocket endpoint with streaming and takeover |
| `src/exo_web/routes/logs.py` | Dedicated log streaming WebSocket |
| `src/exo_web/routes/workflow_runs.py` | Workflow run stream and debug WebSocket |
| `src/exo_web/routes/crews.py` | Crew run stream WebSocket |
| `src/exo_web/routes/runs.py` | Agent run stream WebSocket |
| `src/utils/websocket.ts` | Client-side `ExoSocket` singleton |
