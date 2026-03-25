# Agent-to-Agent (A2A)

The `exo-a2a` package implements the Agent-to-Agent protocol, enabling agents to discover, communicate with, and delegate tasks to other agents over HTTP. It provides an `A2AServer` for exposing agents as network services, an `A2AClient` for consuming them, and a `RemoteAgent` wrapper that makes remote agents look like local ones.

## Basic Usage

### Server Side

```python
from exo.a2a import A2AServer, AgentCard, AgentCapabilities, AgentSkill, ServingConfig

# Define what your agent can do
card = AgentCard(
    name="researcher",
    description="Searches and summarizes information",
    url="http://localhost:8001",
    capabilities=AgentCapabilities(streaming=True, batch=False),
    skills=[
        AgentSkill(name="search", description="Search the web"),
        AgentSkill(name="summarize", description="Summarize text"),
    ],
)

# Create and start server
config = ServingConfig(host="0.0.0.0", port=8001)
server = A2AServer(card=card, config=config)

# Register the agent executor
server.register_executor(my_agent_executor)

# Run (blocks)
await server.start()
```

### Client Side

```python
from exo.a2a import A2AClient, ClientConfig

config = ClientConfig(timeout=30.0)
client = A2AClient(config=config)

# Discover agent capabilities
card = await client.get_card("http://localhost:8001")
print(f"Agent: {card.name}")
print(f"Skills: {[s.name for s in card.skills]}")

# Send a task
result = await client.send_task(
    url="http://localhost:8001",
    input="What are the latest trends in AI?",
)
print(result)
```

## Agent Cards

An `AgentCard` describes an agent's identity, capabilities, and skills:

```python
from exo.a2a import AgentCard, AgentCapabilities, AgentSkill

card = AgentCard(
    name="data-analyst",
    description="Analyzes datasets and generates reports",
    url="http://localhost:8002",
    version="1.0.0",
    capabilities=AgentCapabilities(
        streaming=True,
        batch=True,
        push_notifications=False,
    ),
    skills=[
        AgentSkill(
            name="analyze",
            description="Analyze a dataset",
            input_schema={"type": "object", "properties": {"data_url": {"type": "string"}}},
        ),
        AgentSkill(
            name="visualize",
            description="Create data visualizations",
        ),
    ],
)
```

### Agent Discovery

Clients discover agents by fetching their card from a well-known endpoint:

```python
# Cards are served at /.well-known/agent-card (or /agent-card)
card = await client.get_card("http://agent-host:8001")
```

Cards can also be loaded from local files:

```python
card = await client.get_card("file:///path/to/agent-card.json")
```

## Transport Modes

| Mode | Value | Description |
|------|-------|-------------|
| `HTTP` | `"http"` | Standard HTTP request/response |
| `SSE` | `"sse"` | Server-Sent Events for streaming |
| `WEBSOCKET` | `"websocket"` | WebSocket for bidirectional communication |

## Task Lifecycle

Tasks follow a state machine:

| State | Description |
|-------|-------------|
| `PENDING` | Task created, waiting to execute |
| `RUNNING` | Currently executing |
| `COMPLETED` | Finished successfully |
| `FAILED` | Execution failed |
| `CANCELLED` | Cancelled by client |

```python
from exo.a2a import TaskState

# Task status updates are emitted during execution
# TaskStatusUpdateEvent: task_id, state, message
# TaskArtifactUpdateEvent: task_id, artifact_name, content
```

## A2AServer

The server uses FastAPI to expose agent capabilities:

```python
from exo.a2a import A2AServer, AgentCard, ServingConfig

server = A2AServer(
    card=agent_card,
    config=ServingConfig(
        host="0.0.0.0",
        port=8001,
        cors_origins=["*"],
    ),
)
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/agent-card` | GET | Return the agent card |
| `/.well-known/agent-card` | GET | Agent card discovery endpoint |
| `/tasks` | POST | Submit a new task |
| `/tasks/{id}` | GET | Get task status |
| `/tasks/{id}/stream` | GET | Stream task results (SSE) |

### Agent Executor

Register an executor that handles incoming tasks:

```python
from exo.a2a.server import AgentExecutor

class MyExecutor(AgentExecutor):
    async def execute(self, task_id: str, input: str) -> str:
        result = await my_agent.run(input)
        return result.output

    async def stream(self, task_id: str, input: str):
        async for chunk in my_agent.stream(input):
            yield chunk

server.register_executor(MyExecutor())
```

### Task Store

The server uses an in-memory task store by default. Implement the `TaskStore` protocol for persistence:

```python
from exo.a2a.server import TaskStore

class RedisTaskStore(TaskStore):
    async def create(self, task_id: str, input: str) -> None: ...
    async def get(self, task_id: str) -> dict | None: ...
    async def update(self, task_id: str, **kwargs) -> None: ...
    async def delete(self, task_id: str) -> None: ...
```

## A2AClient

The client handles HTTP communication and card resolution:

```python
from exo.a2a import A2AClient, ClientConfig

client = A2AClient(
    config=ClientConfig(
        timeout=30.0,
        max_retries=3,
    ),
)

# Get agent card
card = await client.get_card("http://agent:8001")

# Send task
result = await client.send_task("http://agent:8001", input="Hello")

# Stream results
async for event in client.stream_task("http://agent:8001", input="Hello"):
    print(event)
```

## RemoteAgent

`RemoteAgent` wraps an A2A client connection to make remote agents behave like local `Agent` instances:

```python
from exo.a2a import RemoteAgent

remote = RemoteAgent(
    url="http://researcher:8001",
    client=client,
)

# Use like a local agent
result = await remote.run("What is quantum computing?")
print(result)

# Describe returns the agent card info
info = remote.describe()
print(info["name"])  # "researcher"
```

## ClientManager

Thread-safe manager for multiple A2A client connections:

```python
from exo.a2a import ClientManager

manager = ClientManager()

# Register remote agents
manager.register("researcher", "http://researcher:8001")
manager.register("coder", "http://coder:8002")

# Get client for a specific agent
client = manager.get("researcher")
result = await client.send_task("http://researcher:8001", input="Find papers on LLMs")
```

## Advanced Patterns

### Multi-Agent Orchestration

Use A2A to coordinate multiple specialized agents:

```python
from exo.a2a import A2AClient, RemoteAgent

client = A2AClient()

researcher = RemoteAgent(url="http://researcher:8001", client=client)
writer = RemoteAgent(url="http://writer:8002", client=client)
reviewer = RemoteAgent(url="http://reviewer:8003", client=client)

# Pipeline: research -> write -> review
research = await researcher.run("Latest AI trends")
draft = await writer.run(f"Write an article based on: {research}")
review = await reviewer.run(f"Review this article: {draft}")
```

### Service Discovery

Dynamically discover and connect to agents:

```python
agent_urls = ["http://agent1:8001", "http://agent2:8002", "http://agent3:8003"]

available_agents = {}
for url in agent_urls:
    try:
        card = await client.get_card(url)
        available_agents[card.name] = RemoteAgent(url=url, client=client)
    except Exception:
        print(f"Agent at {url} unavailable")
```

### Streaming Integration

Stream results from a remote agent:

```python
async for event in client.stream_task("http://agent:8001", input="Explain Python"):
    if hasattr(event, "text"):
        print(event.text, end="", flush=True)
print()  # final newline
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `AgentCard` | `exo.a2a` | Agent identity, capabilities, and skills |
| `AgentCapabilities` | `exo.a2a` | Streaming, batch, push notification support |
| `AgentSkill` | `exo.a2a` | Named skill with optional input schema |
| `ServingConfig` | `exo.a2a` | Server configuration: host, port, CORS |
| `ClientConfig` | `exo.a2a` | Client configuration: timeout, retries |
| `A2AServer` | `exo.a2a` | FastAPI-based agent server |
| `A2AClient` | `exo.a2a` | HTTP client for A2A communication |
| `RemoteAgent` | `exo.a2a` | Agent-compatible wrapper for remote agents |
| `ClientManager` | `exo.a2a` | Thread-safe multi-client manager |
| `TaskState` | `exo.a2a` | Enum: `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED` |
| `TransportMode` | `exo.a2a` | Enum: `HTTP`, `SSE`, `WEBSOCKET` |
| `TaskStore` | `exo.a2a.server` | Protocol for task persistence |
| `AgentExecutor` | `exo.a2a.server` | Protocol for task execution |
