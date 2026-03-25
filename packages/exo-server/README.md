# exo-server

FastAPI-based HTTP server for serving [Exo](../../README.md) agents over HTTP, SSE, and WebSocket.

## Installation

```bash
pip install exo-server
```

Requires Python 3.11+, `exo-core`, `exo-models`, `fastapi>=0.115`, and `uvicorn>=0.30`.

## What's Included

- **Agent server** -- serve agents over REST API with chat completion endpoints.
- **Session management** -- stateful multi-turn conversations with session persistence.
- **SSE streaming** -- Server-Sent Events for real-time streaming responses.
- **WebSocket streaming** -- bidirectional WebSocket connections for interactive agents.
- **Agent registry** -- register and manage multiple agents on a single server.

## Quick Example

```python
from exo import Agent
from exo_server import create_app

agent = Agent(name="assistant", model="openai:gpt-4o-mini")
app = create_app(agents=[agent])

# Run with: uvicorn app:app --port 8000
```

## Documentation

- [Server Guide](../../docs/guides/server.md)
- [API Reference](../../docs/reference/server/)
