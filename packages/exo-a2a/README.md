# exo-a2a

Agent-to-Agent (A2A) protocol for the [Exo](../../README.md) multi-agent framework. Enable agents to communicate and delegate tasks across process and network boundaries.

## Installation

```bash
pip install exo-a2a
```

Requires Python 3.11+, `exo-core`, and `httpx>=0.27`.

## What's Included

- **A2A Server** -- expose agents as HTTP endpoints with standardized request/response schemas.
- **A2A Client** -- connect to remote agents and invoke them as if they were local.
- **RemoteAgent** -- wrapper that makes a remote agent behave like a local `Agent` for seamless integration into swarms.
- **Agent discovery** -- registry-based agent lookup for dynamic routing.

## Quick Example

```python
# Server side
from exo.a2a import A2AServer

server = A2AServer(agents=[my_agent])
server.run(port=8080)

# Client side
from exo.a2a import RemoteAgent

remote = RemoteAgent(url="http://localhost:8080", name="remote-agent")
result = await run(remote, "Hello from the client!")
```

## Documentation

- [A2A Guide](../../docs/guides/a2a.md)
- [API Reference](../../docs/reference/a2a/)
