"""Serve an agent via A2A protocol — FastAPI-based HTTP server.

Demonstrates ``A2AServer`` which exposes an agent as an HTTP
endpoint with agent-card discovery at ``/.well-known/agent-card``.

Usage:
    pip install fastapi uvicorn
    export OPENAI_API_KEY=sk-...
    uv run python examples/quickstart/serving.py
"""

from exo import Agent
from exo.a2a.server import A2AServer, AgentExecutor  # pyright: ignore[reportMissingImports]
from exo.a2a.types import ServingConfig  # pyright: ignore[reportMissingImports]

agent = Agent(
    name="weather-bot",
    model="openai:gpt-4o-mini",
    instructions="You are a helpful weather assistant.",
)

server = A2AServer(
    executor=AgentExecutor(agent),
    config=ServingConfig(host="0.0.0.0", port=8000),
)

app = server.build_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
