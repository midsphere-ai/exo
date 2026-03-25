"""Simple chatbot running via distributed execution with result streaming.

Submits an agent to the distributed queue and streams events back
in real time.  A Redis server and at least one worker must be running.

Prerequisites:
    # Terminal 1 — start Redis
    docker run -p 6379:6379 redis:7

    # Terminal 2 — start a worker
    export EXO_REDIS_URL=redis://localhost:6379
    exo start worker

Usage:
    export OPENAI_API_KEY=sk-...
    export EXO_REDIS_URL=redis://localhost:6379
    uv run python examples/distributed/simple_chat.py
"""

import asyncio

from exo import Agent, tool
from exo.distributed import distributed  # pyright: ignore[reportMissingImports]
from exo.types import (
    ErrorEvent,
    StatusEvent,
    StepEvent,
    TextEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)

# -- Define a simple agent with one tool -----------------------------------

@tool
async def search_knowledge_base(query: str) -> str:
    """Search an internal knowledge base for information."""
    # Stub implementation — replace with a real search in production.
    return f"Found 3 results for '{query}': [result-1, result-2, result-3]"


agent = Agent(
    name="chat-bot",
    model="openai:gpt-4o-mini",
    instructions=(
        "You are a helpful support assistant. Use the search_knowledge_base "
        "tool when the user asks a question you don't know the answer to."
    ),
    tools=[search_knowledge_base],
)


# -- Submit to the distributed queue and stream events ---------------------

async def main() -> None:
    # Submit the agent task to the distributed queue.
    # detailed=True enables rich streaming events (steps, tool results, etc.)
    handle = await distributed(
        agent,
        "How do I reset my password?",
        detailed=True,
    )

    print(f"Task submitted: {handle.task_id}\n")

    # Stream events as they arrive from the worker.
    async for event in handle.stream():
        match event:
            case TextEvent():
                # Incremental text tokens from the model
                print(event.text, end="", flush=True)

            case ToolCallEvent():
                print(f"\n[tool call] {event.tool_name}")

            case ToolResultEvent():
                status = "ok" if event.success else "FAILED"
                print(f"[tool result] {event.tool_name} -> {status} ({event.duration_ms:.0f}ms)")

            case StepEvent():
                if event.status == "started":
                    print(f"\n--- step {event.step_number} ---")

            case StatusEvent():
                print(f"\n[status] {event.status}: {event.message}")

            case UsageEvent():
                u = event.usage
                print(f"[usage] {u.input_tokens} in / {u.output_tokens} out tokens")

            case ErrorEvent():
                print(f"\n[error] {event.error_type}: {event.error}")

    # Alternatively, wait for the final result without streaming:
    #   result = await handle.result()
    #   print(result)


if __name__ == "__main__":
    asyncio.run(main())
