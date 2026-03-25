"""FastAPI endpoint that submits a distributed task and streams SSE events.

Demonstrates how to bridge Exo's distributed streaming with
Server-Sent Events so a web frontend can consume agent output in
real time.

Prerequisites:
    # Terminal 1 — start Redis
    docker run -p 6379:6379 redis:7

    # Terminal 2 — start a worker
    export EXO_REDIS_URL=redis://localhost:6379
    exo start worker

    # Terminal 3 — install extra deps and run this server
    pip install fastapi uvicorn
    export OPENAI_API_KEY=sk-...
    export EXO_REDIS_URL=redis://localhost:6379
    uv run uvicorn examples.distributed.fastapi_sse:app --reload

    # Terminal 4 — test with curl
    curl -N "http://localhost:8000/chat?message=How+do+I+reset+my+password"
"""

import json
from collections.abc import AsyncIterator

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

from exo import Agent, tool
from exo.distributed import distributed  # pyright: ignore[reportMissingImports]
from exo.types import StreamEvent

app = FastAPI(title="Exo Distributed SSE Example")


# -- Agent definition -------------------------------------------------------

@tool
async def lookup_docs(query: str) -> str:
    """Search the documentation for relevant articles."""
    return f"Found article: 'How to {query}' — follow the steps in Settings > Account."


agent = Agent(
    name="support-bot",
    model="openai:gpt-4o-mini",
    instructions=(
        "You are a helpful support assistant. Use the lookup_docs tool "
        "when the user asks a question."
    ),
    tools=[lookup_docs],
)


# -- SSE streaming helper ---------------------------------------------------

async def _event_generator(handle_stream: AsyncIterator[StreamEvent]) -> AsyncIterator[str]:
    """Convert Exo StreamEvents into SSE-formatted strings.

    Each event is sent as a JSON object with ``type`` and ``data`` fields,
    following the standard ``text/event-stream`` format.
    """
    async for event in handle_stream:
        payload = event.model_dump()
        event_type = payload.get("type", "message")
        data = json.dumps(payload)
        # SSE format: "event: <type>\ndata: <json>\n\n"
        yield f"event: {event_type}\ndata: {data}\n\n"


# -- Endpoints ---------------------------------------------------------------

@app.get("/chat")
async def chat(message: str = Query(..., description="User message")) -> StreamingResponse:
    """Submit a chat message and stream agent events as SSE.

    The endpoint returns a ``text/event-stream`` response.  Each SSE event
    has a ``type`` matching the Exo event type (``text``, ``tool_call``,
    ``tool_result``, ``status``, etc.) and a JSON ``data`` payload.

    Example JavaScript client::

        const source = new EventSource("/chat?message=Hello");
        source.addEventListener("text", (e) => {
            const data = JSON.parse(e.data);
            document.getElementById("output").textContent += data.text;
        });
        source.addEventListener("status", (e) => {
            const data = JSON.parse(e.data);
            if (data.status === "completed") source.close();
        });
    """
    # Submit to the distributed queue with rich events enabled.
    handle = await distributed(agent, message, detailed=True)

    return StreamingResponse(
        _event_generator(handle.stream()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Task-Id": handle.task_id,
        },
    )


@app.get("/task/{task_id}/status")
async def task_status(task_id: str) -> dict:
    """Check the status of a previously submitted task.

    Useful for polling when the SSE connection is lost.
    """
    import os

    from exo.distributed import TaskStore  # pyright: ignore[reportMissingImports]

    redis_url = os.environ["EXO_REDIS_URL"]
    store = TaskStore(redis_url)
    await store.connect()
    try:
        result = await store.get_status(task_id)
        if result is None:
            return {"task_id": task_id, "status": "not_found"}
        return result.model_dump()
    finally:
        await store.disconnect()
