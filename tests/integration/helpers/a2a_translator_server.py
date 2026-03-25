"""A2A translator agent server for integration tests.

Exposes a Spanish translator agent via A2A protocol on port 8766.
Run standalone: uvicorn tests.integration.helpers.a2a_translator_server:app --port 8766
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="A2A Translator Integration Test App")

_MODEL = "vertex:gemini-2.0-flash"
_PORT = 8766


@app.get("/.well-known/agent-card")
async def agent_card() -> dict:
    """A2A agent discovery endpoint."""
    return {
        "name": "translator",
        "description": "Spanish translator agent",
        "version": "1.0",
        "url": f"http://localhost:{_PORT}",
        "capabilities": {"streaming": False},
        "skills": [],
        "default_input_modes": ["text"],
        "default_output_modes": ["text"],
        "supported_transports": ["jsonrpc"],
    }


@app.post("/")
async def execute_task(payload: dict) -> JSONResponse:
    """Execute a translation task via the A2A protocol."""
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]

    text = payload.get("text", "")
    task_id = payload.get("task_id") or str(uuid.uuid4())

    try:
        provider = get_provider(_MODEL)
        agent = Agent(
            name="translator",
            model=_MODEL,
            instructions=(
                "You are a Spanish translator. "
                "Translate the input text to Spanish. "
                "Respond with ONLY the translated text, nothing else."
            ),
        )
        result = await agent.run(text, provider=provider)
        return JSONResponse(
            {
                "task_id": task_id,
                "status": {"state": "completed"},
                "artifact": {"text": result.text or "", "last_chunk": True},
            }
        )
    except Exception as exc:
        return JSONResponse(
            {
                "task_id": task_id,
                "status": {"state": "failed", "reason": str(exc)},
            },
            status_code=500,
        )
