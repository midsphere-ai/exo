"""FastAPI server with OpenAI-compatible chat completions API."""
from __future__ import annotations
import json
import time
import uuid
import asyncio
import logging
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import DeepSearchConfig
from .types import ChatCompletionRequest, ResearchResult

logger = logging.getLogger("deepsearch")

app = FastAPI(title="DeepSearch API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "deepsearch",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "deepsearch",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    from .engine import DeepSearchEngine

    config = DeepSearchConfig(
        token_budget=request.budget_tokens or 1_000_000,
        max_bad_attempts=request.max_attempts or 2,
        no_direct_answer=request.no_direct_answer,
        max_returned_urls=request.max_returned_urls,
        max_references=request.max_annotations,
        min_relevance_score=request.min_annotation_relevance,
        with_images=request.with_images,
        team_size=request.team_size,
    )

    if request.search_provider:
        config.search_provider = request.search_provider

    engine = DeepSearchEngine(config)

    # Extract question from messages
    question = ""
    messages = []
    for msg in request.messages:
        if msg.role == "system":
            continue
        messages.append({"role": msg.role, "content": msg.content})
        if msg.role == "user":
            question = msg.content

    if not question:
        raise HTTPException(status_code=400, detail="No user message found")

    if request.stream:
        return StreamingResponse(
            stream_research(engine, question, messages),
            media_type="text/event-stream",
        )

    # Non-streaming
    result = await engine.research(question, messages=messages)

    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "deepsearch",
        "system_fingerprint": "deepsearch-v1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.md_answer or result.answer,
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url_citation": {
                                "url": ref.url,
                                "title": ref.title,
                                "exact_quote": ref.exact_quote,
                            },
                        }
                        for ref in result.references
                    ],
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": result.usage,
        "visitedURLs": result.visited_urls,
        "readURLs": result.read_urls,
    }


async def stream_research(
    engine, question: str, messages: list[dict]
) -> AsyncIterator[str]:
    """Stream research progress as SSE events."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def make_chunk(content: str = "", finish_reason: str | None = None, chunk_type: str = "text") -> str:
        chunk = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "deepsearch",
            "system_fingerprint": "deepsearch-v1",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content, "type": chunk_type} if content else {},
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    # Stream thinking steps
    def on_action(action_data):
        think = action_data.get("think", "")
        if think:
            # Will be sent via the queue
            pass

    think_queue: asyncio.Queue = asyncio.Queue()

    original_track = None

    async def run_research():
        try:
            result = await engine.research(question, messages=messages)
            await think_queue.put(("result", result))
        except Exception as e:
            await think_queue.put(("error", str(e)))

    # Start research in background
    task = asyncio.create_task(run_research())

    # Stream think events
    yield make_chunk("Researching...\n\n", chunk_type="think")

    # Wait for result
    try:
        result_type, result_data = await think_queue.get()

        if result_type == "error":
            yield make_chunk(f"Error: {result_data}", chunk_type="error")
            yield make_chunk(finish_reason="error")
        else:
            # Stream the answer
            answer = result_data.md_answer or result_data.answer
            # Send in chunks for streaming effect
            chunk_size = 50
            for i in range(0, len(answer), chunk_size):
                yield make_chunk(answer[i:i + chunk_size])
                await asyncio.sleep(0.01)

            # Send annotations
            if result_data.references:
                annotations = [
                    {
                        "type": "url_citation",
                        "url_citation": {
                            "url": ref.url,
                            "title": ref.title,
                        },
                    }
                    for ref in result_data.references
                ]
                anno_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "deepsearch",
                    "system_fingerprint": "deepsearch-v1",
                    "choices": [{
                        "index": 0,
                        "delta": {"annotations": annotations},
                        "logprobs": None,
                        "finish_reason": None,
                    }],
                    "visitedURLs": result_data.visited_urls,
                    "readURLs": result_data.read_urls,
                }
                yield f"data: {json.dumps(anno_chunk)}\n\n"

            yield make_chunk(finish_reason="stop")
    except Exception as e:
        yield make_chunk(f"Error: {e}", chunk_type="error")
        yield make_chunk(finish_reason="error")

    yield "data: [DONE]\n\n"
    await task


@app.get("/health")
async def health():
    return {"status": "ok"}
