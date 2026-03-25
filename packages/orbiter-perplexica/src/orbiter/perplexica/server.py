"""FastAPI server for Perplexica search engine."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from orbiter.observability.logging import get_logger  # pyright: ignore[reportMissingImports]

from .config import PerplexicaConfig
from .conversation import ConversationManager
from .pipeline import run_search_pipeline

_log = get_logger(__name__)

app = FastAPI(
    title="Perplexica - AI Search Engine",
    description="AI-powered search engine with source citations, matching Perplexica's architecture.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_conversations: dict[str, ConversationManager] = {}


def _get_conversation(session_id: str) -> ConversationManager:
    if session_id not in _conversations:
        _log.debug("new session %s", session_id)
        _conversations[session_id] = ConversationManager()
    return _conversations[session_id]


@app.get("/search")
async def search_endpoint(
    q: str = Query(..., description="Search query"),
    quality: str = Query("balanced", description="Research quality: speed, balanced, quality"),
    sources: str = Query("web", description="Comma-separated sources: web,academic,discussions"),
):
    """Search endpoint returning JSON with answer, sources, and suggestions."""
    _log.info("search request q=%r quality=%s sources=%s", q, quality, sources)
    cfg = PerplexicaConfig()
    cfg.sources = [s.strip() for s in sources.split(",")]

    result = await run_search_pipeline(
        query=q,
        mode=quality,
        config=cfg,
    )
    return result.model_dump()


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    quality: str = "balanced"
    sources: str = "web"


@app.post("/chat")
async def chat_endpoint(body: ChatRequest):
    """Multi-turn chat with conversation memory."""
    _log.info("chat request session=%s q=%r", body.session_id, body.query)
    cfg = PerplexicaConfig()
    cfg.sources = [s.strip() for s in body.sources.split(",")]
    conversation = _get_conversation(body.session_id)

    result = await run_search_pipeline(
        query=body.query,
        chat_history=conversation.turns,
        mode=body.quality,
        config=cfg,
    )

    conversation.add_turn(body.query, result.answer)
    return result.model_dump()


@app.get("/search/stream")
async def search_stream_endpoint(
    q: str = Query(..., description="Search query"),
    quality: str = Query("balanced", description="Research quality mode"),
    sources: str = Query("web", description="Comma-separated sources"),
):
    """Streaming search with SSE progress updates."""
    _log.info("stream request q=%r quality=%s sources=%s", q, quality, sources)
    cfg = PerplexicaConfig()
    cfg.sources = [s.strip() for s in sources.split(",")]

    async def event_stream() -> AsyncIterator[str]:
        from .agents.classifier import classify
        from .agents.researcher import research
        from .agents.suggestion_generator import generate_suggestions
        from .agents.writer import write_answer

        # Step 1: Classify
        yield f"event: status\ndata: {json.dumps({'stage': 'classifier', 'status': 'starting'})}\n\n"
        classification = await classify(q, [], cfg)
        effective_query = classification.standalone_follow_up or q
        yield f"event: status\ndata: {json.dumps({'stage': 'classifier', 'status': 'done', 'skip_search': classification.classification.skip_search})}\n\n"

        # Step 2: Research
        search_results = []
        if not classification.classification.skip_search:
            yield f"event: status\ndata: {json.dumps({'stage': 'researcher', 'status': 'starting'})}\n\n"
            search_results = await research(
                query=effective_query,
                classification=classification,
                chat_history=[],
                mode=quality,
                config=cfg,
            )
            yield f"event: status\ndata: {json.dumps({'stage': 'researcher', 'status': 'done', 'results_count': len(search_results)})}\n\n"

        # Step 3: Write
        yield f"event: status\ndata: {json.dumps({'stage': 'writer', 'status': 'starting'})}\n\n"
        answer = await write_answer(
            query=effective_query,
            search_results=search_results,
            chat_history=[],
            system_instructions=cfg.system_instructions,
            mode=quality,
            config=cfg,
        )
        yield f"event: answer\ndata: {json.dumps({'answer': answer})}\n\n"

        # Step 4: Sources
        sources_data = [
            {"title": r.title, "url": r.url, "content": r.content}
            for r in search_results
        ]
        yield f"event: sources\ndata: {json.dumps({'sources': sources_data})}\n\n"

        # Step 5: Suggestions
        yield f"event: status\ndata: {json.dumps({'stage': 'suggestions', 'status': 'starting'})}\n\n"
        suggestions = await generate_suggestions([(q, answer)], cfg)
        yield f"event: suggestions\ndata: {json.dumps({'suggestions': suggestions})}\n\n"

        yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/modes")
async def list_modes():
    """List available research quality modes."""
    return [
        {"name": "speed", "description": "Fast search with 1-2 searches"},
        {"name": "balanced", "description": "Balanced research with 2-6 searches"},
        {"name": "quality", "description": "Deep research with up to 25 iterations"},
    ]


@app.delete("/chat/{session_id}")
async def clear_chat(session_id: str):
    """Clear conversation history for a session."""
    if session_id in _conversations:
        _conversations[session_id].clear()
    _log.info("session cleared %s", session_id)
    return {"status": "cleared", "session_id": session_id}
