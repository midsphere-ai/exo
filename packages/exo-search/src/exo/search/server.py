"""FastAPI server for Exo Search search engine."""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]

from .config import SearchConfig
from .conversation import ConversationManager
from .pipeline import run_search_pipeline, stream_search_pipeline

_log = get_logger(__name__)

app = FastAPI(
    title="Exo Search - AI Search Engine",
    description="AI-powered search engine with source citations, matching Exo Search's architecture.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_conversations: dict[str, ConversationManager] = {}
_config_cache: dict[str, tuple[SearchConfig, float]] = {}
_CONFIG_TTL = 3600  # 1 hour


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
    cfg = SearchConfig()
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
    cfg = SearchConfig()
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
    cfg = SearchConfig()
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
            {"title": r.title, "url": r.url, "content": r.content} for r in search_results
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


# ---------------------------------------------------------------------------
# UI API models
# ---------------------------------------------------------------------------


class UISearchRequest(BaseModel):
    query: str
    mode: str = "balanced"
    session_id: str = "default"
    config: dict = {}


class UIConfigRequest(BaseModel):
    serper_api_key: str = ""
    jina_api_key: str = ""
    searxng_url: str = ""
    jina_reader_url: str = ""
    model: str = ""
    fast_model: str = ""
    embedding_model: str = ""
    api_key: str = ""
    base_url: str = ""


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

_PROVIDER_ENV_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def _build_config(src: UIConfigRequest | dict) -> SearchConfig:
    """Build a ``SearchConfig`` from a UI config request or raw dict."""
    if isinstance(src, dict):
        src = UIConfigRequest(**src)

    # Side-effect: inject provider credentials into env so the model layer picks
    # them up automatically.
    if src.api_key:
        parts = src.model.split(":", 1) if src.model else ["openai"]
        provider = parts[0] if len(parts) > 1 else "openai"
        env_var = _PROVIDER_ENV_MAP.get(provider, "OPENAI_API_KEY")
        os.environ[env_var] = src.api_key

    if src.base_url:
        os.environ["OPENAI_BASE_URL"] = src.base_url

    return SearchConfig(
        serper_api_key=src.serper_api_key,
        jina_api_key=src.jina_api_key,
        searxng_url=src.searxng_url,
        jina_reader_url=src.jina_reader_url,
        model=src.model,
        fast_model=src.fast_model,
        embedding_model=src.embedding_model,
    )


def _clean_stale_configs() -> None:
    """Remove entries from ``_config_cache`` older than ``_CONFIG_TTL``."""
    now = time.time()
    stale = [sid for sid, (_, ts) in _config_cache.items() if now - ts > _CONFIG_TTL]
    for sid in stale:
        del _config_cache[sid]


# ---------------------------------------------------------------------------
# UI endpoints
# ---------------------------------------------------------------------------


@app.post("/api/config/{session_id}")
async def set_config(session_id: str, body: UIConfigRequest):
    """Store a search configuration for a session."""
    _clean_stale_configs()
    cfg = _build_config(body)
    _config_cache[session_id] = (cfg, time.time())
    _log.info("config stored session=%s", session_id)
    return {"status": "ok", "session_id": session_id}


@app.post("/api/search")
async def ui_search_endpoint(body: UISearchRequest):
    """Non-streaming search for the UI."""
    _log.info("ui search session=%s q=%r mode=%s", body.session_id, body.query, body.mode)
    cfg = _build_config(body.config)
    conversation = _get_conversation(body.session_id)

    result = await run_search_pipeline(
        query=body.query,
        chat_history=conversation.turns,
        mode=body.mode,
        config=cfg,
    )

    conversation.add_turn(body.query, result.answer)
    return result.model_dump()


@app.get("/api/search/stream")
async def ui_search_stream_endpoint(
    q: str = Query(..., description="Search query"),
    mode: str = Query("balanced", description="Research quality mode"),
    session_id: str = Query("default", description="Session ID"),
):
    """Streaming search for the UI — uses stream_search_pipeline."""
    _log.info("ui stream session=%s q=%r mode=%s", session_id, q, mode)

    # Resolve config from cache (or default)
    if session_id in _config_cache:
        cfg, _ts = _config_cache[session_id]
        # Refresh timestamp and re-apply env vars
        _config_cache[session_id] = (cfg, time.time())
        _build_config(
            UIConfigRequest(
                serper_api_key=cfg.serper_api_key,
                jina_api_key=cfg.jina_api_key,
                searxng_url=cfg.searxng_url,
                jina_reader_url=cfg.jina_reader_url,
                model=cfg.model,
                fast_model=cfg.fast_model,
                embedding_model=cfg.embedding_model,
            )
        )
    else:
        cfg = SearchConfig()

    conversation = _get_conversation(session_id)
    history = conversation.turns

    async def event_stream() -> AsyncIterator[str]:
        from exo.types import TextEvent

        from .types import PipelineEvent, SearchResponse

        async for event in stream_search_pipeline(
            query=q,
            chat_history=history,
            mode=mode,
            config=cfg,
        ):
            if isinstance(event, PipelineEvent):
                yield (
                    f"event: status\n"
                    f"data: {json.dumps({'stage': event.stage, 'status': event.status, 'message': event.message})}\n\n"
                )
            elif isinstance(event, TextEvent):
                yield f"event: token\ndata: {json.dumps({'text': event.text})}\n\n"
            elif isinstance(event, SearchResponse):
                sources_data = [
                    {"title": s.title, "url": s.url, "content": s.content} for s in event.sources
                ]
                yield f"event: sources\ndata: {json.dumps({'sources': sources_data})}\n\n"
                yield f"event: suggestions\ndata: {json.dumps({'suggestions': event.suggestions})}\n\n"
                yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"

                # Persist conversation turn
                conversation.add_turn(q, event.answer)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/search/{session_id}")
async def ui_clear_session(session_id: str):
    """Clear conversation history and config for a UI session."""
    if session_id in _conversations:
        _conversations[session_id].clear()
    _config_cache.pop(session_id, None)
    _log.info("ui session cleared %s", session_id)
    return {"status": "cleared", "session_id": session_id}


# ---------------------------------------------------------------------------
# Static file mount (catch-all — MUST be last)
# ---------------------------------------------------------------------------

_ui_dir = Path(__file__).resolve().parent.parent.parent.parent / "ui"
if _ui_dir.exists():
    app.mount("/", StaticFiles(directory=str(_ui_dir), html=True), name="ui")
