"""Playground WebSocket endpoint for streaming chat with agents."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from exo_web.database import get_db
from exo_web.routes.tools import BUILTIN_TOOLS
from exo_web.services.memory import memory_service

router = APIRouter(tags=["playground"])


# ---- Persistence helpers ----


async def _create_conversation(agent_id: str, user_id: str) -> str:
    """Create a new conversation and return its id."""
    conv_id = str(uuid.uuid4())
    async with get_db() as db:
        await db.execute(
            "INSERT INTO conversations (id, agent_id, user_id) VALUES (?, ?, ?)",
            (conv_id, agent_id, user_id),
        )
        await db.commit()
    return conv_id


async def _save_message(
    conversation_id: str,
    role: str,
    content: str,
    usage_json: str | None = None,
) -> str:
    """Save a message to the database and return its id."""
    msg_id = str(uuid.uuid4())
    async with get_db() as db:
        await db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, usage_json) VALUES (?, ?, ?, ?, ?)",
            (msg_id, conversation_id, role, content, usage_json),
        )
        await db.execute(
            "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
            (conversation_id,),
        )
        await db.commit()
    return msg_id


async def _load_conversation_messages(conversation_id: str) -> list[dict[str, str]]:
    """Load all messages from a conversation for the history."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        )
        rows = await cursor.fetchall()
    return [{"id": row["id"], "role": row["role"], "content": row["content"]} for row in rows]


class _TokenCollector:
    """Wraps a WebSocket to collect streamed token content and trace data."""

    def __init__(self, websocket: WebSocket, cancel_event: asyncio.Event | None = None) -> None:
        self._ws = websocket
        self.collected = ""
        self.usage: dict[str, int] | None = None
        self._cancel = cancel_event

    async def send_json(self, data: Any) -> None:
        # Check if takeover was requested — stop forwarding tokens
        if self._cancel and self._cancel.is_set():
            raise asyncio.CancelledError("takeover")
        if data.get("type") == "token":
            self.collected += data.get("content", "")
        elif data.get("type") == "done":
            self.usage = data.get("usage")
        await self._ws.send_json(data)


async def _get_user_from_cookie(websocket: WebSocket) -> dict[str, Any] | None:
    """Extract user from session cookie on the WebSocket connection."""
    session_id = websocket.cookies.get("exo_session")
    if not session_id:
        return None

    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT u.id, u.email, u.created_at
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.id = ? AND s.expires_at > datetime('now')
            """,
            (session_id,),
        )
        row = await cursor.fetchone()

    return dict(row) if row else None


async def _get_agent(agent_id: str, user_id: str) -> dict[str, Any] | None:
    """Load agent config from DB."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM agents WHERE id = ? AND user_id = ?",
            (agent_id, user_id),
        )
        row = await cursor.fetchone()
    return dict(row) if row else None


async def _resolve_api_key(provider_id: str, user_id: str) -> tuple[dict[str, Any] | None, str]:
    """Resolve provider config and API key. Returns (provider_dict, api_key)."""
    from exo_web.crypto import decrypt_api_key

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM providers WHERE id = ? AND user_id = ?",
            (provider_id, user_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return None, ""
        provider = dict(row)

        api_key = ""
        if provider.get("encrypted_api_key"):
            api_key = decrypt_api_key(provider["encrypted_api_key"])
        else:
            cursor = await db.execute(
                "SELECT encrypted_api_key FROM provider_keys WHERE provider_id = ? AND status = 'active' LIMIT 1",
                (provider_id,),
            )
            key_row = await cursor.fetchone()
            if key_row:
                api_key = decrypt_api_key(key_row["encrypted_api_key"])

    return provider, api_key


async def _find_provider_by_type(provider_type: str, user_id: str) -> str | None:
    """Find a provider ID by type for the given user."""
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT id FROM providers
            WHERE provider_type = ? AND user_id = ?
            AND (
                encrypted_api_key IS NOT NULL AND encrypted_api_key != ''
                OR EXISTS (
                    SELECT 1 FROM provider_keys pk
                    WHERE pk.provider_id = providers.id AND pk.status = 'active'
                )
            )
            LIMIT 1
            """,
            (provider_type, user_id),
        )
        row = await cursor.fetchone()
    return row["id"] if row else None


async def _resolve_agent_tool_schemas(
    tools_json: str | None,
) -> list[dict[str, Any]]:
    """Build OpenAI-format tool schemas from the agent's tools_json.

    Returns a list of tool definitions suitable for the OpenAI ``tools``
    parameter.  Built-in tools (``builtin:*``) are resolved from
    ``BUILTIN_TOOLS``; user-defined tools are looked up in the database.
    """
    try:
        tool_ids: list[str] = json.loads(tools_json or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    if not tool_ids:
        return []

    schemas: list[dict[str, Any]] = []
    db_ids: list[str] = []

    for tid in tool_ids:
        if isinstance(tid, dict):
            # Sub-agent tool or inline tool definition
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tid.get("name", tid.get("id", "unknown")),
                        "description": tid.get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task": {
                                    "type": "string",
                                    "description": "The task to delegate",
                                },
                            },
                            "required": ["task"],
                        },
                    },
                }
            )
        elif isinstance(tid, str) and tid.startswith("builtin:"):
            for bt in BUILTIN_TOOLS:
                if bt["id"] == tid:
                    try:
                        params = json.loads(bt["schema_json"])
                    except (json.JSONDecodeError, TypeError):
                        params = {"type": "object", "properties": {}}
                    schemas.append(
                        {
                            "type": "function",
                            "function": {
                                "name": bt["name"],
                                "description": bt["description"],
                                "parameters": params,
                            },
                        }
                    )
                    break
        elif isinstance(tid, str):
            db_ids.append(tid)

    if db_ids:
        async with get_db() as db:
            placeholders = ", ".join("?" for _ in db_ids)
            cursor = await db.execute(
                f"SELECT name, description, schema_json FROM tools WHERE id IN ({placeholders})",
                db_ids,
            )
            rows = await cursor.fetchall()
        for row in rows:
            r = dict(row)
            try:
                params = json.loads(r.get("schema_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                params = {"type": "object", "properties": {}}
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": r["name"],
                        "description": r.get("description", ""),
                        "parameters": params,
                    },
                }
            )

    return schemas


async def _execute_sandbox(code: str, user_id: str) -> dict[str, Any]:
    """Execute code in the sandbox and return a result dict."""
    from exo_web.services.sandbox import SandboxConfig, execute_code

    # Load user sandbox config
    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM sandbox_configs WHERE user_id = ?", (user_id,))
        row = await cursor.fetchone()

    if row:
        row_d = dict(row)
        try:
            libs = json.loads(row_d.get("allowed_libraries", "[]"))
        except (json.JSONDecodeError, TypeError):
            libs = SandboxConfig().allowed_libraries
        config = SandboxConfig(
            allowed_libraries=libs,
            timeout_seconds=row_d.get("timeout_seconds", 30),
            memory_limit_mb=row_d.get("memory_limit_mb", 256),
        )
    else:
        config = SandboxConfig()

    result = execute_code(code, config)
    return {
        "success": result.success,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": result.error,
        "generated_files": result.generated_files,
        "execution_time_ms": result.execution_time_ms,
    }


async def _stream_openai(
    websocket: WebSocket,
    api_key: str,
    base_url: str,
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
    tools: list[dict[str, Any]] | None = None,
    user_id: str = "",
) -> None:
    """Stream a response from an OpenAI-compatible API."""
    import httpx

    url = (base_url or "https://api.openai.com") + "/v1/chat/completions"
    body: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": temperature,
    }
    if max_tokens:
        body["max_tokens"] = max_tokens
    if tools:
        body["tools"] = tools

    prompt_tokens = 0
    completion_tokens = 0
    finish_reason: str | None = None
    tool_calls_acc: dict[int, dict[str, Any]] = {}
    start_time = time.monotonic()

    async with (
        httpx.AsyncClient(timeout=120.0) as client,
        client.stream(
            "POST",
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
        ) as resp,
    ):
        if resp.status_code >= 400:
            error_body = await resp.aread()
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"API error ({resp.status_code}): {error_body.decode()[:300]}",
                }
            )
            return

        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # Extract usage from final chunk if present
            if usage := chunk.get("usage"):
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

            choices = chunk.get("choices", [])
            if choices:
                choice = choices[0]
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content:
                    await websocket.send_json({"type": "token", "content": content})

                # Accumulate tool call deltas
                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"name": "", "arguments": "", "id": ""}
                    if tc.get("id"):
                        tool_calls_acc[idx]["id"] = tc["id"]
                    if tc.get("function", {}).get("name"):
                        tool_calls_acc[idx]["name"] = tc["function"]["name"]
                    if tc.get("function", {}).get("arguments"):
                        tool_calls_acc[idx]["arguments"] += tc["function"]["arguments"]

                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

    latency_ms = round((time.monotonic() - start_time) * 1000)

    # Execute code_interpreter tool calls via sandbox
    for tc in tool_calls_acc.values():
        if tc["name"] == "code_interpreter" and user_id:
            try:
                args = json.loads(tc["arguments"])
                code = args.get("code", "")
            except (json.JSONDecodeError, TypeError):
                code = ""
            if code:
                sandbox_result = await _execute_sandbox(code, user_id)
                tc["result"] = json.dumps(sandbox_result, default=str)
                # Send sandbox result with generated files to client
                await websocket.send_json(
                    {
                        "type": "sandbox_result",
                        "success": sandbox_result["success"],
                        "stdout": sandbox_result["stdout"],
                        "stderr": sandbox_result["stderr"],
                        "error": sandbox_result.get("error"),
                        "generated_files": sandbox_result["generated_files"],
                        "execution_time_ms": sandbox_result["execution_time_ms"],
                    }
                )

    # Send tool call trace events
    for tc in tool_calls_acc.values():
        await websocket.send_json(
            {
                "type": "tool_call",
                "name": tc["name"],
                "arguments": tc["arguments"],
                "result": tc.get("result"),
                "duration_ms": round((time.monotonic() - start_time) * 1000)
                if tc.get("result")
                else None,
            }
        )

    await websocket.send_json(
        {
            "type": "done",
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            "model": model_name,
            "finish_reason": finish_reason or "stop",
            "latency_ms": latency_ms,
        }
    )


async def _stream_anthropic(
    websocket: WebSocket,
    api_key: str,
    model_name: str,
    system_prompt: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
    tools: list[dict[str, Any]] | None = None,
    user_id: str = "",
) -> None:
    """Stream a response from the Anthropic API."""
    import httpx

    url = "https://api.anthropic.com/v1/messages"
    body: dict[str, Any] = {
        "model": model_name,
        "messages": [m for m in messages if m["role"] != "system"],
        "stream": True,
        "max_tokens": max_tokens or 1024,
        "temperature": temperature,
    }
    if system_prompt:
        body["system"] = system_prompt
    # Convert OpenAI tool format to Anthropic tool format
    if tools:
        anthropic_tools = []
        for t in tools:
            fn = t.get("function", {})
            anthropic_tools.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        body["tools"] = anthropic_tools

    input_tokens = 0
    output_tokens = 0
    finish_reason: str | None = None
    # Track content blocks for tool_use and thinking
    current_block_type: str | None = None
    current_tool_name = ""
    current_tool_input = ""
    thinking_content = ""
    start_time = time.monotonic()

    async with (
        httpx.AsyncClient(timeout=120.0) as client,
        client.stream(
            "POST",
            url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=body,
        ) as resp,
    ):
        if resp.status_code >= 400:
            error_body = await resp.aread()
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"API error ({resp.status_code}): {error_body.decode()[:300]}",
                }
            )
            return

        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")
            if event_type == "content_block_start":
                block = event.get("content_block", {})
                current_block_type = block.get("type")
                if current_block_type == "tool_use":
                    current_tool_name = block.get("name", "")
                    current_tool_input = ""
                elif current_block_type == "thinking":
                    thinking_content = block.get("thinking", "")
            elif event_type == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        await websocket.send_json({"type": "token", "content": text})
                elif delta.get("type") == "input_json_delta":
                    current_tool_input += delta.get("partial_json", "")
                elif delta.get("type") == "thinking_delta":
                    thinking_content += delta.get("thinking", "")
            elif event_type == "content_block_stop":
                if current_block_type == "tool_use" and current_tool_name:
                    tool_result_str: str | None = None
                    # Execute code_interpreter via sandbox
                    if current_tool_name == "code_interpreter" and user_id:
                        try:
                            args = json.loads(current_tool_input)
                            code = args.get("code", "")
                        except (json.JSONDecodeError, TypeError):
                            code = ""
                        if code:
                            sandbox_result = await _execute_sandbox(code, user_id)
                            tool_result_str = json.dumps(sandbox_result, default=str)
                            await websocket.send_json(
                                {
                                    "type": "sandbox_result",
                                    "success": sandbox_result["success"],
                                    "stdout": sandbox_result["stdout"],
                                    "stderr": sandbox_result["stderr"],
                                    "error": sandbox_result.get("error"),
                                    "generated_files": sandbox_result["generated_files"],
                                    "execution_time_ms": sandbox_result["execution_time_ms"],
                                }
                            )
                    await websocket.send_json(
                        {
                            "type": "tool_call",
                            "name": current_tool_name,
                            "arguments": current_tool_input,
                            "result": tool_result_str,
                        }
                    )
                    current_tool_name = ""
                    current_tool_input = ""
                elif current_block_type == "thinking" and thinking_content:
                    await websocket.send_json(
                        {
                            "type": "reasoning",
                            "content": thinking_content,
                        }
                    )
                    thinking_content = ""
                current_block_type = None
            elif event_type == "message_start":
                usage = event.get("message", {}).get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
            elif event_type == "message_delta":
                usage = event.get("usage", {})
                output_tokens = usage.get("output_tokens", 0)
                if event.get("delta", {}).get("stop_reason"):
                    finish_reason = event["delta"]["stop_reason"]

    latency_ms = round((time.monotonic() - start_time) * 1000)

    await websocket.send_json(
        {
            "type": "done",
            "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens},
            "model": model_name,
            "finish_reason": finish_reason or "end_turn",
            "latency_ms": latency_ms,
        }
    )


async def _stream_gemini(
    websocket: WebSocket,
    api_key: str,
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
) -> None:
    """Stream a response from the Gemini API."""
    import httpx

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse&key={api_key}"

    # Convert messages to Gemini format
    contents = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        if msg["role"] == "system":
            role = "user"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    body: dict[str, Any] = {"contents": contents}
    if temperature or max_tokens:
        gen_config: dict[str, Any] = {}
        if temperature is not None:
            gen_config["temperature"] = temperature
        if max_tokens:
            gen_config["maxOutputTokens"] = max_tokens
        body["generationConfig"] = gen_config

    prompt_tokens = 0
    completion_tokens = 0
    finish_reason: str | None = None
    start_time = time.monotonic()

    async with (
        httpx.AsyncClient(timeout=120.0) as client,
        client.stream("POST", url, json=body) as resp,
    ):
        if resp.status_code >= 400:
            error_body = await resp.aread()
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"API error ({resp.status_code}): {error_body.decode()[:300]}",
                }
            )
            return

        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            candidates = chunk.get("candidates", [])
            if candidates:
                candidate = candidates[0]
                parts = candidate.get("content", {}).get("parts", [])
                for part in parts:
                    text = part.get("text", "")
                    if text:
                        await websocket.send_json({"type": "token", "content": text})
                    # Gemini function calls
                    if fc := part.get("functionCall"):
                        await websocket.send_json(
                            {
                                "type": "tool_call",
                                "name": fc.get("name", ""),
                                "arguments": json.dumps(fc.get("args", {})),
                            }
                        )
                if candidate.get("finishReason"):
                    finish_reason = candidate["finishReason"]

            usage_meta = chunk.get("usageMetadata", {})
            if usage_meta:
                prompt_tokens = usage_meta.get("promptTokenCount", prompt_tokens)
                completion_tokens = usage_meta.get("candidatesTokenCount", completion_tokens)

    latency_ms = round((time.monotonic() - start_time) * 1000)

    await websocket.send_json(
        {
            "type": "done",
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            "model": model_name,
            "finish_reason": finish_reason or "STOP",
            "latency_ms": latency_ms,
        }
    )


async def _stream_ollama(
    websocket: WebSocket,
    base_url: str,
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> None:
    """Stream a response from an Ollama API."""
    import httpx

    url = (base_url or "http://localhost:11434") + "/api/chat"
    body: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": True,
    }
    if temperature is not None:
        body["options"] = {"temperature": temperature}

    prompt_tokens = 0
    completion_tokens = 0
    finish_reason = "stop"
    start_time = time.monotonic()

    async with (
        httpx.AsyncClient(timeout=120.0) as client,
        client.stream("POST", url, json=body) as resp,
    ):
        if resp.status_code >= 400:
            error_body = await resp.aread()
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"API error ({resp.status_code}): {error_body.decode()[:300]}",
                }
            )
            return

        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = chunk.get("message", {}).get("content", "")
            if content:
                await websocket.send_json({"type": "token", "content": content})

            if chunk.get("done"):
                prompt_tokens = chunk.get("prompt_eval_count", 0)
                completion_tokens = chunk.get("eval_count", 0)
                if chunk.get("done_reason"):
                    finish_reason = chunk["done_reason"]

    latency_ms = round((time.monotonic() - start_time) * 1000)

    await websocket.send_json(
        {
            "type": "done",
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            "model": model_name,
            "finish_reason": finish_reason,
            "latency_ms": latency_ms,
        }
    )


async def _run_stream(
    provider_type: str,
    collector: _TokenCollector,
    api_key: str,
    base_url: str,
    model_name: str,
    system_prompt: str,
    history: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
    tools: list[dict[str, Any]] | None = None,
    user_id: str = "",
) -> None:
    """Dispatch to the appropriate streaming function."""
    if provider_type in ("openai", "custom"):
        await _stream_openai(
            collector,
            api_key,
            base_url,
            model_name,
            history,
            temperature,
            max_tokens,  # type: ignore[arg-type]
            tools=tools,
            user_id=user_id,
        )
    elif provider_type == "anthropic":
        await _stream_anthropic(
            collector,
            api_key,
            model_name,
            system_prompt,
            history,
            temperature,
            max_tokens,  # type: ignore[arg-type]
            tools=tools,
            user_id=user_id,
        )
    elif provider_type == "gemini":
        await _stream_gemini(
            collector,
            api_key,
            model_name,
            history,
            temperature,
            max_tokens,  # type: ignore[arg-type]
        )
    elif provider_type == "ollama":
        await _stream_ollama(collector, base_url, model_name, history, temperature)  # type: ignore[arg-type]
    else:
        msg = f"Unsupported provider: {provider_type}"
        raise ValueError(msg)


# ---- Compare mode helpers ----
# NOTE: The compare endpoint MUST be defined before the {agent_id} route
# so that FastAPI doesn't match "compare" as an agent_id.


class _IndexedCollector:
    """Wraps a WebSocket to prefix all messages with a model_index."""

    def __init__(self, websocket: WebSocket, model_index: int) -> None:
        self._ws = websocket
        self._idx = model_index
        self.collected = ""
        self.usage: dict[str, int] | None = None

    async def send_json(self, data: Any) -> None:
        if data.get("type") == "token":
            self.collected += data.get("content", "")
        elif data.get("type") == "done":
            self.usage = data.get("usage")
        data["model_index"] = self._idx
        await self._ws.send_json(data)


async def _stream_for_model(
    websocket: WebSocket,
    model_index: int,
    provider_type: str,
    api_key: str,
    base_url: str,
    model_name: str,
    system_prompt: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int | None,
) -> None:
    """Stream a single model response, tagged with model_index."""
    collector = _IndexedCollector(websocket, model_index)
    try:
        if provider_type in ("openai", "custom"):
            await _stream_openai(
                collector,
                api_key,
                base_url,
                model_name,
                messages,
                temperature,
                max_tokens,  # type: ignore[arg-type]
            )
        elif provider_type == "anthropic":
            await _stream_anthropic(
                collector,
                api_key,
                model_name,
                system_prompt,
                messages,
                temperature,
                max_tokens,  # type: ignore[arg-type]
            )
        elif provider_type == "gemini":
            await _stream_gemini(
                collector,
                api_key,
                model_name,
                messages,
                temperature,
                max_tokens,  # type: ignore[arg-type]
            )
        elif provider_type == "ollama":
            await _stream_ollama(collector, base_url, model_name, messages, temperature)  # type: ignore[arg-type]
        else:
            await websocket.send_json(
                {
                    "type": "error",
                    "model_index": model_index,
                    "message": f"Unsupported provider: {provider_type}",
                }
            )
    except Exception as exc:
        await websocket.send_json(
            {"type": "error", "model_index": model_index, "message": f"Stream error: {exc!s}"}
        )


@router.websocket("/api/v1/playground/compare/chat")
async def playground_compare(websocket: WebSocket) -> None:
    """WebSocket endpoint for multi-model comparison.

    Client sends: { models: [{provider_type, model_name}], content: "..." }
    Server streams responses tagged with model_index for each model.
    """
    import asyncio

    user = await _get_user_from_cookie(websocket)
    if user is None:
        await websocket.close(code=4001, reason="Not authenticated")
        return

    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            content = msg.get("content", "").strip()
            models = msg.get("models", [])

            if not content:
                await websocket.send_json({"type": "error", "message": "Empty message"})
                continue
            if not models or len(models) < 2:
                await websocket.send_json({"type": "error", "message": "Select at least 2 models"})
                continue
            if len(models) > 4:
                await websocket.send_json({"type": "error", "message": "Maximum 4 models"})
                continue

            # Resolve each model's provider and key
            tasks = []
            for i, m in enumerate(models):
                ptype = m.get("provider_type", "")
                mname = m.get("model_name", "")
                if not ptype or not mname:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "model_index": i,
                            "message": "Missing provider or model name",
                        }
                    )
                    continue

                pid = await _find_provider_by_type(ptype, user["id"])
                if not pid:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "model_index": i,
                            "message": f"No {ptype} provider configured",
                        }
                    )
                    continue

                provider, api_key = await _resolve_api_key(pid, user["id"])
                if not api_key:
                    await websocket.send_json(
                        {"type": "error", "model_index": i, "message": "No API key for provider"}
                    )
                    continue

                burl = (provider or {}).get("base_url", "") or ""
                system_prompt = msg.get("system_prompt", "")
                temperature = m.get("temperature", 0.7)
                max_tokens = m.get("max_tokens")

                messages = [{"role": "user", "content": content}]
                if system_prompt:
                    messages.insert(0, {"role": "system", "content": system_prompt})

                tasks.append(
                    _stream_for_model(
                        websocket,
                        i,
                        ptype,
                        api_key,
                        burl,
                        mname,
                        system_prompt,
                        messages,
                        temperature,
                        max_tokens,
                    )
                )

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    except WebSocketDisconnect:
        pass


@router.websocket("/api/v1/playground/{agent_id}/chat")
async def playground_chat(websocket: WebSocket, agent_id: str) -> None:
    """WebSocket endpoint for streaming chat with an agent."""
    # Authenticate via session cookie
    user = await _get_user_from_cookie(websocket)
    if user is None:
        await websocket.close(code=4001, reason="Not authenticated")
        return

    # Load agent
    agent = await _get_agent(agent_id, user["id"])
    if agent is None:
        await websocket.close(code=4004, reason="Agent not found")
        return

    await websocket.accept()

    # Resolve provider and API key
    provider_type = agent.get("model_provider", "")
    model_name = agent.get("model_name", "")
    if not provider_type or not model_name:
        await websocket.send_json({"type": "error", "message": "Agent has no model configured"})
        await websocket.close()
        return

    provider_id = await _find_provider_by_type(provider_type, user["id"])
    if not provider_id:
        await websocket.send_json(
            {"type": "error", "message": f"No {provider_type} provider configured"}
        )
        await websocket.close()
        return

    provider, api_key = await _resolve_api_key(provider_id, user["id"])
    if not api_key:
        await websocket.send_json(
            {"type": "error", "message": "No API key configured for provider"}
        )
        await websocket.close()
        return

    base_url = (provider or {}).get("base_url", "") or ""
    temperature = agent.get("temperature") or 0.7
    max_tokens = agent.get("max_tokens")

    # Memory strategy from agent config
    memory_type = agent.get("context_memory_type", "conversation")

    # Resolve tool schemas for function calling
    tool_schemas = await _resolve_agent_tool_schemas(agent.get("tools_json"))

    # Build system prompt from agent config
    system_prompt = ""
    persona_parts = []
    if agent.get("persona_role"):
        persona_parts.append(f"Role: {agent['persona_role']}")
    if agent.get("persona_goal"):
        persona_parts.append(f"Goal: {agent['persona_goal']}")
    if agent.get("persona_backstory"):
        persona_parts.append(f"Backstory: {agent['persona_backstory']}")
    if persona_parts:
        system_prompt = "## Persona\n" + "\n".join(persona_parts) + "\n\n"
    if agent.get("instructions"):
        system_prompt += agent["instructions"]

    # Conversation persistence
    conversation_id: str | None = None

    # Conversation history (backed by DB when conversation_id is set)
    history: list[dict[str, str]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    # Takeover state
    cancel_event = asyncio.Event()
    collector: _TokenCollector | None = None
    stream_task: asyncio.Task[None] | None = None
    is_paused = False

    # Message queue for commands received while streaming
    msg_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def _receive_loop() -> None:
        """Read from WebSocket and dispatch messages."""
        while True:
            raw = await websocket.receive_text()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue
            await msg_queue.put(parsed)

    receive_task = asyncio.create_task(_receive_loop())

    try:
        while True:
            msg = await msg_queue.get()

            # Handle conversation loading
            if msg.get("type") == "load_conversation":
                conv_id = msg.get("conversation_id", "")
                if conv_id:
                    saved_messages = await _load_conversation_messages(conv_id)
                    if saved_messages:
                        conversation_id = conv_id
                        # Rebuild history: system prompt + memory-managed messages
                        history = []
                        if system_prompt:
                            history.append({"role": "system", "content": system_prompt})
                        # Use memory service to build context-aware history
                        if memory_type != "conversation":
                            mem = await memory_service.get_memory(
                                agent_id,
                                conv_id,
                                memory_type=memory_type,
                            )
                            history.extend(mem)
                        else:
                            history.extend(saved_messages)
                        await websocket.send_json(
                            {
                                "type": "conversation_loaded",
                                "conversation_id": conv_id,
                                "messages": saved_messages,
                            }
                        )
                    else:
                        await websocket.send_json(
                            {"type": "error", "message": "Conversation not found"}
                        )
                continue

            # ---- Takeover commands (only valid while streaming) ----
            if msg.get("type") == "takeover":
                if stream_task and not stream_task.done():
                    cancel_event.set()
                    stream_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await stream_task
                    is_paused = True
                    # Save partial response so far
                    partial = collector.collected if collector else ""
                    await websocket.send_json(
                        {
                            "type": "takeover_ack",
                            "partial_content": partial,
                        }
                    )
                    await websocket.send_json(
                        {
                            "type": "takeover_event",
                            "action": "paused",
                            "timestamp": time.time(),
                        }
                    )
                else:
                    await websocket.send_json(
                        {"type": "error", "message": "No active stream to take over"}
                    )
                continue

            if msg.get("type") == "stop":
                if is_paused:
                    # Save partial content as the assistant response
                    partial = collector.collected if collector else ""
                    if partial and conversation_id:
                        history.append({"role": "assistant", "content": partial})
                        usage_str = (
                            json.dumps(collector.usage) if (collector and collector.usage) else None
                        )
                        await _save_message(conversation_id, "assistant", partial, usage_str)
                    is_paused = False
                    cancel_event.clear()
                    await websocket.send_json(
                        {
                            "type": "takeover_stopped",
                            "partial_content": partial,
                        }
                    )
                    await websocket.send_json(
                        {
                            "type": "takeover_event",
                            "action": "stopped",
                            "timestamp": time.time(),
                        }
                    )
                else:
                    await websocket.send_json({"type": "error", "message": "Not in takeover mode"})
                continue

            if msg.get("type") == "resume":
                if is_paused:
                    # Save partial as assistant message, then resume agent
                    partial = collector.collected if collector else ""
                    if partial and conversation_id:
                        history.append({"role": "assistant", "content": partial})
                        usage_str = (
                            json.dumps(collector.usage) if (collector and collector.usage) else None
                        )
                        await _save_message(conversation_id, "assistant", partial, usage_str)
                    is_paused = False
                    cancel_event.clear()
                    await websocket.send_json(
                        {
                            "type": "takeover_event",
                            "action": "resumed",
                            "timestamp": time.time(),
                        }
                    )
                    # Re-stream: send the history back to the model for a fresh response
                    cancel_event = asyncio.Event()
                    collector = _TokenCollector(websocket, cancel_event)
                    try:
                        await _run_stream(
                            provider_type,
                            collector,
                            api_key,
                            base_url,
                            model_name,
                            system_prompt,
                            history,
                            temperature,
                            max_tokens,
                            tools=tool_schemas or None,
                            user_id=user["id"],
                        )
                    except (asyncio.CancelledError, Exception) as exc:
                        if not cancel_event.is_set():
                            await websocket.send_json(
                                {"type": "error", "message": f"Stream error: {exc!s}"}
                            )
                        continue
                    # Persist the resumed assistant response
                    assistant_content = collector.collected or "(empty response)"
                    history.append({"role": "assistant", "content": assistant_content})
                    usage_str = json.dumps(collector.usage) if collector.usage else None
                    if conversation_id:
                        await _save_message(
                            conversation_id, "assistant", assistant_content, usage_str
                        )
                else:
                    await websocket.send_json({"type": "error", "message": "Not in takeover mode"})
                continue

            if msg.get("type") == "inject":
                if is_paused:
                    inject_content = msg.get("content", "").strip()
                    if not inject_content:
                        await websocket.send_json(
                            {"type": "error", "message": "Empty inject message"}
                        )
                        continue
                    # Save partial assistant response
                    partial = collector.collected if collector else ""
                    if partial and conversation_id:
                        history.append({"role": "assistant", "content": partial})
                        usage_str = (
                            json.dumps(collector.usage) if (collector and collector.usage) else None
                        )
                        await _save_message(conversation_id, "assistant", partial, usage_str)
                    # Add injected user message
                    history.append({"role": "user", "content": inject_content})
                    if conversation_id:
                        await _save_message(conversation_id, "user", inject_content)
                    is_paused = False
                    cancel_event.clear()
                    await websocket.send_json(
                        {
                            "type": "takeover_event",
                            "action": "injected",
                            "content": inject_content,
                            "timestamp": time.time(),
                        }
                    )
                    # Stream response with injected context
                    cancel_event = asyncio.Event()
                    collector = _TokenCollector(websocket, cancel_event)
                    try:
                        await _run_stream(
                            provider_type,
                            collector,
                            api_key,
                            base_url,
                            model_name,
                            system_prompt,
                            history,
                            temperature,
                            max_tokens,
                            tools=tool_schemas or None,
                            user_id=user["id"],
                        )
                    except (asyncio.CancelledError, Exception) as exc:
                        if not cancel_event.is_set():
                            await websocket.send_json(
                                {"type": "error", "message": f"Stream error: {exc!s}"}
                            )
                        continue
                    # Persist the new assistant response
                    assistant_content = collector.collected or "(empty response)"
                    history.append({"role": "assistant", "content": assistant_content})
                    usage_str = json.dumps(collector.usage) if collector.usage else None
                    if conversation_id:
                        await _save_message(
                            conversation_id, "assistant", assistant_content, usage_str
                        )
                else:
                    await websocket.send_json({"type": "error", "message": "Not in takeover mode"})
                continue

            # ---- Regular message ----
            content = msg.get("content", "").strip()
            if not content:
                await websocket.send_json({"type": "error", "message": "Empty message"})
                continue

            # Create conversation on first real message if not already loaded
            if conversation_id is None:
                conversation_id = await _create_conversation(agent_id, user["id"])
                await websocket.send_json(
                    {"type": "conversation_created", "conversation_id": conversation_id}
                )
                # Save system prompt as first message
                if system_prompt:
                    await _save_message(conversation_id, "system", system_prompt)
                # Inject persisted memory into history
                memory_messages = await memory_service.get_memory(
                    agent_id,
                    conversation_id,
                    memory_type=memory_type,
                )
                if memory_messages:
                    # Insert memory after system prompt but before new messages
                    insert_idx = 1 if system_prompt else 0
                    for mm in memory_messages:
                        history.insert(insert_idx, mm)
                        insert_idx += 1

            # Add user message to history and persist
            history.append({"role": "user", "content": content})
            user_msg_id = await _save_message(conversation_id, "user", content)
            await websocket.send_json(
                {"type": "message_saved", "message_id": user_msg_id, "role": "user"}
            )

            # Stream response — run in background to allow takeover
            cancel_event = asyncio.Event()
            collector = _TokenCollector(websocket, cancel_event)
            stream_task = asyncio.create_task(
                _run_stream(
                    provider_type,
                    collector,
                    api_key,
                    base_url,
                    model_name,
                    system_prompt,
                    history,
                    temperature,
                    max_tokens,
                    tools=tool_schemas or None,
                    user_id=user["id"],
                )
            )

            # Wait for either stream completion or a queued command
            while not stream_task.done():
                # Check for incoming messages with a short timeout
                try:
                    cmd = await asyncio.wait_for(msg_queue.get(), timeout=0.1)
                    # Re-queue the command so the main loop processes it
                    await msg_queue.put(cmd)
                    if cmd.get("type") == "takeover":
                        # Break out to let the main loop handle takeover
                        break
                except TimeoutError:
                    continue

            if stream_task.done():
                # Stream completed normally — check for errors
                exc = stream_task.exception() if not stream_task.cancelled() else None
                if exc:
                    await websocket.send_json(
                        {"type": "error", "message": f"Stream error: {exc!s}"}
                    )
                    continue

                # Persist the full assistant response
                assistant_content = collector.collected or "(empty response)"
                history.append({"role": "assistant", "content": assistant_content})
                usage_str = json.dumps(collector.usage) if collector.usage else None
                asst_msg_id = await _save_message(
                    conversation_id, "assistant", assistant_content, usage_str
                )
                await websocket.send_json(
                    {"type": "message_saved", "message_id": asst_msg_id, "role": "assistant"}
                )
                # Persist turn to agent memory
                await memory_service.save_turn(
                    agent_id,
                    conversation_id,
                    content,
                    assistant_content,
                )
                stream_task = None

    except WebSocketDisconnect:
        pass
    finally:
        receive_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receive_task
        if stream_task and not stream_task.done():
            stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stream_task
