"""Plugins REST API.

Plugin manifest, isolation, install/uninstall, and local-directory loading.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from exo_web.database import get_db
from exo_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/v1/plugins", tags=["plugins"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_PLUGIN_TYPES = {"model", "tool", "strategy", "extension", "bundle"}
VALID_STATUSES = {"installed", "enabled", "disabled", "error"}

# Static marketplace catalog — will be replaced with remote registry later
MARKETPLACE_CATALOG: list[dict[str, Any]] = [
    {
        "id": "mp-openai-provider",
        "name": "OpenAI Provider",
        "author": "Exo Team",
        "description": "Connect to OpenAI GPT-4o, GPT-4, and GPT-3.5 models with streaming support and function calling.",
        "category": "Models",
        "type": "model",
        "version": "1.2.0",
        "install_count": 12840,
        "rating": 4.8,
        "permissions": ["network", "api_keys"],
        "changelog": "v1.2.0: Added GPT-4o support\nv1.1.0: Streaming improvements\nv1.0.0: Initial release",
    },
    {
        "id": "mp-anthropic-provider",
        "name": "Anthropic Provider",
        "author": "Exo Team",
        "description": "Claude 4.5 Sonnet, Opus, and Haiku models with tool use, vision, and extended thinking.",
        "category": "Models",
        "type": "model",
        "version": "2.0.1",
        "install_count": 9720,
        "rating": 4.9,
        "permissions": ["network", "api_keys"],
        "changelog": "v2.0.1: Bug fixes\nv2.0.0: Claude 4.5 support\nv1.0.0: Initial release",
    },
    {
        "id": "mp-ollama-local",
        "name": "Ollama Local Models",
        "author": "Exo Team",
        "description": "Run open-source LLMs locally with Ollama. Supports Llama 3, Mistral, CodeLlama, and more.",
        "category": "Models",
        "type": "model",
        "version": "1.0.3",
        "install_count": 6510,
        "rating": 4.5,
        "permissions": ["network"],
        "changelog": "v1.0.3: Added model auto-pull\nv1.0.0: Initial release",
    },
    {
        "id": "mp-web-search",
        "name": "Web Search Tool",
        "author": "Exo Team",
        "description": "Search the web using Google, Bing, or DuckDuckGo APIs. Returns structured results with snippets.",
        "category": "Tools",
        "type": "tool",
        "version": "1.1.0",
        "install_count": 8430,
        "rating": 4.6,
        "permissions": ["network", "api_keys"],
        "changelog": "v1.1.0: Added DuckDuckGo\nv1.0.0: Initial release",
    },
    {
        "id": "mp-code-interpreter",
        "name": "Code Interpreter",
        "author": "Exo Team",
        "description": "Execute Python code in a sandboxed environment. Supports file I/O, plotting, and data analysis.",
        "category": "Tools",
        "type": "tool",
        "version": "2.1.0",
        "install_count": 11200,
        "rating": 4.7,
        "permissions": ["sandbox", "file_system"],
        "changelog": "v2.1.0: Matplotlib support\nv2.0.0: Sandbox isolation\nv1.0.0: Initial release",
    },
    {
        "id": "mp-file-manager",
        "name": "File Manager",
        "author": "Community",
        "description": "Read, write, and manage files in the agent workspace. Supports text, CSV, JSON, and binary files.",
        "category": "Tools",
        "type": "tool",
        "version": "1.3.2",
        "install_count": 7650,
        "rating": 4.4,
        "permissions": ["file_system"],
        "changelog": "v1.3.2: CSV parsing fix\nv1.3.0: Binary support\nv1.0.0: Initial release",
    },
    {
        "id": "mp-rag-pipeline",
        "name": "RAG Pipeline",
        "author": "Exo Team",
        "description": "Retrieval-augmented generation with vector search, chunking strategies, and multi-source ingestion.",
        "category": "Tools",
        "type": "tool",
        "version": "1.0.0",
        "install_count": 4320,
        "rating": 4.3,
        "permissions": ["network", "file_system", "api_keys"],
        "changelog": "v1.0.0: Initial release with ChromaDB backend",
    },
    {
        "id": "mp-react-strategy",
        "name": "ReAct Strategy",
        "author": "Exo Team",
        "description": "Reasoning + Acting loop for complex multi-step tasks. Alternates between thinking and tool calls.",
        "category": "Agent Strategies",
        "type": "strategy",
        "version": "1.1.0",
        "install_count": 5890,
        "rating": 4.6,
        "permissions": [],
        "changelog": "v1.1.0: Improved retry logic\nv1.0.0: Initial release",
    },
    {
        "id": "mp-tree-of-thought",
        "name": "Tree of Thought",
        "author": "Community",
        "description": "Explore multiple reasoning paths simultaneously and select the best solution. Great for math and logic.",
        "category": "Agent Strategies",
        "type": "strategy",
        "version": "0.9.0",
        "install_count": 2140,
        "rating": 4.2,
        "permissions": [],
        "changelog": "v0.9.0: Beta release with BFS and DFS modes",
    },
    {
        "id": "mp-prompt-caching",
        "name": "Prompt Caching",
        "author": "Exo Team",
        "description": "Cache LLM responses for identical prompts. Reduces costs and latency for repetitive queries.",
        "category": "Extensions",
        "type": "extension",
        "version": "1.0.1",
        "install_count": 3870,
        "rating": 4.5,
        "permissions": ["file_system"],
        "changelog": "v1.0.1: TTL configuration\nv1.0.0: Initial release",
    },
    {
        "id": "mp-observability",
        "name": "Observability Suite",
        "author": "Exo Team",
        "description": "OpenTelemetry tracing, structured logging, and metrics for agent runs. Export to Jaeger or Datadog.",
        "category": "Extensions",
        "type": "extension",
        "version": "1.2.0",
        "install_count": 4590,
        "rating": 4.7,
        "permissions": ["network"],
        "changelog": "v1.2.0: Datadog exporter\nv1.1.0: Custom span attributes\nv1.0.0: Initial release",
    },
    {
        "id": "mp-starter-bundle",
        "name": "Starter Bundle",
        "author": "Exo Team",
        "description": "Essential plugins to get started: OpenAI provider, web search, file manager, and ReAct strategy.",
        "category": "Bundles",
        "type": "bundle",
        "version": "1.0.0",
        "install_count": 15200,
        "rating": 4.8,
        "permissions": ["network", "api_keys", "file_system"],
        "changelog": "v1.0.0: Initial curated bundle",
    },
]

MANIFEST_REQUIRED_FIELDS = {"name", "version", "type", "entry_point"}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PluginManifest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Display name")
    version: str = Field(default="0.1.0", description="Version identifier")
    type: str = Field(default="extension", description="Type")
    permissions: list[str] = Field(default_factory=list, description="Permissions")
    entry_point: str = Field(default="main.py", description="Entry point")
    description: str = Field("", description="Human-readable description")
    author: str = Field("", description="Author")


class PluginInstallRequest(BaseModel):
    manifest: PluginManifest = Field(description="Manifest")
    directory: str = Field("", description="Directory")


class PluginLoadDirRequest(BaseModel):
    directory: str = Field(..., min_length=1, description="Directory")


class PluginResponse(BaseModel):
    id: str = Field(description="Unique identifier")
    name: str = Field(description="Display name")
    version: str = Field(description="Version identifier")
    type: str = Field(description="Type")
    manifest_json: str = Field(description="Manifest json")
    status: str = Field(description="Current status")
    entry_point: str = Field(description="Entry point")
    directory: str = Field(description="Directory")
    permissions_json: str = Field(description="Permissions json")
    description: str = Field(description="Human-readable description")
    author: str = Field(description="Author")
    user_id: str = Field(description="Owning user identifier")
    installed_at: str = Field(description="Installed at")


class PluginIsolationResult(BaseModel):
    success: bool = Field(description="Whether the operation succeeded")
    output: str = Field("", description="Output text or data")
    error: str | None = Field(None, description="Error message if failed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict."""
    return dict(row)


def _validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate that a manifest dict has required fields and valid type."""
    missing = MANIFEST_REQUIRED_FIELDS - set(manifest.keys())
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Manifest missing required fields: {', '.join(sorted(missing))}",
        )
    if manifest.get("type") not in VALID_PLUGIN_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid plugin type. Must be one of: {', '.join(sorted(VALID_PLUGIN_TYPES))}",
        )


async def _run_plugin_subprocess(
    entry_point: str, directory: str, timeout: float = 10.0
) -> PluginIsolationResult:
    """Run a plugin entry point in an isolated subprocess.

    Each plugin runs in its own subprocess with a timeout to prevent runaway
    processes. The subprocess is given only the plugin directory as cwd.
    """
    entry_path = Path(directory) / entry_point if directory else Path(entry_point)
    if not entry_path.exists():
        return PluginIsolationResult(
            success=False,
            error=f"Entry point not found: {entry_path}",
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            "python",
            str(entry_path),
            "--validate",
            cwd=directory or None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode == 0:
            return PluginIsolationResult(
                success=True,
                output=stdout.decode(errors="replace").strip(),
            )
        return PluginIsolationResult(
            success=False,
            output=stdout.decode(errors="replace").strip(),
            error=stderr.decode(errors="replace").strip() or f"Exit code {proc.returncode}",
        )
    except TimeoutError:
        proc.kill()  # type: ignore[possibly-undefined]
        return PluginIsolationResult(success=False, error="Plugin validation timed out")
    except Exception as exc:
        return PluginIsolationResult(success=False, error=str(exc))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[PluginResponse])
async def list_plugins(
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all installed plugins for the current user."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM plugins WHERE user_id = ? ORDER BY installed_at DESC",
            (user["id"],),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


@router.get("/marketplace")
async def list_marketplace() -> list[dict[str, Any]]:
    """Return the marketplace catalog of available plugins."""
    return MARKETPLACE_CATALOG


@router.get("/marketplace/{plugin_id}")
async def get_marketplace_plugin(plugin_id: str) -> dict[str, Any]:
    """Return a single marketplace plugin by ID."""
    for plugin in MARKETPLACE_CATALOG:
        if plugin["id"] == plugin_id:
            return plugin
    raise HTTPException(status_code=404, detail="Marketplace plugin not found")


@router.post("/install", response_model=PluginResponse, status_code=201)
async def install_plugin(
    body: PluginInstallRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Install a plugin from a manifest package.

    Validates the manifest, optionally runs the entry point in an isolated
    subprocess for validation, and stores the plugin in the database.
    """
    manifest = body.manifest
    manifest_dict = manifest.model_dump()
    _validate_manifest(manifest_dict)

    # If a directory is provided and the entry point exists, run validation
    if body.directory:
        entry_path = Path(body.directory) / manifest.entry_point
        if entry_path.exists():
            result = await _run_plugin_subprocess(manifest.entry_point, body.directory)
            if not result.success:
                raise HTTPException(
                    status_code=422,
                    detail=f"Plugin validation failed: {result.error}",
                )

    plugin_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        # Check for duplicate name+version for this user
        cursor = await db.execute(
            "SELECT id FROM plugins WHERE name = ? AND version = ? AND user_id = ?",
            (manifest.name, manifest.version, user["id"]),
        )
        if await cursor.fetchone() is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Plugin '{manifest.name}' v{manifest.version} is already installed",
            )

        await db.execute(
            """
            INSERT INTO plugins (
                id, name, version, type, manifest_json, status,
                entry_point, directory, permissions_json,
                description, author, user_id, installed_at
            ) VALUES (?, ?, ?, ?, ?, 'installed', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plugin_id,
                manifest.name,
                manifest.version,
                manifest.type,
                json.dumps(manifest_dict),
                manifest.entry_point,
                body.directory,
                json.dumps(manifest.permissions),
                manifest.description,
                manifest.author,
                user["id"],
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM plugins WHERE id = ?", (plugin_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.post("/load-directory", response_model=PluginResponse, status_code=201)
async def load_from_directory(
    body: PluginLoadDirRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Load a plugin from a local directory for development.

    Reads plugin.json manifest from the directory, validates it,
    and installs the plugin.
    """
    dir_path = Path(body.directory)
    if not dir_path.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")

    manifest_path = dir_path / "plugin.json"
    if not manifest_path.exists():
        raise HTTPException(
            status_code=422,
            detail="No plugin.json found in directory",
        )

    try:
        manifest_raw = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse plugin.json: {exc}") from exc

    _validate_manifest(manifest_raw)

    manifest = PluginManifest(**manifest_raw)
    install_req = PluginInstallRequest(manifest=manifest, directory=str(dir_path.resolve()))
    return await install_plugin(install_req, user)


@router.get("/{plugin_id}", response_model=PluginResponse)
async def get_plugin(
    plugin_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single plugin by ID."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM plugins WHERE id = ? AND user_id = ?",
            (plugin_id, user["id"]),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Plugin not found")
        return _row_to_dict(row)


@router.delete("/{plugin_id}", status_code=204)
async def uninstall_plugin(
    plugin_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Uninstall (delete) a plugin."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM plugins WHERE id = ? AND user_id = ?",
            (plugin_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Plugin not found")
        await db.execute("DELETE FROM plugins WHERE id = ?", (plugin_id,))
        await db.commit()


@router.put("/{plugin_id}/status")
async def update_plugin_status(
    plugin_id: str,
    status: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update plugin status (enable/disable)."""
    if status not in VALID_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status. Must be one of: {', '.join(sorted(VALID_STATUSES))}",
        )
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM plugins WHERE id = ? AND user_id = ?",
            (plugin_id, user["id"]),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Plugin not found")

        await db.execute(
            "UPDATE plugins SET status = ? WHERE id = ?",
            (status, plugin_id),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM plugins WHERE id = ?", (plugin_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)
