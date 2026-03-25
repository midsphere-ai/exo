"""Integration test fixtures for the Exo monorepo.

All fixtures work against real external dependencies:
- Vertex AI (gemini-2.0-flash) via GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION
- SQLite (via aiosqlite)
- ChromaDB (via chromadb)
- Real stdio MCP subprocesses (via mcp)
- Real Uvicorn HTTP server
- Real Redis (via Docker)
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HELPERS_DIR = Path(__file__).parent / "helpers"
_MCP_SERVER_SCRIPT = str(_HELPERS_DIR / "mcp_test_server.py")
_WEB_APP_MODULE = "tests.integration.helpers.web_app:app"
_UVICORN_PORT = 8765
_REDIS_PORT = 6380


# ---------------------------------------------------------------------------
# vertex_model — session fixture; skips if Vertex creds absent
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def vertex_model() -> str:
    """Return the model string for integration tests.

    Skips if GOOGLE_CLOUD_PROJECT or GOOGLE_CLOUD_LOCATION are absent.

    When VERTEX_MODEL is an ``openai:`` model (i.e. a Vertex AI Model Garden
    model accessed via the OpenAI-compatible endpoint), this fixture generates
    a short-lived OAuth2 bearer token from GOOGLE_SERVICE_ACCOUNT_BASE64 and
    sets OPENAI_API_KEY / OPENAI_BASE_URL so that ``get_provider()`` builds an
    OpenAIProvider correctly without any per-test changes.
    """
    import base64
    import json

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    if not project or not location:
        pytest.skip(
            "Vertex AI integration tests require GOOGLE_CLOUD_PROJECT and "
            "GOOGLE_CLOUD_LOCATION environment variables."
        )

    model = os.environ.get("VERTEX_MODEL", "vertex:gemini-2.0-flash")

    # Model Garden path: openai:<model> via /endpoints/openapi
    if model.startswith("openai:"):
        sa_b64 = os.environ.get("GOOGLE_SERVICE_ACCOUNT_BASE64")
        if not sa_b64:
            pytest.skip(
                "Model Garden tests require GOOGLE_SERVICE_ACCOUNT_BASE64."
            )
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account

        sa_info = json.loads(base64.b64decode(sa_b64))
        creds = service_account.Credentials.from_service_account_info(
            sa_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        creds.refresh(Request())
        os.environ["OPENAI_API_KEY"] = creds.token
        os.environ["OPENAI_BASE_URL"] = (
            f"https://aiplatform.googleapis.com/v1/projects/{project}"
            f"/locations/global/endpoints/openapi"
        )

    return model


# ---------------------------------------------------------------------------
# tmp_sqlite_db — function fixture; yields a temp file path
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_sqlite_db() -> str:  # type: ignore[return]
    """Create a temporary SQLite database file and clean it up after the test."""
    fd, path = tempfile.mkstemp(suffix=".db", prefix="exo_test_")
    os.close(fd)
    try:
        yield path  # type: ignore[misc]
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)


# ---------------------------------------------------------------------------
# memory_store — function fixture; fresh SQLiteMemoryStore
# ---------------------------------------------------------------------------


@pytest.fixture
async def memory_store(tmp_sqlite_db: str):  # type: ignore[return]
    """Yield a fresh, initialised SQLiteMemoryStore backed by a temp file."""
    from exo.memory.backends.sqlite import (  # pyright: ignore[reportMissingImports]
        SQLiteMemoryStore,
    )

    store = SQLiteMemoryStore(db_path=tmp_sqlite_db)
    await store.init()
    try:
        yield store
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# vector_store — function fixture; fresh ChromaVectorMemoryStore
# ---------------------------------------------------------------------------


@pytest.fixture
async def vector_store():  # type: ignore[return]
    """Yield a fresh ChromaVectorMemoryStore backed by a temp directory."""
    from exo.memory.backends.vector import (  # pyright: ignore[reportMissingImports]
        ChromaVectorMemoryStore,
        SentenceTransformerEmbeddingProvider,
    )

    tmp_dir = tempfile.mkdtemp(prefix="exo_chroma_")
    embedding_provider = SentenceTransformerEmbeddingProvider()
    store = ChromaVectorMemoryStore(
        embedding_provider,
        collection_name="test_collection",
        path=tmp_dir,
    )
    try:
        yield store
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# mcp_server_process — session fixture; MCPServerConfig for stdio MCP server
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mcp_server_process():  # type: ignore[return]
    """Yield an MCPServerConfig pointing to the stdio test MCP server.

    The MCP server script is launched as a subprocess by the MCP client
    on each connection.  No long-running process is needed here — the
    config describes *how* to launch the subprocess.
    """
    from exo.mcp import MCPServerConfig  # pyright: ignore[reportMissingImports]

    config = MCPServerConfig(
        name="test-server",
        transport="stdio",
        command=sys.executable,
        args=[_MCP_SERVER_SCRIPT],
    )
    yield config  # type: ignore[misc]


# ---------------------------------------------------------------------------
# uvicorn_server — session fixture; real HTTP server on port 8765
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def uvicorn_server():  # type: ignore[return]
    """Start a real Uvicorn server and wait until /health returns 200."""
    import urllib.request

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            _WEB_APP_MODULE,
            "--host",
            "0.0.0.0",
            "--port",
            str(_UVICORN_PORT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    health_url = f"http://localhost:{_UVICORN_PORT}/health"
    deadline = time.time() + 15.0
    started = False
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1) as resp:
                if resp.status == 200:
                    started = True
                    break
        except Exception:
            time.sleep(0.5)

    if not started:
        proc.kill()
        proc.wait()
        pytest.skip(f"Uvicorn server did not start within 15s on port {_UVICORN_PORT}")

    try:
        yield f"http://localhost:{_UVICORN_PORT}"  # type: ignore[misc]
    finally:
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# redis_container — session fixture; Docker Redis on port 6380
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def redis_container():  # type: ignore[return]
    """Start a Redis container via Docker and yield the connection URL.

    Skips the test if Docker is unavailable on the host.
    """
    docker_check = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        timeout=10,
    )
    if docker_check.returncode != 0:
        pytest.skip("Docker is not available on this host.")

    container_name = "exo_test_redis"

    # Ensure no leftover container from a previous run
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        check=False,
    )

    proc = subprocess.Popen(
        [
            "docker",
            "run",
            "--rm",
            "--name",
            container_name,
            "-p",
            f"{_REDIS_PORT}:6379",
            "redis:7-alpine",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for Redis to become ready
    time.sleep(2)

    try:
        yield f"redis://localhost:{_REDIS_PORT}"  # type: ignore[misc]
    finally:
        subprocess.run(
            ["docker", "stop", container_name],
            capture_output=True,
            check=False,
        )
        proc.wait(timeout=10)


# ---------------------------------------------------------------------------
# http_client — function fixture; httpx.AsyncClient for uvicorn_server
# ---------------------------------------------------------------------------


@pytest.fixture
async def http_client(uvicorn_server: str):  # type: ignore[return]
    """Yield an async httpx client pointed at the integration test server."""
    import httpx  # pyright: ignore[reportMissingImports]

    async with httpx.AsyncClient(base_url=uvicorn_server, timeout=30.0) as client:
        yield client  # type: ignore[misc]
