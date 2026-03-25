"""Agent-to-Agent (A2A) protocol integration tests.

Verifies Agent A can delegate a task to Agent B running on a separate HTTP
server using the A2A protocol, and the result is correctly propagated.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request

import pytest

_A2A_MODULE = "tests.integration.helpers.a2a_translator_server:app"
_A2A_PORT = 8766


# ---------------------------------------------------------------------------
# Session fixture: Agent B as A2A server on port 8766
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def a2a_server_b():  # type: ignore[return]
    """Start Agent B (translator) as an A2A server on port 8766."""
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            _A2A_MODULE,
            "--host",
            "0.0.0.0",
            "--port",
            str(_A2A_PORT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait until the agent-card endpoint is reachable
    card_url = f"http://localhost:{_A2A_PORT}/.well-known/agent-card"
    deadline = time.time() + 15.0
    started = False
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(card_url, timeout=1) as resp:
                if resp.status == 200:
                    started = True
                    break
        except Exception:
            time.sleep(0.5)

    if not started:
        proc.kill()
        proc.wait()
        pytest.skip(
            f"A2A server B did not start within 15s on port {_A2A_PORT}"
        )

    try:
        yield f"http://localhost:{_A2A_PORT}"  # type: ignore[misc]
    finally:
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_a2a_agent_delegates_and_receives_result(
    vertex_model: str,
    a2a_server_b: str,
) -> None:
    """Agent A delegates translation to Agent B via A2A HTTP and uses the result."""
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]
    from exo.tool import tool  # pyright: ignore[reportMissingImports]

    a2a_url = a2a_server_b

    @tool
    def call_translator_agent(text: str) -> str:
        """Call the A2A translator agent on port 8766 to translate text to Spanish.

        Args:
            text: The text to translate to Spanish.
        """
        import httpx  # pyright: ignore[reportMissingImports]

        with httpx.Client(timeout=45.0) as client:
            resp = client.post(f"{a2a_url}/", json={"text": text})
            resp.raise_for_status()
            data = resp.json()
        artifact = data.get("artifact") or {}
        return artifact.get("text", "") or data.get("result", "")

    provider = get_provider(vertex_model)
    agent = Agent(
        name="orchestrator",
        model=vertex_model,
        instructions=(
            "You are an orchestrator. When asked to translate text to Spanish, "
            "you MUST use the call_translator_agent tool. "
            "Never translate the text yourself."
        ),
        tools=[call_translator_agent],
    )

    result = await agent.run(
        "Translate 'Hello World' to Spanish using the translator agent.",
        provider=provider,
    )

    # Verify the translation contains expected Spanish words
    combined = result.text.lower()
    assert "hola" in combined or "mundo" in combined, (
        f"Expected 'hola' or 'mundo' in output, got: {result.text!r}"
    )

    # Verify the A2A tool was called (HTTP tool call to port 8766)
    tool_names = [tc.name for tc in result.tool_calls]
    assert any("translator" in name for name in tool_names), (
        f"Expected call_translator_agent tool call to port 8766, got: {tool_names}"
    )
