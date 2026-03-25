"""Integration tests for MCP large output workspace offloading.

US-INT-015: Verifies that an MCP tool returning a large blob (>10 KB)
triggers workspace offload, that the agent correctly calls retrieve_artifact
to access the content, and that the retrieved content is used in the final
response.

The get_large_dataset tool added to mcp_test_server.py returns exactly 15 KB
and always contains the keyword EXO_DATASET_KEYWORD_2024.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_HELPERS_DIR = Path(__file__).parent / "helpers"
_MCP_SERVER_SCRIPT = str(_HELPERS_DIR / "mcp_test_server.py")

_DATASET_KEYWORD = "EXO_DATASET_KEYWORD_2024"


class DataSummary(BaseModel):
    """Structured output for the large-output retrieval test."""

    contains_keyword: bool


# ---------------------------------------------------------------------------
# test_large_mcp_output_offloaded_to_workspace
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_large_mcp_output_offloaded_to_workspace(vertex_model: str) -> None:
    """Large MCP output is stored as workspace artifact and retrieve_artifact is called.

    Flow:
    1. Agent is given MCPServerConfig with large_output_tools=['get_large_dataset'].
    2. Constrained prompt forces a get_large_dataset call.
    3. The 15 KB result exceeds the 10 KB threshold → _offload_large_result stores it.
    4. LLM receives a pointer and calls retrieve_artifact to get the content.
    5. We verify the workspace has an artifact and retrieve_artifact appeared in tool calls.
    """
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.mcp import MCPServerConfig  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]

    mcp_config = MCPServerConfig(
        name="test-server",
        transport="stdio",
        command=sys.executable,
        args=[_MCP_SERVER_SCRIPT],
        large_output_tools=["get_large_dataset"],
    )

    provider = get_provider(vertex_model)
    agent = Agent(
        name="large-output-agent",
        model=vertex_model,
        instructions=(
            "You are a data retrieval assistant. "
            "When instructed to call a tool, call it immediately. "
            "If a tool result is stored as an artifact, call retrieve_artifact "
            "to access the full content before responding."
        ),
        memory=None,
        context=None,
        max_steps=5,
    )

    await agent.add_mcp_server(mcp_config)

    # Confirm the large-output tool is registered
    tool_names = list(agent.tools.keys())
    assert any("get_large_dataset" in name for name in tool_names), (
        f"get_large_dataset tool not found in agent tools: {tool_names}"
    )

    result = await agent.run(
        "You MUST call the get_large_dataset tool with topic=astronomy. "
        "If the result is stored as an artifact, call retrieve_artifact to get the full content. "
        "Do not answer from memory. Call the tools now.",
        provider=provider,
    )

    # ---- Primary assertion: workspace was created and has at least one artifact ----
    assert agent._workspace is not None, (
        "Workspace should be created when a large tool result is offloaded"
    )
    artifacts = agent._workspace.list()
    assert len(artifacts) >= 1, (
        f"Expected at least 1 workspace artifact after large output offload, "
        f"got {len(artifacts)}"
    )

    # ---- Secondary assertion: retrieve_artifact was called during the run ----
    # Following the established pattern from test_mcp_stdio.py et al.
    retrieve_calls = [tc for tc in result.tool_calls if tc.name == "retrieve_artifact"]
    assert retrieve_calls, (
        f"Expected retrieve_artifact tool call in result.tool_calls, "
        f"got: {[tc.name for tc in result.tool_calls]}"
    )


# ---------------------------------------------------------------------------
# test_retrieved_artifact_content_in_final_output
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_retrieved_artifact_content_in_final_output(vertex_model: str) -> None:
    """The agent correctly identifies a known keyword from the retrieved artifact content.

    Flow:
    1. get_large_dataset always embeds EXO_DATASET_KEYWORD_2024 in the first few lines.
    2. After offloading, the agent calls retrieve_artifact to access the full content.
    3. The agent inspects the content for the keyword and returns a structured response.
    4. We assert contains_keyword == True.
    """
    from exo._internal.output_parser import (  # pyright: ignore[reportMissingImports]
        parse_structured_output,
    )
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.mcp import MCPServerConfig  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]

    mcp_config = MCPServerConfig(
        name="test-server",
        transport="stdio",
        command=sys.executable,
        args=[_MCP_SERVER_SCRIPT],
        large_output_tools=["get_large_dataset"],
    )

    provider = get_provider(vertex_model)
    agent = Agent(
        name="keyword-checker-agent",
        model=vertex_model,
        instructions=(
            "You are a data analysis assistant. "
            "When instructed to call a tool, call it immediately. "
            "If a tool result is stored as an artifact, call retrieve_artifact "
            "to access the full content. "
            "After retrieving the content, check whether it contains a specific keyword. "
            "Respond ONLY with valid JSON matching the schema: "
            '{"contains_keyword": true} or {"contains_keyword": false}. '
            "No explanation, no markdown — just the JSON object."
        ),
        memory=None,
        context=None,
        max_steps=5,
    )

    await agent.add_mcp_server(mcp_config)

    result = await agent.run(
        f"Call get_large_dataset with topic=science. "
        f"If the result is stored as an artifact, call retrieve_artifact to get the full content. "
        f"Then check if the retrieved text contains the exact string '{_DATASET_KEYWORD}'. "
        f"Respond ONLY with JSON: "
        f'{{"contains_keyword": true}} if the keyword is present, '
        f'{{"contains_keyword": false}} if it is not.',
        provider=provider,
    )

    output = parse_structured_output(result.text, DataSummary)
    assert output.contains_keyword is True, (
        f"Expected contains_keyword=True (keyword '{_DATASET_KEYWORD}' is always in dataset), "
        f"but got contains_keyword={output.contains_keyword}. "
        f"Agent response: {result.text!r}"
    )
