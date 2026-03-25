"""Use MCP (Model Context Protocol) tools with an agent — Exo quickstart.

Shows how to connect to an MCP server, load its tools, and pass
them to an Agent for function-calling.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python examples/quickstart/mcp_tool.py
"""

import asyncio

from exo import Agent, run
from exo.mcp.client import MCPClient, MCPServerConfig  # pyright: ignore[reportMissingImports]
from exo.mcp.tools import load_tools_from_client  # pyright: ignore[reportMissingImports]


async def main() -> None:
    # 1. Configure an MCP server (stdio transport, running a Python module).
    config = MCPServerConfig(
        name="example-server",
        transport="stdio",
        command="python",
        args=["-m", "my_mcp_server"],  # replace with your MCP server module
    )

    # 2. Create a client and register the server.
    client = MCPClient()
    client.add_server(config)

    # 3. Connect, discover tools, and run the agent.
    async with client:
        tools = await load_tools_from_client(client)
        print(f"Discovered {len(tools)} MCP tool(s):")
        for t in tools:
            print(f"  - {t.name}: {t.description}")

        agent = Agent(
            name="mcp-bot",
            model="openai:gpt-4o-mini",
            instructions="Use the available MCP tools to help the user.",
            tools=tools,
        )

        result = await run(agent, "Hello, use one of your tools!")
        print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
