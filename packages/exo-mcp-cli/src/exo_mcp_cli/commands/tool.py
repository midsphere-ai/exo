"""Tool interaction commands: list, call."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated

import typer

from exo_mcp_cli.connection import MCPConnectionError, connect_to_server
from exo_mcp_cli.output import (
    console,
    print_error,
    print_json,
    print_tool_result,
    print_tools_table,
)

tool_app = typer.Typer(
    name="tool",
    help="List and execute MCP tools.",
    no_args_is_help=True,
)


@tool_app.command("list")
def tool_list(
    ctx: typer.Context,
    server: Annotated[str, typer.Argument(help="Server name.")],
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON."),
    ] = False,
) -> None:
    """List available tools from an MCP server."""
    from exo_mcp_cli.main import get_server, get_vault

    entry = get_server(ctx, server)
    vault = get_vault(ctx)

    async def _run() -> None:
        async with connect_to_server(entry, vault) as session:
            result = await session.list_tools()
            if as_json:
                data = []
                for t in result.tools:
                    item = {"name": t.name, "description": t.description}
                    schema = getattr(t, "inputSchema", None)
                    if schema:
                        item["inputSchema"] = schema
                    data.append(item)
                print_json(data)
            else:
                print_tools_table(result.tools)

    try:
        asyncio.run(_run())
    except MCPConnectionError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        print_error(f"Failed to list tools: {exc}")
        if ctx.obj.get("verbose"):
            console.print_exception()
        raise typer.Exit(code=1) from exc


@tool_app.command("call")
def tool_call(
    ctx: typer.Context,
    server: Annotated[str, typer.Argument(help="Server name.")],
    tool: Annotated[str, typer.Argument(help="Tool name.")],
    arg: Annotated[
        list[str] | None,
        typer.Option("--arg", "-a", help="Argument as KEY=VALUE (repeatable)."),
    ] = None,
    json_args: Annotated[
        str | None,
        typer.Option("--json", "-j", help="Arguments as a JSON string."),
    ] = None,
    raw: Annotated[
        bool,
        typer.Option("--raw", help="Output raw JSON result."),
    ] = False,
) -> None:
    """Execute a tool on an MCP server."""
    from exo_mcp_cli.main import get_server, get_vault

    entry = get_server(ctx, server)
    vault = get_vault(ctx)

    # Build arguments dict
    arguments: dict[str, str] = {}
    if json_args:
        try:
            parsed = json.loads(json_args)
            if not isinstance(parsed, dict):
                print_error("--json must be a JSON object")
                raise typer.Exit(code=1)
            arguments.update(parsed)
        except json.JSONDecodeError as exc:
            print_error(f"Invalid JSON: {exc}")
            raise typer.Exit(code=1) from exc
    if arg:
        for a in arg:
            if "=" not in a:
                print_error(f"Invalid argument format (expected KEY=VALUE): {a}")
                raise typer.Exit(code=1)
            k, v = a.split("=", 1)
            arguments[k] = v

    async def _run() -> None:
        async with connect_to_server(entry, vault) as session:
            result = await session.call_tool(tool, arguments or None)
            if raw:
                data = {
                    "isError": getattr(result, "isError", False),
                    "content": [],
                }
                for item in getattr(result, "content", []):
                    text = getattr(item, "text", None)
                    if text is not None:
                        data["content"].append({"type": "text", "text": text})
                    else:
                        data["content"].append({"type": str(type(item).__name__)})
                print_json(data)
            else:
                print_tool_result(result)

    try:
        asyncio.run(_run())
    except MCPConnectionError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        print_error(f"Tool call failed: {exc}")
        if ctx.obj.get("verbose"):
            console.print_exception()
        raise typer.Exit(code=1) from exc
