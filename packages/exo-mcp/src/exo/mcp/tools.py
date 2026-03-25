"""MCP tool schema extraction, conversion to Exo Tool format, and filtering."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from exo.tool import Tool, ToolError
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPTool

logger = logging.getLogger(__name__)

# Default namespace prefix for MCP tools
DEFAULT_NAMESPACE = "mcp"


class MCPToolError(ToolError):
    """Error raised during MCP tool operations."""


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class MCPToolFilter:
    """Filter for including/excluding MCP tools by name.

    Args:
        include: Whitelist of tool names (if non-empty, only these are included).
        exclude: Blacklist of tool names (always excluded, takes priority over include).
    """

    __slots__ = ("_exclude", "_include")

    def __init__(
        self,
        *,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        self._include = set(include) if include else set()
        self._exclude = set(exclude) if exclude else set()

    def accepts(self, name: str) -> bool:
        """Check if a tool name passes the filter."""
        if name in self._exclude:
            return False
        return not (self._include and name not in self._include)

    def apply(self, tools: list[MCPTool]) -> list[MCPTool]:
        """Filter a list of MCP tools, returning only accepted ones."""
        return [t for t in tools if self.accepts(t.name)]

    def __repr__(self) -> str:
        return f"MCPToolFilter(include={sorted(self._include)}, exclude={sorted(self._exclude)})"


# ---------------------------------------------------------------------------
# Namespace mapping
# ---------------------------------------------------------------------------


def namespace_tool_name(
    tool_name: str,
    server_name: str,
    *,
    namespace: str = DEFAULT_NAMESPACE,
) -> str:
    """Create a namespaced tool name: ``{namespace}__{server}__{tool}``.

    Args:
        tool_name: Original tool name.
        server_name: MCP server name.
        namespace: Namespace prefix (default "mcp").

    Returns:
        Namespaced tool name with non-alphanumeric chars replaced by underscores.
    """
    safe_server = re.sub(r"[^a-zA-Z0-9]", "_", server_name)
    safe_tool = re.sub(r"[^a-zA-Z0-9]", "_", tool_name)
    return f"{namespace}__{safe_server}__{safe_tool}"


def parse_namespaced_name(
    namespaced: str,
) -> tuple[str, str, str]:
    """Parse a namespaced tool name back into (namespace, server, tool).

    Args:
        namespaced: A name like "mcp__server__tool".

    Returns:
        Tuple of (namespace, server_name, tool_name).

    Raises:
        MCPToolError: If the name doesn't match the expected format.
    """
    parts = namespaced.split("__", 2)
    if len(parts) != 3:
        raise MCPToolError(
            f"Invalid namespaced tool name '{namespaced}': "
            "expected format 'namespace__server__tool'"
        )
    return parts[0], parts[1], parts[2]


# ---------------------------------------------------------------------------
# Schema extraction + conversion
# ---------------------------------------------------------------------------


def extract_schema(mcp_tool: MCPTool) -> dict[str, Any]:
    """Extract the JSON Schema parameters from an MCP tool.

    Args:
        mcp_tool: An MCP tool definition.

    Returns:
        JSON Schema parameters dict (``type: "object"``, ``properties``, etc.).
    """
    schema = mcp_tool.inputSchema
    if isinstance(schema, dict):
        return dict(schema)
    # Fallback: empty schema
    return {"type": "object", "properties": {}}


def _format_call_result(result: CallToolResult) -> str:
    """Convert a CallToolResult to a string for tool output.

    Args:
        result: MCP call_tool result.

    Returns:
        String representation of the result content.
    """
    parts: list[str] = []
    for item in result.content:
        if isinstance(item, TextContent):
            parts.append(item.text)
        elif (text := getattr(item, "text", None)) is not None:
            parts.append(str(text))
        else:
            parts.append(str(item))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# MCPToolWrapper — Exo Tool wrapping an MCP tool
# ---------------------------------------------------------------------------


class MCPToolWrapper(Tool):
    """An Exo Tool that wraps an MCP tool for execution.

    The tool delegates execution to an MCP server connection's ``call_tool``
    method. Schema is extracted from the MCP tool definition.

    Args:
        mcp_tool: The MCP tool definition.
        server_name: Name of the MCP server providing this tool.
        call_fn: Async callable that invokes the tool on the server.
        namespace: Namespace prefix for the tool name.
    """

    __slots__ = (
        "_call_fn",
        "_connection",
        "_original_name",
        "_reconnect_lock",
        "_server_config",
        "_server_name",
        "description",
        "large_output",
        "name",
        "parameters",
        "progress_queue",
    )

    def __init__(
        self,
        mcp_tool: MCPTool,
        server_name: str,
        call_fn: Any,
        *,
        namespace: str = DEFAULT_NAMESPACE,
        server_config: Any | None = None,
    ) -> None:
        self._original_name = mcp_tool.name
        self._server_name = server_name
        self._call_fn = call_fn
        self._server_config = server_config
        self._connection = None
        self._reconnect_lock = asyncio.Lock()
        self.name = namespace_tool_name(mcp_tool.name, server_name, namespace=namespace)
        self.description = mcp_tool.description or f"MCP tool: {mcp_tool.name}"
        self.parameters = extract_schema(mcp_tool)
        # Set large_output from server_config.large_output_tools membership
        large_output_tools = getattr(server_config, "large_output_tools", None) or []
        self.large_output: bool = mcp_tool.name in large_output_tools
        # Queue for progress notifications captured during execute(); drained by agent.stream()
        self.progress_queue: asyncio.Queue[Any] = asyncio.Queue()

    @property
    def original_name(self) -> str:
        """The original (un-namespaced) tool name from the MCP server."""
        return self._original_name

    @property
    def server_name(self) -> str:
        """The MCP server providing this tool."""
        return self._server_name

    def to_dict(self) -> dict[str, Any]:
        """Serialize the MCP tool wrapper to a dict for distributed execution.

        Returns:
            A JSON-serializable dict with an ``__mcp_tool__`` marker.
        """
        data: dict[str, Any] = {
            "__mcp_tool__": True,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "original_name": self._original_name,
            "server_name": self._server_name,
            "large_output": self.large_output,
        }
        if self._server_config is not None:
            data["server_config"] = self._server_config.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MCPToolWrapper:
        """Reconstruct an MCPToolWrapper from a dict produced by ``to_dict()``.

        Uses ``object.__new__()`` to bypass ``__init__`` (which requires an
        ``MCPTool`` instance). The ``_call_fn`` is left as ``None`` — on first
        ``execute()`` the wrapper will lazily reconnect using ``_server_config``.

        Args:
            data: Dict as produced by ``to_dict()``.

        Returns:
            A reconstructed ``MCPToolWrapper`` instance.
        """
        from exo.mcp.client import MCPServerConfig

        wrapper = object.__new__(cls)
        wrapper.name = data["name"]
        wrapper.description = data["description"]
        wrapper.parameters = data["parameters"]
        wrapper._original_name = data["original_name"]
        wrapper._server_name = data["server_name"]
        wrapper._call_fn = None
        wrapper._connection = None
        wrapper._reconnect_lock = asyncio.Lock()
        wrapper.large_output = data.get("large_output", False)
        wrapper.progress_queue = asyncio.Queue()
        if "server_config" in data:
            wrapper._server_config = MCPServerConfig.from_dict(data["server_config"])
        else:
            wrapper._server_config = None
        return wrapper

    async def execute(self, **kwargs: Any) -> str | dict[str, Any]:
        """Execute the MCP tool via the server connection.

        If ``_call_fn`` is ``None`` (e.g. after deserialization), attempts to
        lazily reconnect using ``_server_config``.

        Args:
            **kwargs: Tool arguments forwarded to the MCP server.

        Returns:
            String result from the MCP tool.

        Raises:
            MCPToolError: If execution fails or the result indicates an error.
        """
        if self._call_fn is None:
            async with self._reconnect_lock:
                if self._call_fn is None:  # double-check after acquiring lock
                    if self._server_config is not None:
                        from exo.mcp.client import MCPServerConnection

                        logger.debug(
                            "Lazy reconnecting MCP tool '%s' on server '%s'",
                            self._original_name,
                            self._server_name,
                        )
                        conn = MCPServerConnection(self._server_config)
                        await conn.connect()
                        self._connection = conn
                        self._call_fn = conn.call_tool
                    else:
                        raise MCPToolError(
                            f"MCP tool '{self._original_name}' has no call function and no server config "
                            "for reconnection. Cannot execute."
                        )

        logger.debug("Calling MCP tool '%s' on server '%s'", self._original_name, self._server_name)

        # Build a progress callback that enqueues MCPProgressEvent objects.
        # Import lazily to avoid a hard dependency on exo-core from exo-mcp.
        try:
            from exo.types import MCPProgressEvent  # pyright: ignore[reportMissingImports]

            _queue = self.progress_queue
            _tool_name = self.name

            async def _progress_callback(
                progress: float, total: float | None, message: str | None
            ) -> None:
                event = MCPProgressEvent(
                    tool_name=_tool_name,
                    progress=int(progress),
                    total=int(total) if total is not None else None,
                    message=message or "",
                )
                await _queue.put(event)

        except ImportError:
            _progress_callback = None  # type: ignore[assignment]

        try:
            result: CallToolResult = await self._call_fn(
                self._original_name, kwargs or None, progress_callback=_progress_callback
            )
        except Exception as exc:
            logger.error(
                "MCP tool '%s' on server '%s' failed: %s",
                self._original_name,
                self._server_name,
                exc,
                exc_info=True,
            )
            raise MCPToolError(
                f"MCP tool '{self._original_name}' on server '{self._server_name}' failed: {exc}"
            ) from exc

        if result.isError:
            error_text = _format_call_result(result)
            raise MCPToolError(f"MCP tool '{self._original_name}' returned error: {error_text}")

        return _format_call_result(result)

    async def cleanup(self) -> None:
        """Close any owned connection created by lazy reconnection."""
        if self._connection is not None:
            try:
                await self._connection.cleanup()
            except Exception:
                logger.warning(
                    "Failed to clean up MCP connection for tool '%s' on server '%s'",
                    self._original_name,
                    self._server_name,
                    exc_info=True,
                )
            finally:
                self._connection = None
                self._call_fn = None


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------


def convert_mcp_tools(
    mcp_tools: list[MCPTool],
    server_name: str,
    call_fn: Any,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    tool_filter: MCPToolFilter | None = None,
    server_config: Any | None = None,
) -> list[MCPToolWrapper]:
    """Convert a list of MCP tools to Exo MCPToolWrapper instances.

    Args:
        mcp_tools: MCP tool definitions from ``list_tools()``.
        server_name: Name of the MCP server.
        call_fn: Async callable ``(tool_name, arguments) -> CallToolResult``.
        namespace: Namespace prefix for tool names.
        tool_filter: Optional filter for including/excluding tools.
        server_config: Optional MCPServerConfig for serialization support.

    Returns:
        List of MCPToolWrapper instances.
    """
    if tool_filter:
        mcp_tools = tool_filter.apply(mcp_tools)

    return [
        MCPToolWrapper(t, server_name, call_fn, namespace=namespace, server_config=server_config)
        for t in mcp_tools
    ]


async def load_tools_from_connection(
    connection: Any,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    tool_filter: MCPToolFilter | None = None,
) -> list[MCPToolWrapper]:
    """Load and convert tools from a live MCP server connection.

    Args:
        connection: An MCPServerConnection (duck-typed: needs ``name``,
            ``list_tools()``, ``call_tool()``).
        namespace: Namespace prefix for tool names.
        tool_filter: Optional filter for including/excluding tools.

    Returns:
        List of MCPToolWrapper instances ready for agent use.

    Raises:
        MCPToolError: If the connection fails to list tools.
    """
    try:
        mcp_tools: list[MCPTool] = await connection.list_tools()
    except Exception as exc:
        raise MCPToolError(f"Failed to list tools from server '{connection.name}': {exc}") from exc

    server_config = getattr(connection, "config", None)

    tools = convert_mcp_tools(
        mcp_tools,
        connection.name,
        connection.call_tool,
        namespace=namespace,
        tool_filter=tool_filter,
        server_config=server_config,
    )
    logger.debug("Loaded %d tools from server '%s'", len(tools), connection.name)
    return tools


async def load_tools_from_client(
    client: Any,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    tool_filter: MCPToolFilter | None = None,
) -> list[MCPToolWrapper]:
    """Load and convert tools from all servers in an MCP client.

    Args:
        client: An MCPClient (duck-typed: needs ``server_names``,
            ``connect(name)``, connection has ``list_tools()`` and ``call_tool()``).
        namespace: Namespace prefix for tool names.
        tool_filter: Optional filter for including/excluding tools.

    Returns:
        List of MCPToolWrapper instances from all servers.

    Raises:
        MCPToolError: If tool loading fails for any server.
    """
    all_tools: list[MCPToolWrapper] = []
    for server_name in client.server_names:
        conn = await client.connect(server_name)
        tools = await load_tools_from_connection(conn, namespace=namespace, tool_filter=tool_filter)
        all_tools.extend(tools)
        logger.debug("Server '%s': loaded %d tools", server_name, len(tools))
    logger.debug(
        "Total tools loaded from client: %d across %d servers",
        len(all_tools),
        len(client.server_names),
    )
    return all_tools
