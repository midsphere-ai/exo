"""@mcp_server() class decorator and MCPServerRegistry for exposing tools as MCP servers."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


class MCPServerError(Exception):
    """Error raised by MCP server operations."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class MCPServerRegistry:
    """Singleton registry for @mcp_server-decorated classes.

    Stores class references and lazily-created singleton instances.
    """

    __slots__ = ("_classes", "_instances")

    def __init__(self) -> None:
        self._classes: dict[str, type] = {}
        self._instances: dict[str, Any] = {}

    def register(self, name: str, cls: type) -> None:
        """Register a server class by name."""
        self._classes[name] = cls

    def get_class(self, name: str) -> type:
        """Get a registered server class.

        Raises:
            MCPServerError: If the name is not registered.
        """
        if name not in self._classes:
            raise MCPServerError(f"MCP server '{name}' not registered")
        return self._classes[name]

    def get_instance(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Get or create a singleton instance of a registered server.

        Raises:
            MCPServerError: If the name is not registered.
        """
        if name not in self._instances:
            cls = self.get_class(name)
            self._instances[name] = cls(*args, **kwargs)
        return self._instances[name]

    @property
    def names(self) -> list[str]:
        """All registered server names."""
        return list(self._classes)

    def has(self, name: str) -> bool:
        """Check if a server name is registered."""
        return name in self._classes

    def clear(self) -> None:
        """Remove all registrations and instances."""
        self._classes.clear()
        self._instances.clear()

    def __len__(self) -> int:
        return len(self._classes)

    def __repr__(self) -> str:
        return f"MCPServerRegistry(servers={sorted(self._classes)})"


# Module-level global registry
server_registry = MCPServerRegistry()


# ---------------------------------------------------------------------------
# @mcp_server() decorator
# ---------------------------------------------------------------------------


def _register_methods(instance: Any, mcp: FastMCP) -> list[str]:
    """Discover public methods on *instance* and register them as MCP tools.

    Returns the list of registered tool names.
    """
    tool_names: list[str] = []

    for method_name, method in inspect.getmembers(instance, inspect.ismethod):
        if method_name.startswith("_") or method_name in ("run", "stop"):
            continue

        description = (inspect.getdoc(method) or f"{method_name} tool").strip().split("\n")[0]
        is_async = asyncio.iscoroutinefunction(method)

        # Register as MCP tool; description param exists at runtime (mcp>=1.0)
        tool_decorator = mcp.tool(name=method_name, description=description)  # pyright: ignore[reportCallIssue]
        if is_async:

            @tool_decorator
            @functools.wraps(method)
            async def async_wrapper(*args: Any, _m: Any = method, **kwargs: Any) -> Any:
                return await _m(*args, **kwargs)

        else:

            @tool_decorator
            @functools.wraps(method)
            def sync_wrapper(*args: Any, _m: Any = method, **kwargs: Any) -> Any:
                return _m(*args, **kwargs)

        tool_names.append(method_name)

    return tool_names


def mcp_server(
    name: str | None = None,
    *,
    transport: str = "stdio",
) -> Any:
    """Class decorator that converts a Python class into an MCP server.

    Public methods (non-underscored, excluding ``run``/``stop``) are
    registered as MCP tools via FastMCP.

    After decoration the class gains:

    * ``_mcp`` -- the ``FastMCP`` instance
    * ``_tool_names`` -- list of registered tool names
    * ``run(**kwargs)`` -- start the server (``transport`` kwarg overrides default)
    * ``stop()`` -- placeholder for graceful shutdown

    The class is also registered in the module-level ``server_registry``.

    Args:
        name: Server name. Defaults to the class name.
        transport: Default transport mode (``"stdio"`` or ``"sse"``).

    Returns:
        The decorated class.

    Example::

        @mcp_server(name="calculator")
        class Calculator:
            \"\"\"A simple calculator server.\"\"\"

            def add(self, a: int, b: int) -> int:
                \"\"\"Add two numbers.\"\"\"
                return a + b
    """

    def decorator(cls: type) -> type:
        server_name = name or cls.__name__
        server_desc = (cls.__doc__ or f"{server_name} MCP Server").strip()
        default_transport = transport

        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)

            mcp = FastMCP(server_name, instructions=server_desc)
            tool_names = _register_methods(self, mcp)

            self._mcp = mcp
            self._tool_names = tool_names

            logger.info(
                "MCP server %r created with tools: %s",
                server_name,
                ", ".join(tool_names) if tool_names else "(none)",
            )

        cls.__init__ = new_init  # type: ignore[assignment]

        def run(self: Any, *, transport: str = default_transport, **kwargs: Any) -> None:
            """Run the MCP server.

            Args:
                transport: "stdio" or "sse".
                **kwargs: Passed to ``FastMCP.run()``.
            """
            if not hasattr(self, "_mcp") or self._mcp is None:
                raise MCPServerError("MCP server not initialized")
            self._mcp.run(transport=transport, **kwargs)

        def stop(self: Any) -> None:
            """Stop the MCP server (placeholder)."""
            logger.info("Stopping MCP server %r", server_name)

        cls.run = run  # type: ignore[attr-defined]
        cls.stop = stop  # type: ignore[attr-defined]

        server_registry.register(server_name, cls)
        return cls

    return decorator
