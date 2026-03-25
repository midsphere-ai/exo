"""Fluent builder for constructing Sandbox instances with lazy evaluation."""

from __future__ import annotations

import logging
from typing import Any

from exo.sandbox.base import LocalSandbox, Sandbox  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


class SandboxBuilder:
    """Fluent builder for creating :class:`Sandbox` instances.

    Supports method chaining for configuration and lazy evaluation —
    the sandbox is automatically built on the first lifecycle or tool
    API call (any attribute not part of the builder itself).

    Example::

        sb = (
            SandboxBuilder()
            .with_workspace(["/tmp/ws"])
            .with_timeout(60.0)
            .with_mcp_config({"server": "local"})
        )
        # Not built yet — lazy until first use:
        await sb.start()  # triggers build(), then start()
    """

    __slots__ = (
        "_agents",
        "_built",
        "_mcp_config",
        "_sandbox_class",
        "_sandbox_id",
        "_sandbox_kwargs",
        "_timeout",
        "_workspace",
    )

    def __init__(self, sandbox_class: type[Sandbox] | None = None) -> None:
        self._sandbox_class: type[Sandbox] = sandbox_class or LocalSandbox
        self._sandbox_id: str | None = None
        self._workspace: list[str] | None = None
        self._mcp_config: dict[str, Any] | None = None
        self._agents: dict[str, Any] | None = None
        self._timeout: float | None = None
        self._sandbox_kwargs: dict[str, Any] = {}
        self._built: Sandbox | None = None

    # -- fluent setters (return self for chaining) --------------------------

    def with_sandbox_id(self, sandbox_id: str) -> SandboxBuilder:
        """Set the sandbox ID."""
        self._sandbox_id = sandbox_id
        return self

    def with_workspace(self, workspace: list[str]) -> SandboxBuilder:
        """Set allowed workspace directories."""
        self._workspace = list(workspace)
        return self

    def with_mcp_config(self, mcp_config: dict[str, Any]) -> SandboxBuilder:
        """Set MCP server configuration."""
        self._mcp_config = dict(mcp_config)
        return self

    def with_agents(self, agents: dict[str, Any]) -> SandboxBuilder:
        """Set agent configurations."""
        self._agents = dict(agents)
        return self

    def with_timeout(self, timeout: float) -> SandboxBuilder:
        """Set execution timeout in seconds."""
        self._timeout = timeout
        return self

    def with_sandbox_class(self, cls: type[Sandbox]) -> SandboxBuilder:
        """Override the sandbox implementation class."""
        self._sandbox_class = cls
        return self

    def with_extra(self, **kwargs: Any) -> SandboxBuilder:
        """Pass additional keyword arguments to the sandbox constructor."""
        self._sandbox_kwargs.update(kwargs)
        return self

    # -- build --------------------------------------------------------------

    def build(self) -> Sandbox:
        """Construct and return the :class:`Sandbox` instance.

        Raises :class:`SandboxError` if the builder has already been built.
        Call :meth:`reset` first to reuse the builder.
        """
        if self._built is not None:
            return self._built

        kwargs: dict[str, Any] = {}
        if self._sandbox_id is not None:
            kwargs["sandbox_id"] = self._sandbox_id
        if self._workspace is not None:
            kwargs["workspace"] = self._workspace
        if self._mcp_config is not None:
            kwargs["mcp_config"] = self._mcp_config
        if self._agents is not None:
            kwargs["agents"] = self._agents
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout
        kwargs.update(self._sandbox_kwargs)

        self._built = self._sandbox_class(**kwargs)
        logger.debug("Built %s (id=%s)", self._sandbox_class.__name__, self._built.sandbox_id)
        return self._built

    def reset(self) -> SandboxBuilder:
        """Clear the built instance so the builder can be reused."""
        self._built = None
        return self

    # -- lazy evaluation via __getattr__ ------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Auto-build and delegate attribute access to the sandbox.

        Builder methods (``with_*``, ``build``, ``reset``) are resolved
        normally.  Any other attribute triggers :meth:`build` and is
        forwarded to the resulting :class:`Sandbox`.
        """
        # __getattr__ is only called when normal lookup fails, so builder
        # methods are never intercepted here.
        sandbox = self.build()
        return getattr(sandbox, name)

    def __repr__(self) -> str:
        status = "built" if self._built is not None else "pending"
        return f"SandboxBuilder(class={self._sandbox_class.__name__}, status={status})"
