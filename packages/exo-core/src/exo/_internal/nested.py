"""Support for nested swarms — using a Swarm as a node in another Swarm.

A :class:`SwarmNode` wraps a :class:`~exo.swarm.Swarm` so it can be
placed in another Swarm's agent list.  The outer Swarm treats it like
any other agent node: it has a ``name``, an ``is_swarm`` marker, and a
``run()`` method that delegates to the inner Swarm.

Context isolation is achieved by never sharing mutable state between
the inner and outer swarms — each execution creates its own
``RunState`` and message history.

Usage::

    inner = Swarm(agents=[a, b], flow="a >> b")
    node = SwarmNode(swarm=inner, name="inner_pipeline")
    outer = Swarm(agents=[c, node, d], flow="c >> inner_pipeline >> d")
    result = await run(outer, "Hello!")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from exo.types import ExoError, Message, RunResult


class NestedSwarmError(ExoError):
    """Raised for nested swarm errors."""


class SwarmNode:
    """Wraps a Swarm so it can be used as a node in another Swarm.

    Provides the same interface that :class:`~exo.swarm.Swarm`
    expects from agent nodes — ``name`` attribute and ``run()`` method.
    An ``is_swarm`` marker allows the outer Swarm to detect nested
    swarms via duck-typing.

    Context isolation: the inner Swarm creates its own ``RunState``
    and message history on each ``run()`` invocation.  No mutable
    state leaks between inner and outer executions.

    Args:
        swarm: The inner Swarm to wrap.
        name: Node name for the outer Swarm's flow DSL.
            Defaults to the inner Swarm's ``name`` attribute.
    """

    def __init__(
        self,
        *,
        swarm: Any,
        name: str | None = None,
    ) -> None:
        if not hasattr(swarm, "flow_order"):
            raise NestedSwarmError("SwarmNode requires a Swarm instance (object with flow_order)")

        self._swarm = swarm
        self.name = name or swarm.name

        # Marker for outer Swarm detection (duck-typing)
        self.is_swarm = True

    async def run(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Execute the inner swarm with context isolation.

        Each call creates a fresh execution context — the inner
        Swarm builds its own ``RunState`` and does not share mutable
        state with the outer Swarm.

        Args:
            input: User query string.
            messages: Not forwarded to inner swarm (context isolation).
                The inner swarm starts with a clean message history.
            provider: LLM provider, forwarded to inner swarm.
            max_retries: Retry attempts, forwarded to inner swarm.

        Returns:
            ``RunResult`` from the inner swarm's execution.
        """
        # Context isolation: inner swarm gets fresh message history.
        # The outer swarm's messages are NOT forwarded — each swarm
        # maintains its own conversation context.
        return await self._swarm.run(
            input,
            provider=provider,
            max_retries=max_retries,
        )

    def describe(self) -> dict[str, Any]:
        """Return a summary including the inner swarm's description."""
        return {
            "type": "nested_swarm",
            "name": self.name,
            "inner": self._swarm.describe(),
        }

    def __repr__(self) -> str:
        return f"SwarmNode(name={self.name!r}, inner={self._swarm!r})"
