"""Parallel and serial agent group primitives.

Provides :class:`ParallelGroup` and :class:`SerialGroup` for
expressing concurrent-then-sequential (``(a | b) >> c``) execution
patterns within a :class:`~exo.swarm.Swarm` flow.

Groups behave like agents from the Swarm's perspective — they have
a ``name`` attribute and can be placed in the agent list and flow DSL.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from exo._internal.call_runner import call_runner
from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]
from exo.types import ExoError, Message, RunResult, Usage

_log = get_logger(__name__)


class GroupError(ExoError):
    """Raised for agent group errors."""


class ParallelGroup:
    """Concurrent execution of multiple agents.

    All agents receive the same input and run concurrently via
    ``asyncio.TaskGroup``.  Results are aggregated by joining
    outputs with the specified separator, or via a custom
    aggregation function.

    Args:
        name: Group name (used as a node in flow DSL).
        agents: List of agents to run concurrently.
        separator: String used to join agent outputs.
        aggregate_fn: Optional custom ``(list[RunResult]) -> str``
            aggregation function.  Overrides *separator* when provided.
    """

    def __init__(
        self,
        *,
        name: str,
        agents: list[Any],
        separator: str = "\n\n",
        aggregate_fn: Any = None,
    ) -> None:
        if not agents:
            raise GroupError("ParallelGroup requires at least one agent")

        self.name = name
        self.agents = {a.name: a for a in agents}
        self.agent_order = [a.name for a in agents]
        self.separator = separator
        self.aggregate_fn = aggregate_fn

        # Marker so Swarm can detect groups vs regular agents
        self.is_group = True

    async def run(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Run all agents concurrently and aggregate results.

        Args:
            input: User query string (sent to every agent).
            messages: Prior conversation history.
            provider: LLM provider for all agents.
            max_retries: Retry attempts for transient errors.

        Returns:
            ``RunResult`` with aggregated output, merged usage, and
            combined step count.
        """
        results: list[RunResult] = [RunResult() for _ in range(len(self.agent_order))]

        async def _run_one(idx: int) -> None:
            agent_name = self.agent_order[idx]
            try:
                agent = self.agents[agent_name]
                results[idx] = await call_runner(
                    agent,
                    input,
                    messages=messages,
                    provider=provider,
                    max_retries=max_retries,
                )
            except Exception as exc:
                raise GroupError(f"Agent '{agent_name}' failed: {exc}") from exc

        _log.debug("ParallelGroup '%s' starting: agents=%s", self.name, self.agent_order)
        try:
            async with asyncio.TaskGroup() as tg:
                for i in range(len(self.agent_order)):
                    tg.create_task(_run_one(i))
        except* GroupError as eg:
            msgs = "; ".join(str(e) for e in eg.exceptions)
            raise GroupError(f"ParallelGroup '{self.name}' failed: {msgs}") from eg
        _log.debug("ParallelGroup '%s' completed", self.name)

        return self._aggregate(results)

    def _aggregate(self, results: list[RunResult]) -> RunResult:
        """Combine multiple RunResults into one.

        Uses *aggregate_fn* if provided, otherwise joins outputs
        with *separator*.
        """
        if self.aggregate_fn is not None:
            output = self.aggregate_fn(results)
        else:
            output = self.separator.join(r.output for r in results)

        # Merge usage: sum all tokens
        total_input = sum(r.usage.input_tokens for r in results)
        total_output = sum(r.usage.output_tokens for r in results)
        total_steps = sum(r.steps for r in results)

        # Collect all messages from all agents
        all_messages: list[Message] = []
        for r in results:
            all_messages.extend(r.messages)

        return RunResult(
            output=output,
            messages=all_messages,
            usage=Usage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_input + total_output,
            ),
            steps=total_steps,
        )

    def describe(self) -> dict[str, Any]:
        """Return a summary of the group's configuration."""
        return {
            "type": "parallel",
            "name": self.name,
            "agents": list(self.agents.keys()),
        }

    def __repr__(self) -> str:
        return f"ParallelGroup(name={self.name!r}, agents={self.agent_order})"


class SerialGroup:
    """Sequential execution of agents with output→input chaining.

    Agents execute in order; each agent's output becomes the next
    agent's input.  The final agent's output is the group output.

    Args:
        name: Group name (used as a node in flow DSL).
        agents: List of agents to run sequentially (in given order).
    """

    def __init__(
        self,
        *,
        name: str,
        agents: list[Any],
    ) -> None:
        if not agents:
            raise GroupError("SerialGroup requires at least one agent")

        self.name = name
        self.agents = {a.name: a for a in agents}
        self.agent_order = [a.name for a in agents]

        # Marker so Swarm can detect groups vs regular agents
        self.is_group = True

    async def run(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Run agents sequentially, chaining output→input.

        Args:
            input: User query string for the first agent.
            messages: Prior conversation history.
            provider: LLM provider for all agents.
            max_retries: Retry attempts for transient errors.

        Returns:
            ``RunResult`` from the last agent, with accumulated
            usage and step count from all agents.
        """
        current_input = input
        total_input_tokens = 0
        total_output_tokens = 0
        total_steps = 0
        all_messages: list[Message] = []

        last_result: RunResult | None = None

        for i, agent_name in enumerate(self.agent_order):
            _log.debug(
                "SerialGroup '%s' step %d/%d: agent='%s'",
                self.name,
                i + 1,
                len(self.agent_order),
                agent_name,
            )
            agent = self.agents[agent_name]
            result = await call_runner(
                agent,
                current_input,
                messages=messages,
                provider=provider,
                max_retries=max_retries,
            )
            current_input = result.output
            total_input_tokens += result.usage.input_tokens
            total_output_tokens += result.usage.output_tokens
            total_steps += result.steps
            all_messages.extend(result.messages)
            last_result = result

        assert last_result is not None
        return RunResult(
            output=last_result.output,
            messages=all_messages,
            usage=Usage(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_input_tokens + total_output_tokens,
            ),
            steps=total_steps,
        )

    def describe(self) -> dict[str, Any]:
        """Return a summary of the group's configuration."""
        return {
            "type": "serial",
            "name": self.name,
            "agents": list(self.agents.keys()),
        }

    def __repr__(self) -> str:
        return f"SerialGroup(name={self.name!r}, agents={self.agent_order})"
