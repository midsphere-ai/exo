"""Harness ABC: composable orchestration for agent runs.

:class:`Harness` provides a single abstract method —
:meth:`execute` — where developers write orchestration logic as an
async generator.  :class:`HarnessContext` is the runtime handle
passed to ``execute()`` with utilities for running agents, managing
state, and emitting events.

Usage::

    class Router(Harness):
        async def execute(self, ctx):
            result = await ctx.run_agent(self.agents["classifier"], ctx.input)
            target = self.agents[result.output.strip()]
            async for event in ctx.stream_agent(target, ctx.input):
                yield event

    harness = Router(name="router", agents=[classifier, agent_a, agent_b])
    result = await run(harness, "Hello!")
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import Any

from exo.harness.types import (
    HarnessCheckpoint,
    HarnessEvent,
    SessionState,
    SubAgentResult,
    SubAgentTask,
)
from exo.types import (
    ErrorEvent,
    ExoError,
    Message,
    MessageContent,
    RunResult,
    StreamEvent,
    TextEvent,
    Usage,
    UsageEvent,
)

logger = logging.getLogger(__name__)


class HarnessError(ExoError):
    """Raised for harness-level errors (cancellation, invalid config, etc.)."""


class HarnessContext:
    """Runtime context passed to :meth:`Harness.execute`.

    Provides utilities for running agents, managing state,
    and emitting events — all without touching agent internals.

    The ``messages`` parameter on :meth:`run_agent` and
    :meth:`stream_agent` controls history visibility:

    - ``None`` — fresh conversation (no history)
    - ``ctx.messages`` — shared reference to harness history
    - ``list(ctx.messages)`` — forked copy of harness history

    Args:
        input: The user's input.
        messages: The message history.
        state: The mutable session state.
        harness: The parent harness instance.
        provider: LLM provider to use as default.
        detailed: Whether to emit rich event types.
        max_steps: Maximum LLM-tool round-trips per agent.
    """

    def __init__(
        self,
        *,
        input: MessageContent,
        messages: list[Message],
        state: SessionState,
        harness: Harness,
        provider: Any = None,
        detailed: bool = False,
        max_steps: int | None = None,
    ) -> None:
        self.input = input
        self.messages = messages
        self.state = state
        self._harness = harness
        self.provider = provider
        self.detailed = detailed
        self.max_steps = max_steps
        self._cancel_agents: dict[str, asyncio.Event] = {}

    async def run_agent(
        self,
        agent: Any,
        input: MessageContent,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
    ) -> RunResult:
        """Run an agent and return the result.

        Uses the harness provider if none is specified.
        Does NOT modify the agent in any way.

        Args:
            agent: An ``Agent``, ``Swarm``, or ``Harness`` instance.
            input: User query string.
            messages: Prior conversation history.  ``None`` means
                fresh conversation.
            provider: LLM provider override.

        Returns:
            ``RunResult`` with the agent's output, message history,
            usage stats, and step count.
        """
        from exo.runner import run

        return await run(
            agent,
            input,
            messages=messages,
            provider=provider or self.provider,
        )

    async def stream_agent(
        self,
        agent: Any,
        input: MessageContent,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        detailed: bool | None = None,
        max_steps: int | None = None,
        event_types: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream an agent's execution.

        Returns an async iterator of ``StreamEvent``.  Iterate this
        inside ``execute()`` and yield events through for transparent
        streaming.  Or inspect events to make routing decisions.

        Args:
            agent: An ``Agent``, ``Swarm``, or ``Harness`` instance.
            input: User query string.
            messages: Prior conversation history.
            provider: LLM provider override.
            detailed: Whether to emit rich event types.  Defaults to
                the harness-level setting.
            max_steps: Maximum LLM-tool round-trips.
            event_types: When provided, only events whose ``type``
                matches are yielded.

        Yields:
            ``StreamEvent`` instances from the agent's execution.
        """
        from exo.runner import run

        async for event in run.stream(
            agent,
            input,
            messages=messages,
            provider=provider or self.provider,
            detailed=detailed if detailed is not None else self.detailed,
            max_steps=max_steps or self.max_steps,
            event_types=event_types,
        ):
            yield event

    async def run_agents_parallel(
        self,
        tasks: list[SubAgentTask],
        *,
        continue_on_error: bool = False,
        max_concurrency: int | None = None,
    ) -> list[SubAgentResult]:
        """Run multiple agents concurrently and collect results.

        Args:
            tasks: Agent tasks to execute in parallel.
            continue_on_error: When ``False`` (default), raises
                ``SubAgentError`` on first failure.  When ``True``,
                captures errors per-agent and returns all results.
            max_concurrency: Limit concurrent agents via semaphore.
                ``None`` means unlimited.

        Returns:
            Results in the same order as *tasks*.

        Raises:
            SubAgentError: When ``continue_on_error=False`` and any
                agent fails.
        """
        from exo.harness.parallel import run_parallel

        return await run_parallel(
            self,
            tasks,
            continue_on_error=continue_on_error,
            max_concurrency=max_concurrency,
        )

    async def stream_agents_parallel(
        self,
        tasks: list[SubAgentTask],
        *,
        continue_on_error: bool = False,
        max_concurrency: int | None = None,
        queue_size: int = 256,
    ) -> AsyncIterator[StreamEvent]:
        """Stream events from multiple agents running concurrently.

        Events are multiplexed in arrival order.  Each event's
        ``agent_name`` field identifies which agent produced it.

        **Output contract:**

        - Each sub-agent writes events to a log file at
          ``/tmp/exo_subagent_{name}_{id}.log`` for real-time
          progress monitoring by the parent.
        - When a sub-agent completes, an ``AssistantMessage`` is
          appended to ``ctx.messages`` with the agent's output.
        - The ``SubAgentResult.log_path`` field holds the path
          to the agent's log file.

        Args:
            tasks: Agent tasks to execute in parallel.
            continue_on_error: When ``False``, cancels all agents on
                first error.  When ``True``, other agents continue.
            max_concurrency: Limit concurrent agents via semaphore.
            queue_size: Bounded queue size for backpressure.

        Yields:
            ``StreamEvent`` instances from all agents, interleaved by
            arrival order.
        """
        from exo.harness.parallel import stream_parallel

        async for event in stream_parallel(
            self,
            tasks,
            continue_on_error=continue_on_error,
            max_concurrency=max_concurrency,
            queue_size=queue_size,
        ):
            yield event

    def cancel_agent(self, agent_name: str) -> None:
        """Cancel a specific parallel sub-agent by name.

        Only effective for agents currently running via
        :meth:`stream_agents_parallel`.

        Args:
            agent_name: Name of the agent to cancel.
        """
        cancel_ev = self._cancel_agents.get(agent_name)
        if cancel_ev is not None:
            cancel_ev.set()

    @property
    def cancelled(self) -> bool:
        """Whether the harness has been cancelled."""
        return self._harness.cancelled

    def check_cancelled(self) -> None:
        """Raise :class:`HarnessError` if the harness has been cancelled."""
        if self._harness.cancelled:
            raise HarnessError("Harness execution cancelled")

    def emit(self, kind: str, **data: Any) -> HarnessEvent:
        """Create a :class:`HarnessEvent` for yielding from ``execute()``.

        Args:
            kind: Developer-defined event sub-kind.
            **data: Arbitrary payload stored in ``event.data``.

        Returns:
            A frozen ``HarnessEvent`` ready to be yielded.
        """
        return HarnessEvent(
            kind=kind,
            agent_name=self._harness.name,
            data=data,
        )

    async def checkpoint(
        self,
        *,
        pending_agent: str | None = None,
        pending_agents: list[str] | None = None,
    ) -> None:
        """Save a checkpoint of the current harness state.

        Args:
            pending_agent: Name of a single agent about to execute.
            pending_agents: Names of multiple agents about to execute
                in parallel.  Takes precedence over *pending_agent*.
        """
        await self._harness.save_checkpoint(
            pending_agent=pending_agent, pending_agents=pending_agents
        )


class Harness(ABC):
    """Abstract orchestration harness for agent runs.

    Subclass and implement :meth:`execute` to define orchestration
    logic.  The harness provides :meth:`run`, :meth:`stream`, and
    :meth:`cancel` with the same interface as ``Agent`` and ``Swarm``.

    Args:
        name: Unique name for this harness.
        agents: Agents available for orchestration.  Accepts a list
            (indexed by ``agent.name``) or a dict.
        state: Initial session state.  Accepts a ``SessionState``
            or a plain dict.
        checkpoint_store: Optional ``MemoryStore`` backend for
            checkpoint persistence.
        middleware: List of :class:`Middleware` instances to wrap
            the event stream.
    """

    # Marker for runner.py duck-typing detection (parallels Swarm.flow_order)
    is_harness: bool = True

    def __init__(
        self,
        *,
        name: str,
        agents: dict[str, Any] | list[Any] | None = None,
        state: SessionState | dict[str, Any] | None = None,
        checkpoint_store: Any = None,
        middleware: list[Any] | None = None,
    ) -> None:
        self.name = name

        # Index agents by name
        if agents is None:
            self.agents: dict[str, Any] = {}
        elif isinstance(agents, list):
            self.agents = {}
            for agent in agents:
                agent_name = agent.name
                if agent_name in self.agents:
                    raise HarnessError(f"Duplicate agent name '{agent_name}' in harness")
                self.agents[agent_name] = agent
        else:
            self.agents = dict(agents)

        # Session state
        if isinstance(state, dict):
            self.session = SessionState(data=state)
        elif state is not None:
            self.session = state
        else:
            self.session = SessionState()

        self._checkpoint_store = checkpoint_store
        self._middleware = list(middleware or [])
        self._cancel_event = asyncio.Event()

    # === Abstract method: the ONE thing developers implement ===

    @abstractmethod
    async def execute(self, ctx: HarnessContext) -> AsyncIterator[StreamEvent]:
        """Orchestration logic.  Yields ``StreamEvent`` instances.

        Use ``ctx`` to run agents, manage state, and emit events.
        Normal Python control flow (if/else, loops, try/except)
        drives orchestration decisions.

        Args:
            ctx: Runtime context with agent execution utilities.

        Yields:
            ``StreamEvent`` instances — both standard agent events
            (passed through from ``ctx.stream_agent()``) and custom
            ``HarnessEvent`` instances (created via ``ctx.emit()``).
        """
        yield  # type: ignore[misc]  # pragma: no cover

    # === Public API (same shape as Agent / Swarm) ===

    async def run(
        self,
        input: MessageContent,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Execute the harness and return a :class:`RunResult`.

        Drains the streaming generator, accumulates text from
        ``TextEvent`` instances and usage from ``UsageEvent``
        instances, then returns the aggregated result.

        Args:
            input: User query — a string or list of ContentBlock
                objects.
            messages: Prior conversation history.
            provider: LLM provider passed to agents.
            max_retries: Not used directly — available for
                ``execute()`` logic via ``ctx``.

        Returns:
            ``RunResult`` with the harness's output, message
            history, usage stats, and step count.
        """
        logger.debug("harness.run() starting harness='%s'", self.name)
        ctx = self._build_context(input=input, messages=messages, provider=provider)

        output_parts: list[str] = []
        total_input = 0
        total_output = 0
        total_total = 0

        async for event in self._execute_with_middleware(ctx):
            if isinstance(event, TextEvent):
                output_parts.append(event.text)
            elif isinstance(event, UsageEvent):
                total_input += event.usage.input_tokens
                total_output += event.usage.output_tokens
                total_total += event.usage.total_tokens

        return RunResult(
            output="".join(output_parts),
            messages=ctx.messages,
            usage=Usage(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=total_total,
            ),
        )

    async def stream(
        self,
        input: MessageContent,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        detailed: bool = False,
        max_steps: int | None = None,
        event_types: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream harness execution, same interface as Agent/Swarm.

        Each event includes the correct ``agent_name`` of the agent
        that produced it.  Custom ``HarnessEvent`` instances carry
        the harness name.

        Args:
            input: User query string.
            messages: Prior conversation history.
            provider: LLM provider passed to agents.
            detailed: When ``True``, emit rich event types.
            max_steps: Maximum LLM-tool round-trips per agent.
            event_types: When provided, only events whose ``type``
                field matches one of the given strings are yielded.

        Yields:
            ``StreamEvent`` instances from agent execution and
            custom ``HarnessEvent`` instances.
        """
        logger.debug("harness.stream() starting harness='%s'", self.name)
        ctx = self._build_context(
            input=input,
            messages=messages,
            provider=provider,
            detailed=detailed,
            max_steps=max_steps,
        )

        async for event in self._execute_with_middleware(ctx):
            if event_types is None or event.type in event_types:
                yield event

    # === Cancellation ===

    def cancel(self) -> None:
        """Signal cancellation to the running ``execute()``."""
        self._cancel_event.set()

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been signalled."""
        return self._cancel_event.is_set()

    def reset(self) -> None:
        """Clear the cancellation signal for reuse."""
        self._cancel_event.clear()

    # === Checkpoint support ===

    async def save_checkpoint(
        self,
        *,
        pending_agent: str | None = None,
        pending_agents: list[str] | None = None,
    ) -> None:
        """Persist current state to the checkpoint store.

        No-op if no checkpoint store is configured.

        Args:
            pending_agent: Name of a single agent about to execute.
            pending_agents: Names of multiple agents about to execute
                in parallel.
        """
        if self._checkpoint_store is None:
            return
        from exo.harness.checkpoint import CheckpointAdapter

        adapter = CheckpointAdapter(store=self._checkpoint_store, harness_name=self.name)
        checkpoint = HarnessCheckpoint(
            harness_name=self.name,
            session_state=dict(self.session.data),
            completed_agents=list(self.session.get("_completed_agents", [])),
            pending_agent=pending_agent,
            pending_agents=pending_agents or [],
        )
        await adapter.save(checkpoint)
        self.session.mark_clean()
        logger.debug("Checkpoint saved for harness='%s'", self.name)

    async def restore_checkpoint(self) -> HarnessCheckpoint | None:
        """Load the latest checkpoint from the store.

        Returns:
            The latest checkpoint, or ``None`` if none exists
            or no store is configured.
        """
        if self._checkpoint_store is None:
            return None
        from exo.harness.checkpoint import CheckpointAdapter

        adapter = CheckpointAdapter(store=self._checkpoint_store, harness_name=self.name)
        return await adapter.load_latest()

    # === Internal ===

    def _build_context(
        self,
        *,
        input: MessageContent,
        messages: Sequence[Message] | None,
        provider: Any = None,
        detailed: bool = False,
        max_steps: int | None = None,
    ) -> HarnessContext:
        return HarnessContext(
            input=input,
            messages=list(messages) if messages else [],
            state=self.session,
            harness=self,
            provider=provider,
            detailed=detailed,
            max_steps=max_steps,
        )

    async def _execute_with_middleware(self, ctx: HarnessContext) -> AsyncIterator[StreamEvent]:
        """Wrap ``execute()`` with the middleware chain."""
        stream: AsyncIterator[StreamEvent] = self.execute(ctx)
        for mw in reversed(self._middleware):
            stream = mw.wrap(stream, ctx)
        try:
            async for event in stream:
                yield event
        except HarnessError:
            raise
        except Exception as exc:
            yield ErrorEvent(
                error=str(exc),
                error_type=type(exc).__name__,
                agent_name=self.name,
                recoverable=False,
            )
            raise

    def __repr__(self) -> str:
        agents = ", ".join(self.agents.keys())
        return f"Harness(name={self.name!r}, agents=[{agents}])"


class HarnessNode:
    """Wraps a :class:`Harness` for use as a node in a Swarm.

    Follows the same pattern as ``SwarmNode`` and ``RalphNode``
    from ``exo._internal.nested``.  The ``is_group = True`` marker
    makes the Swarm's duck-typing check route to ``.stream()``
    during streaming and ``.run()`` during non-streaming execution.

    Args:
        harness: The harness to wrap.
        name: Node name for the outer Swarm's flow DSL.
            Defaults to the harness's ``name``.
    """

    def __init__(self, *, harness: Harness, name: str | None = None) -> None:
        self._harness = harness
        self.name = name or harness.name
        self.is_group = True

    async def run(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Execute the inner harness with context isolation.

        Args:
            input: User query string.
            messages: Not forwarded (context isolation).
            provider: LLM provider, forwarded to inner harness.
            max_retries: Forwarded to inner harness.

        Returns:
            ``RunResult`` from the inner harness.
        """
        return await self._harness.run(input, provider=provider, max_retries=max_retries)

    async def stream(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        detailed: bool = False,
        max_steps: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream inner harness events with context isolation.

        Args:
            input: User query string.
            messages: Not forwarded (context isolation).
            provider: LLM provider, forwarded to inner harness.
            detailed: When ``True``, emit rich event types.
            max_steps: Maximum LLM-tool round-trips per agent.

        Yields:
            ``StreamEvent`` instances from the inner harness.
        """
        async for event in self._harness.stream(
            input, provider=provider, detailed=detailed, max_steps=max_steps
        ):
            yield event

    def __repr__(self) -> str:
        return f"HarnessNode(name={self.name!r})"
