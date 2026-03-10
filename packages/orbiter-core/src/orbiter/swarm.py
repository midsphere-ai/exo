"""Swarm: multi-agent orchestration with flow DSL.

A ``Swarm`` groups multiple agents and defines their execution
topology using a simple DSL (``"a >> b >> c"``).  Supports
``mode='workflow'`` (sequential pipeline), ``mode='handoff'``
(agent-driven delegation), and ``mode='team'`` (lead-worker
delegation).

Usage::

    swarm = Swarm(
        agents=[agent_a, agent_b, agent_c],
        flow="a >> b >> c",
    )
    result = await run(swarm, "Hello!")
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from orbiter._internal.call_runner import call_runner
from orbiter._internal.graph import GraphError, parse_flow_dsl, topological_sort
from orbiter._internal.workflow_checkpoint import WorkflowCheckpoint, WorkflowCheckpointStore
from orbiter._internal.workflow_state import WorkflowState
from orbiter.observability.logging import get_logger  # pyright: ignore[reportMissingImports]
from orbiter.tool import Tool
from orbiter.types import Message, OrbiterError, RunResult, StatusEvent, StreamEvent

_log = get_logger(__name__)


class SwarmError(OrbiterError):
    """Raised for swarm-level errors (invalid flow, missing agents, etc.)."""


class Swarm:
    """Multi-agent orchestration container.

    Groups agents and defines their execution topology via a flow DSL.
    In workflow mode, agents run sequentially with output→input chaining.
    In handoff mode, agents delegate dynamically via handoff targets.
    In team mode, the first agent is the lead and others are workers;
    the lead can delegate to workers via auto-generated tools.

    Args:
        agents: List of ``Agent`` instances to include in the swarm.
        flow: Flow DSL string defining execution order
            (e.g., ``"a >> b >> c"``).  If not provided, agents
            run in the order they are given.
        mode: Execution mode — ``"workflow"``, ``"handoff"``, or ``"team"``.
        max_handoffs: Maximum number of handoff transitions before
            raising an error (handoff mode only).
    """

    def __init__(
        self,
        *,
        agents: list[Any],
        flow: str | None = None,
        mode: str = "workflow",
        max_handoffs: int = 10,
    ) -> None:
        if not agents:
            raise SwarmError("Swarm requires at least one agent")

        self.mode = mode
        self.max_handoffs = max_handoffs

        # Index agents by name for O(1) lookup
        self.agents: dict[str, Any] = {}
        for agent in agents:
            name = agent.name
            if name in self.agents:
                _log.error("Duplicate agent name '%s' in swarm", name)
                raise SwarmError(f"Duplicate agent name '{name}' in swarm")
            self.agents[name] = agent

        # Resolve execution order from flow DSL or agent list order
        if flow is not None:
            try:
                graph = parse_flow_dsl(flow)
            except GraphError as exc:
                raise SwarmError(f"Invalid flow DSL: {exc}") from exc

            # Validate all flow nodes are known agents
            for node_name in graph.nodes:
                if node_name not in self.agents:
                    raise SwarmError(f"Flow references unknown agent '{node_name}'")

            try:
                self.flow_order = topological_sort(graph)
            except GraphError as exc:
                _log.error("Cycle detected in flow DSL: %s", exc)
                raise SwarmError(f"Cycle in flow DSL: {exc}") from exc
        else:
            # Default: run in the order agents were provided
            self.flow_order = [a.name for a in agents]

        self.flow = flow

        # Set name from the first agent for compatibility with runner
        self.name = f"swarm({self.flow_order[0]}...)"

    async def run(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
        checkpoint: bool | WorkflowCheckpointStore = False,
    ) -> RunResult:
        """Execute the swarm according to its mode.

        In workflow mode, agents execute in topological order.
        In handoff mode, the first agent runs and can hand off
        to other agents dynamically.

        Args:
            input: User query string.
            messages: Prior conversation history.
            provider: LLM provider for all agents.
            max_retries: Retry attempts for transient errors.
            checkpoint: Enable state checkpointing before each node.
                Pass ``True`` to create a new store, or pass an existing
                :class:`WorkflowCheckpointStore` instance.  Only applies
                to workflow mode.

        Returns:
            ``RunResult`` from the final agent in the chain.

        Raises:
            SwarmError: If mode is unsupported or an agent fails.
        """
        if self.mode == "workflow":
            checkpoint_store: WorkflowCheckpointStore | None = None
            if checkpoint is True:
                checkpoint_store = WorkflowCheckpointStore()
            elif isinstance(checkpoint, WorkflowCheckpointStore):
                checkpoint_store = checkpoint
            return await self._run_workflow(
                input, messages=messages, provider=provider,
                max_retries=max_retries, checkpoint_store=checkpoint_store,
            )
        if self.mode == "handoff":
            return await self._run_handoff(
                input, messages=messages, provider=provider, max_retries=max_retries
            )
        if self.mode == "team":
            return await self._run_team(
                input, messages=messages, provider=provider, max_retries=max_retries
            )
        raise SwarmError(f"Unsupported swarm mode: {self.mode!r}")

    async def stream(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        detailed: bool = False,
        max_steps: int | None = None,
        event_types: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream swarm execution, yielding events from each sub-agent.

        Each event includes the correct ``agent_name`` of the sub-agent
        that produced it.  ``StatusEvent`` is emitted for agent handoffs
        and delegations.

        Args:
            input: User query string.
            messages: Prior conversation history.
            provider: LLM provider for all agents.
            detailed: When ``True``, emit rich event types.
            max_steps: Maximum LLM-tool round-trips per agent.
            event_types: When provided, only events whose ``type`` field
                matches one of the given strings are yielded.

        Yields:
            ``StreamEvent`` instances from sub-agent execution.
        """
        if self.mode == "workflow":
            async for event in self._stream_workflow(
                input, messages=messages, provider=provider,
                detailed=detailed, max_steps=max_steps,
                event_types=event_types,
            ):
                yield event
        elif self.mode == "handoff":
            async for event in self._stream_handoff(
                input, messages=messages, provider=provider,
                detailed=detailed, max_steps=max_steps,
                event_types=event_types,
            ):
                yield event
        elif self.mode == "team":
            async for event in self._stream_team(
                input, messages=messages, provider=provider,
                detailed=detailed, max_steps=max_steps,
                event_types=event_types,
            ):
                yield event
        else:
            raise SwarmError(f"Unsupported swarm mode: {self.mode!r}")

    async def _stream_workflow(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        detailed: bool = False,
        max_steps: int | None = None,
        event_types: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream workflow mode: agents run sequentially, chaining output→input.

        Collects text from ``TextEvent`` objects during streaming to build
        the output for chaining to the next agent — avoiding double execution.

        A :class:`WorkflowState` accumulates agent outputs so that
        :class:`BranchNode` and :class:`LoopNode` conditions can
        reference any prior agent's output.
        """
        from orbiter.runner import run
        from orbiter.types import TextEvent

        def _passes_filter(event: StreamEvent) -> bool:
            return event_types is None or event.type in event_types

        current_input = input
        skip_until: str | None = None
        workflow_state = WorkflowState({"input": current_input})

        for agent_name in self.flow_order:
            # Branch routing: skip agents until we reach the target
            if skip_until is not None:
                if agent_name != skip_until:
                    continue
                skip_until = None

            agent = self.agents[agent_name]

            # Branch node: evaluate condition and route to target
            if getattr(agent, "is_branch", False):
                state = workflow_state.to_dict()
                state["input"] = current_input
                target = agent.evaluate(state)
                if target not in self.agents:
                    raise SwarmError(
                        f"Branch '{agent_name}' targets unknown agent '{target}'"
                    )
                if detailed:
                    _ev = StatusEvent(
                        status="running",
                        agent_name=agent_name,
                        message=f"Branch '{agent_name}' routing to '{target}'",
                    )
                    if _passes_filter(_ev):
                        yield _ev
                # Check if target is ahead in the flow — skip to it
                remaining = self.flow_order[self.flow_order.index(agent_name) + 1 :]
                if target in remaining:
                    skip_until = target
                    continue
                # Target not in remaining flow — stream it directly and stop
                target_agent = self.agents[target]
                text_parts_branch: list[str] = []
                async for event in run.stream(
                    target_agent, current_input, messages=messages,
                    provider=provider, detailed=detailed, max_steps=max_steps,
                ):
                    if isinstance(event, TextEvent):
                        text_parts_branch.append(event.text)
                    if _passes_filter(event):
                        yield event
                workflow_state.set(target, "".join(text_parts_branch))
                return

            if detailed:
                _ev = StatusEvent(
                    status="running",
                    agent_name=agent_name,
                    message=f"Workflow executing agent '{agent_name}'",
                )
                if _passes_filter(_ev):
                    yield _ev

            # Loop node: iterate body agents with streaming
            if getattr(agent, "is_loop", False):
                from orbiter._internal.loop_node import BREAK_SENTINEL

                loop_state = workflow_state.to_dict()
                loop_state["input"] = current_input
                n_iters = agent._resolve_iterations(loop_state)
                iteration = 0
                while True:
                    if iteration >= agent.max_iterations:
                        break
                    if n_iters is not None and iteration >= n_iters:
                        break
                    if agent.condition is not None and not agent._check_condition(loop_state):
                        break
                    loop_state["iteration"] = iteration
                    if agent.items is not None:
                        arr = loop_state.get(agent.items, [])
                        loop_state["item"] = arr[iteration] if iteration < len(arr) else None
                    body_input = current_input
                    broke = False
                    for body_name in agent.body:
                        if body_name not in self.agents:
                            raise SwarmError(
                                f"Loop '{agent_name}' body references unknown agent '{body_name}'"
                            )
                        body_agent = self.agents[body_name]
                        loop_text_parts: list[str] = []
                        async for event in run.stream(
                            body_agent, body_input, messages=messages,
                            provider=provider, detailed=detailed,
                            max_steps=max_steps,
                        ):
                            if isinstance(event, TextEvent):
                                loop_text_parts.append(event.text)
                            if _passes_filter(event):
                                yield event
                        body_input = "".join(loop_text_parts)
                        workflow_state.set(body_name, body_input)
                        if BREAK_SENTINEL in body_input:
                            broke = True
                            break
                    current_input = body_input
                    loop_state["input"] = current_input
                    iteration += 1
                    if broke:
                        break
                workflow_state.set(agent_name, current_input)
                continue

            # For groups/nested swarms, delegate to their stream if available
            if getattr(agent, "is_group", False) or getattr(agent, "is_swarm", False):
                if hasattr(agent, "stream"):
                    text_parts: list[str] = []
                    async for event in agent.stream(
                        current_input, messages=messages, provider=provider,
                        detailed=detailed, max_steps=max_steps,
                    ):
                        if isinstance(event, TextEvent):
                            text_parts.append(event.text)
                        if _passes_filter(event):
                            yield event
                    current_input = "".join(text_parts)
                else:
                    result = await agent.run(
                        current_input, messages=messages, provider=provider,
                    )
                    current_input = result.output
                workflow_state.set(agent_name, current_input)
                continue

            # Stream the agent and collect text for chaining
            text_parts = []
            async for event in run.stream(
                agent, current_input, messages=messages, provider=provider,
                detailed=detailed, max_steps=max_steps,
            ):
                if isinstance(event, TextEvent):
                    text_parts.append(event.text)
                if _passes_filter(event):
                    yield event

            current_input = "".join(text_parts)
            workflow_state.set(agent_name, current_input)

    async def _stream_handoff(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        detailed: bool = False,
        max_steps: int | None = None,
        event_types: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream handoff mode: agents delegate dynamically.

        Collects text from ``TextEvent`` objects during streaming to detect
        handoff targets without double execution.
        """
        from orbiter.runner import run
        from orbiter.types import TextEvent

        def _passes_filter(event: StreamEvent) -> bool:
            return event_types is None or event.type in event_types

        current_agent_name = self.flow_order[0]
        current_input = input
        handoff_count = 0

        while True:
            agent = self.agents[current_agent_name]

            if detailed:
                _ev = StatusEvent(
                    status="running",
                    agent_name=current_agent_name,
                    message=f"Handoff executing agent '{current_agent_name}'",
                )
                if _passes_filter(_ev):
                    yield _ev

            # Stream the agent's execution and collect text
            text_parts: list[str] = []
            async for event in run.stream(
                agent, current_input, messages=messages, provider=provider,
                detailed=detailed, max_steps=max_steps,
            ):
                if isinstance(event, TextEvent):
                    text_parts.append(event.text)
                if _passes_filter(event):
                    yield event

            output_text = "".join(text_parts)

            # Check for handoff using a lightweight RunResult
            from orbiter.types import RunResult

            fake_result = RunResult(output=output_text)
            next_agent = self._detect_handoff(agent, fake_result)
            if next_agent is None:
                return

            handoff_count += 1
            if handoff_count > self.max_handoffs:
                raise SwarmError(f"Max handoffs ({self.max_handoffs}) exceeded in swarm")

            if detailed:
                _ev = StatusEvent(
                    status="running",
                    agent_name=next_agent,
                    message=f"Handoff from '{current_agent_name}' to '{next_agent}'",
                )
                if _passes_filter(_ev):
                    yield _ev

            current_agent_name = next_agent
            current_input = output_text

    async def _stream_team(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        detailed: bool = False,
        max_steps: int | None = None,
        event_types: set[str] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream team mode: lead delegates to workers via tools."""
        from orbiter.runner import run

        def _passes_filter(event: StreamEvent) -> bool:
            return event_types is None or event.type in event_types

        if len(self.agents) < 2:
            raise SwarmError("Team mode requires at least two agents (lead + workers)")

        lead_name = self.flow_order[0]
        lead = self.agents[lead_name]
        worker_names = [n for n in self.flow_order if n != lead_name]

        # Create delegate tools for each worker and add to lead
        delegate_tools: list[Tool] = []
        for worker_name in worker_names:
            worker = self.agents[worker_name]
            dtool = _DelegateTool(
                worker=worker,
                provider=provider,
                max_retries=3,
            )
            delegate_tools.append(dtool)

        # Temporarily add delegate tools to the lead agent
        original_tools = dict(lead.tools)
        for dtool in delegate_tools:
            lead.tools[dtool.name] = dtool

        try:
            if detailed:
                _ev = StatusEvent(
                    status="running",
                    agent_name=lead_name,
                    message=f"Team lead '{lead_name}' starting execution",
                )
                if _passes_filter(_ev):
                    yield _ev

            async for event in run.stream(
                lead, input, messages=messages, provider=provider,
                detailed=detailed, max_steps=max_steps,
                event_types=event_types,
            ):
                yield event
        finally:
            # Restore original tools
            lead.tools = original_tools

    async def _run_workflow(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
        checkpoint_store: WorkflowCheckpointStore | None = None,
        _resume_completed: set[str] | None = None,
        _resume_state: dict[str, Any] | None = None,
    ) -> RunResult:
        """Execute agents sequentially, chaining output→input.

        Supports regular agents, group nodes (``ParallelGroup``,
        ``SerialGroup``), nested swarms, branch nodes
        (``BranchNode``), and loop nodes (``LoopNode``).  Branch
        nodes evaluate a condition and skip to the chosen target
        agent in the flow.  Loop nodes iterate body agents.

        A :class:`WorkflowState` is created at the start and
        propagated through execution.  Each agent's output is stored
        as ``state.set(agent_name, output)`` so downstream nodes can
        access the full workflow context for condition evaluation.

        When *checkpoint_store* is provided, a
        :class:`WorkflowCheckpoint` is saved before each node
        executes.

        Returns the ``RunResult`` from the last agent in the flow.
        """
        current_input = input
        last_result: RunResult | None = None
        skip_until: str | None = None

        # Restore state when resuming from a checkpoint
        if _resume_state is not None:
            workflow_state = WorkflowState(_resume_state)
            current_input = _resume_state.get("input", input)
        else:
            workflow_state = WorkflowState({"input": current_input})

        completed_nodes: list[str] = list(_resume_completed) if _resume_completed else []

        for agent_name in self.flow_order:
            # Branch routing: skip agents until we reach the target
            if skip_until is not None:
                if agent_name != skip_until:
                    continue
                skip_until = None

            # Resume: skip already-completed nodes
            if agent_name in completed_nodes:
                saved = workflow_state.get(agent_name)
                if saved is not None:
                    current_input = saved
                continue

            agent = self.agents[agent_name]

            # Save checkpoint before executing this node
            if checkpoint_store is not None:
                checkpoint_store.save(
                    WorkflowCheckpoint(
                        node_name=agent_name,
                        state=workflow_state.to_dict(),
                        completed_nodes=list(completed_nodes),
                    )
                )

            # Branch node: evaluate condition and route to target
            if getattr(agent, "is_branch", False):
                state = workflow_state.to_dict()
                state["input"] = current_input
                target = agent.evaluate(state)
                if target not in self.agents:
                    raise SwarmError(
                        f"Branch '{agent_name}' targets unknown agent '{target}'"
                    )
                # Check if target is ahead in the flow — skip to it
                remaining = self.flow_order[self.flow_order.index(agent_name) + 1 :]
                if target in remaining:
                    skip_until = target
                    continue
                # Target not in remaining flow — execute directly and stop
                target_agent = self.agents[target]
                last_result = await call_runner(
                    target_agent,
                    current_input,
                    messages=messages,
                    provider=provider,
                    max_retries=max_retries,
                )
                workflow_state.set(target, last_result.output)
                return last_result

            # Loop node: iterate body agents
            if getattr(agent, "is_loop", False):
                from orbiter._internal.loop_node import BREAK_SENTINEL

                loop_state = workflow_state.to_dict()
                loop_state["input"] = current_input
                n_iters = agent._resolve_iterations(loop_state)
                iteration = 0
                while True:
                    if iteration >= agent.max_iterations:
                        break
                    if n_iters is not None and iteration >= n_iters:
                        break
                    if agent.condition is not None and not agent._check_condition(loop_state):
                        break
                    loop_state["iteration"] = iteration
                    if agent.items is not None:
                        arr = loop_state.get(agent.items, [])
                        loop_state["item"] = arr[iteration] if iteration < len(arr) else None
                    body_input = current_input
                    broke = False
                    for body_name in agent.body:
                        if body_name not in self.agents:
                            raise SwarmError(
                                f"Loop '{agent_name}' body references unknown agent '{body_name}'"
                            )
                        body_agent = self.agents[body_name]
                        last_result = await call_runner(
                            body_agent,
                            body_input,
                            messages=messages,
                            provider=provider,
                            max_retries=max_retries,
                        )
                        body_input = last_result.output
                        workflow_state.set(body_name, last_result.output)
                        if BREAK_SENTINEL in last_result.output:
                            broke = True
                            break
                    current_input = body_input
                    loop_state["input"] = current_input
                    iteration += 1
                    if broke:
                        break
                workflow_state.set(agent_name, current_input)
                completed_nodes.append(agent_name)
                if last_result is None:
                    last_result = RunResult(output=current_input)
                continue

            if getattr(agent, "is_group", False) or getattr(agent, "is_swarm", False):
                last_result = await agent.run(
                    current_input,
                    messages=messages,
                    provider=provider,
                    max_retries=max_retries,
                )
            else:
                last_result = await call_runner(
                    agent,
                    current_input,
                    messages=messages,
                    provider=provider,
                    max_retries=max_retries,
                )
            current_input = last_result.output  # type: ignore[union-attr]
            workflow_state.set(agent_name, current_input)
            completed_nodes.append(agent_name)

        assert last_result is not None  # guaranteed since agents is non-empty
        return last_result

    async def resume(
        self,
        checkpoint_store: WorkflowCheckpointStore,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
        checkpoint: bool | WorkflowCheckpointStore = False,
    ) -> RunResult:
        """Resume workflow execution from the latest checkpoint.

        Loads the latest checkpoint from *checkpoint_store*, restores
        workflow state, and re-executes ``_run_workflow`` starting from
        the checkpoint node (skipping already-completed nodes).

        Args:
            checkpoint_store: Store containing checkpoints from a
                previous run.
            messages: Prior conversation history.
            provider: LLM provider for all agents.
            max_retries: Retry attempts for transient errors.
            checkpoint: Enable checkpointing during the resumed run.
                Pass ``True`` to create a new store, or an existing
                :class:`WorkflowCheckpointStore`.

        Returns:
            ``RunResult`` from the final agent in the resumed chain.

        Raises:
            SwarmError: If no checkpoints exist or mode is not workflow.
        """
        if self.mode != "workflow":
            raise SwarmError("resume() is only supported in workflow mode")

        latest = checkpoint_store.latest()
        if latest is None:
            raise SwarmError("No checkpoints available to resume from")

        resume_store: WorkflowCheckpointStore | None = None
        if checkpoint is True:
            resume_store = WorkflowCheckpointStore()
        elif isinstance(checkpoint, WorkflowCheckpointStore):
            resume_store = checkpoint

        return await self._run_workflow(
            latest.state.get("input", ""),
            messages=messages,
            provider=provider,
            max_retries=max_retries,
            checkpoint_store=resume_store,
            _resume_completed=set(latest.completed_nodes),
            _resume_state=latest.state,
        )

    async def _run_handoff(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Execute agents following handoff chains.

        Starts with the first agent in flow_order.  If an agent's
        output matches a handoff target name (declared on the agent),
        control transfers to that target with the full conversation
        history.  Stops when an agent produces output that is not a
        handoff target, or when ``max_handoffs`` is exceeded.

        Returns the ``RunResult`` from the last agent that ran.
        """
        current_agent_name = self.flow_order[0]
        current_input = input
        all_messages: list[Message] = list(messages) if messages else []
        handoff_count = 0

        while True:
            agent = self.agents[current_agent_name]
            result = await call_runner(
                agent,
                current_input,
                messages=all_messages,
                provider=provider,
                max_retries=max_retries,
            )

            # Accumulate conversation history from this agent's run
            all_messages = list(result.messages)

            # Check for handoff
            next_agent = self._detect_handoff(agent, result)
            if next_agent is None:
                return result

            handoff_count += 1
            if handoff_count > self.max_handoffs:
                raise SwarmError(f"Max handoffs ({self.max_handoffs}) exceeded in swarm")

            current_agent_name = next_agent
            current_input = result.output

    def _detect_handoff(self, agent: Any, result: RunResult) -> str | None:
        """Check if an agent's result indicates a handoff.

        Matches the agent's output (stripped) against its declared
        handoff target names.  The target must also exist in the
        swarm's agents dict.

        Returns:
            Target agent name, or ``None`` if no handoff detected.
        """
        handoffs: dict[str, Any] = getattr(agent, "handoffs", {})
        if not handoffs:
            return None

        output = result.output.strip()
        for target_name in handoffs:
            if target_name in self.agents and output == target_name:
                return target_name

        return None

    async def _run_team(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> RunResult:
        """Execute team mode: lead delegates to workers via tools.

        The first agent in flow_order is the lead.  Other agents are
        workers.  The lead receives auto-generated ``delegate_to_{name}``
        tools that invoke worker agents.  When the lead calls a delegate
        tool, the worker runs and its output is returned as the tool
        result.  The lead then synthesizes the final output.

        Returns the ``RunResult`` from the lead agent.
        """
        if len(self.agents) < 2:
            raise SwarmError("Team mode requires at least two agents (lead + workers)")

        lead_name = self.flow_order[0]
        lead = self.agents[lead_name]
        worker_names = [n for n in self.flow_order if n != lead_name]

        # Create delegate tools for each worker and add to lead
        delegate_tools: list[Tool] = []
        for worker_name in worker_names:
            worker = self.agents[worker_name]
            dtool = _DelegateTool(
                worker=worker,
                provider=provider,
                max_retries=max_retries,
            )
            delegate_tools.append(dtool)

        # Temporarily add delegate tools to the lead agent
        original_tools = dict(lead.tools)
        for dtool in delegate_tools:
            lead.tools[dtool.name] = dtool

        try:
            result = await call_runner(
                lead,
                input,
                messages=messages,
                provider=provider,
                max_retries=max_retries,
            )
        finally:
            # Restore original tools
            lead.tools = original_tools

        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize the swarm configuration to a dict.

        All agents are serialized via ``Agent.to_dict()``.

        Returns:
            A dict suitable for JSON serialization and later reconstruction
            via ``Swarm.from_dict()``.
        """
        return {
            "agents": [agent.to_dict() for agent in self.agents.values()],
            "flow": self.flow,
            "mode": self.mode,
            "max_handoffs": self.max_handoffs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Swarm:
        """Reconstruct a Swarm from a dict produced by ``to_dict()``.

        Args:
            data: Dict as produced by ``Swarm.to_dict()``.

        Returns:
            A reconstructed ``Swarm`` instance.
        """
        from orbiter.agent import Agent

        agents = [Agent.from_dict(a) for a in data["agents"]]
        return cls(
            agents=agents,
            flow=data.get("flow"),
            mode=data.get("mode", "workflow"),
            max_handoffs=data.get("max_handoffs", 10),
        )

    def describe(self) -> dict[str, Any]:
        """Return a summary of the swarm's configuration.

        Returns:
            Dict with mode, flow order, and agent descriptions.
        """
        return {
            "mode": self.mode,
            "flow": self.flow,
            "flow_order": self.flow_order,
            "agents": {name: agent.describe() for name, agent in self.agents.items()},
        }

    def __repr__(self) -> str:
        return f"Swarm(mode={self.mode!r}, agents={list(self.agents.keys())}, flow={self.flow!r})"


class _DelegateTool(Tool):
    """Auto-generated tool that delegates work to a worker agent.

    When the lead agent calls this tool, the worker agent runs with
    the provided task description and its output is returned as the
    tool result.
    """

    def __init__(
        self,
        *,
        worker: Any,
        provider: Any = None,
        max_retries: int = 3,
    ) -> None:
        worker_name: str = worker.name
        self.name = f"delegate_to_{worker_name}"
        self.description = f"Delegate a task to the '{worker_name}' worker agent."
        self.parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": f"The task description to send to '{worker_name}'.",
                },
            },
            "required": ["task"],
        }
        self._worker = worker
        self._provider = provider
        self._max_retries = max_retries

    async def execute(self, **kwargs: Any) -> str:
        """Run the worker agent with the given task.

        Args:
            **kwargs: Must include ``task`` (str).

        Returns:
            The worker agent's output text.
        """
        task: str = kwargs.get("task", "")
        result = await call_runner(
            self._worker,
            task,
            provider=self._provider,
            max_retries=self._max_retries,
        )
        return result.output
