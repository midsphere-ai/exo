"""Public entry point for running agents.

Provides ``run()`` (async), ``run.sync()`` (blocking), and
``run.stream()`` (async generator) as the primary API for executing
an ``Agent``.  Internally delegates to
:func:`orbiter._internal.call_runner.call_runner` for state tracking
and loop detection.

Usage::

    result = await run(agent, "Hello!")
    result = run.sync(agent, "Hello!")
    async for event in run.stream(agent, "Hello!"):
        print(event)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from orbiter._internal.call_runner import call_runner
from orbiter._internal.message_builder import build_messages
from orbiter._internal.output_parser import parse_tool_arguments
from orbiter.observability.metrics import (  # pyright: ignore[reportMissingImports]
    HAS_OTEL,
    _collector,
    _get_meter,
)
from orbiter.observability.semconv import (  # pyright: ignore[reportMissingImports]
    METRIC_STREAM_EVENTS_EMITTED,
    STREAM_EVENT_TYPE,
)
from orbiter.types import (
    AssistantMessage,
    ErrorEvent,
    Message,
    ReasoningEvent,
    RunResult,
    StatusEvent,
    StepEvent,
    StreamEvent,
    TextEvent,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    Usage,
    UsageEvent,
    UserMessage,
)


async def run(
    agent: Any,
    input: str,
    *,
    messages: Sequence[Message] | None = None,
    provider: Any = None,
    max_retries: int = 3,
    loop_threshold: int = 3,
) -> RunResult:
    """Execute an agent (or swarm) and return the result.

    This is the primary async API for running agents.  For a blocking
    variant, use ``run.sync()``.

    If *provider* is ``None``, a default provider is resolved from the
    agent's ``provider_name`` using the model registry (if available).

    Args:
        agent: An ``Agent`` (or ``Swarm``) instance.
        input: User query string.
        messages: Prior conversation history to continue from.
        provider: LLM provider with ``async complete()`` method.
            When ``None``, auto-resolved from the agent's model string.
        max_retries: Retry attempts for transient LLM errors.
        loop_threshold: Consecutive identical tool-call patterns
            before raising a loop error.

    Returns:
        ``RunResult`` with the agent's output, message history,
        usage stats, and step count.
    """
    resolved_provider = provider or _resolve_provider(agent)

    # Detect Swarm: delegate to its own run() method
    if hasattr(agent, "flow_order"):
        return await agent.run(
            input,
            messages=messages,
            provider=resolved_provider,
            max_retries=max_retries,
        )

    return await call_runner(
        agent,
        input,
        messages=messages,
        provider=resolved_provider,
        max_retries=max_retries,
        loop_threshold=loop_threshold,
    )


def _sync(
    agent: Any,
    input: str,
    *,
    messages: Sequence[Message] | None = None,
    provider: Any = None,
    max_retries: int = 3,
    loop_threshold: int = 3,
) -> RunResult:
    """Execute an agent synchronously (blocking wrapper).

    Calls ``run()`` via ``asyncio.run()``.  This is a convenience for
    scripts and notebooks where an event loop is not already running.

    Args:
        agent: An ``Agent`` (or ``Swarm``) instance.
        input: User query string.
        messages: Prior conversation history to continue from.
        provider: LLM provider with ``async complete()`` method.
        max_retries: Retry attempts for transient LLM errors.
        loop_threshold: Consecutive identical tool-call patterns
            before raising a loop error.

    Returns:
        ``RunResult`` with the agent's output, message history,
        usage stats, and step count.
    """
    return asyncio.run(
        run(
            agent,
            input,
            messages=messages,
            provider=provider,
            max_retries=max_retries,
            loop_threshold=loop_threshold,
        )
    )


async def _stream(
    agent: Any,
    input: str,
    *,
    messages: Sequence[Message] | None = None,
    provider: Any = None,
    max_steps: int | None = None,
    detailed: bool = False,
    event_types: set[str] | None = None,
) -> AsyncIterator[StreamEvent]:
    """Stream agent execution, yielding events in real-time.

    Uses the provider's ``stream()`` method to deliver text deltas
    as ``TextEvent`` objects and emit ``ToolCallEvent`` for each tool
    invocation. When tool calls are detected, tools are executed and
    the LLM is re-streamed with the results — looping until a
    text-only response or *max_steps* is reached.

    When *detailed* is ``True``, additional event types are emitted:
    ``StatusEvent``, ``StepEvent``, ``UsageEvent``, and
    ``ToolResultEvent``.  ``ErrorEvent`` is emitted on errors
    regardless of the *detailed* flag.

    Args:
        agent: An ``Agent`` instance.
        input: User query string.
        messages: Prior conversation history.
        provider: LLM provider with an ``async stream()`` method.
            When ``None``, auto-resolved from the agent's model string.
        max_steps: Maximum LLM-tool round-trips. Defaults to
            ``agent.max_steps``.
        detailed: When ``True``, emit rich event types (StepEvent,
            UsageEvent, ToolResultEvent, StatusEvent) in addition to
            the default TextEvent and ToolCallEvent.
        event_types: When provided, only events whose ``type`` field
            matches one of the given strings are yielded.  When
            ``None`` (default), all events pass through (respecting
            the *detailed* flag).

    Yields:
        ``TextEvent`` for text chunks and ``ToolCallEvent`` for tool
        invocations.  When *detailed* is ``True``, also yields
        ``StepEvent``, ``UsageEvent``, ``ToolResultEvent``, and
        ``StatusEvent``.  ``ErrorEvent`` is yielded on errors
        regardless of the *detailed* flag.
    """
    resolved = provider or _resolve_provider(agent)

    # Track total events emitted for metrics (only recorded when detailed=True).
    events_emitted: dict[str, int] = {}

    def _passes_filter(event: StreamEvent) -> bool:
        passes = event_types is None or event.type in event_types
        if passes and detailed:
            events_emitted[event.type] = events_emitted.get(event.type, 0) + 1
        return passes

    def _record_stream_metrics() -> None:
        """Record total events emitted during this stream run."""
        if not detailed or not events_emitted:
            return
        for evt_type, count in events_emitted.items():
            attrs: dict[str, str] = {STREAM_EVENT_TYPE: evt_type}
            if HAS_OTEL:
                meter = _get_meter()
                meter.create_counter(
                    name=METRIC_STREAM_EVENTS_EMITTED,
                    unit="1",
                    description="Number of streaming events emitted",
                ).add(count, attrs)
            else:
                _collector.add_counter(METRIC_STREAM_EVENTS_EMITTED, float(count), attrs)

    # Detect Swarm: delegate to its stream() method
    if hasattr(agent, "flow_order"):
        async for event in agent.stream(
            input,
            messages=messages,
            provider=resolved,
            detailed=detailed,
            max_steps=max_steps,
            event_types=event_types,
        ):
            yield event
        return

    if resolved is None:
        from orbiter.agent import AgentError

        raise AgentError(f"Agent '{agent.name}' requires a provider for stream()")

    steps = max_steps if max_steps is not None else agent.max_steps

    # Resolve instructions
    instr: str = ""
    raw_instr = agent.instructions
    if callable(raw_instr):
        instr = str(raw_instr(agent.name))
    elif raw_instr:
        instr = str(raw_instr)

    # Build initial message list
    history: list[Message] = list(messages) if messages else []
    history.append(UserMessage(content=input))
    msg_list = build_messages(instr, history)

    tool_schemas = agent.get_tool_schemas() or None

    model_name = getattr(agent, "model", "") or ""

    if detailed:
        _ev = StatusEvent(
            status="starting",
            agent_name=agent.name,
            message=f"Agent '{agent.name}' starting execution",
        )
        if _passes_filter(_ev):
            yield _ev

    for step_num in range(steps):
        step_started_at = time.time()

        if detailed:
            _ev = StepEvent(
                step_number=step_num + 1,
                agent_name=agent.name,
                status="started",
                started_at=step_started_at,
            )
            if _passes_filter(_ev):
                yield _ev

        try:
            # Accumulate text and tool call deltas from the stream
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            thought_sigs: list[bytes] = []
            # dict of index -> accumulated tool call data
            tc_acc: dict[int, dict[str, str]] = {}
            step_usage = Usage()

            async for chunk in resolved.stream(
                msg_list,
                tools=tool_schemas,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens,
            ):
                # Yield text deltas
                if chunk.delta:
                    text_parts.append(chunk.delta)
                    _ev = TextEvent(text=chunk.delta, agent_name=agent.name)
                    if _passes_filter(_ev):
                        yield _ev

                # Accumulate reasoning/thought data from stream chunks
                if chunk.reasoning_delta:
                    reasoning_parts.append(chunk.reasoning_delta)
                    _ev = ReasoningEvent(text=chunk.reasoning_delta, agent_name=agent.name)
                    if _passes_filter(_ev):
                        yield _ev
                if chunk.thought_signatures:
                    thought_sigs.extend(chunk.thought_signatures)

                # Accumulate tool call deltas
                for tcd in chunk.tool_call_deltas:
                    idx = tcd.index
                    if idx not in tc_acc:
                        tc_acc[idx] = {"id": "", "name": "", "arguments": "", "thought_signature": None}
                    if tcd.id is not None:
                        tc_acc[idx]["id"] = tcd.id
                    if tcd.name is not None:
                        tc_acc[idx]["name"] = tcd.name
                    tc_acc[idx]["arguments"] += tcd.arguments
                    if tcd.thought_signature is not None:
                        tc_acc[idx]["thought_signature"] = tcd.thought_signature

                # Capture usage from final chunk
                if chunk.usage and chunk.usage.total_tokens > 0:
                    step_usage = Usage(
                        input_tokens=chunk.usage.input_tokens,
                        output_tokens=chunk.usage.output_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )

            if detailed:
                _ev = UsageEvent(
                    usage=step_usage,
                    agent_name=agent.name,
                    step_number=step_num + 1,
                    model=model_name,
                )
                if _passes_filter(_ev):
                    yield _ev

            # Build completed tool calls
            tool_calls = [
                ToolCall(
                    id=data["id"],
                    name=data["name"],
                    arguments=data["arguments"],
                    thought_signature=data.get("thought_signature"),
                )
                for data in tc_acc.values()
                if data["id"]
            ]

            # No tool calls — done streaming
            if not tool_calls:
                if detailed:
                    _ev = StepEvent(
                        step_number=step_num + 1,
                        agent_name=agent.name,
                        status="completed",
                        started_at=step_started_at,
                        completed_at=time.time(),
                        usage=step_usage,
                    )
                    if _passes_filter(_ev):
                        yield _ev
                    _ev = StatusEvent(
                        status="completed",
                        agent_name=agent.name,
                        message=f"Agent '{agent.name}' completed execution",
                    )
                    if _passes_filter(_ev):
                        yield _ev
                _record_stream_metrics()
                return

            # Yield ToolCallEvent for each tool call
            for tc in tool_calls:
                _ev = ToolCallEvent(
                    tool_name=tc.name,
                    tool_call_id=tc.id,
                    agent_name=agent.name,
                )
                if _passes_filter(_ev):
                    yield _ev

            # Execute tools and feed results back
            full_text = "".join(text_parts)
            actions = parse_tool_arguments(tool_calls)
            tool_exec_start = time.time()
            tool_results = await agent._execute_tools(actions)
            tool_exec_end = time.time()

            # Emit ToolResultEvent for each tool execution when detailed
            if detailed:
                total_tool_duration_ms = (tool_exec_end - tool_exec_start) * 1000
                per_tool_duration_ms = (
                    total_tool_duration_ms / len(tool_results)
                    if tool_results
                    else 0.0
                )
                for action, tr in zip(actions, tool_results):
                    _ev = ToolResultEvent(
                        tool_name=tr.tool_name,
                        tool_call_id=tr.tool_call_id,
                        arguments=action.arguments,
                        result=tr.content,
                        error=tr.error,
                        success=tr.error is None,
                        duration_ms=per_tool_duration_ms,
                        agent_name=agent.name,
                    )
                    if _passes_filter(_ev):
                        yield _ev

            if detailed:
                _ev = StepEvent(
                    step_number=step_num + 1,
                    agent_name=agent.name,
                    status="completed",
                    started_at=step_started_at,
                    completed_at=time.time(),
                    usage=step_usage,
                )
                if _passes_filter(_ev):
                    yield _ev

            # Append assistant message + tool results to conversation
            msg_list.append(
                AssistantMessage(
                    content=full_text,
                    tool_calls=tool_calls,
                    reasoning_content="".join(reasoning_parts),
                    thought_signatures=thought_sigs,
                )
            )
            msg_list.extend(tool_results)

        except Exception as exc:
            _ev = ErrorEvent(
                error=str(exc),
                error_type=type(exc).__name__,
                agent_name=agent.name,
                step_number=step_num + 1,
                recoverable=False,
            )
            if _passes_filter(_ev):
                yield _ev
            if detailed:
                _ev = StatusEvent(
                    status="error",
                    agent_name=agent.name,
                    message=str(exc),
                )
                if _passes_filter(_ev):
                    yield _ev
            _record_stream_metrics()
            raise


def _resolve_provider(agent: Any) -> Any:
    """Attempt to auto-resolve a provider from the agent's model config.

    Tries the model registry from ``orbiter.models`` if available.
    Returns ``None`` if auto-resolution fails (call_runner will then
    let Agent.run() raise its own error for missing provider).
    """
    try:
        from orbiter.models import get_provider  # pyright: ignore[reportMissingImports]

        return get_provider(agent.model)
    except Exception:
        return None


# Attach sync and stream as attributes of the run function
run.sync = _sync  # type: ignore[attr-defined]
run.stream = _stream  # type: ignore[attr-defined]
