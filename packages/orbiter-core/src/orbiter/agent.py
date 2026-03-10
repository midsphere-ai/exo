"""Agent class: the core autonomous unit in Orbiter."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel

from orbiter._internal.message_builder import build_messages
from orbiter._internal.output_parser import parse_response, parse_tool_arguments
from orbiter.config import parse_model_string
from orbiter.hooks import Hook, HookManager, HookPoint
from orbiter.observability.logging import get_logger  # pyright: ignore[reportMissingImports]
from orbiter.guardrail.types import GuardrailError  # pyright: ignore[reportMissingImports]
from orbiter.rail import Rail, RailAbortError, RailManager
from orbiter.tool import Tool, ToolError
from orbiter.types import (
    AgentOutput,
    AssistantMessage,
    Message,
    OrbiterError,
    ToolResult,
    UserMessage,
)

_log = get_logger(__name__)


class AgentError(OrbiterError):
    """Raised for agent-level errors (duplicate tools, invalid config, etc.)."""


class Agent:
    """An autonomous LLM-powered agent with tools and lifecycle hooks.

    Agents are the core building block in Orbiter. Each agent wraps an LLM
    model, a set of tools, optional handoff targets, and lifecycle hooks.
    The ``run()`` method (added in a later session) executes the agent's
    tool loop.

    All parameters are keyword-only; only ``name`` is required.

    Args:
        name: Unique identifier for this agent.
        model: Model string in ``"provider:model_name"`` format.
        instructions: System prompt. Can be a string or an async callable
            that receives a context dict and returns a string.
        tools: Tools available to this agent.
        handoffs: Other agents this agent can delegate to via handoff.
        hooks: Lifecycle hooks as ``(HookPoint, Hook)`` tuples.
        rails: Optional list of :class:`~orbiter.rail.Rail` instances.
            If provided, a :class:`~orbiter.rail.RailManager` is created
            and registered as hooks on the agent's hook_manager.
        output_type: Pydantic model class for structured output validation.
        max_steps: Maximum LLM-tool round-trips before stopping.
        temperature: LLM sampling temperature.
        max_tokens: Maximum output tokens per LLM call.
        memory: Optional memory store for persistent memory across sessions.
        context: Optional context engine for hierarchical state and prompt building.
    """

    def __init__(
        self,
        *,
        name: str,
        model: str = "openai:gpt-4o",
        instructions: str | Callable[..., str] = "",
        tools: list[Tool] | None = None,
        handoffs: list[Agent] | None = None,
        hooks: list[tuple[HookPoint, Hook]] | None = None,
        rails: list[Rail] | None = None,
        output_type: type[BaseModel] | None = None,
        max_steps: int = 10,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        memory: Any = None,
        context: Any = None,
    ) -> None:
        self.name = name
        self.model = model
        self.provider_name, self.model_name = parse_model_string(model)
        self.instructions = instructions
        self.output_type = output_type
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory = memory
        self.context = context

        # Tools indexed by name for O(1) lookup during execution
        self.tools: dict[str, Tool] = {}
        if tools:
            for t in tools:
                self._register_tool(t)

        # Handoff targets indexed by name
        self.handoffs: dict[str, Agent] = {}
        if handoffs:
            for agent in handoffs:
                self._register_handoff(agent)

        # Lifecycle hooks
        self.hook_manager = HookManager()
        if hooks:
            for point, hook in hooks:
                self.hook_manager.add(point, hook)

        # Rails — structured lifecycle guards registered as hooks
        self.rail_manager: RailManager | None = None
        if rails:
            self.rail_manager = RailManager()
            for rail in rails:
                self.rail_manager.add(rail)
            # Register rail hooks on every hook point so rails fire
            # alongside (before or after) traditional hooks.
            for point in HookPoint:
                self.hook_manager.add(point, self.rail_manager.hook_for(point))

    def _register_tool(self, t: Tool) -> None:
        """Add a tool, raising on duplicate names.

        Args:
            t: The tool to register.

        Raises:
            AgentError: If a tool with the same name is already registered.
        """
        if t.name in self.tools:
            raise AgentError(f"Duplicate tool name '{t.name}' on agent '{self.name}'")
        self.tools[t.name] = t

    def _register_handoff(self, agent: Agent) -> None:
        """Add a handoff target, raising on duplicate names.

        Args:
            agent: The target agent.

        Raises:
            AgentError: If a handoff with the same name is already registered.
        """
        if agent.name in self.handoffs:
            raise AgentError(f"Duplicate handoff agent '{agent.name}' on agent '{self.name}'")
        self.handoffs[agent.name] = agent

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-format tool schemas for all registered tools.

        Returns:
            A list of tool schema dicts suitable for LLM function calling.
        """
        return [t.to_schema() for t in self.tools.values()]

    def describe(self) -> dict[str, Any]:
        """Return a summary of the agent's capabilities.

        Useful for debugging, logging, and capability advertisement
        in multi-agent systems.

        Returns:
            A dict with the agent's name, model, tools, and configuration.
        """
        return {
            "name": self.name,
            "model": self.model,
            "tools": list(self.tools.keys()),
            "handoffs": list(self.handoffs.keys()),
            "max_steps": self.max_steps,
            "output_type": (self.output_type.__name__ if self.output_type else None),
        }

    async def run(
        self,
        input: str,
        *,
        messages: Sequence[Message] | None = None,
        provider: Any = None,
        max_retries: int = 3,
    ) -> AgentOutput:
        """Execute the agent's LLM-tool loop with retry logic.

        Builds the message list, calls the LLM, and if tool calls are
        returned, executes them in parallel, feeds results back, and
        re-calls the LLM. The loop continues until a text-only response
        is produced or ``max_steps`` is reached.

        Args:
            input: User query string for this turn.
            messages: Prior conversation history.
            provider: An object with an ``async complete()`` method
                (e.g. a ``ModelProvider`` instance).
            max_retries: Maximum retry attempts for transient errors.

        Returns:
            Parsed ``AgentOutput`` from the final LLM response.

        Raises:
            AgentError: If no provider is supplied or all retries are exhausted.
        """
        if provider is None:
            raise AgentError(f"Agent '{self.name}' requires a provider for run()")

        # Resolve instructions
        instructions = self.instructions
        if callable(instructions):
            instructions = instructions(self.name)

        # Build initial message list
        history: list[Message] = list(messages) if messages else []
        history.append(UserMessage(content=input))
        msg_list = build_messages(instructions, history)

        # Tool schemas
        tool_schemas = self.get_tool_schemas() or None

        # Tool loop — iterate up to max_steps
        for _step in range(self.max_steps):
            output = await self._call_llm(msg_list, tool_schemas, provider, max_retries)

            # No tool calls — return the final text response
            if not output.tool_calls:
                return output

            # Execute tool calls and collect results
            actions = parse_tool_arguments(output.tool_calls)
            tool_results = await self._execute_tools(actions)

            # Append assistant message (with tool calls) and results to history
            msg_list.append(AssistantMessage(content=output.text, tool_calls=output.tool_calls))
            msg_list.extend(tool_results)

        # max_steps exhausted — return last output as-is
        return output

    async def _call_llm(
        self,
        msg_list: list[Message],
        tool_schemas: list[dict[str, Any]] | None,
        provider: Any,
        max_retries: int,
    ) -> AgentOutput:
        """Single LLM call with retry logic and lifecycle hooks."""
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                await self.hook_manager.run(HookPoint.PRE_LLM_CALL, agent=self, messages=msg_list)

                response = await provider.complete(
                    msg_list,
                    tools=tool_schemas,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                await self.hook_manager.run(HookPoint.POST_LLM_CALL, agent=self, response=response)

                return parse_response(
                    content=response.content,
                    tool_calls=response.tool_calls,
                    usage=response.usage,
                )

            except (RailAbortError, GuardrailError):
                raise

            except Exception as exc:
                if _is_context_length_error(exc):
                    _log.error("Context length exceeded on '%s'", self.name)
                    raise AgentError(
                        f"Context length exceeded on agent '{self.name}': {exc}"
                    ) from exc

                last_error = exc
                if attempt < max_retries - 1:
                    _log.warning(
                        "Retry %d/%d for '%s': %s", attempt + 1, max_retries, self.name, exc
                    )
                    delay = 2**attempt
                    await asyncio.sleep(delay)

        _log.error("Agent '%s' failed after %d retries", self.name, max_retries)
        raise AgentError(
            f"Agent '{self.name}' failed after {max_retries} retries: {last_error}"
        ) from last_error

    async def _execute_tools(
        self,
        actions: list[Any],
    ) -> list[ToolResult]:
        """Execute tool calls in parallel, catching errors per-tool."""
        results: list[ToolResult] = [ToolResult(tool_call_id="", tool_name="")] * len(actions)

        async def _run_one(idx: int) -> None:
            action = actions[idx]
            tool = self.tools.get(action.tool_name)

            # PRE_TOOL_CALL hook
            await self.hook_manager.run(
                HookPoint.PRE_TOOL_CALL,
                agent=self,
                tool_name=action.tool_name,
                arguments=action.arguments,
            )

            if tool is None:
                result = ToolResult(
                    tool_call_id=action.tool_call_id,
                    tool_name=action.tool_name,
                    error=f"Unknown tool '{action.tool_name}'",
                )
            else:
                try:
                    output = await tool.execute(**action.arguments)
                    content = output if isinstance(output, str) else str(output)
                    result = ToolResult(
                        tool_call_id=action.tool_call_id,
                        tool_name=action.tool_name,
                        content=content,
                    )
                except (ToolError, Exception) as exc:
                    _log.warning("Tool '%s' failed on '%s': %s", action.tool_name, self.name, exc)
                    result = ToolResult(
                        tool_call_id=action.tool_call_id,
                        tool_name=action.tool_name,
                        error=str(exc),
                    )

            # POST_TOOL_CALL hook
            await self.hook_manager.run(
                HookPoint.POST_TOOL_CALL,
                agent=self,
                tool_name=action.tool_name,
                result=result,
            )

            results[idx] = result

        async with asyncio.TaskGroup() as tg:
            for i in range(len(actions)):
                tg.create_task(_run_one(i))

        return results

    def to_dict(self) -> dict[str, Any]:
        """Serialize the agent configuration to a dict.

        Tools are serialized as importable dotted paths. Callable instructions,
        hooks, memory, and context cannot be serialized and will raise ValueError.

        Returns:
            A dict suitable for JSON serialization and later reconstruction
            via ``Agent.from_dict()``.

        Raises:
            ValueError: If the agent contains non-serializable components
                (callable instructions, hooks, closure-based tools, memory, context).
        """
        if callable(self.instructions):
            raise ValueError(
                f"Agent '{self.name}' has callable instructions which cannot be serialized. "
                "Use a string instruction instead."
            )
        if self.hook_manager.has_hooks(HookPoint.START) or any(
            self.hook_manager.has_hooks(hp) for hp in HookPoint
        ):
            raise ValueError(
                f"Agent '{self.name}' has hooks which cannot be serialized."
            )
        if self.memory is not None:
            raise ValueError(
                f"Agent '{self.name}' has a memory store which cannot be serialized."
            )
        if self.context is not None:
            raise ValueError(
                f"Agent '{self.name}' has a context engine which cannot be serialized."
            )

        data: dict[str, Any] = {
            "name": self.name,
            "model": self.model,
            "instructions": self.instructions,
            "max_steps": self.max_steps,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Serialize tools as importable dotted paths
        if self.tools:
            data["tools"] = [_serialize_tool(t) for t in self.tools.values()]

        # Serialize handoffs recursively
        if self.handoffs:
            data["handoffs"] = [agent.to_dict() for agent in self.handoffs.values()]

        # Serialize output_type as importable dotted path
        if self.output_type is not None:
            data["output_type"] = f"{self.output_type.__module__}.{self.output_type.__qualname__}"

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Agent:
        """Reconstruct an Agent from a dict produced by ``to_dict()``.

        Tools are resolved by importing dotted paths. Handoff agents are
        reconstructed recursively.

        Args:
            data: Dict as produced by ``Agent.to_dict()``.

        Returns:
            A reconstructed ``Agent`` instance.

        Raises:
            ValueError: If a tool or output_type path cannot be imported.
        """
        tools: list[Tool] | None = None
        if "tools" in data:
            tools = [_deserialize_tool(t) for t in data["tools"]]

        handoffs: list[Agent] | None = None
        if "handoffs" in data:
            handoffs = [Agent.from_dict(h) for h in data["handoffs"]]

        output_type: type[BaseModel] | None = None
        if "output_type" in data:
            output_type = _import_object(data["output_type"])

        return cls(
            name=data["name"],
            model=data.get("model", "openai:gpt-4o"),
            instructions=data.get("instructions", ""),
            tools=tools,
            handoffs=handoffs,
            output_type=output_type,
            max_steps=data.get("max_steps", 10),
            temperature=data.get("temperature", 1.0),
            max_tokens=data.get("max_tokens"),
        )

    def __repr__(self) -> str:
        parts = [f"name={self.name!r}", f"model={self.model!r}"]
        if self.tools:
            parts.append(f"tools={list(self.tools.keys())}")
        if self.handoffs:
            parts.append(f"handoffs={list(self.handoffs.keys())}")
        return f"Agent({', '.join(parts)})"


def _is_context_length_error(exc: Exception) -> bool:
    """Check if an exception represents a context-length overflow.

    Detects errors with a ``code`` attribute of ``"context_length"``
    (set by ``ModelError``) or common provider error messages.
    """
    code = getattr(exc, "code", "")
    if code == "context_length":
        return True
    msg = str(exc).lower()
    return "context_length" in msg or "context length" in msg


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_tool(t: Tool) -> str:
    """Serialize a tool to an importable dotted path.

    For ``FunctionTool``, uses the wrapped function's module and qualname.
    For custom ``Tool`` subclasses, uses the class's module and qualname.

    Raises:
        ValueError: If the tool cannot be serialized (e.g., closures, lambdas).
    """
    from orbiter.tool import FunctionTool

    if isinstance(t, FunctionTool):
        fn = t._fn
        module = getattr(fn, "__module__", None)
        qualname = getattr(fn, "__qualname__", None)
        if not module or not qualname:
            raise ValueError(
                f"Tool '{t.name}' wraps a function without __module__ or __qualname__ "
                "and cannot be serialized."
            )
        # Detect closures/lambdas (qualname contains '<')
        if "<" in qualname:
            raise ValueError(
                f"Tool '{t.name}' wraps a closure or lambda ({qualname}) "
                "which cannot be serialized. Use a module-level function instead."
            )
        return f"{module}.{qualname}"

    # Custom Tool subclass — serialize the class itself
    cls = type(t)
    module = cls.__module__
    qualname = cls.__qualname__
    if "<" in qualname:
        raise ValueError(
            f"Tool '{t.name}' is a locally-defined class ({qualname}) "
            "which cannot be serialized."
        )
    return f"{module}.{qualname}"


def _deserialize_tool(path: str) -> Tool:
    """Deserialize a tool from an importable dotted path.

    If the imported object is a callable (function), wraps it as a FunctionTool.
    If it's already a Tool instance, returns it directly.
    If it's a Tool subclass, instantiates it.

    Raises:
        ValueError: If the path cannot be imported or doesn't resolve to a tool.
    """
    from orbiter.tool import FunctionTool

    obj = _import_object(path)

    # Already a Tool instance (e.g., @tool decorated at module level)
    if isinstance(obj, Tool):
        return obj

    # A Tool subclass — instantiate it
    if isinstance(obj, type) and issubclass(obj, Tool):
        return obj()

    # A plain callable — wrap it
    if callable(obj):
        return FunctionTool(obj)

    raise ValueError(
        f"Imported '{path}' is not a callable or Tool instance: {type(obj)}"
    )


def _import_object(dotted_path: str) -> Any:
    """Import an object from a dotted path like 'package.module.ClassName'.

    Tries progressively shorter module paths, resolving the remainder
    via getattr.

    Raises:
        ValueError: If the path cannot be resolved.
    """
    parts = dotted_path.rsplit(".", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid dotted path: {dotted_path!r}")

    module_path, attr_name = parts
    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    except (ImportError, AttributeError):
        pass

    # Try splitting further for nested attributes (e.g., module.Class.method)
    parts = dotted_path.split(".")
    for i in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:i])
        try:
            obj = importlib.import_module(module_path)
            for attr in parts[i:]:
                obj = getattr(obj, attr)
            return obj
        except (ImportError, AttributeError):
            continue

    raise ValueError(f"Cannot import '{dotted_path}'")
