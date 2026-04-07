"""Programmatic Tool Calling (PTC) for Exo agents.

When ``ptc=True`` on an Agent, the LLM writes Python code that calls
tools as async functions inside a single ``__exo_ptc__`` tool call,
instead of requiring separate LLM round-trips per tool.  This reduces
latency and token consumption — especially for batch operations,
filtering, and multi-step workflows.

The PTC tool is fully transparent to the event stream: consumers see
individual ``ToolCallEvent``/``ToolResultEvent`` per inner tool call,
as if PTC were not enabled.

Provider-agnostic: works with any LLM that supports tool calling.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import datetime
import io
import itertools
import json
import math
import re
import textwrap
import time
import traceback
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from exo.tool import FunctionTool, Tool
from exo.types import ToolCallEvent, ToolResultEvent

if TYPE_CHECKING:
    from exo.agent import Agent

# ---------------------------------------------------------------------------
# PTC tool name — intentionally obscure to avoid collision with user tools.
# Imported by agent.py and runner.py.
# ---------------------------------------------------------------------------

PTC_TOOL_NAME = "__exo_ptc__"

# ---------------------------------------------------------------------------
# Internal tool names that should NOT be wrapped by PTC.
# These remain as direct tool schemas sent to the LLM.
# ---------------------------------------------------------------------------

_PTC_EXCLUDED_NAMES: frozenset[str] = frozenset(
    {
        PTC_TOOL_NAME,
        "retrieve_artifact",
        "spawn_self",
        "activate_skill",
    }
)

# JSON Schema type → Python type annotation string
_SCHEMA_TYPE_MAP: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_ptc_eligible_tools(agent: Agent) -> dict[str, Tool]:
    """Return tools that should be available as functions inside PTC code.

    Excludes framework-internal tools, HITL tools, and handoff targets.
    """
    hitl: frozenset[str] = getattr(agent, "hitl_tools", frozenset())
    handoff_names: frozenset[str] = frozenset(getattr(agent, "handoffs", {}).keys())

    return {
        name: tool
        for name, tool in agent.tools.items()
        if name not in _PTC_EXCLUDED_NAMES
        and name not in hitl
        and name not in handoff_names
        and not getattr(tool, "_is_context_tool", False)
        and not getattr(tool, "_is_ptc_tool", False)
    }


def _schema_type(prop: dict[str, Any]) -> str:
    """Convert a JSON Schema property to a Python type hint string."""
    raw = prop.get("type", "Any")
    if isinstance(raw, list):
        # e.g. ["string", "null"]
        types = [_SCHEMA_TYPE_MAP.get(t, t) for t in raw if t != "null"]
        return types[0] if len(types) == 1 else f"{' | '.join(types)}"
    return _SCHEMA_TYPE_MAP.get(raw, "Any")


def schema_to_python_sig(tool: Tool) -> str:
    """Convert a Tool's JSON Schema parameters into a Python function signature.

    Returns a string like ``async def search(query: str, max_results: int = 10) -> str``.
    """
    params = tool.parameters
    properties: dict[str, Any] = params.get("properties", {})
    required: set[str] = set(params.get("required", []))

    parts: list[str] = []
    # Required params first, then optional
    for pname in sorted(properties, key=lambda p: (p not in required, p)):
        ptype = _schema_type(properties[pname])
        if pname in required:
            parts.append(f"{pname}: {ptype}")
        else:
            default = properties[pname].get("default")
            if default is None:
                parts.append(f"{pname}: {ptype} | None = None")
            else:
                parts.append(f"{pname}: {ptype} = {default!r}")

    sig = ", ".join(parts)
    return f"async def {tool.name}({sig}) -> str"


def build_tool_signatures(tools: dict[str, Tool]) -> str:
    """Build a description block listing all PTC-eligible tools as function signatures."""
    if not tools:
        return "(no tools available)"

    lines: list[str] = []
    for tool in tools.values():
        sig = schema_to_python_sig(tool)
        lines.append(f"  {sig}")
        if tool.description:
            lines.append(f'    """{tool.description}"""')
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PTCTool — the synthetic tool sent to the LLM
# ---------------------------------------------------------------------------

_PTC_PREAMBLE = """\
Execute Python code that calls the agent's tools as async functions.

Use `await` to call tool functions. Use `print()` to output results.
The printed output (and/or the return value) will be returned.
Standard library modules are available: json, math, re, asyncio,
collections, itertools, datetime.

Available tool functions:

"""


class PTCTool(Tool):
    """Synthetic ``__exo_ptc__`` tool injected when ``ptc=True``.

    Its description is built dynamically from the agent's current tools
    so that dynamic tool mutations (``add_tool`` / ``remove_tool``) are
    reflected automatically when the schema cache is invalidated.

    The tool is fully transparent to the event stream: individual
    ``ToolCallEvent``/``ToolResultEvent`` are emitted per inner tool
    call via the agent's event queue, and the runner suppresses events
    for the outer PTC tool itself.
    """

    _is_ptc_tool: bool = True

    def __init__(self, agent: Agent, timeout: int = 60) -> None:
        self.name = PTC_TOOL_NAME
        self._agent = agent
        self._timeout = timeout
        self.parameters: dict[str, Any] = {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python code to execute. "
                        "Call tools with `await tool_name(arg=value)`. "
                        "Use `print()` to output results."
                    ),
                },
            },
            "required": ["code"],
        }

    @property
    def description(self) -> str:  # type: ignore[override]
        """Dynamically build description from current PTC-eligible tools."""
        eligible = get_ptc_eligible_tools(self._agent)
        return _PTC_PREAMBLE + build_tool_signatures(eligible)

    @description.setter
    def description(self, _value: str) -> None:
        # Ignored — description is always computed dynamically.
        pass

    def to_schema(self) -> dict[str, Any]:
        """Override to use the dynamic description."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        """Run user code via PTCExecutor."""
        code: str = kwargs.get("code", "")
        executor = PTCExecutor(self._agent, timeout=self._timeout)
        return await executor.run(code)


# ---------------------------------------------------------------------------
# PTCExecutor — runs user code with tool functions in the namespace
# ---------------------------------------------------------------------------

# Stdlib modules available in the PTC namespace
_PTC_STDLIB = {
    "json": json,
    "math": math,
    "re": re,
    "asyncio": asyncio,
    "collections": collections,
    "itertools": itertools,
    "datetime": datetime,
}


class PTCExecutor:
    """Execute user-written Python code with agent tools as async functions.

    Each :meth:`run` call creates a fresh namespace so tool mutations
    between runs are reflected automatically.
    """

    def __init__(self, agent: Agent, timeout: int = 60) -> None:
        self._agent = agent
        self._timeout = timeout

    async def run(self, code: str) -> str:
        """Execute *code* and return captured stdout + return value.

        Returns:
            Combined stdout output and ``repr()`` of the return value (if any).
            On error, returns the error message / traceback.
        """
        namespace = self._build_namespace()

        # Wrap in async function so `await` works at the top level.
        indented = textwrap.indent(textwrap.dedent(code), "    ")
        wrapped = f"async def __ptc_main__():\n{indented}\n"

        try:
            compiled = compile(wrapped, "<ptc>", "exec")
        except SyntaxError as exc:
            return f"SyntaxError: {exc}"

        exec(compiled, namespace)

        stdout_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buf):
                result = await asyncio.wait_for(
                    namespace["__ptc_main__"](),
                    timeout=self._timeout,
                )
        except TimeoutError:
            captured = stdout_buf.getvalue()
            return (captured + f"\nTimeoutError: execution exceeded {self._timeout}s").strip()
        except Exception:
            captured = stdout_buf.getvalue()
            tb = traceback.format_exc()
            return (captured + "\n" + tb).strip()

        captured = stdout_buf.getvalue()
        if result is not None:
            if captured:
                return f"{captured.rstrip()}\n{result!r}"
            return repr(result)
        return captured.rstrip() or "(no output)"

    # -- namespace construction --

    def _build_namespace(self) -> dict[str, Any]:
        """Create the execution namespace with tool wrappers and stdlib."""
        ns: dict[str, Any] = {"__builtins__": __builtins__}
        ns.update(_PTC_STDLIB)

        eligible = get_ptc_eligible_tools(self._agent)
        for tool in eligible.values():
            ns[tool.name] = self._make_tool_fn(tool)

        return ns

    def _make_tool_fn(self, tool_obj: Tool) -> Callable[..., Any]:
        """Create an async wrapper function for a tool.

        The wrapper emits ``ToolCallEvent`` / ``ToolResultEvent`` to the
        agent's event queue so the stream is transparent — consumers see
        individual tool events instead of the opaque PTC tool.
        """
        agent = self._agent

        async def wrapper(**kwargs: Any) -> str:
            tool_call_id = f"ptc_{uuid.uuid4().hex[:8]}"

            # Emit ToolCallEvent BEFORE execution (transparent to stream)
            agent._event_queue.put_nowait(
                ToolCallEvent(
                    tool_name=tool_obj.name,
                    tool_call_id=tool_call_id,
                    arguments=json.dumps(kwargs),
                    agent_name=agent.name,
                )
            )

            # Fire PRE_TOOL_CALL hook
            await agent.hook_manager.run(
                _hook_point("PRE_TOOL_CALL"),
                agent=agent,
                tool_name=tool_obj.name,
                arguments=kwargs,
            )

            # Inject ToolContext if the tool declares one
            if isinstance(tool_obj, FunctionTool) and tool_obj._tool_context_param:
                from exo.tool_context import ToolContext

                kwargs[tool_obj._tool_context_param] = ToolContext(
                    agent_name=agent.name,
                    queue=agent._event_queue,
                )

            start = time.time()
            try:
                raw = await tool_obj.execute(**kwargs)
            except Exception as exc:
                duration_ms = (time.time() - start) * 1000

                # Emit error ToolResultEvent (transparent to stream)
                agent._event_queue.put_nowait(
                    ToolResultEvent(
                        tool_name=tool_obj.name,
                        tool_call_id=tool_call_id,
                        arguments=kwargs,
                        result="",
                        error=str(exc),
                        success=False,
                        duration_ms=duration_ms,
                        agent_name=agent.name,
                    )
                )

                # Fire POST_TOOL_CALL with error result
                from exo.types import ToolResult

                err_result = ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name=tool_obj.name,
                    error=str(exc),
                )
                await agent.hook_manager.run(
                    _hook_point("POST_TOOL_CALL"),
                    agent=agent,
                    tool_name=tool_obj.name,
                    result=err_result,
                )
                raise

            duration_ms = (time.time() - start) * 1000

            # Normalise to string
            if isinstance(raw, str):
                output = raw
            elif isinstance(raw, (dict, list)):
                output = json.dumps(raw)
            else:
                output = str(raw)

            # Emit success ToolResultEvent (transparent to stream)
            agent._event_queue.put_nowait(
                ToolResultEvent(
                    tool_name=tool_obj.name,
                    tool_call_id=tool_call_id,
                    arguments=kwargs,
                    result=output,
                    error=None,
                    success=True,
                    duration_ms=duration_ms,
                    agent_name=agent.name,
                )
            )

            # Fire POST_TOOL_CALL hook
            from exo.types import ToolResult

            result_obj = ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_obj.name,
                content=output,
            )
            await agent.hook_manager.run(
                _hook_point("POST_TOOL_CALL"),
                agent=agent,
                tool_name=tool_obj.name,
                result=result_obj,
            )

            return output

        wrapper.__name__ = tool_obj.name
        wrapper.__doc__ = tool_obj.description
        return wrapper


def _hook_point(name: str) -> Any:
    """Resolve a HookPoint by name (deferred import to avoid circular deps)."""
    from exo.hooks import HookPoint

    return HookPoint[name]
