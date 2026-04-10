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

import ast
import asyncio
import collections
import contextlib
import datetime
import io
import itertools
import json
import logging
import math
import re
import textwrap
import time
import traceback
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from RestrictedPython.Guards import safer_getattr

from exo.hooks import HookPoint
from exo.tool import Tool
from exo.types import ToolCallEvent, ToolResultEvent

if TYPE_CHECKING:
    from exo.agent import Agent

_ptc_log = logging.getLogger(__name__)

# Resolve hook points once at import — avoids repeated lookups per tool call.
_PRE_TOOL_CALL = HookPoint.PRE_TOOL_CALL
_POST_TOOL_CALL = HookPoint.POST_TOOL_CALL

# ---------------------------------------------------------------------------
# PTC tool name — intentionally obscure to avoid collision with user tools.
# Imported by agent.py and runner.py.
# ---------------------------------------------------------------------------

PTC_TOOL_NAME = "__exo_ptc__"

# Default limits — overridable via Agent parameters.
DEFAULT_PTC_TIMEOUT = 60
DEFAULT_PTC_MAX_OUTPUT_BYTES = 200_000
DEFAULT_PTC_MAX_TOOL_CALLS = 200
DEFAULT_PTC_MAX_CODE_BYTES = 100_000

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

# Track which invalid tool names we've already warned about, to avoid log spam.
_excluded_name_warnings: set[str] = set()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MaxToolCallsExceeded(RuntimeError):  # noqa: N818  -- intentional name
    """Raised when PTC code exceeds the configured ``max_tool_calls``."""


class _PTCBaseExceptionTrap(Exception):  # noqa: N818  -- internal trap funnel
    """Internal: funnels ``SystemExit`` / ``KeyboardInterrupt`` from user code.

    ``asyncio.Task`` re-raises ``SystemExit`` and ``KeyboardInterrupt``
    directly into the event loop, bypassing any ``except`` on ``wait_for``.
    We wrap user code in an inner ``try`` that converts those to this
    regular ``Exception`` subclass so our outer handler can catch and
    format them normally.
    """

    def __init__(self, kind: str, detail: str = "") -> None:
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}" if detail else kind)


class PTCSandboxError(RuntimeError):
    """Raised when user code hits a PTC sandbox restriction.

    This is a regular ``Exception`` subclass so user code can
    ``try/except`` and continue, and the agent sees a clean error
    message in the tool result pointing at ``default_api``.
    """


# ---------------------------------------------------------------------------
# Sandbox: restricted builtins, blocked imports, blocked attributes
# ---------------------------------------------------------------------------
#
# PTC code runs in a locked-down namespace so agents cannot lazily escape
# via ``import os`` / ``open(...)`` / ``eval(...)`` / ``().__class__.__bases__``.
# Defense layers:
#
#   1. ``__builtins__`` is a curated whitelist — dangerous names like
#      ``open``/``eval``/``exec``/``compile``/``__import__``/``globals``/
#      ``locals``/``vars``/``dir``/``breakpoint`` are replaced with a stub
#      that raises ``PTCSandboxError`` pointing at ``default_api``.
#   2. ``__import__`` is replaced with ``_ptc_import`` which rejects any
#      module outside the pre-loaded stdlib allowlist with a clear error.
#   3. ``getattr`` is replaced with ``RestrictedPython.Guards.safer_getattr``
#      which blocks any attribute whose name starts with ``_`` — this stops
#      dynamic dunder lookups like ``getattr(obj, "__class__")``.
#   4. A static AST pre-scan walks the user code before compile and rejects
#      attribute access on ``__class__``/``__bases__``/``__subclasses__``/
#      ``__globals__`` etc., direct references to blocked builtins, and
#      ``import`` / ``from`` statements naming blocked modules.
#
# This closes the lazy escape paths. A determined attacker with deep Python
# introspection can still find edge-case bypasses — for those, route PTC
# through a subprocess sandbox (future ``exo-sandbox`` backend work).
# ---------------------------------------------------------------------------

# Curated safe builtins — anything NOT in this dict is unavailable in PTC code.
_PTC_SAFE_BUILTINS: dict[str, Any] = {
    # Constants
    "None": None,
    "True": True,
    "False": False,
    "Ellipsis": Ellipsis,
    "NotImplemented": NotImplemented,
    # Type constructors
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "str": str,
    "bytes": bytes,
    "bytearray": bytearray,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "frozenset": frozenset,
    # Numeric ops
    "abs": abs,
    "round": round,
    "divmod": divmod,
    "pow": pow,
    "min": min,
    "max": max,
    "sum": sum,
    # Sequence / iteration
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "sorted": sorted,
    "reversed": reversed,
    "map": map,
    "filter": filter,
    "iter": iter,
    "next": next,
    "slice": slice,
    # Logic
    "any": any,
    "all": all,
    # String / format
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "oct": oct,
    "bin": bin,
    "ascii": ascii,
    "repr": repr,
    "format": format,
    # Introspection (hasattr also goes through our guarded getattr)
    "isinstance": isinstance,
    "issubclass": issubclass,
    "callable": callable,
    "type": type,
    "object": object,
    # NOTE: ``getattr`` / ``hasattr`` are bound to safer_getattr-based wrappers
    # at namespace-build time so runtime dunder lookup is also blocked.
    # Safe exceptions (so user code can try/except).
    # SystemExit and KeyboardInterrupt are included because the internal
    # wrapper catches them — they are funnelled through ``_PTCBaseExceptionTrap``
    # so user code ``raise SystemExit(...)`` becomes a returned error string,
    # not a process exit.
    "Exception": Exception,
    "BaseException": BaseException,
    "ArithmeticError": ArithmeticError,
    "AttributeError": AttributeError,
    "ImportError": ImportError,
    "IndexError": IndexError,
    "KeyError": KeyError,
    "KeyboardInterrupt": KeyboardInterrupt,
    "LookupError": LookupError,
    "NameError": NameError,
    "NotImplementedError": NotImplementedError,
    "OverflowError": OverflowError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "StopAsyncIteration": StopAsyncIteration,
    "SystemExit": SystemExit,
    "TimeoutError": TimeoutError,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError,
    # Output (captured via redirect_stdout — no special print collector)
    "print": print,
    # Helpers used by format-strings
    "id": id,
}

# Explicitly blocked builtins — replaced by a stub that raises PTCSandboxError
# with a helpful message pointing at default_api.
_PTC_BLOCKED_BUILTINS: frozenset[str] = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "globals",
        "locals",
        "vars",
        "dir",
        "breakpoint",
        "input",
        "exit",
        "quit",
        "help",
        "memoryview",
        "__build_class__",
    }
)

# Expanded stdlib blocklist — filesystem, process, network, concurrency,
# introspection, codegen. Imports of any name whose top package is in here
# raise ImportError with a pointer to default_api.
_PTC_BLOCKED_IMPORTS: frozenset[str] = frozenset(
    {
        # Filesystem / process
        "os",
        "sys",
        "subprocess",
        "shutil",
        "pathlib",
        "tempfile",
        "glob",
        "fnmatch",
        "fcntl",
        "resource",
        # I/O bypass
        "io",
        "builtins",
        "ctypes",
        "mmap",
        # Network
        "socket",
        "urllib",
        "http",
        "httplib",
        "ssl",
        "ftplib",
        "smtplib",
        "poplib",
        "imaplib",
        "telnetlib",
        "nntplib",
        # Code / import / exec
        "importlib",
        "pkgutil",
        "runpy",
        "code",
        "codeop",
        # Introspection / gc
        "inspect",
        "gc",
        "traceback",
        # Concurrency (asyncio is pre-imported)
        "threading",
        "multiprocessing",
        "concurrent",
        # System-level
        "signal",
        "pty",
        "tty",
        "select",
        "termios",
        "pwd",
        "grp",
    }
)

# Dunder attributes that expose the object graph / internals. Access via
# direct attribute syntax (``x.__class__``) is caught statically by the AST
# pre-scan; dynamic access via ``getattr(x, "__class__")`` is caught at
# runtime by ``safer_getattr`` which blocks all underscore-prefixed names.
_PTC_BLOCKED_ATTRS: frozenset[str] = frozenset(
    {
        "__class__",
        "__bases__",
        "__base__",
        "__subclasses__",
        "__mro__",
        "__globals__",
        "__builtins__",
        "__import__",
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "__dict__",
        "__code__",
        "__closure__",
        "__func__",
        "__self__",
        "__loader__",
        "__spec__",
        "__init_subclass__",
        "__new__",
        "__reduce__",
        "__reduce_ex__",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_ptc_eligible_tools(agent: Agent) -> dict[str, Tool]:
    """Return tools that should be available as functions inside PTC code.

    Excludes framework-internal tools, HITL tools, handoff targets, tools
    with ``_ptc_exclude=True``, and tools whose name is not a valid Python
    identifier (those stay as direct tool schemas so the LLM can still call
    them).
    """
    hitl: frozenset[str] = getattr(agent, "hitl_tools", frozenset())
    handoff_names: frozenset[str] = frozenset(getattr(agent, "handoffs", {}).keys())

    eligible: dict[str, Tool] = {}
    for name, tool in agent.tools.items():
        if name in _PTC_EXCLUDED_NAMES:
            continue
        if name in hitl:
            continue
        if name in handoff_names:
            continue
        if getattr(tool, "_is_context_tool", False):
            continue
        if getattr(tool, "_is_ptc_tool", False):
            continue
        if getattr(tool, "_ptc_exclude", False):
            continue
        if not name.isidentifier():
            if name not in _excluded_name_warnings:
                _ptc_log.warning(
                    "Tool %r excluded from PTC: name is not a valid Python "
                    "identifier. It remains available as a direct tool schema.",
                    name,
                )
                _excluded_name_warnings.add(name)
            continue
        eligible[name] = tool
    return eligible


def _ast_scan_user_code(code: str) -> list[str]:
    """Walk the user code AST and return sandbox violations.

    Catches attribute access on any name in ``_PTC_BLOCKED_ATTRS`` —
    direct forms like ``x.__class__``, ``obj.__bases__``, ``f.__globals__``
    that plain ``compile()`` would otherwise emit as a bare ``LOAD_ATTR``
    bytecode (bypassing our runtime ``safer_getattr`` guard).

    Imports of blocked stdlib modules and calls to blocked builtins
    (``open``, ``eval``, ``exec``, …) are NOT caught here — they are
    handled at runtime by the ``_ptc_import`` hook and the blocked-builtin
    stubs respectively.  This keeps those errors *catchable* by user code
    (the agent can ``try/except`` and fall back to ``default_api``), while
    dunder attribute escapes remain hard-blocked because they are a pure
    escape attempt with no legitimate use case in PTC.

    Line numbers in the returned errors match the agent's own view of the
    code (before the async-def wrapper is added).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Let the compile step emit the SyntaxError with our own line-offset.
        return []

    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _PTC_BLOCKED_ATTRS:
            errors.append(
                f"line {node.lineno}: access to `.{node.attr}` is blocked in "
                f"PTC (dunder escape hatch). Use `default_api.*` tools instead."
            )
    return errors


def _schema_type(prop: dict[str, Any]) -> str:
    """Convert a JSON Schema property to a Python type hint string."""
    raw = prop.get("type", "Any")
    if isinstance(raw, list):
        # e.g. ["string", "null"]
        types = [_SCHEMA_TYPE_MAP.get(t, t) for t in raw if t != "null"]
        base = types[0] if len(types) == 1 else " | ".join(types)
    else:
        base = _SCHEMA_TYPE_MAP.get(raw, "Any")

    # For array types, include item type when known (e.g., "list[str]").
    if base == "list":
        items = prop.get("items")
        if isinstance(items, dict):
            item_raw = items.get("type", "")
            item_type = _SCHEMA_TYPE_MAP.get(item_raw)
            if item_type:
                return f"list[{item_type}]"
    return base


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
            # "Any" already implies optional, so skip the "| None" suffix.
            if ptype == "Any":
                type_hint = "Any"
            elif default is None:
                type_hint = f"{ptype} | None"
            else:
                type_hint = ptype
            if default is None:
                parts.append(f"{pname}: {type_hint} = None")
            else:
                parts.append(f"{pname}: {type_hint} = {default!r}")

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
        lines.append(f"    # usage: await default_api.{tool.name}(...)")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PTCTool — the synthetic tool sent to the LLM
# ---------------------------------------------------------------------------

_PTC_PREAMBLE = """\
Execute Python code that calls the agent's tools as async functions in a single round-trip.

<instructions>
- Write MINIMAL Python code — only what is required to orchestrate tool calls and return a result.
- ONLY write PTC-related code: tool calls via `default_api`, simple filtering/aggregation of tool results, and `print()` for output. DO NOT write general-purpose Python programs, define unused classes/functions, or add helper abstractions.
- Prefer the shortest correct form. If a task fits in one or two lines, do not write ten.
- Call tools via the `default_api` namespace: `result = await default_api.tool_name(arg=value)`.
- ALL tool arguments MUST be passed as keyword arguments (e.g., `query="foo"`, not positional).
- Use `print()` to output results — captured stdout is returned to the model.
- Tool results are strings. For structured data, the tool returns JSON — parse with `json.loads(result)`.
- Standard library is pre-imported: `json`, `math`, `re`, `asyncio`, `collections`, `itertools`, `datetime`. DO NOT re-import them.
- Use `asyncio.gather(default_api.a(...), default_api.b(...))` for independent parallel tool calls.
- Tool errors are catchable with `try/except Exception`; execution continues after the except block.
- Each PTC invocation runs in a FRESH namespace — DO NOT assume state persists between calls.
- The code runs in-process with a hard timeout; long operations are terminated with TimeoutError.
- DO NOT use blocking sync code (`time.sleep`, CPU-bound loops) — use `await asyncio.sleep(...)` if you must pause.
- DO NOT attempt direct file I/O, subprocess, or network calls — use the provided tools instead.
- DO NOT add top-level `import` statements for the pre-loaded stdlib modules.
- DO NOT define top-level functions or classes unless strictly necessary for the orchestration.
</instructions>

<recommended_usage>
- Use PTC when calling MULTIPLE tools in one step (batching, loops over items, parallel fetches).
- Use PTC to FILTER or AGGREGATE large tool results so only a summary returns to the model.
- Use `asyncio.gather` for parallel independent tool calls to cut latency.
- Emit results with a SINGLE `print(json.dumps(summary))` at the end for clean structured output.
- Prefer PTC for numeric computation, data transformation, and multi-step workflows where each step builds on the previous.
- AVOID PTC for a SINGLE tool call — a direct tool call has less overhead.
- AVOID PTC when the model must inspect intermediate tool output before deciding the next step (clarification flows, dynamic plans).
- For error-tolerant batch operations, wrap each call in try/except and collect successes/failures explicitly.
</recommended_usage>

Available tool functions (call via `default_api`):

"""


_CODE_PARAM_DESCRIPTION = (
    "Python source code to execute inside an async function (top-level `await` works). "
    "Write MINIMAL code — only tool orchestration via `default_api`. "
    "See the tool description for the full usage rules.\n\n"
    "EXAMPLES:\n\n"
    "1. Batch + filter:\n"
    "    items = json.loads(await default_api.search(query='x', limit=50))\n"
    "    print(json.dumps([i for i in items if i['score'] > 0.8][:5]))\n\n"
    "2. Parallel:\n"
    "    us, eu = await asyncio.gather(\n"
    "        default_api.fetch_stats(region='us'),\n"
    "        default_api.fetch_stats(region='eu'),\n"
    "    )\n"
    "    print(f'us={us} eu={eu}')\n\n"
    "3. Error-tolerant batch:\n"
    "    results = {}\n"
    "    for item_id in [1, 2, 3]:\n"
    "        try:\n"
    "            results[item_id] = await default_api.fetch(id=item_id)\n"
    "        except Exception as e:\n"
    "            results[item_id] = f'error: {e}'\n"
    "    print(json.dumps(results))"
)


class PTCTool(Tool):
    """Synthetic ``__exo_ptc__`` tool injected when ``ptc=True``.

    Its description is built dynamically from the agent's current tools
    so that dynamic tool mutations (``add_tool`` / ``remove_tool``) are
    reflected automatically.  The computed description is cached and
    invalidated when the set of PTC-eligible tools changes.

    The tool is fully transparent to the event stream: individual
    ``ToolCallEvent``/``ToolResultEvent`` are emitted per inner tool
    call via the agent's event queue, and the runner suppresses events
    for the outer PTC tool itself.
    """

    _is_ptc_tool: bool = True

    def __init__(
        self,
        agent: Agent,
        timeout: int = DEFAULT_PTC_TIMEOUT,
        max_output_bytes: int = DEFAULT_PTC_MAX_OUTPUT_BYTES,
        max_tool_calls: int = DEFAULT_PTC_MAX_TOOL_CALLS,
        max_code_bytes: int = DEFAULT_PTC_MAX_CODE_BYTES,
    ) -> None:
        self.name = PTC_TOOL_NAME
        self._agent = agent
        self._timeout = max(1, int(timeout))
        self._max_output_bytes = max(256, int(max_output_bytes))
        self._max_tool_calls = max(1, int(max_tool_calls))
        self._max_code_bytes = max(1024, int(max_code_bytes))
        self._desc_cache: str | None = None
        self._desc_cache_key: tuple[tuple[str, str], ...] | None = None
        self.parameters: dict[str, Any] = {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": _CODE_PARAM_DESCRIPTION,
                },
            },
            "required": ["code"],
        }

    @property
    def description(self) -> str:  # type: ignore[override]
        """Dynamically build description from current PTC-eligible tools.

        The result is cached and keyed on (tool_name, tool_description)
        pairs so that add/remove/description-change invalidates correctly
        without needing explicit cache busts from the agent.  If the
        cache key computation fails (e.g., a tool's ``description``
        property raises), we fall back to an uncached rebuild so the
        tool still works.
        """
        try:
            eligible = get_ptc_eligible_tools(self._agent)
        except Exception:
            _ptc_log.exception("PTC description: get_ptc_eligible_tools failed")
            return _PTC_PREAMBLE + "(tool list unavailable)"

        try:
            current_key = tuple(
                (name, tool.description or "") for name, tool in sorted(eligible.items())
            )
        except Exception:
            current_key = None

        if (
            current_key is not None
            and self._desc_cache is not None
            and self._desc_cache_key == current_key
        ):
            return self._desc_cache

        try:
            desc = _PTC_PREAMBLE + build_tool_signatures(eligible)
        except Exception:
            _ptc_log.exception("PTC description: build_tool_signatures failed")
            return _PTC_PREAMBLE + "(signature rendering failed)"

        if current_key is not None:
            self._desc_cache = desc
            self._desc_cache_key = current_key
        return desc

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
        code_arg = kwargs.get("code", "")
        code: str = code_arg if isinstance(code_arg, str) else str(code_arg)
        executor = PTCExecutor(
            self._agent,
            timeout=self._timeout,
            max_output_bytes=self._max_output_bytes,
            max_tool_calls=self._max_tool_calls,
            max_code_bytes=self._max_code_bytes,
        )
        return await executor.run(code)


# ---------------------------------------------------------------------------
# Tool namespace — isolates tool functions from Python builtins/keywords
# ---------------------------------------------------------------------------


class _ToolNamespace:
    """Simple attribute-based namespace for PTC tool functions.

    Tools are set as attributes so the LLM writes
    ``await default_api.search(...)`` instead of bare ``await search(...)``.
    This prevents collisions with Python builtins (``map``, ``list``,
    ``filter``, ``type``, ``id``, etc.), keywords (``return``, ``class``),
    and stdlib modules (``json``, ``math``, ``re``, etc.).
    """

    pass


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

    def __init__(
        self,
        agent: Agent,
        timeout: int = DEFAULT_PTC_TIMEOUT,
        max_output_bytes: int = DEFAULT_PTC_MAX_OUTPUT_BYTES,
        max_tool_calls: int = DEFAULT_PTC_MAX_TOOL_CALLS,
        max_code_bytes: int = DEFAULT_PTC_MAX_CODE_BYTES,
    ) -> None:
        self._agent = agent
        # Clamp all limits to safe minimums so bad config cannot break the tool.
        self._timeout = max(1, int(timeout))
        self._max_output_bytes = max(256, int(max_output_bytes))
        self._max_tool_calls = max(1, int(max_tool_calls))
        self._max_code_bytes = max(1024, int(max_code_bytes))
        self._tool_call_count = 0

    async def run(self, code: str) -> str:
        """Execute *code* and return captured output.

        Returns:
            Combined stdout+stderr and ``repr()`` of the return value (if any).
            On error (tool raise, SystemExit, KeyboardInterrupt, timeout,
            max-tool-calls exceeded), returns a formatted error string.

        Raises:
            asyncio.CancelledError: Propagated after orphan-task cleanup so
                the outer scope can cancel the agent cleanly.
        """
        if not isinstance(code, str):
            return "Error: code must be a string"
        if not code or not code.strip():
            return "Error: empty code"
        if len(code) > self._max_code_bytes:
            return f"Error: code length {len(code)} exceeds max_code_bytes={self._max_code_bytes}"

        # Static sandbox pre-scan: reject dunder escape hatches, blocked
        # builtin name references, and imports of blocked stdlib modules
        # before any code runs.  Line numbers match the agent's own view
        # (no wrapper offset), so the error is actionable.
        scan_errors = _ast_scan_user_code(code)
        if scan_errors:
            agent_name = getattr(self._agent, "name", "<unknown>")
            _ptc_log.warning(
                "PTC: sandbox pre-scan rejected code from agent %s (%d errors)",
                agent_name,
                len(scan_errors),
            )
            return self._truncate("PTC sandbox error:\n" + "\n".join(scan_errors))

        try:
            namespace = self._build_namespace()
        except Exception as exc:
            _ptc_log.exception("PTC namespace build failed")
            return f"Error: failed to build PTC namespace: {exc}"

        # Wrap user code inside an async function with an inner try/except
        # that traps SystemExit / KeyboardInterrupt. asyncio.Task re-raises
        # those directly into the event loop, so we must convert them to a
        # regular Exception *before* they escape the coroutine.  The user's
        # code is at line 3 of the wrapper, so the line-number rewrites
        # below subtract 2.
        indented = textwrap.indent(textwrap.dedent(code), "        ")
        wrapped = (
            "async def __ptc_main__():\n"
            "    try:\n"
            f"{indented}\n"
            "    except SystemExit as __ptc_exit__:\n"
            "        raise __PTC_TRAP__("
            "'SystemExit', f'code={__ptc_exit__.code}')\n"
            "    except KeyboardInterrupt:\n"
            "        raise __PTC_TRAP__('KeyboardInterrupt', '')\n"
        )

        try:
            compiled = compile(wrapped, "<ptc>", "exec")
        except SyntaxError as exc:
            fixed_line = exc.lineno - 2 if (exc.lineno and exc.lineno > 2) else exc.lineno
            return self._truncate(f"SyntaxError at line {fixed_line}: {exc.msg}")

        exec(compiled, namespace)

        # Fresh per-run counter for the max-tool-calls cap.
        self._tool_call_count = 0
        output_buf = io.StringIO()

        # Snapshot active tasks so we can identify & cancel orphans afterwards.
        try:
            tasks_before = asyncio.all_tasks()
        except RuntimeError:
            tasks_before = set()

        try:
            with (
                contextlib.redirect_stdout(output_buf),
                contextlib.redirect_stderr(output_buf),
            ):
                result = await asyncio.wait_for(
                    namespace["__ptc_main__"](),
                    timeout=self._timeout,
                )
        except BaseException as exc:
            # asyncio.CancelledError MUST propagate — honour outer cancellation.
            if isinstance(exc, asyncio.CancelledError):
                await self._cleanup_orphan_tasks(tasks_before)
                raise
            captured = output_buf.getvalue()
            await self._cleanup_orphan_tasks(tasks_before)
            try:
                msg = self._format_exception(exc)
            except Exception:
                msg = f"{type(exc).__name__}: (unprintable)"
            combined = (captured + "\n" + msg).strip() if captured else msg
            return self._truncate(combined)

        # Happy path
        await self._cleanup_orphan_tasks(tasks_before)
        captured = output_buf.getvalue()
        if result is not None:
            # Defensive repr — a badly-written __repr__ should not break PTC.
            try:
                result_repr = repr(result)
            except Exception as exc:
                result_repr = f"<unreprable {type(result).__name__}: {exc}>"
            if captured:
                return self._truncate(f"{captured.rstrip()}\n{result_repr}")
            return self._truncate(result_repr)
        return self._truncate(captured.rstrip() or "(no output)")

    # -- helpers --

    def _format_exception(self, exc: BaseException) -> str:
        """Turn an exception into a user-facing error string."""
        if isinstance(exc, _PTCBaseExceptionTrap):
            if exc.detail:
                return f"{exc.kind}: {exc.detail} (blocked inside PTC)"
            return f"{exc.kind} (blocked inside PTC)"
        if isinstance(exc, TimeoutError):
            return f"TimeoutError: execution exceeded {self._timeout}s"
        if isinstance(exc, SystemExit):
            return f"SystemExit: code={exc.code} (blocked inside PTC)"
        if isinstance(exc, KeyboardInterrupt):
            return "KeyboardInterrupt (blocked inside PTC)"
        if isinstance(exc, MaxToolCallsExceeded):
            return str(exc)
        try:
            tb = traceback.format_exc()
        except Exception:
            return f"{type(exc).__name__}: {exc}"
        return self._rewrite_traceback(tb)

    def _rewrite_traceback(self, tb: str) -> str:
        """Subtract 2 from ``<ptc>`` line numbers to account for the wrapper.

        The wrapper adds two lines before the user code:
            line 1: ``async def __ptc_main__():``
            line 2: ``    try:``
            line 3: first line of user code
        """
        return re.sub(
            r'File "<ptc>", line (\d+)',
            lambda m: f'File "<ptc>", line {max(1, int(m.group(1)) - 2)}',
            tb,
        )

    def _truncate(self, text: str) -> str:
        """Cap output to ``max_output_bytes`` characters with a trailing marker."""
        limit = self._max_output_bytes
        if len(text) <= limit:
            return text
        head = text[: max(0, limit - 80)]
        dropped = len(text) - len(head)
        return f"{head}\n...[truncated {dropped} chars]"

    async def _cleanup_orphan_tasks(self, tasks_before: set[asyncio.Task[Any]]) -> None:
        """Cancel any asyncio tasks created by user code still running."""
        try:
            tasks_after = asyncio.all_tasks()
        except RuntimeError:
            return
        current = asyncio.current_task()
        orphans = {t for t in (tasks_after - tasks_before) if t is not current and not t.done()}
        if not orphans:
            return
        for task in orphans:
            task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*orphans, return_exceptions=True),
                timeout=1.0,
            )
        except TimeoutError:
            _ptc_log.warning(
                "PTC cleanup: %d orphan task(s) did not cancel within 1s",
                len(orphans),
            )
        else:
            _ptc_log.debug("PTC cleanup: cancelled %d orphan task(s)", len(orphans))

    # -- namespace construction --

    def _build_namespace(self) -> dict[str, Any]:
        """Create the sandboxed execution namespace with tool wrappers.

        The namespace uses a curated ``__builtins__`` whitelist — dangerous
        builtins are replaced with stubs that raise :class:`PTCSandboxError`,
        ``__import__`` is hooked to reject non-allowlisted modules, and
        ``getattr``/``hasattr`` go through ``safer_getattr`` which blocks
        dynamic dunder lookup.  Tools are exposed via a ``default_api``
        namespace object so tool names can never collide with builtins,
        keywords, or the stdlib modules.
        """
        agent_name = getattr(self._agent, "name", "<unknown>")

        # 1. Start with the curated safe builtins — NOT `builtins.__dict__`.
        safe_builtins: dict[str, Any] = dict(_PTC_SAFE_BUILTINS)

        # 2. Install stubs for every explicitly blocked builtin so user code
        #    gets a helpful PTCSandboxError rather than a bare NameError.
        def _make_blocked_stub(name: str) -> Callable[..., Any]:
            def _blocked(*_args: Any, **_kwargs: Any) -> Any:
                _ptc_log.warning(
                    "PTC: blocked builtin %r called by agent %s",
                    name,
                    agent_name,
                )
                raise PTCSandboxError(
                    f"PTC: `{name}()` is blocked for security. "
                    f"Use `default_api.*` tools registered on the agent instead."
                )

            _blocked.__name__ = name
            return _blocked

        for bname in _PTC_BLOCKED_BUILTINS:
            safe_builtins[bname] = _make_blocked_stub(bname)

        # 3. Install the restricted ``__import__`` hook.
        preloaded = set(_PTC_STDLIB)

        def _ptc_import(
            name: str,
            globals_: dict[str, Any] | None = None,
            locals_: dict[str, Any] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> Any:
            top = name.split(".", 1)[0]
            if top in _PTC_BLOCKED_IMPORTS:
                _ptc_log.warning("PTC: blocked import of %r by agent %s", name, agent_name)
                raise ImportError(
                    f"PTC: import of {name!r} is blocked for security. "
                    f"Use `default_api.*` tools registered on the agent instead "
                    f"(e.g. `await default_api.read_file(...)`). "
                    f"If you need this capability, ask the user to register "
                    f"a tool for it."
                )
            if top not in preloaded:
                raise ImportError(
                    f"PTC: import of {name!r} is not allowed. "
                    f"Available modules: {sorted(preloaded)}. "
                    f"Use `default_api.*` for anything else."
                )
            # Re-import of a pre-loaded module is a no-op — return the cached
            # reference so ``import json as j`` still works.
            return __import__(name, globals_, locals_, fromlist, level)

        safe_builtins["__import__"] = _ptc_import

        # 4. Install safer_getattr as the runtime attribute accessor.
        #    This catches dynamic dunder access like ``getattr(x, "__class__")``
        #    even though the static form ``x.__class__`` is caught separately
        #    by the AST pre-scan.
        def _ptc_getattr(obj: Any, name: str, *default: Any) -> Any:
            if name in _PTC_BLOCKED_ATTRS:
                raise PTCSandboxError(f"PTC: access to `.{name}` is blocked (dunder escape hatch).")
            if default:
                return safer_getattr(obj, name, default[0])
            return safer_getattr(obj, name)

        def _ptc_hasattr(obj: Any, name: str) -> bool:
            try:
                _ptc_getattr(obj, name)
                return True
            except (AttributeError, PTCSandboxError):
                return False

        safe_builtins["getattr"] = _ptc_getattr
        safe_builtins["hasattr"] = _ptc_hasattr

        # 5. Compose the final namespace: builtins, stdlib, trap, tools.
        ns: dict[str, Any] = {"__builtins__": safe_builtins}
        ns.update(_PTC_STDLIB)
        ns["__PTC_TRAP__"] = _PTCBaseExceptionTrap

        eligible = get_ptc_eligible_tools(self._agent)
        api = _ToolNamespace()
        for tool in eligible.values():
            try:
                setattr(api, tool.name, self._make_tool_fn(tool))
            except Exception:
                _ptc_log.exception("PTC: failed to wrap tool %r — skipping", tool.name)
        ns["default_api"] = api

        return ns

    def _make_tool_fn(self, tool_obj: Tool) -> Callable[..., Any]:
        """Create an async wrapper function for a tool.

        The wrapper emits ``ToolCallEvent`` / ``ToolResultEvent`` to the
        agent's event queue so the stream is transparent — consumers see
        individual tool events instead of the opaque PTC tool.
        """
        executor = self
        agent = self._agent

        async def wrapper(**kwargs: Any) -> str:
            # Enforce the per-run tool-call cap before any work is done.
            executor._tool_call_count += 1
            if executor._tool_call_count > executor._max_tool_calls:
                raise MaxToolCallsExceeded(
                    f"PTC code exceeded max_tool_calls={executor._max_tool_calls}"
                )

            tool_call_id = f"ptc_{uuid.uuid4().hex}"

            # Emit ToolCallEvent BEFORE execution (transparent to stream)
            _ptc_log.debug(
                "PTC enqueue ToolCallEvent: tool=%s id=%s agent=%s queue_size=%d",
                tool_obj.name,
                tool_call_id,
                agent.name,
                agent._event_queue.qsize(),
            )
            agent._event_queue.put_nowait(
                ToolCallEvent(
                    tool_name=tool_obj.name,
                    tool_call_id=tool_call_id,
                    arguments=json.dumps(kwargs, default=str),
                    agent_name=agent.name,
                )
            )

            # Fire PRE_TOOL_CALL hook
            await agent.hook_manager.run(
                _PRE_TOOL_CALL,
                agent=agent,
                tool_name=tool_obj.name,
                arguments=kwargs,
            )

            # Inject ToolContext if the tool declares one (any Tool subclass).
            ctx_param = getattr(tool_obj, "_tool_context_param", None)
            if ctx_param:
                from exo.tool_context import ToolContext

                kwargs[ctx_param] = ToolContext(
                    agent_name=agent.name,
                    queue=agent._event_queue,
                    human_input_handler=getattr(agent, "_human_input_handler", None),
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
                    _POST_TOOL_CALL,
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
                # ``default=str`` handles datetimes, Path, Decimal, Pydantic, etc.
                try:
                    output = json.dumps(raw, default=str)
                except (TypeError, ValueError):
                    output = str(raw)
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
                _POST_TOOL_CALL,
                agent=agent,
                tool_name=tool_obj.name,
                result=result_obj,
            )

            return output

        wrapper.__name__ = tool_obj.name
        wrapper.__doc__ = tool_obj.description
        return wrapper
