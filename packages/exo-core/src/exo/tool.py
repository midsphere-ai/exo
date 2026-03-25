"""Tool system: ABC, decorator, schema generation, and execution."""

from __future__ import annotations

import asyncio
import inspect
import re
import types
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Union, get_type_hints, overload

from exo.types import ContentBlock, ExoError


class ToolError(ExoError):
    """Raised when a tool execution fails."""


# ---------------------------------------------------------------------------
# Schema generation helpers (private)
# ---------------------------------------------------------------------------


def _extract_description(fn: Callable[..., Any]) -> str:
    """Return the first non-empty line of the function's docstring.

    Args:
        fn: The function to extract a description from.

    Returns:
        The first line, or empty string if no docstring.
    """
    doc = inspect.getdoc(fn)
    if not doc:
        return ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _parse_docstring_args(fn: Callable[..., Any]) -> dict[str, str]:
    """Parse Google-style ``Args:`` section from a docstring.

    Args:
        fn: The function whose docstring to parse.

    Returns:
        Mapping of parameter name to its description string.
    """
    doc = inspect.getdoc(fn)
    if not doc:
        return {}

    result: dict[str, str] = {}
    in_args = False
    current_name: str | None = None
    current_desc: list[str] = []

    for line in doc.splitlines():
        stripped = line.strip()

        # Detect start of Args: section
        if stripped == "Args:":
            in_args = True
            continue

        # Detect end of Args: section (another top-level section)
        if in_args and re.match(r"^[A-Z]\w*:\s*$", stripped):
            if current_name is not None:
                result[current_name] = " ".join(current_desc).strip()
            break

        if not in_args:
            continue

        # Match "param_name: description" or "param_name (type): description"
        match = re.match(r"^(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)", stripped)
        if match:
            if current_name is not None:
                result[current_name] = " ".join(current_desc).strip()
            current_name = match.group(1)
            current_desc = [match.group(2)] if match.group(2) else []
        elif current_name is not None and stripped:
            current_desc.append(stripped)

    # Flush last param
    if current_name is not None and current_name not in result:
        result[current_name] = " ".join(current_desc).strip()

    return result


def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema type dict.

    Args:
        annotation: A Python type annotation.

    Returns:
        A JSON Schema type dictionary.
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    # Handle Union types (X | None) — unwrap Optional
    # Python 3.10+ pipe syntax produces types.UnionType (no __origin__)
    if isinstance(annotation, types.UnionType):
        args = [a for a in annotation.__args__ if a is not type(None)]
        if args:
            return _python_type_to_json_schema(args[0])
        return {"type": "string"}

    origin = getattr(annotation, "__origin__", None)

    # typing.Union / typing.Optional
    if origin is Union:
        args = [a for a in annotation.__args__ if a is not type(None)]
        if args:
            return _python_type_to_json_schema(args[0])
        return {"type": "string"}

    # list[X]
    if origin is list:
        item_args = getattr(annotation, "__args__", None)
        items = _python_type_to_json_schema(item_args[0]) if item_args else {"type": "string"}
        return {"type": "array", "items": items}

    # dict[K, V]
    if origin is dict:
        return {"type": "object"}

    # Simple scalar types
    type_map: dict[type, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }
    json_type = type_map.get(annotation)  # type: ignore[arg-type]
    if json_type:
        return {"type": json_type}

    return {"type": "string"}


def _generate_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Generate a JSON Schema ``parameters`` object from a function signature.

    Inspects the function's signature, type hints, and Google-style docstring
    to produce a complete JSON Schema object with property types, descriptions,
    and required fields.

    Args:
        fn: The function to generate a schema for.

    Returns:
        A JSON Schema object dict with ``type``, ``properties``, and ``required``.
    """
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    doc_args = _parse_docstring_args(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    skip = {"self", "cls"}

    for name, param in sig.parameters.items():
        if name in skip:
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        annotation = hints.get(name, inspect.Parameter.empty)
        prop = _python_type_to_json_schema(annotation)

        if name in doc_args:
            prop["description"] = doc_args[name]

        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# ---------------------------------------------------------------------------
# Tool ABC and FunctionTool
# ---------------------------------------------------------------------------


class Tool(ABC):
    """Abstract base class for all tools.

    Subclasses must implement ``execute()``. The ``name``, ``description``,
    and ``parameters`` attributes describe the tool for LLM function calling.

    Args:
        name: Unique tool name.
        description: Human-readable description.
        parameters: JSON Schema dict describing accepted arguments.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    async def execute(self, **kwargs: Any) -> str | dict[str, Any] | list[ContentBlock]:
        """Execute the tool with the given keyword arguments.

        Args:
            **kwargs: Tool-specific arguments.

        Returns:
            A string, dict, or list of ContentBlock results.
        """

    def to_schema(self) -> dict[str, Any]:
        """Return the tool schema in OpenAI function-calling format.

        Returns:
            A dict with ``type``, ``function.name``, ``function.description``,
            and ``function.parameters``.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class FunctionTool(Tool):
    """A tool that wraps a plain sync or async function.

    Sync functions are automatically wrapped via ``asyncio.to_thread()``
    so all execution goes through the single async ``execute()`` interface.

    Args:
        fn: The function to wrap.
        name: Override the tool name (defaults to ``fn.__name__``).
        description: Override the description (defaults to docstring).
        large_output: When ``True``, the tool's result will be offloaded to
            the agent's workspace instead of being injected directly into the
            LLM context.  A pointer string is returned in its place.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        large_output: bool = False,
    ) -> None:
        self._fn = fn
        self._is_async = asyncio.iscoroutinefunction(fn)
        self.name = name or fn.__name__
        self.description = description or _extract_description(fn)
        self.parameters = _generate_schema(fn)
        self.large_output: bool = large_output

    async def execute(self, **kwargs: Any) -> str | dict[str, Any] | list[ContentBlock]:
        """Execute the wrapped function.

        Args:
            **kwargs: Arguments forwarded to the wrapped function.

        Returns:
            The function's return value.

        Raises:
            ToolError: If the wrapped function raises any exception.
        """
        try:
            if self._is_async:
                return await self._fn(**kwargs)
            return await asyncio.to_thread(self._fn, **kwargs)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"Tool '{self.name}' failed: {exc}") from exc


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------


@overload
def tool(fn: Callable[..., Any], /) -> FunctionTool: ...


@overload
def tool(
    fn: None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    large_output: bool = False,
) -> Callable[[Callable[..., Any]], FunctionTool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    large_output: bool = False,
) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
    """Decorator to turn a function into a ``FunctionTool``.

    Supports bare ``@tool``, ``@tool()``, and ``@tool(name="x")`` forms.

    Args:
        fn: The function (when used as bare ``@tool``).
        name: Override tool name.
        description: Override tool description.
        large_output: When ``True``, the tool's result will be offloaded to
            the agent's workspace.  A pointer string referencing the artifact
            is returned in its place so the LLM context window is not flooded.

    Returns:
        A ``FunctionTool`` instance, or a decorator that produces one.
    """
    if fn is not None:
        return FunctionTool(fn, name=name, description=description, large_output=large_output)

    def decorator(func: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(func, name=name, description=description, large_output=large_output)

    return decorator
