"""Tests for exo.tool — tool system, decorator, schema, execution."""

from typing import Any

import pytest

from exo.registry import Registry
from exo.tool import (
    FunctionTool,
    Tool,
    ToolError,
    _generate_schema,
    tool,
)
from exo.types import ExoError

# --- ToolError ---


class TestToolError:
    def test_inherits_exo_error(self) -> None:
        assert issubclass(ToolError, ExoError)

    def test_raise_and_catch(self) -> None:
        with pytest.raises(ToolError, match="broken"):
            raise ToolError("broken")


# --- @tool decorator forms ---


class TestToolDecorator:
    def test_bare_decorator(self) -> None:
        @tool
        def greet(name: str) -> str:
            """Say hello."""
            return f"hi {name}"

        assert isinstance(greet, FunctionTool)
        assert greet.name == "greet"

    def test_empty_parens_decorator(self) -> None:
        @tool()
        def greet(name: str) -> str:
            """Say hello."""
            return f"hi {name}"

        assert isinstance(greet, FunctionTool)
        assert greet.name == "greet"

    def test_named_decorator(self) -> None:
        @tool(name="my_greet", description="Custom desc")
        def greet(name: str) -> str:
            """Say hello."""
            return f"hi {name}"

        assert greet.name == "my_greet"
        assert greet.description == "Custom desc"

    def test_no_docstring(self) -> None:
        @tool
        def nodoc(x: int) -> int:
            return x

        assert nodoc.description == ""


# --- Schema generation ---


class TestSchemaGeneration:
    def test_str_param(self) -> None:
        def fn(x: str) -> None: ...

        schema = _generate_schema(fn)
        assert schema["properties"]["x"] == {"type": "string"}

    def test_int_param(self) -> None:
        def fn(x: int) -> None: ...

        schema = _generate_schema(fn)
        assert schema["properties"]["x"] == {"type": "integer"}

    def test_float_param(self) -> None:
        def fn(x: float) -> None: ...

        schema = _generate_schema(fn)
        assert schema["properties"]["x"] == {"type": "number"}

    def test_bool_param(self) -> None:
        def fn(x: bool) -> None: ...

        schema = _generate_schema(fn)
        assert schema["properties"]["x"] == {"type": "boolean"}

    def test_list_param(self) -> None:
        def fn(x: list[str]) -> None: ...

        schema = _generate_schema(fn)
        assert schema["properties"]["x"] == {
            "type": "array",
            "items": {"type": "string"},
        }

    def test_dict_param(self) -> None:
        def fn(x: dict[str, Any]) -> None: ...

        schema = _generate_schema(fn)
        assert schema["properties"]["x"] == {"type": "object"}

    def test_optional_unwraps(self) -> None:
        def fn(x: int | None = None) -> None: ...

        schema = _generate_schema(fn)
        assert schema["properties"]["x"] == {"type": "integer"}
        # Has a default, so should NOT be in required
        assert "required" not in schema or "x" not in schema.get("required", [])

    def test_no_type_hint_defaults_to_string(self) -> None:
        def fn(x) -> None: ...  # type: ignore[no-untyped-def]

        schema = _generate_schema(fn)
        assert schema["properties"]["x"] == {"type": "string"}

    def test_default_value_not_required(self) -> None:
        def fn(a: str, b: int = 5) -> None: ...

        schema = _generate_schema(fn)
        assert schema["required"] == ["a"]

    def test_docstring_descriptions(self) -> None:
        def fn(query: str, limit: int = 10) -> None:
            """Search things.

            Args:
                query: The search query.
                limit: Max results to return.
            """

        schema = _generate_schema(fn)
        assert schema["properties"]["query"]["description"] == "The search query."
        assert schema["properties"]["limit"]["description"] == "Max results to return."

    def test_tool_context_param_excluded(self) -> None:
        """ToolContext-typed parameters are excluded from the generated schema."""
        from exo.tool_context import ToolContext

        def fn(query: str, ctx: ToolContext) -> str:
            """Search with context."""
            return query

        schema = _generate_schema(fn)
        assert "ctx" not in schema["properties"]
        assert schema["required"] == ["query"]

    def test_tool_context_not_in_schema_with_default(self) -> None:
        """ToolContext param with default is also excluded."""
        from exo.tool_context import ToolContext

        def fn(query: str, ctx: ToolContext = None) -> str:  # type: ignore[assignment]
            return query

        schema = _generate_schema(fn)
        assert "ctx" not in schema["properties"]


# --- FunctionTool ToolContext detection ---


class TestFunctionToolContextDetection:
    def test_detects_tool_context_param(self) -> None:
        """FunctionTool detects a ToolContext-typed parameter."""
        from exo.tool_context import ToolContext

        @tool
        async def research(query: str, ctx: ToolContext) -> str:
            """Research something."""
            return query

        assert research._tool_context_param == "ctx"

    def test_no_tool_context_param(self) -> None:
        """FunctionTool sets _tool_context_param to None for normal tools."""

        @tool
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        assert add._tool_context_param is None

    def test_custom_param_name(self) -> None:
        """Detection works regardless of parameter name."""
        from exo.tool_context import ToolContext

        @tool
        async def fetch(url: str, tool_ctx: ToolContext) -> str:
            """Fetch."""
            return url

        assert fetch._tool_context_param == "tool_ctx"


# --- Execution ---


class TestExecution:
    async def test_sync_function(self) -> None:
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = await add.execute(a=2, b=3)
        assert result == 5

    async def test_async_function(self) -> None:
        @tool
        async def fetch(url: str) -> str:
            """Fetch a URL."""
            return f"response from {url}"

        result = await fetch.execute(url="https://example.com")
        assert result == "response from https://example.com"

    async def test_str_passthrough(self) -> None:
        @tool
        def echo(msg: str) -> str:
            """Echo."""
            return msg

        result = await echo.execute(msg="hello")
        assert isinstance(result, str)
        assert result == "hello"

    async def test_dict_passthrough(self) -> None:
        @tool
        def info() -> dict[str, int]:
            """Return info."""
            return {"count": 42}

        result = await info.execute()
        assert isinstance(result, dict)
        assert result == {"count": 42}

    async def test_error_wrapping(self) -> None:
        @tool
        def bad() -> str:
            """Fail."""
            raise ValueError("oops")

        with pytest.raises(ToolError, match="bad"):
            await bad.execute()


# --- to_schema ---


class TestToSchema:
    def test_schema_structure(self) -> None:
        @tool
        def search(query: str, limit: int = 10) -> str:
            """Search for items."""
            return query

        schema = search.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert schema["function"]["description"] == "Search for items."
        assert "properties" in schema["function"]["parameters"]
        assert "query" in schema["function"]["parameters"]["properties"]


# --- Custom Tool subclass ---


class TestToolSubclass:
    async def test_custom_subclass(self) -> None:
        class UpperTool(Tool):
            def __init__(self) -> None:
                self.name = "upper"
                self.description = "Uppercase a string."
                self.parameters = {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                }

            async def execute(self, **kwargs: Any) -> str:
                return kwargs["text"].upper()

        t = UpperTool()
        result = await t.execute(text="hello")
        assert result == "HELLO"
        assert t.to_schema()["function"]["name"] == "upper"


# --- Registry integration ---


class TestRegistryIntegration:
    def test_manual_registration(self) -> None:
        reg: Registry[Tool] = Registry("test_tools")

        @tool
        def ping() -> str:
            """Ping."""
            return "pong"

        reg.register("ping", ping)
        assert reg.get("ping") is ping
        assert isinstance(reg.get("ping"), FunctionTool)
