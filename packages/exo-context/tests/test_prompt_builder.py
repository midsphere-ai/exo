"""Tests for exo.context.prompt_builder — PromptBuilder."""

from __future__ import annotations

from typing import Any

import pytest

from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
from exo.context.context import Context  # pyright: ignore[reportMissingImports]
from exo.context.neuron import (  # pyright: ignore[reportMissingImports]
    Neuron,
)
from exo.context.prompt_builder import (  # pyright: ignore[reportMissingImports]
    PromptBuilder,
    PromptBuilderError,
)
from exo.context.variables import (  # pyright: ignore[reportMissingImports]
    DynamicVariableRegistry,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _make_ctx(
    task_id: str = "test-task",
    state: dict[str, Any] | None = None,
    config: ContextConfig | None = None,
) -> Context:
    ctx = Context(task_id, config=config)
    if state:
        for k, v in state.items():
            ctx.state.set(k, v)
    return ctx


class FixedNeuron(Neuron):
    """Neuron that returns a fixed string."""

    def __init__(self, name: str, text: str, *, priority: int = 50) -> None:
        super().__init__(name, priority=priority)
        self._text = text

    async def format(self, ctx: Context, **kwargs: Any) -> str:
        return self._text


class EmptyNeuron(Neuron):
    """Neuron that returns empty string."""

    def __init__(self, name: str = "empty", *, priority: int = 50) -> None:
        super().__init__(name, priority=priority)

    async def format(self, ctx: Context, **kwargs: Any) -> str:
        return ""


class KwargsCapturingNeuron(Neuron):
    """Neuron that captures kwargs for test verification."""

    def __init__(self, name: str = "capture", *, priority: int = 50) -> None:
        super().__init__(name, priority=priority)
        self.captured_kwargs: dict[str, Any] = {}

    async def format(self, ctx: Context, **kwargs: Any) -> str:
        self.captured_kwargs = dict(kwargs)
        return f"captured:{kwargs}"


# ── TestPromptBuilder — construction ─────────────────────────────────


class TestPromptBuilderInit:
    def test_empty_builder(self) -> None:
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)
        assert len(builder) == 0
        assert builder.ctx is ctx

    def test_custom_separator(self) -> None:
        ctx = _make_ctx()
        builder = PromptBuilder(ctx, separator="---")
        assert builder._separator == "---"

    def test_repr_empty(self) -> None:
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)
        assert "PromptBuilder" in repr(builder)
        assert "neurons=[]" in repr(builder)


# ── TestPromptBuilder — add neurons ──────────────────────────────────


class TestPromptBuilderAdd:
    def test_add_by_name(self) -> None:
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)
        result = builder.add("task")
        assert len(builder) == 1
        assert result is builder  # method chaining

    def test_add_unknown_name_raises(self) -> None:
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)
        with pytest.raises(PromptBuilderError, match="not found"):
            builder.add("nonexistent_neuron_xyz")

    def test_add_neuron_instance(self) -> None:
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)
        neuron = FixedNeuron("custom", "hello")
        result = builder.add_neuron(neuron)
        assert len(builder) == 1
        assert result is builder

    def test_chaining(self) -> None:
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)
        builder.add("task").add("history").add("system")
        assert len(builder) == 3

    def test_repr_with_neurons(self) -> None:
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)
        builder.add_neuron(FixedNeuron("alpha", "a"))
        builder.add_neuron(FixedNeuron("beta", "b"))
        r = repr(builder)
        assert "alpha" in r
        assert "beta" in r


# ── TestPromptBuilder — build ────────────────────────────────────────


class TestPromptBuilderBuild:
    async def test_build_empty(self) -> None:
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)
        result = await builder.build()
        assert result == ""

    async def test_build_single_neuron(self) -> None:
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)
        builder.add_neuron(FixedNeuron("a", "Hello world"))
        result = await builder.build()
        assert result == "Hello world"

    async def test_build_multiple_neurons(self) -> None:
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)
        builder.add_neuron(FixedNeuron("a", "AAA", priority=10))
        builder.add_neuron(FixedNeuron("b", "BBB", priority=20))
        result = await builder.build()
        assert result == "AAA\n\nBBB"

    async def test_build_custom_separator(self) -> None:
        ctx = _make_ctx()
        builder = PromptBuilder(ctx, separator="---")
        builder.add_neuron(FixedNeuron("a", "X", priority=1))
        builder.add_neuron(FixedNeuron("b", "Y", priority=2))
        result = await builder.build()
        assert result == "X---Y"

    async def test_build_empty_neurons_filtered(self) -> None:
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)
        builder.add_neuron(FixedNeuron("a", "content", priority=1))
        builder.add_neuron(EmptyNeuron(priority=2))
        builder.add_neuron(FixedNeuron("c", "more", priority=3))
        result = await builder.build()
        assert result == "content\n\nmore"

    async def test_all_empty_neurons(self) -> None:
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)
        builder.add_neuron(EmptyNeuron("e1", priority=1))
        builder.add_neuron(EmptyNeuron("e2", priority=2))
        result = await builder.build()
        assert result == ""


# ── TestPromptBuilder — priority ordering ────────────────────────────


class TestPromptBuilderPriority:
    async def test_priority_ordering(self) -> None:
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)
        # Add in reverse priority order
        builder.add_neuron(FixedNeuron("high", "HIGH", priority=100))
        builder.add_neuron(FixedNeuron("low", "LOW", priority=1))
        builder.add_neuron(FixedNeuron("mid", "MID", priority=50))
        result = await builder.build()
        assert result == "LOW\n\nMID\n\nHIGH"

    async def test_same_priority_preserves_insertion_order(self) -> None:
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)
        builder.add_neuron(FixedNeuron("first", "FIRST", priority=10))
        builder.add_neuron(FixedNeuron("second", "SECOND", priority=10))
        builder.add_neuron(FixedNeuron("third", "THIRD", priority=10))
        result = await builder.build()
        assert result == "FIRST\n\nSECOND\n\nTHIRD"

    async def test_registered_neurons_priority(self) -> None:
        """task (p1), history (p10), system (p100) should sort correctly."""
        ctx = _make_ctx(
            state={
                "history": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            }
        )
        builder = PromptBuilder(ctx)
        builder.add("system")  # priority 100
        builder.add("task")  # priority 1
        builder.add("history")  # priority 10
        result = await builder.build()
        # Task (p1) should come first
        parts = result.split("\n\n")
        assert parts[0].startswith("<task_info>")
        assert any("<conversation_history>" in p for p in parts)
        assert any("<system_info>" in p for p in parts)


# ── TestPromptBuilder — variable resolution ──────────────────────────


class TestPromptBuilderVariables:
    async def test_variable_resolution(self) -> None:
        ctx = _make_ctx(state={"user_name": "Alice"})
        variables = DynamicVariableRegistry()
        variables.register("user.name", lambda state: state.get("user_name", "anon"))

        builder = PromptBuilder(ctx, variables=variables)
        builder.add_neuron(FixedNeuron("greeting", "Hello ${user.name}!", priority=1))
        result = await builder.build()
        assert result == "Hello Alice!"

    async def test_unresolvable_variable_left_as_is(self) -> None:
        ctx = _make_ctx()
        variables = DynamicVariableRegistry()

        builder = PromptBuilder(ctx, variables=variables)
        builder.add_neuron(FixedNeuron("msg", "Value: ${unknown.var}", priority=1))
        result = await builder.build()
        assert result == "Value: ${unknown.var}"

    async def test_no_variables_no_resolution(self) -> None:
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)  # no variables registry
        builder.add_neuron(FixedNeuron("msg", "Keep ${this}", priority=1))
        result = await builder.build()
        assert result == "Keep ${this}"

    async def test_multiple_variables(self) -> None:
        ctx = _make_ctx(state={"a": "X", "b": "Y"})
        variables = DynamicVariableRegistry()
        variables.register("va", lambda state: state.get("a"))
        variables.register("vb", lambda state: state.get("b"))

        builder = PromptBuilder(ctx, variables=variables)
        builder.add_neuron(FixedNeuron("msg", "${va} and ${vb}", priority=1))
        result = await builder.build()
        assert result == "X and Y"

    async def test_variable_resolution_across_neurons(self) -> None:
        ctx = _make_ctx(state={"name": "Bob"})
        variables = DynamicVariableRegistry()
        variables.register("name", lambda state: state.get("name"))

        builder = PromptBuilder(ctx, variables=variables)
        builder.add_neuron(FixedNeuron("a", "Hi ${name}", priority=1))
        builder.add_neuron(FixedNeuron("b", "Bye ${name}", priority=2))
        result = await builder.build()
        assert result == "Hi Bob\n\nBye Bob"


# ── TestPromptBuilder — kwargs forwarding ────────────────────────────


class TestPromptBuilderKwargs:
    async def test_kwargs_forwarded_to_neuron(self) -> None:
        ctx = _make_ctx()
        capture = KwargsCapturingNeuron("cap", priority=1)
        builder = PromptBuilder(ctx)
        builder.add_neuron(capture, extra_key="extra_value", limit=5)
        await builder.build()
        assert capture.captured_kwargs == {"extra_key": "extra_value", "limit": 5}


# ── TestPromptBuilder — clear ────────────────────────────────────────


class TestPromptBuilderClear:
    def test_clear(self) -> None:
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)
        builder.add_neuron(FixedNeuron("a", "x"))
        builder.add_neuron(FixedNeuron("b", "y"))
        assert len(builder) == 2
        builder.clear()
        assert len(builder) == 0

    async def test_build_after_clear(self) -> None:
        ctx = _make_ctx()
        builder = PromptBuilder(ctx)
        builder.add_neuron(FixedNeuron("a", "x"))
        builder.clear()
        result = await builder.build()
        assert result == ""


# ── TestPromptBuilder — context traversal ────────────────────────────


class TestPromptBuilderContextTraversal:
    async def test_variable_resolution_with_parent_state(self) -> None:
        """Variables should resolve from context state including parent chain."""
        parent_ctx = _make_ctx(task_id="parent", state={"shared": "inherited"})
        child_ctx = parent_ctx.fork("child")

        variables = DynamicVariableRegistry()
        # This should resolve through child -> parent state chain
        builder = PromptBuilder(child_ctx, variables=variables)
        builder.add_neuron(FixedNeuron("msg", "Value: ${shared}", priority=1))
        result = await builder.build()
        assert result == "Value: inherited"

    async def test_forked_context_with_overridden_state(self) -> None:
        parent_ctx = _make_ctx(task_id="parent", state={"key": "parent_val"})
        child_ctx = parent_ctx.fork("child")
        child_ctx.state.set("key", "child_val")

        variables = DynamicVariableRegistry()
        builder = PromptBuilder(child_ctx, variables=variables)
        builder.add_neuron(FixedNeuron("msg", "Key: ${key}", priority=1))
        result = await builder.build()
        assert result == "Key: child_val"


# ── TestPromptBuilder — integration with built-in neurons ────────────


class TestPromptBuilderIntegration:
    async def test_task_neuron_via_registry(self) -> None:
        ctx = _make_ctx(
            task_id="my-task",
            state={"task_input": "Do something useful"},
        )
        builder = PromptBuilder(ctx)
        builder.add("task")
        result = await builder.build()
        assert "my-task" in result
        assert "Do something useful" in result

    async def test_full_prompt_composition(self) -> None:
        """Build a prompt with task + history + system neurons."""
        ctx = _make_ctx(
            task_id="compose-test",
            state={
                "task_input": "Summarize the meeting",
                "history": [
                    {"role": "user", "content": "Hello agent"},
                    {"role": "assistant", "content": "Hello! How can I help?"},
                ],
            },
        )
        builder = PromptBuilder(ctx)
        builder.add("task").add("history").add("system")
        result = await builder.build()

        # Verify ordering: task (p1) < history (p10) < system (p100)
        task_pos = result.index("<task_info>")
        history_pos = result.index("<conversation_history>")
        system_pos = result.index("<system_info>")
        assert task_pos < history_pos < system_pos

    async def test_add_duplicate_neuron_names(self) -> None:
        """Adding same neuron name twice should include it twice."""
        ctx = _make_ctx(task_id="dup-test")
        builder = PromptBuilder(ctx)
        builder.add("task")
        builder.add("task")
        assert len(builder) == 2
        result = await builder.build()
        assert result.count("<task_info>") == 2
