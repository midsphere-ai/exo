"""Tests for extended neurons and dynamic variable registry."""

from __future__ import annotations

from typing import Any

import pytest

from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
from exo.context.context import Context  # pyright: ignore[reportMissingImports]
from exo.context.neuron import (  # pyright: ignore[reportMissingImports]
    EntityNeuron,
    FactNeuron,
    KnowledgeNeuron,
    SkillNeuron,
    TodoNeuron,
    WorkspaceNeuron,
    neuron_registry,
)
from exo.context.state import ContextState  # pyright: ignore[reportMissingImports]
from exo.context.variables import (  # pyright: ignore[reportMissingImports]
    DynamicVariableRegistry,
    VariableResolveError,
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


# ── TodoNeuron ──────────────────────────────────────────────────────


class TestTodoNeuron:
    async def test_default_name_and_priority(self) -> None:
        neuron = TodoNeuron()
        assert neuron.name == "todo"
        assert neuron.priority == 2

    async def test_empty_state(self) -> None:
        neuron = TodoNeuron()
        ctx = _make_ctx()
        result = await neuron.format(ctx)
        assert result == ""

    async def test_empty_list(self) -> None:
        neuron = TodoNeuron()
        ctx = _make_ctx(state={"todos": []})
        result = await neuron.format(ctx)
        assert result == ""

    async def test_basic_todos(self) -> None:
        neuron = TodoNeuron()
        ctx = _make_ctx(
            state={
                "todos": [
                    {"item": "Write tests", "done": False},
                    {"item": "Fix bugs", "done": True},
                ]
            }
        )
        result = await neuron.format(ctx)
        assert "<todo_list>" in result
        assert "</todo_list>" in result
        assert "[ ] Write tests" in result
        assert "[x] Fix bugs" in result

    async def test_todo_default_done(self) -> None:
        """Items without 'done' key default to False."""
        neuron = TodoNeuron()
        ctx = _make_ctx(state={"todos": [{"item": "New task"}]})
        result = await neuron.format(ctx)
        assert "[ ] New task" in result


# ── KnowledgeNeuron ─────────────────────────────────────────────────


class TestKnowledgeNeuron:
    async def test_default_name_and_priority(self) -> None:
        neuron = KnowledgeNeuron()
        assert neuron.name == "knowledge"
        assert neuron.priority == 20

    async def test_empty_state(self) -> None:
        neuron = KnowledgeNeuron()
        ctx = _make_ctx()
        result = await neuron.format(ctx)
        assert result == ""

    async def test_basic_knowledge(self) -> None:
        neuron = KnowledgeNeuron()
        ctx = _make_ctx(
            state={
                "knowledge_items": [
                    {"source": "docs/api.md", "content": "API uses REST"},
                    {"source": "README.md", "content": "Install via pip"},
                ]
            }
        )
        result = await neuron.format(ctx)
        assert "<knowledge>" in result
        assert "</knowledge>" in result
        assert "[docs/api.md]: API uses REST" in result
        assert "[README.md]: Install via pip" in result

    async def test_knowledge_default_source(self) -> None:
        neuron = KnowledgeNeuron()
        ctx = _make_ctx(state={"knowledge_items": [{"content": "Some info"}]})
        result = await neuron.format(ctx)
        assert "[unknown]: Some info" in result


# ── WorkspaceNeuron ─────────────────────────────────────────────────


class TestWorkspaceNeuron:
    async def test_default_name_and_priority(self) -> None:
        neuron = WorkspaceNeuron()
        assert neuron.name == "workspace"
        assert neuron.priority == 30

    async def test_empty_state(self) -> None:
        neuron = WorkspaceNeuron()
        ctx = _make_ctx()
        result = await neuron.format(ctx)
        assert result == ""

    async def test_basic_artifacts(self) -> None:
        neuron = WorkspaceNeuron()
        ctx = _make_ctx(
            state={
                "workspace_artifacts": [
                    {"name": "report.md", "type": "markdown", "size": 1024},
                    {"name": "data.csv", "type": "csv"},
                ]
            }
        )
        result = await neuron.format(ctx)
        assert "<workspace>" in result
        assert "</workspace>" in result
        assert "report.md [markdown] (1024 bytes)" in result
        assert "data.csv [csv]" in result
        # No size info for data.csv
        assert "data.csv [csv] (" not in result

    async def test_workspace_defaults(self) -> None:
        """Artifacts without type default to 'file'."""
        neuron = WorkspaceNeuron()
        ctx = _make_ctx(state={"workspace_artifacts": [{"name": "unknown"}]})
        result = await neuron.format(ctx)
        assert "unknown [file]" in result


# ── SkillNeuron ─────────────────────────────────────────────────────


class TestSkillNeuron:
    async def test_default_name_and_priority(self) -> None:
        neuron = SkillNeuron()
        assert neuron.name == "skill"
        assert neuron.priority == 40

    async def test_empty_state(self) -> None:
        neuron = SkillNeuron()
        ctx = _make_ctx()
        result = await neuron.format(ctx)
        assert result == ""

    async def test_basic_skills(self) -> None:
        neuron = SkillNeuron()
        ctx = _make_ctx(
            state={
                "skills": [
                    {"name": "search", "description": "Web search", "active": True},
                    {"name": "code_gen", "description": "Generate code", "active": False},
                ]
            }
        )
        result = await neuron.format(ctx)
        assert "<available_skills>" in result
        assert "</available_skills>" in result
        assert "- search: Web search" in result
        assert "- code_gen: Generate code [inactive]" in result

    async def test_skill_default_active(self) -> None:
        """Skills default to active if not specified."""
        neuron = SkillNeuron()
        ctx = _make_ctx(state={"skills": [{"name": "s1", "description": "Skill 1"}]})
        result = await neuron.format(ctx)
        assert "- s1: Skill 1" in result
        assert "[inactive]" not in result


# ── FactNeuron ──────────────────────────────────────────────────────


class TestFactNeuron:
    async def test_default_name_and_priority(self) -> None:
        neuron = FactNeuron()
        assert neuron.name == "fact"
        assert neuron.priority == 50

    async def test_empty_state(self) -> None:
        neuron = FactNeuron()
        ctx = _make_ctx()
        result = await neuron.format(ctx)
        assert result == ""

    async def test_basic_facts(self) -> None:
        neuron = FactNeuron()
        ctx = _make_ctx(
            state={"facts": ["Python is dynamically typed", "UV is a fast package manager"]}
        )
        result = await neuron.format(ctx)
        assert "<facts>" in result
        assert "</facts>" in result
        assert "- Python is dynamically typed" in result
        assert "- UV is a fast package manager" in result


# ── EntityNeuron ────────────────────────────────────────────────────


class TestEntityNeuron:
    async def test_default_name_and_priority(self) -> None:
        neuron = EntityNeuron()
        assert neuron.name == "entity"
        assert neuron.priority == 60

    async def test_empty_state(self) -> None:
        neuron = EntityNeuron()
        ctx = _make_ctx()
        result = await neuron.format(ctx)
        assert result == ""

    async def test_basic_entities(self) -> None:
        neuron = EntityNeuron()
        ctx = _make_ctx(
            state={
                "entities": [
                    {"name": "Alice", "type": "person"},
                    {"name": "Acme Corp", "type": "organization"},
                ]
            }
        )
        result = await neuron.format(ctx)
        assert "<entities>" in result
        assert "</entities>" in result
        assert "Alice (person)" in result
        assert "Acme Corp (organization)" in result

    async def test_entity_defaults(self) -> None:
        """Entities without name/type default to 'unknown'."""
        neuron = EntityNeuron()
        ctx = _make_ctx(state={"entities": [{}]})
        result = await neuron.format(ctx)
        assert "unknown (unknown)" in result


# ── Extended registry ───────────────────────────────────────────────


class TestExtendedNeuronRegistry:
    def test_all_extended_registered(self) -> None:
        for name in ("todo", "knowledge", "workspace", "skill", "fact", "entity"):
            assert name in neuron_registry

    def test_get_todo(self) -> None:
        assert isinstance(neuron_registry.get("todo"), TodoNeuron)

    def test_get_knowledge(self) -> None:
        assert isinstance(neuron_registry.get("knowledge"), KnowledgeNeuron)

    def test_get_workspace(self) -> None:
        assert isinstance(neuron_registry.get("workspace"), WorkspaceNeuron)

    def test_get_skill(self) -> None:
        assert isinstance(neuron_registry.get("skill"), SkillNeuron)

    def test_get_fact(self) -> None:
        assert isinstance(neuron_registry.get("fact"), FactNeuron)

    def test_get_entity(self) -> None:
        assert isinstance(neuron_registry.get("entity"), EntityNeuron)


# ── Priority ordering (extended) ────────────────────────────────────


class TestExtendedPriorityOrdering:
    def test_full_priority_order(self) -> None:
        """All neurons sorted by priority."""
        all_names = neuron_registry.list_all()
        neurons = [neuron_registry.get(n) for n in all_names]
        sorted_neurons = sorted(neurons, key=lambda n: n.priority)
        priorities = [(n.name, n.priority) for n in sorted_neurons]
        # Expected order: task(1), todo(2), history(10), knowledge(20),
        # workspace(30), skill(40), fact(50), entity(60), system(100)
        expected = [
            ("task", 1),
            ("todo", 2),
            ("history", 10),
            ("knowledge", 20),
            ("workspace", 30),
            ("skill", 40),
            ("fact", 50),
            ("entity", 60),
            ("system", 100),
        ]
        assert priorities == expected


# ── DynamicVariableRegistry ─────────────────────────────────────────


class TestDynamicVariableRegistry:
    def test_register_and_resolve_callable(self) -> None:
        reg = DynamicVariableRegistry()
        reg.register("user.name", lambda state: state.get("user_name", "anon"))
        state = ContextState({"user_name": "Alice"})
        assert reg.resolve("user.name", state) == "Alice"

    def test_register_and_resolve_static(self) -> None:
        reg = DynamicVariableRegistry()
        reg.register("app.version", "1.0.0")
        state = ContextState()
        assert reg.resolve("app.version", state) == "1.0.0"

    def test_register_decorator_form(self) -> None:
        reg = DynamicVariableRegistry()

        @reg.register("env.mode")
        def _resolve_mode(state: Any) -> str:
            return state.get("mode", "dev")

        state = ContextState({"mode": "production"})
        assert reg.resolve("env.mode", state) == "production"

    def test_nested_path_resolution(self) -> None:
        reg = DynamicVariableRegistry()
        state = ContextState({"user": {"name": "Bob", "age": 30}})
        assert reg.resolve("user.name", state) == "Bob"
        assert reg.resolve("user.age", state) == 30

    def test_nested_path_with_dict(self) -> None:
        reg = DynamicVariableRegistry()
        state = {"config": {"db": {"host": "localhost"}}}
        assert reg.resolve("config.db.host", state) == "localhost"

    def test_resolver_takes_priority(self) -> None:
        """Registered resolvers take priority over nested path lookup."""
        reg = DynamicVariableRegistry()
        reg.register("user.name", lambda state: "from-resolver")
        state = ContextState({"user": {"name": "from-state"}})
        assert reg.resolve("user.name", state) == "from-resolver"

    def test_missing_path_raises(self) -> None:
        reg = DynamicVariableRegistry()
        state = ContextState()
        with pytest.raises(VariableResolveError, match="not found"):
            reg.resolve("nonexistent.path", state)

    def test_has(self) -> None:
        reg = DynamicVariableRegistry()
        reg.register("x.y", "val")
        assert reg.has("x.y")
        assert not reg.has("a.b")

    def test_list_all(self) -> None:
        reg = DynamicVariableRegistry()
        reg.register("a", "1")
        reg.register("b", "2")
        assert reg.list_all() == ["a", "b"]

    def test_resolve_template(self) -> None:
        reg = DynamicVariableRegistry()
        reg.register("name", "World")
        reg.register("version", "2.0")
        state = ContextState()
        result = reg.resolve_template("Hello ${name}! v${version}", state)
        assert result == "Hello World! v2.0"

    def test_resolve_template_missing_var(self) -> None:
        """Unresolvable variables are left as-is."""
        reg = DynamicVariableRegistry()
        state = ContextState()
        result = reg.resolve_template("Hello ${missing}", state)
        assert result == "Hello ${missing}"

    def test_resolve_template_mixed(self) -> None:
        """Template with some resolvable, some not."""
        reg = DynamicVariableRegistry()
        reg.register("found", "yes")
        state = ContextState()
        result = reg.resolve_template("${found} and ${not_found}", state)
        assert result == "yes and ${not_found}"

    def test_repr(self) -> None:
        reg = DynamicVariableRegistry()
        reg.register("a", "1")
        reg.register("b", "2")
        assert "2" in repr(reg)

    def test_nested_path_partial_raises(self) -> None:
        """Nested path where an intermediate segment is missing raises."""
        reg = DynamicVariableRegistry()
        state = ContextState({"user": {"name": "Alice"}})
        with pytest.raises(VariableResolveError, match="not found"):
            reg.resolve("user.email", state)
