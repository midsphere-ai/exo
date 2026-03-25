"""Tests for exo.registry — generic registry."""

import pytest

from exo.registry import Registry, RegistryError, agent_registry, tool_registry
from exo.types import ExoError

# --- RegistryError ---


class TestRegistryError:
    def test_inherits_exo_error(self) -> None:
        assert issubclass(RegistryError, ExoError)

    def test_raise_and_catch(self) -> None:
        with pytest.raises(RegistryError, match="not found"):
            raise RegistryError("not found")


# --- Registry basics ---


class TestRegistryBasics:
    def test_register_and_get(self) -> None:
        reg: Registry[int] = Registry("test")
        reg.register("x", 42)
        assert reg.get("x") == 42

    def test_contains(self) -> None:
        reg: Registry[str] = Registry("test")
        reg.register("a", "alpha")
        assert "a" in reg
        assert "b" not in reg

    def test_list_all_insertion_order(self) -> None:
        reg: Registry[int] = Registry("test")
        reg.register("c", 3)
        reg.register("a", 1)
        reg.register("b", 2)
        assert reg.list_all() == ["c", "a", "b"]

    def test_list_all_empty(self) -> None:
        reg: Registry[int] = Registry("test")
        assert reg.list_all() == []

    def test_register_returns_item(self) -> None:
        reg: Registry[str] = Registry("test")
        result = reg.register("x", "hello")
        assert result == "hello"


# --- Decorator form ---


class TestRegistryDecorator:
    def test_decorator_class(self) -> None:
        reg: Registry[type] = Registry("test")

        @reg.register("my_class")
        class MyClass:
            pass

        assert reg.get("my_class") is MyClass

    def test_decorator_function(self) -> None:
        reg: Registry[object] = Registry("test")

        @reg.register("my_func")
        def my_func() -> str:
            return "hello"

        assert reg.get("my_func") is my_func
        assert my_func() == "hello"


# --- Error cases ---


class TestRegistryErrors:
    def test_duplicate_raises(self) -> None:
        reg: Registry[int] = Registry("test")
        reg.register("x", 1)
        with pytest.raises(RegistryError, match="already registered"):
            reg.register("x", 2)

    def test_missing_raises(self) -> None:
        reg: Registry[int] = Registry("test")
        with pytest.raises(RegistryError, match="not found"):
            reg.get("nonexistent")

    def test_error_includes_registry_name(self) -> None:
        reg: Registry[int] = Registry("my_registry")
        with pytest.raises(RegistryError, match="my_registry"):
            reg.get("missing")

    def test_duplicate_error_includes_registry_name(self) -> None:
        reg: Registry[int] = Registry("my_registry")
        reg.register("x", 1)
        with pytest.raises(RegistryError, match="my_registry"):
            reg.register("x", 2)


# --- Global instances ---


class TestGlobalRegistries:
    def test_agent_registry_exists(self) -> None:
        assert isinstance(agent_registry, Registry)

    def test_tool_registry_exists(self) -> None:
        assert isinstance(tool_registry, Registry)

    def test_registries_are_separate(self) -> None:
        assert agent_registry is not tool_registry
