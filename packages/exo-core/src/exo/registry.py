"""Generic registry for named items."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from exo.types import ExoError

T = TypeVar("T")


class RegistryError(ExoError):
    """Raised on registry operations that fail (duplicate or missing items)."""


class Registry(Generic[T]):
    """A named, typed registry for items.

    Stores items by unique string name with fail-fast duplicate detection.
    Supports both direct registration and decorator-style usage.

    Args:
        name: Human-readable name for error messages.
    """

    def __init__(self, name: str = "registry") -> None:
        self._name = name
        self._items: dict[str, T] = {}

    def _set(self, name: str, item: T) -> None:
        """Store an item, raising on duplicate names."""
        if name in self._items:
            raise RegistryError(f"'{name}' is already registered in {self._name}")
        self._items[name] = item

    def register(self, name: str, item: T | None = None) -> Any:
        """Register an item directly or use as a decorator.

        When ``item`` is provided, registers it immediately. When omitted,
        returns a decorator that registers the decorated object.

        Args:
            name: Unique name for the item.
            item: The item to register (if not using decorator form).

        Returns:
            The item when called directly, or a decorator when ``item`` is None.

        Raises:
            RegistryError: If ``name`` is already registered.
        """
        if item is not None:
            self._set(name, item)
            return item

        def decorator(obj: T) -> T:
            self._set(name, obj)
            return obj

        return decorator

    def get(self, name: str) -> T:
        """Retrieve an item by name.

        Args:
            name: The registered name to look up.

        Returns:
            The registered item.

        Raises:
            RegistryError: If ``name`` is not found.
        """
        if name not in self._items:
            raise RegistryError(f"'{name}' not found in {self._name}")
        return self._items[name]

    def __contains__(self, name: str) -> bool:
        """Check whether a name is registered."""
        return name in self._items

    def list_all(self) -> list[str]:
        """Return all registered names in insertion order."""
        return list(self._items.keys())


agent_registry: Registry[Any] = Registry("agent_registry")
tool_registry: Registry[Any] = Registry("tool_registry")
