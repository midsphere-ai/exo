# exo.registry

Generic named registry for items with duplicate detection and decorator support.

**Module:** `exo.registry`

```python
from exo.registry import Registry, RegistryError, agent_registry, tool_registry
```

---

## RegistryError

```python
class RegistryError(ExoError)
```

Raised on registry operations that fail (duplicate or missing items). Inherits from `ExoError`.

---

## Registry[T]

```python
class Registry(Generic[T])
```

A named, typed registry for items. Stores items by unique string name with fail-fast duplicate detection. Supports both direct registration and decorator-style usage.

### Constructor

```python
def __init__(self, name: str = "registry") -> None
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | `"registry"` | Human-readable name used in error messages. |

### Methods

#### register()

```python
def register(self, name: str, item: T | None = None) -> Any
```

Register an item directly or use as a decorator.

When `item` is provided, registers it immediately. When omitted, returns a decorator that registers the decorated object.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | *(required)* | Unique name for the item. |
| `item` | `T \| None` | `None` | The item to register (if not using decorator form). |

**Returns:** The item when called directly, or a decorator when `item` is `None`.

**Raises:** `RegistryError` if `name` is already registered.

#### get()

```python
def get(self, name: str) -> T
```

Retrieve an item by name.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | *(required)* | The registered name to look up. |

**Returns:** The registered item.

**Raises:** `RegistryError` if `name` is not found.

#### \_\_contains\_\_()

```python
def __contains__(self, name: str) -> bool
```

Check whether a name is registered. Supports `in` operator.

#### list_all()

```python
def list_all(self) -> list[str]
```

Return all registered names in insertion order.

**Returns:** List of name strings.

### Example

```python
from exo.registry import Registry

# Create a typed registry
my_registry: Registry[str] = Registry("my_items")

# Direct registration
my_registry.register("greeting", "hello")

# Decorator-style registration
@my_registry.register("farewell")
def farewell():
    return "goodbye"

# Lookup
value = my_registry.get("greeting")  # "hello"

# Check membership
assert "greeting" in my_registry
assert "unknown" not in my_registry

# List all
names = my_registry.list_all()  # ["greeting", "farewell"]
```

---

## Module-Level Instances

### agent_registry

```python
agent_registry: Registry[Any] = Registry("agent_registry")
```

Global registry for agent instances. Pre-instantiated for framework-wide agent registration.

### tool_registry

```python
tool_registry: Registry[Any] = Registry("tool_registry")
```

Global registry for tool instances. Pre-instantiated for framework-wide tool registration.

### Example

```python
from exo.registry import agent_registry, tool_registry

# Register an agent
agent_registry.register("my_agent", my_agent_instance)

# Register a tool
tool_registry.register("my_tool", my_tool_instance)

# Retrieve
agent = agent_registry.get("my_agent")
tool = tool_registry.get("my_tool")
```
