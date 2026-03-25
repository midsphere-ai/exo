# exo.context.variables

Dynamic variable registry with nested path resolution for template substitution.

## Module Path

```python
from exo.context.variables import DynamicVariableRegistry, VariableResolveError
```

---

## VariableResolveError

Exception raised when a variable path cannot be resolved.

```python
class VariableResolveError(Exception): ...
```

---

## DynamicVariableRegistry

Registry of named variable resolvers with nested path support. Variables are registered as dot-separated paths and resolved from a `ContextState` or a flat dict.

### Constructor

```python
DynamicVariableRegistry()
```

No parameters.

### Methods

#### register()

```python
def register(self, path: str, resolver: Any = None) -> Any
```

Register a resolver for a path.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | *(required)* | Dot-separated variable path (e.g. `"user.name"`) |
| `resolver` | `Any` | `None` | Callable `(state) -> value`, static value, or `None` (decorator mode) |

**When `resolver` is not `None`:** Registers it directly, returns the resolver.

**When `resolver` is `None`:** Returns a decorator function.

```python
# Direct registration with callable
reg.register("user.name", lambda state: state.get("user_name", "anon"))

# Direct registration with static value
reg.register("app.version", "1.0.0")

# Decorator form
@reg.register("user.email")
def get_email(state):
    return state.get("email", "unknown@example.com")
```

#### resolve()

```python
def resolve(self, path: str, state: ContextState | dict[str, Any]) -> Any
```

Resolve a variable path to its value.

**Resolution order:**
1. Exact match in registered resolvers (callable invoked with state)
2. Nested path lookup in state (e.g. `"a.b"` maps to `state["a"]["b"]`)

| Parameter | Type | Description |
|---|---|---|
| `path` | `str` | Dot-separated variable path |
| `state` | `ContextState \| dict` | State to resolve from |

**Raises:** `VariableResolveError` if the path cannot be resolved.

#### has()

```python
def has(self, path: str) -> bool
```

Check if a resolver is registered for the path.

#### list_all()

```python
def list_all(self) -> list[str]
```

Return all registered variable paths.

#### resolve_template()

```python
def resolve_template(self, template: str, state: ContextState | dict[str, Any]) -> str
```

Resolve `${path}` placeholders in a template string. Unresolvable variables are left as-is.

| Parameter | Type | Description |
|---|---|---|
| `template` | `str` | Template string with `${path}` placeholders |
| `state` | `ContextState \| dict` | State to resolve from |

**Returns:** The template with resolved values.

### Dunder Methods

| Method | Description |
|---|---|
| `__repr__` | `DynamicVariableRegistry(variables=3)` |

### Example

```python
from exo.context.state import ContextState
from exo.context.variables import DynamicVariableRegistry

reg = DynamicVariableRegistry()

# Register resolvers
reg.register("user.name", lambda state: state.get("user_name", "anon"))
reg.register("app.version", "2.0.0")

# Create state
state = ContextState({"user_name": "Alice", "config": {"debug": True}})

# Resolve individual variables
name = reg.resolve("user.name", state)    # "Alice"
version = reg.resolve("app.version", state)  # "2.0.0"

# Nested path resolution (no explicit resolver needed)
debug = reg.resolve("config.debug", state)  # True

# Template resolution
template = "Hello ${user.name}, running v${app.version}"
result = reg.resolve_template(template, state)
# "Hello Alice, running v2.0.0"

# Unresolvable placeholders are left as-is
template2 = "Hello ${unknown.path}"
result2 = reg.resolve_template(template2, state)
# "Hello ${unknown.path}"
```
