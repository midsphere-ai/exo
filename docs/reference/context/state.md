# exo.context.state

Hierarchical key-value state with parent inheritance.

## Module Path

```python
from exo.context.state import ContextState
```

---

## ContextState

Hierarchical key-value state with parent chain lookup. Reads search local data first, then walk up the parent chain. Writes always target local data only.

### Constructor

```python
ContextState(
    initial: dict[str, Any] | None = None,
    *,
    parent: ContextState | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `initial` | `dict[str, Any] \| None` | `None` | Initial local key-value pairs |
| `parent` | `ContextState \| None` | `None` | Parent state for inheritance chain |

### Properties

| Property | Type | Description |
|---|---|---|
| `parent` | `ContextState \| None` | The parent state, if any |

### Read Methods

#### get()

```python
def get(self, key: str, default: Any = None) -> Any
```

Get value by key, searching local then parent chain. Returns `default` if not found anywhere.

#### `__getitem__`

```python
def __getitem__(self, key: str) -> Any
```

Dict-style access. Raises `KeyError` if not found in local or parent chain.

#### `__contains__`

```python
def __contains__(self, key: object) -> bool
```

Check if key exists in local or parent chain.

### Write Methods (Local Only)

#### set()

```python
def set(self, key: str, value: Any) -> None
```

Set a value in local state only.

#### `__setitem__`

```python
def __setitem__(self, key: str, value: Any) -> None
```

Dict-style write. Always writes to local state.

#### update()

```python
def update(
    self,
    mapping: dict[str, Any] | ContextState | None = None,
    **kwargs: Any,
) -> None
```

Batch-update local state from a dict, another `ContextState`, or keyword arguments.

#### delete()

```python
def delete(self, key: str) -> None
```

Delete a key from local state. Raises `KeyError` if not in local.

#### pop()

```python
def pop(self, key: str, *args: Any) -> Any
```

Remove and return value from local state.

#### clear()

```python
def clear(self) -> None
```

Clear all local state. Does not affect parent.

### Introspection Methods

#### local_dict()

```python
def local_dict(self) -> dict[str, Any]
```

Return a copy of local-only data (no parent).

#### to_dict()

```python
def to_dict(self) -> dict[str, Any]
```

Return merged dict: parent values overridden by local values. Recursively merges the entire parent chain.

#### keys()

```python
def keys(self) -> set[str]
```

All accessible keys (local + parent chain).

### Dunder Methods

| Method | Description |
|---|---|
| `__len__` | Number of accessible keys (local + inherited) |
| `__iter__` | Iterate over all accessible keys |
| `__bool__` | `True` if any data exists locally or in parents |
| `__repr__` | `ContextState(local=N, inherited=M)` |

### Example

```python
from exo.context.state import ContextState

# Parent state
parent = ContextState({"color": "blue", "size": 10})

# Child state inherits from parent
child = ContextState(parent=parent)

# Reads search up the chain
assert child.get("color") == "blue"
assert "color" in child

# Writes are local
child.set("color", "red")
assert child.get("color") == "red"         # local wins
assert parent.get("color") == "blue"       # parent unchanged

# local_dict vs to_dict
assert child.local_dict() == {"color": "red"}
assert child.to_dict() == {"color": "red", "size": 10}

# Batch update
child.update({"x": 1, "y": 2})
assert child.get("x") == 1

# Delete from local
child.delete("color")
assert child.get("color") == "blue"  # falls back to parent

# Keys include inherited
assert child.keys() == {"color", "size", "x", "y"}
```

### Parent Chain Behavior

```
grandparent = ContextState({"a": 1})
parent      = ContextState({"b": 2}, parent=grandparent)
child       = ContextState({"c": 3}, parent=parent)

child.get("a")  # 1 (from grandparent)
child.get("b")  # 2 (from parent)
child.get("c")  # 3 (local)

child.to_dict()  # {"a": 1, "b": 2, "c": 3}
child.local_dict()  # {"c": 3}
```
