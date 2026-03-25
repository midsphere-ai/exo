# exo_cli.plugins

Plugin system for extending Exo CLI functionality. Plugins hook into CLI lifecycle events (startup, pre-run, post-run, shutdown) and can register additional commands or modify agent configuration.

```python
from exo_cli.plugins import PluginError, PluginHook, PluginManager, PluginSpec
```

**Discovery sources:**

1. **Entry points** -- packages declaring the `exo.plugins` group in their metadata (e.g. `[project.entry-points."exo.plugins"]`).
2. **Python files** -- `.py` files in a plugins directory, each exporting a `plugin` attribute conforming to `PluginSpec`.

---

## PluginHook

```python
class PluginHook(StrEnum)
```

Lifecycle hook points for CLI plugins.

| Value | Description |
|---|---|
| `STARTUP = "startup"` | Called when the CLI starts up |
| `SHUTDOWN = "shutdown"` | Called when the CLI shuts down |
| `PRE_RUN = "pre_run"` | Called before agent execution |
| `POST_RUN = "post_run"` | Called after agent execution |

---

## PluginError

```python
class PluginError(Exception)
```

Raised when plugin loading or lifecycle fails.

---

## PluginSpec

```python
class PluginSpec(
    *,
    name: str,
    hooks: dict[PluginHook, HookFn] | None = None,
    version: str = "0.0.0",
    description: str = "",
)
```

Describes a single plugin with optional lifecycle hooks.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | Unique plugin identifier |
| `hooks` | `dict[PluginHook, HookFn] \| None` | `None` | Mapping of `PluginHook` to async callable |
| `version` | `str` | `"0.0.0"` | Version string |
| `description` | `str` | `""` | One-line description |

### Type alias

```python
HookFn = Callable[..., Coroutine[Any, Any, None]]
```

Async callable invoked at a plugin hook point.

### Properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | Plugin identifier |
| `version` | `str` | Version string |
| `description` | `str` | One-line description |
| `hooks` | `dict[PluginHook, HookFn]` | Copy of the hooks mapping |

### Example

```python
from exo_cli import PluginSpec, PluginHook

async def on_startup(**kwargs):
    print("Plugin starting up!")

async def on_post_run(**kwargs):
    print("Agent run completed!")

plugin = PluginSpec(
    name="my-logger",
    version="1.0.0",
    description="Logs agent lifecycle events",
    hooks={
        PluginHook.STARTUP: on_startup,
        PluginHook.POST_RUN: on_post_run,
    },
)
```

---

## PluginManager

```python
class PluginManager()
```

Discovers, loads, and manages CLI plugin lifecycle. Plugins are identified by name -- duplicates are rejected.

### Properties

| Property | Type | Description |
|---|---|---|
| `plugins` | `dict[str, PluginSpec]` | Mapping of name to loaded plugin specs (copy) |

### Methods

#### get

```python
def get(self, name: str) -> PluginSpec | None
```

Look up a plugin by name.

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | Plugin name |

**Returns:** `PluginSpec` or `None` if not found.

#### register

```python
def register(self, spec: PluginSpec) -> None
```

Register a plugin spec.

| Name | Type | Default | Description |
|---|---|---|---|
| `spec` | `PluginSpec` | *(required)* | Plugin specification to register |

**Raises:** `PluginError` -- If a plugin with the same name is already registered.

#### load_entrypoints

```python
def load_entrypoints(self) -> int
```

Discover and load plugins from `exo.plugins` entry points. Each entry point should reference a `PluginSpec` instance or a callable returning one.

**Returns:** `int` -- Number of plugins successfully loaded.

#### load_directory

```python
def load_directory(self, directory: str | Path) -> int
```

Load plugins from `.py` files in *directory*. Each file must expose a module-level `plugin` attribute that is a `PluginSpec` instance. Files starting with `_` are skipped.

| Name | Type | Default | Description |
|---|---|---|---|
| `directory` | `str \| Path` | *(required)* | Path to the plugins directory |

**Returns:** `int` -- Number of plugins successfully loaded.

**Raises:** `PluginError` -- If the directory does not exist.

#### run_hook

```python
async def run_hook(self, hook: PluginHook, **kwargs: Any) -> None
```

Run a lifecycle hook across all registered plugins. Plugins are invoked in registration order. Errors from a plugin hook propagate immediately.

| Name | Type | Default | Description |
|---|---|---|---|
| `hook` | `PluginHook` | *(required)* | The hook point to run |
| `**kwargs` | `Any` | | Keyword arguments passed to hook functions |

#### startup

```python
async def startup(self, **kwargs: Any) -> None
```

Run the `STARTUP` hook on all plugins.

#### shutdown

```python
async def shutdown(self, **kwargs: Any) -> None
```

Run the `SHUTDOWN` hook on all plugins.

### Example

```python
from exo_cli import PluginManager

manager = PluginManager()

# Discover from installed packages
count = manager.load_entrypoints()
print(f"Loaded {count} plugins from entry points")

# Discover from directory
count = manager.load_directory("/path/to/plugins")
print(f"Loaded {count} plugins from directory")

# Run lifecycle hooks
await manager.startup(config=my_config)

# ... run agent ...

await manager.shutdown()

# Inspect loaded plugins
for name, spec in manager.plugins.items():
    print(f"{name} v{spec.version}: {spec.description}")
```

### Entry point example

In `pyproject.toml`:

```toml
[project.entry-points."exo.plugins"]
my_plugin = "my_package.plugin:plugin"
```

In `my_package/plugin.py`:

```python
from exo_cli import PluginSpec, PluginHook

async def on_startup(**kwargs):
    print("My plugin started!")

plugin = PluginSpec(
    name="my-plugin",
    version="1.0.0",
    hooks={PluginHook.STARTUP: on_startup},
)
```
