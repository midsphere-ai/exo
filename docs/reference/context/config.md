# exo.context.config

Context configuration with automation modes, history windowing, summarization, and retrieval settings.

## Module Path

```python
from exo.context.config import ContextConfig, AutomationMode, make_config
```

---

## AutomationMode

Automation level for context management.

```python
class AutomationMode(StrEnum):
    PILOT = "pilot"
    COPILOT = "copilot"
    NAVIGATOR = "navigator"
```

| Value | Description |
|---|---|
| `PILOT` | Minimal automation, user controls context manually |
| `COPILOT` | Basic automation -- summarization, offloading, history windowing |
| `NAVIGATOR` | Full automation -- all context features enabled |

---

## ContextConfig

Immutable configuration for the context engine. Controls automation level, history windowing, summarization thresholds, context offloading, retrieval, and neuron selection.

**Inherits:** `pydantic.BaseModel` (frozen)

### Constructor

```python
ContextConfig(
    mode: AutomationMode = AutomationMode.COPILOT,
    history_rounds: int = 20,
    summary_threshold: int = 10,
    offload_threshold: int = 50,
    enable_retrieval: bool = False,
    neuron_names: tuple[str, ...] = (),
    extra: dict[str, Any] = {},
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mode` | `AutomationMode` | `COPILOT` | Automation level |
| `history_rounds` | `int` | `20` | Max conversation rounds to keep (>= 1) |
| `summary_threshold` | `int` | `10` | Message count before triggering summarization (>= 1) |
| `offload_threshold` | `int` | `50` | Message count before offloading older context (>= 1) |
| `enable_retrieval` | `bool` | `False` | Enable RAG retrieval from workspace artifacts |
| `neuron_names` | `tuple[str, ...]` | `()` | Names of neurons to include in prompt building |
| `extra` | `dict[str, Any]` | `{}` | Additional configuration for custom processors or neurons |

### Validation Rules

- `summary_threshold` must be `<=` `offload_threshold`
- `neuron_names` accepts both `list` and `tuple` (auto-coerced to `tuple`)

### Example

```python
from exo.context.config import ContextConfig, AutomationMode

# Default copilot config
config = ContextConfig()

# Navigator with retrieval
config = ContextConfig(
    mode=AutomationMode.NAVIGATOR,
    history_rounds=10,
    summary_threshold=5,
    offload_threshold=20,
    enable_retrieval=True,
    neuron_names=("task", "history", "knowledge", "system"),
)

# Custom extra configuration
config = ContextConfig(
    extra={"custom_processor_key": "value"},
)
```

---

## make_config()

Factory function for creating `ContextConfig` with preset defaults per automation mode.

```python
def make_config(
    mode: AutomationMode | str = "copilot",
    **overrides: Any,
) -> ContextConfig
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mode` | `AutomationMode \| str` | `"copilot"` | Automation level |
| `**overrides` | `Any` | -- | Override any preset default |

**Returns:** A `ContextConfig` with mode-appropriate defaults.

### Preset Defaults

| Setting | Pilot | Copilot | Navigator |
|---|---|---|---|
| `history_rounds` | 100 | 20 | 10 |
| `summary_threshold` | 100 | 10 | 5 |
| `offload_threshold` | 100 | 50 | 20 |
| `enable_retrieval` | `False` | `False` | `True` |

### Example

```python
from exo.context.config import make_config

# Preset navigator config
config = make_config("navigator")
assert config.history_rounds == 10
assert config.enable_retrieval is True

# Override a preset value
config = make_config("copilot", history_rounds=50)
assert config.history_rounds == 50

# Pilot mode (minimal automation)
config = make_config("pilot")
assert config.summary_threshold == 100  # effectively disabled
```
