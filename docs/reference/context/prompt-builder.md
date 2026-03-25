# exo.context.prompt_builder

Compose neurons in priority order to build rich system prompts.

## Module Path

```python
from exo.context.prompt_builder import PromptBuilder, PromptBuilderError
```

---

## PromptBuilderError

Exception raised for prompt building failures (e.g., neuron not found in registry).

```python
class PromptBuilderError(Exception): ...
```

---

## PromptBuilder

Composes neurons in priority order to build system prompts. Neurons are resolved from the global `neuron_registry` by name and formatted in ascending priority order.

### Constructor

```python
PromptBuilder(
    ctx: Context,
    *,
    variables: DynamicVariableRegistry | None = None,
    separator: str = "\n\n",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ctx` | `Context` | *(required)* | The context to pass to each neuron's `format()` |
| `variables` | `DynamicVariableRegistry \| None` | `None` | Optional variable registry for `${path}` template resolution |
| `separator` | `str` | `"\n\n"` | String used to join neuron outputs |

### Properties

| Property | Type | Description |
|---|---|---|
| `ctx` | `Context` | The context used for neuron formatting |

### Methods

#### add()

```python
def add(self, neuron_name: str, **kwargs: Any) -> PromptBuilder
```

Register a neuron by name for inclusion in the prompt. The neuron is resolved from `neuron_registry` immediately. Extra kwargs are passed to the neuron's `format()` call.

Returns `self` for method chaining.

| Parameter | Type | Description |
|---|---|---|
| `neuron_name` | `str` | Name of the neuron in the registry |
| `**kwargs` | `Any` | Extra arguments forwarded to `neuron.format()` |

**Raises:** `PromptBuilderError` if the neuron name is not found in the registry.

#### add_neuron()

```python
def add_neuron(self, neuron: Neuron, **kwargs: Any) -> PromptBuilder
```

Register a neuron instance directly (bypassing the registry). Returns `self` for method chaining.

| Parameter | Type | Description |
|---|---|---|
| `neuron` | `Neuron` | A neuron instance |
| `**kwargs` | `Any` | Extra arguments forwarded to `neuron.format()` |

#### build()

```python
async def build(self) -> str
```

Resolve all neurons in priority order and compose the final prompt.

**Steps:**
1. Sort entries by neuron priority (ascending -- lower = earlier)
2. Call each neuron's `format(ctx, **kwargs)`
3. Filter out empty results
4. Join non-empty fragments with the separator
5. If a variable registry is set, resolve `${path}` templates

**Returns:** The assembled prompt string.

#### clear()

```python
def clear(self) -> None
```

Remove all registered neuron entries.

### Dunder Methods

| Method | Description |
|---|---|
| `__len__` | Number of registered neuron entries |
| `__repr__` | `PromptBuilder(neurons=['task', 'history', 'system'])` |

### Example

```python
import asyncio
from exo.context import Context, PromptBuilder

async def main():
    ctx = Context("task-1")
    ctx.state.set("task_input", "Explain quantum computing")
    ctx.state.set("history", [
        {"role": "user", "content": "What is a qubit?"},
        {"role": "assistant", "content": "A qubit is..."},
    ])

    # Method chaining
    builder = PromptBuilder(ctx)
    prompt = await (
        builder
        .add("task")       # priority 1 -- appears first
        .add("history")    # priority 10
        .add("system")     # priority 100 -- appears last
        .build()
    )

    print(prompt)
    # <task_info>
    # Task ID: task-1
    # Input: Explain quantum computing
    # </task_info>
    #
    # <conversation_history>
    # [user]: What is a qubit?
    # [assistant]: A qubit is...
    # </conversation_history>
    #
    # <system_info>
    # Current date: 2025-01-15
    # ...
    # </system_info>

asyncio.run(main())
```

### Template Variable Resolution

When a `DynamicVariableRegistry` is provided, `${path}` placeholders in the final prompt are resolved:

```python
from exo.context.variables import DynamicVariableRegistry

variables = DynamicVariableRegistry()
variables.register("user.name", lambda state: state.get("user_name", "User"))

builder = PromptBuilder(ctx, variables=variables)
# If a neuron produces "Hello ${user.name}", and state has user_name="Alice",
# the final prompt will contain "Hello Alice"
```
