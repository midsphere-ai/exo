# Prompt Building

The `PromptBuilder` assembles structured prompts from composable sections called **neurons**. Each neuron produces a text fragment with a priority that controls ordering. The builder sorts neurons by priority (descending -- higher priority appears first), formats each one, joins them, and resolves dynamic `${variable}` templates.

## Basic Usage

```python
from exo.context import Context, ContextConfig, PromptBuilder

ctx = Context(task_id="task-1", config=ContextConfig())
builder = PromptBuilder(ctx)

# Add neurons by name (from the registry)
builder.add("system")
builder.add("task")
builder.add("history")

# Build the assembled prompt string
prompt = builder.build()
print(prompt)
```

## Neurons

A `Neuron` is an abstract base class with three attributes:

- **name** -- unique identifier used for registry lookup.
- **priority** -- integer controlling sort order (higher = earlier in the prompt).
- **format(ctx)** -- method that returns the neuron's text contribution.

```python
from exo.context import Neuron

class CustomNeuron(Neuron):
    name = "custom"
    priority = 15  # between history (10) and knowledge (20)

    def format(self, ctx) -> str:
        user = ctx.state.get("user_name", "User")
        return f"Current user: {user}"
```

### Built-in Neurons

Exo ships with nine built-in neurons, all pre-registered in the `neuron_registry`:

| Neuron | Priority | Description |
|--------|----------|-------------|
| `SystemNeuron` | 100 | System-level instructions (appears first) |
| `EntityNeuron` | 60 | Extracted entities from context |
| `FactNeuron` | 50 | Known facts and assertions |
| `SkillNeuron` | 40 | Available skills/capabilities |
| `WorkspaceNeuron` | 30 | Workspace artifact listing |
| `KnowledgeNeuron` | 20 | Retrieved knowledge chunks |
| `HistoryNeuron` | 10 | Conversation history |
| `TodoNeuron` | 2 | Current TODO items |
| `TaskNeuron` | 1 | Task description (appears last) |

### Priority Ordering

Neurons are sorted by priority **descending** when building the prompt. A neuron with priority 100 appears before one with priority 1. This ensures system instructions come first and task descriptions come last:

```
SystemNeuron (100)  -->  top of prompt
EntityNeuron (60)
FactNeuron (50)
...
TaskNeuron (1)       -->  bottom of prompt
```

## Registering Custom Neurons

Register your neuron in the `neuron_registry` so it can be added by name:

```python
from exo.context import Neuron, neuron_registry

class ToolListNeuron(Neuron):
    name = "tool_list"
    priority = 35

    def format(self, ctx) -> str:
        tools = ctx.state.get("available_tools", [])
        if not tools:
            return ""
        lines = ["Available tools:"]
        for t in tools:
            lines.append(f"  - {t}")
        return "\n".join(lines)

# Register it
neuron_registry.register("tool_list", ToolListNeuron)

# Now usable by name
builder.add("tool_list")
```

## Adding Neurons to the Builder

There are two ways to add neurons:

```python
# 1. By name (looked up in neuron_registry)
builder.add("system")
builder.add("history")

# 2. By instance (for one-off or parameterized neurons)
custom = CustomNeuron()
builder.add_neuron(custom)
```

## Dynamic Variable Substitution

The `PromptBuilder` resolves `${path}` templates in the assembled prompt using the `DynamicVariableRegistry`:

```python
from exo.context.variables import DynamicVariableRegistry

# Register a variable resolver
registry = DynamicVariableRegistry()
registry.register("user.name", lambda state: state.get("user_name", "Unknown"))
registry.register("date.today", lambda state: "2025-01-15")

# In a neuron's format() output:
# "Hello ${user.name}, today is ${date.today}"
# becomes: "Hello Alice, today is 2025-01-15"
```

The variable registry supports dot-separated nested paths. Resolution is done during `builder.build()`.

## Configuration

The `ContextConfig` controls which neurons are included:

```python
from exo.context import ContextConfig

config = ContextConfig(
    neuron_names=["system", "history", "task", "knowledge"],
    history_rounds=10,
    summary_threshold=15,
)
```

The `make_config()` factory provides presets per automation mode:

```python
from exo.context import make_config, AutomationMode

# PILOT mode: fewer neurons, shorter history
pilot = make_config(AutomationMode.PILOT)

# NAVIGATOR mode: all neurons, longer history
navigator = make_config(AutomationMode.NAVIGATOR)
```

## Advanced Patterns

### Conditional Neurons

Return an empty string from `format()` to exclude a neuron from the output:

```python
class ConditionalNeuron(Neuron):
    name = "conditional"
    priority = 25

    def format(self, ctx) -> str:
        data = ctx.state.get("optional_data")
        if not data:
            return ""  # excluded from prompt
        return f"Additional context: {data}"
```

### Multi-Pass Prompt Assembly

Clear and rebuild prompts between agent steps to reflect updated state:

```python
builder = PromptBuilder(ctx)

# Step 1: initial prompt
builder.add("system")
builder.add("task")
prompt_1 = builder.build()

# Step 2: add knowledge after retrieval
builder.clear()
builder.add("system")
builder.add("knowledge")
builder.add("history")
builder.add("task")
prompt_2 = builder.build()
```

### Custom Separator

By default, neurons are joined with `"\n\n"`. Override via the builder's `_separator`:

```python
builder = PromptBuilder(ctx)
builder._separator = "\n---\n"
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `PromptBuilder` | `exo.context` | Assembles prompts from neurons with priority ordering |
| `PromptBuilder.add(name)` | | Add a neuron by registry name |
| `PromptBuilder.add_neuron(neuron)` | | Add a neuron instance directly |
| `PromptBuilder.build()` | | Sort, format, join, and resolve variables |
| `PromptBuilder.clear()` | | Remove all added neurons |
| `Neuron` | `exo.context` | ABC with `name`, `priority`, `format(ctx)` |
| `neuron_registry` | `exo.context` | Global registry of neuron classes |
| `SystemNeuron` | `exo.context.neuron` | System instructions (priority 100) |
| `TaskNeuron` | `exo.context.neuron` | Task description (priority 1) |
| `HistoryNeuron` | `exo.context.neuron` | Conversation history (priority 10) |
| `KnowledgeNeuron` | `exo.context.neuron` | Retrieved knowledge (priority 20) |
| `WorkspaceNeuron` | `exo.context.neuron` | Workspace artifacts (priority 30) |
| `DynamicVariableRegistry` | `exo.context.variables` | Register and resolve `${path}` templates |
