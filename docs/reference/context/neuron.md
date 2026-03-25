# exo.context.neuron

Modular prompt composition components. A Neuron is a composable unit that produces a prompt fragment from context, ordered by priority.

## Module Path

```python
from exo.context.neuron import Neuron, neuron_registry
```

---

## neuron_registry

```python
neuron_registry: Registry[Neuron]
```

Global registry mapping names to `Neuron` instances. Built-in registrations:

| Name | Class | Priority | Description |
|---|---|---|---|
| `"system"` | `SystemNeuron` | 100 | Date, time, platform info |
| `"task"` | `TaskNeuron` | 1 | Task ID, input, output, plan |
| `"history"` | `HistoryNeuron` | 10 | Windowed conversation history |
| `"todo"` | `TodoNeuron` | 2 | Todo/checklist items |
| `"knowledge"` | `KnowledgeNeuron` | 20 | Knowledge base snippets |
| `"workspace"` | `WorkspaceNeuron` | 30 | Workspace artifact summaries |
| `"skill"` | `SkillNeuron` | 40 | Available skill descriptions |
| `"fact"` | `FactNeuron` | 50 | Extracted facts |
| `"entity"` | `EntityNeuron` | 60 | Named entities |

---

## Neuron (ABC)

Abstract base for prompt neurons. Subclasses implement `format()` to produce a prompt fragment from the given context.

### Constructor

```python
Neuron(name: str, *, priority: int = 50)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | Human-readable name for registry and debugging |
| `priority` | `int` | `50` | Ordering priority (lower = earlier in prompt) |

### Properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | The neuron name |
| `priority` | `int` | Ordering priority |

### Abstract Methods

#### format()

```python
async def format(self, ctx: Context, **kwargs: Any) -> str
```

Produce a prompt fragment from `ctx`. Returns an empty string to signal "nothing to contribute".

| Parameter | Type | Description |
|---|---|---|
| `ctx` | `Context` | The context to extract prompt data from |
| `**kwargs` | `Any` | Extra arguments passed via `PromptBuilder.add()` |

**Returns:** A prompt fragment string.

---

## Built-in Neurons

### SystemNeuron

Provides dynamic system variables: date, time, platform. Priority **100** (low, appended near the end).

```python
SystemNeuron(name: str = "system", *, priority: int = 100)
```

**Output format:**

```xml
<system_info>
Current date: 2025-01-15
Current time: 14:30:00 UTC
Platform: Linux 6.1.0
</system_info>
```

**Reads from state:** Nothing (uses system clock and platform).

### TaskNeuron

Provides task context: task ID, input, output, subtask plan. Priority **1** (highest, appears first).

```python
TaskNeuron(name: str = "task", *, priority: int = 1)
```

**Reads from state:**

| Key | Type | Description |
|---|---|---|
| `task_input` | `str` | Current task input text |
| `task_output` | `str` | Partial output so far |
| `subtasks` | `list[str]` | List of subtask descriptions (plan) |

**Output format:**

```xml
<task_info>
Task ID: task-123
Input: Write a sorting function
Plan:
  <step1>Research algorithms</step1>
  <step2>Implement quicksort</step2>
</task_info>
```

### HistoryNeuron

Provides windowed conversation history. Priority **10**.

```python
HistoryNeuron(name: str = "history", *, priority: int = 10)
```

**Reads from state:**

| Key | Type | Description |
|---|---|---|
| `history` | `list[dict[str, Any]]` | Message dicts with `role` and `content` keys |

Uses `ctx.config.history_rounds` to limit rounds (each round = 2 messages).

**Output format:**

```xml
<conversation_history>
[user]: What is Python?
[assistant]: Python is a programming language.
</conversation_history>
```

### TodoNeuron

Provides todo/checklist items. Priority **2**.

```python
TodoNeuron(name: str = "todo", *, priority: int = 2)
```

**Reads from state:**

| Key | Type | Description |
|---|---|---|
| `todos` | `list[dict[str, Any]]` | Dicts with `item` (str) and optional `done` (bool) keys |

**Output format:**

```xml
<todo_list>
  [x] Research algorithms
  [ ] Implement solution
</todo_list>
```

### KnowledgeNeuron

Provides knowledge base snippets. Priority **20**.

```python
KnowledgeNeuron(name: str = "knowledge", *, priority: int = 20)
```

**Reads from state:**

| Key | Type | Description |
|---|---|---|
| `knowledge_items` | `list[dict[str, str]]` | Dicts with `source` and `content` keys |

### WorkspaceNeuron

Provides workspace artifact summaries. Priority **30**.

```python
WorkspaceNeuron(name: str = "workspace", *, priority: int = 30)
```

**Reads from state:**

| Key | Type | Description |
|---|---|---|
| `workspace_artifacts` | `list[dict[str, Any]]` | Dicts with `name`, optional `type` and `size` keys |

### SkillNeuron

Provides available skill descriptions. Priority **40**.

```python
SkillNeuron(name: str = "skill", *, priority: int = 40)
```

**Reads from state:**

| Key | Type | Description |
|---|---|---|
| `skills` | `list[dict[str, Any]]` | Dicts with `name`, `description`, and optional `active` (bool) keys |

### FactNeuron

Provides extracted facts. Priority **50**.

```python
FactNeuron(name: str = "fact", *, priority: int = 50)
```

**Reads from state:**

| Key | Type | Description |
|---|---|---|
| `facts` | `list[str]` | List of fact strings |

### EntityNeuron

Provides named entities. Priority **60**.

```python
EntityNeuron(name: str = "entity", *, priority: int = 60)
```

**Reads from state:**

| Key | Type | Description |
|---|---|---|
| `entities` | `list[dict[str, str]]` | Dicts with `name` and `type` keys |

---

## Custom Neuron Example

```python
from exo.context.neuron import Neuron, neuron_registry
from exo.context.context import Context

class WeatherNeuron(Neuron):
    def __init__(self):
        super().__init__("weather", priority=70)

    async def format(self, ctx: Context, **kwargs) -> str:
        weather = ctx.state.get("weather")
        if not weather:
            return ""
        return f"<weather>Current weather: {weather}</weather>"

# Register it
neuron_registry.register("weather", WeatherNeuron())

# Use with PromptBuilder
builder = PromptBuilder(ctx)
builder.add("task").add("weather").add("system")
prompt = await builder.build()
```
