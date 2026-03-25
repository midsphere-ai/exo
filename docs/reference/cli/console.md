# exo_cli.console

Interactive REPL console for Exo CLI. Provides a read-eval-print loop for chatting with agents, with slash-command support, streaming output, and Rich formatting.

```python
from exo_cli.console import InteractiveConsole, format_agents_table, parse_command
```

---

## parse_command

```python
def parse_command(text: str) -> tuple[str, str]
```

Parse a slash command from *text*.

| Name | Type | Default | Description |
|---|---|---|---|
| `text` | `str` | *(required)* | Raw input text |

**Returns:** `(command, argument)` tuple. `command` is the slash-command name (lowercase, without `/`) and `argument` is the rest of the line. For non-command input, returns `("", text)`.

### Example

```python
from exo_cli import parse_command

cmd, arg = parse_command("/switch helper")
# cmd = "switch", arg = "helper"

cmd, arg = parse_command("Hello world")
# cmd = "", arg = "Hello world"

cmd, arg = parse_command("/help")
# cmd = "help", arg = ""
```

---

## format_agents_table

```python
def format_agents_table(agents: dict[str, Any]) -> Table
```

Build a Rich `Table` listing available agents with columns: Name, Model, and Description.

| Name | Type | Default | Description |
|---|---|---|---|
| `agents` | `dict[str, Any]` | *(required)* | Mapping of agent name to agent instance |

**Returns:** `rich.table.Table` -- Formatted table ready for printing.

### Example

```python
from exo_cli import format_agents_table
from rich.console import Console

table = format_agents_table({"helper": agent, "coder": code_agent})
Console().print(table)
```

---

## InteractiveConsole

```python
class InteractiveConsole(
    *,
    agents: dict[str, Any],
    run_fn: RunFn,
    stream_fn: StreamFn | None = None,
    console: RichConsole | None = None,
    streaming: bool = False,
)
```

Interactive REPL for chatting with Exo agents. Supports slash commands for switching agents, listing agents, clearing the screen, and getting agent info.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `agents` | `dict[str, Any]` | *(required)* | Mapping of agent name to agent instance (at least one required) |
| `run_fn` | `RunFn` | *(required)* | Async callable `run(agent, input, **kw) -> result` |
| `stream_fn` | `StreamFn \| None` | `None` | Optional async-iterator factory for streaming responses |
| `console` | `RichConsole \| None` | `None` | Rich console (default: stderr for clean piping) |
| `streaming` | `bool` | `False` | Enable streaming mode (requires `stream_fn`) |

**Raises:** `ValueError` -- If `agents` is empty.

### Type aliases

```python
RunFn = Callable[..., Coroutine[Any, Any, Any]]
StreamFn = Callable[..., AsyncIterator[StreamEvent]]
```

### Properties

| Property | Type | Description |
|---|---|---|
| `current_agent_name` | `str` | Name of the currently selected agent |
| `current_agent` | `Any` | The currently selected agent instance |
| `agents` | `dict[str, Any]` | Copy of the agents registry |

### Methods

#### start

```python
async def start(self) -> None
```

Run the interactive REPL until exit. Reads lines from stdin, dispatches slash commands, and sends other input to the current agent.

### Slash commands

| Command | Description |
|---|---|
| `/help` | Show help message |
| `/exit`, `/quit` | Exit the console |
| `/agents` | List available agents |
| `/switch <name>` | Switch to a different agent |
| `/clear` | Clear the screen |
| `/info` | Show current agent info |

### Example

```python
from exo_cli import InteractiveConsole
from exo.runner import run

console = InteractiveConsole(
    agents={"helper": my_agent, "coder": code_agent},
    run_fn=run,
    streaming=False,
)
await console.start()
```
