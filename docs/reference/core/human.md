# exo.human

Human-in-the-loop tool for agent execution with human oversight.

**Module:** `exo.human`

```python
from exo.human import HumanInputHandler, ConsoleHandler, HumanInputTool
```

---

## HumanInputHandler (ABC)

```python
class HumanInputHandler(ABC)
```

Protocol for receiving input from a human during agent execution. Implement this to provide custom input mechanisms (console, web UI, Slack bot, etc.). The handler is called when the agent invokes the `HumanInputTool` to request user confirmation or free-form input.

### Methods

#### get_input() *(abstract)*

```python
@abstractmethod
async def get_input(self, prompt: str, choices: list[str] | None = None) -> str
```

Request input from a human.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `prompt` | `str` | *(required)* | The question or instruction shown to the human. |
| `choices` | `list[str] \| None` | `None` | Optional list of valid choices. If provided, the handler should constrain input to these values. |

**Returns:** The human's response string.

### Example: Custom Handler

```python
from exo.human import HumanInputHandler

class SlackHandler(HumanInputHandler):
    async def get_input(self, prompt: str, choices: list[str] | None = None) -> str:
        # Send prompt to Slack channel, wait for response
        response = await send_slack_message_and_wait(prompt, choices)
        return response
```

---

## ConsoleHandler

```python
class ConsoleHandler(HumanInputHandler)
```

Interactive console handler that reads from stdin. Displays the prompt to stderr and reads a line from stdin. When choices are provided, validates the input against them. If the input does not match any choice, defaults to the first choice.

### Methods

#### get_input()

```python
async def get_input(self, prompt: str, choices: list[str] | None = None) -> str
```

Read input from the console.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `prompt` | `str` | *(required)* | The question to display. |
| `choices` | `list[str] \| None` | `None` | Optional valid choices to display and validate. |

**Returns:** The user's input string. If choices are provided and the input is invalid, returns the first choice.

### Example

```python
from exo.human import ConsoleHandler

handler = ConsoleHandler()
# When run, this prints to stderr and reads from stdin
# response = await handler.get_input("Continue?", choices=["yes", "no"])
```

---

## HumanInputTool

```python
class HumanInputTool(Tool)
```

A tool that pauses agent execution to request human input. When the LLM calls this tool, execution blocks until the human responds via the configured `HumanInputHandler`.

### Constructor

```python
def __init__(
    self,
    *,
    handler: HumanInputHandler | None = None,
    timeout: float | None = None,
) -> None
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `handler` | `HumanInputHandler \| None` | `None` | The input handler to use. Defaults to `ConsoleHandler()`. |
| `timeout` | `float \| None` | `None` | Maximum seconds to wait for input. `None` means no timeout. |

### Tool Schema

The tool registers with the following schema:

- **name:** `"human_input"`
- **description:** `"Ask a human for input, confirmation, or clarification."`
- **parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | `string` | Yes | The question or instruction to show the human. |
| `choices` | `array` of `string` | No | Optional list of valid choices for the human to pick from. |

### Methods

#### execute()

```python
async def execute(self, **kwargs: Any) -> str
```

Execute the human input request.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `prompt` | `str` | `""` | The question to present (from `**kwargs`). |
| `choices` | `list[str] \| None` | `None` | Optional valid choices (from `**kwargs`). |

**Returns:** The human's response string.

**Raises:** `ToolError` if the input request times out.

### Example

```python
from exo.agent import Agent
from exo.human import HumanInputTool, ConsoleHandler

# Default console handler
hitl = HumanInputTool()

# With timeout
hitl = HumanInputTool(timeout=60.0)

# With custom handler
hitl = HumanInputTool(handler=my_custom_handler, timeout=120.0)

# Add to an agent
agent = Agent(
    name="assistant",
    instructions="Ask for human confirmation before executing dangerous operations.",
    tools=[hitl],
)
```

### Full Example: Agent with Human Oversight

```python
from exo import Agent, tool, run
from exo.human import HumanInputTool

@tool
def delete_file(path: str) -> str:
    """Delete a file from the filesystem.

    Args:
        path: Path to the file to delete.
    """
    import os
    os.remove(path)
    return f"Deleted {path}"

agent = Agent(
    name="file_manager",
    instructions=(
        "You manage files. Before deleting any file, "
        "use the human_input tool to get confirmation."
    ),
    tools=[delete_file, HumanInputTool()],
)

# The agent will ask for human confirmation before deletions
# result = run.sync(agent, "Delete /tmp/old_data.csv")
```

---

## ToolContext.require_approval()

On-demand HITL approval inside tools. Unlike `HumanInputTool` (where the LLM decides when to ask), `require_approval()` lets the tool itself gate execution based on its own logic.

### Method

```python
async def require_approval(self, message: str = "Approve this operation?") -> None
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `message` | `str` | `"Approve this operation?"` | Prompt shown to the human. |

**Returns:** `None` (execution continues on approval).

**Raises:** `ToolError` if no handler is set or if the human denies.

### Setup

```python
from exo import Agent, tool, ToolContext, ConsoleHandler

@tool
async def run_command(command: str, ctx: ToolContext) -> str:
    """Execute a shell command.

    Args:
        command: The shell command to run.
    """
    if any(p in command for p in ["rm ", "sudo "]):
        await ctx.require_approval(f"Sensitive: {command}\nApprove?")
    return subprocess.check_output(command, shell=True, text=True)

agent = Agent(
    name="ops",
    tools=[run_command],
    human_input_handler=ConsoleHandler(),
)
```

### Agent Parameter

```python
Agent(
    human_input_handler=ConsoleHandler(),  # or any HumanInputHandler
)
```

The handler is passed through to `ToolContext` when tools are executed. Without it, `require_approval()` raises `ToolError`.
