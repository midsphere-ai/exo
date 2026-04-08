# Human-in-the-Loop

Human-in-the-loop (HITL) allows agents to pause execution and request input from a human. This is essential for workflows that require confirmation before taking actions, gathering additional context, or providing oversight for autonomous systems.

## Basic Usage

Add a `HumanInputTool` to your agent's tool list:

```python
from exo.agent import Agent
from exo.human import HumanInputTool
from exo.runner import run

agent = Agent(
    name="careful_agent",
    instructions=(
        "You help users manage files. Before deleting anything, "
        "use the human_input tool to confirm with the user."
    ),
    tools=[HumanInputTool()],
)

result = await run(agent, "Delete the temp files in /var/log")
# Agent will pause and ask: "Are you sure you want to delete files in /var/log?"
# User types "yes" or "no" in the console
```

## HumanInputTool

The `HumanInputTool` is a `Tool` subclass that pauses agent execution and requests human input via a configurable handler.

### Constructor

```python
class HumanInputTool(Tool):
    def __init__(
        self,
        *,
        handler: HumanInputHandler | None = None,
        timeout: float | None = None,
    ) -> None: ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `handler` | `HumanInputHandler \| None` | `None` | Input handler (defaults to `ConsoleHandler`) |
| `timeout` | `float \| None` | `None` | Maximum seconds to wait for input. `None` means no timeout |

### Tool Schema

The tool is presented to the LLM with this schema:

```json
{
  "name": "human_input",
  "description": "Ask a human for input, confirmation, or clarification.",
  "parameters": {
    "type": "object",
    "properties": {
      "prompt": {
        "type": "string",
        "description": "The question or instruction to show the human."
      },
      "choices": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Optional list of valid choices for the human to pick from."
      }
    },
    "required": ["prompt"]
  }
}
```

### Usage with Choices

The LLM can provide a list of valid choices to constrain the human's response:

```python
# The LLM might call:
# human_input(prompt="How should I proceed?", choices=["continue", "abort", "modify"])
```

When choices are provided and the human enters an invalid value, `ConsoleHandler` defaults to the first choice.

### Timeout

Set a timeout to prevent indefinite blocking:

```python
from exo.human import HumanInputTool
from exo.tool import ToolError

tool = HumanInputTool(timeout=30.0)  # 30-second timeout

# If the human doesn't respond in time:
# ToolError: "Human input request timed out"
```

## HumanInputHandler ABC

The `HumanInputHandler` abstract base class defines the interface for receiving human input. Implement it to integrate with any input mechanism.

```python
from exo.human import HumanInputHandler

class HumanInputHandler(ABC):
    @abstractmethod
    async def get_input(
        self,
        prompt: str,
        choices: list[str] | None = None,
    ) -> str:
        """Request input from a human.

        Args:
            prompt: The question or instruction shown to the human.
            choices: Optional list of valid choices.

        Returns:
            The human's response string.
        """
```

## ConsoleHandler

The built-in `ConsoleHandler` reads from stdin and writes prompts to stderr:

```python
from exo.human import ConsoleHandler

handler = ConsoleHandler()
response = await handler.get_input(
    "Should we proceed?",
    choices=["yes", "no"],
)
# Displays to stderr:
#   Should we proceed?
#   Choices: yes, no
#   >
# Reads a line from stdin
```

The `ConsoleHandler` runs blocking stdin reads in a thread via `asyncio.to_thread()` to avoid blocking the event loop.

## Custom Handlers

### Web UI Handler

```python
from exo.human import HumanInputHandler

class WebUIHandler(HumanInputHandler):
    def __init__(self, websocket):
        self._ws = websocket

    async def get_input(self, prompt: str, choices: list[str] | None = None) -> str:
        await self._ws.send_json({
            "type": "input_request",
            "prompt": prompt,
            "choices": choices,
        })
        response = await self._ws.receive_json()
        return response["value"]

agent = Agent(
    name="web_agent",
    tools=[HumanInputTool(handler=WebUIHandler(ws))],
)
```

### Slack Bot Handler

```python
class SlackHandler(HumanInputHandler):
    def __init__(self, slack_client, channel_id: str):
        self._client = slack_client
        self._channel = channel_id

    async def get_input(self, prompt: str, choices: list[str] | None = None) -> str:
        message = prompt
        if choices:
            message += f"\nOptions: {', '.join(choices)}"

        await self._client.post_message(self._channel, message)
        # Wait for a reply in the channel
        response = await self._client.wait_for_reply(self._channel)
        return response.text
```

### Auto-Approve Handler (for Testing)

```python
class AutoApproveHandler(HumanInputHandler):
    async def get_input(self, prompt: str, choices: list[str] | None = None) -> str:
        if choices:
            return choices[0]  # Always pick the first choice
        return "approved"

# In tests
agent = Agent(
    name="test_agent",
    tools=[HumanInputTool(handler=AutoApproveHandler())],
)
```

## Tool-Level Approval with `require_approval()`

Instead of relying on the LLM to call a separate `HumanInputTool`, tools can gate themselves on human approval using `ToolContext.require_approval()`. The tool decides when approval is needed based on its own logic.

### Setup

Set `human_input_handler` on the Agent:

```python
from exo import Agent, tool, ToolContext, ConsoleHandler

@tool
async def delete_file(path: str, ctx: ToolContext) -> str:
    """Delete a file from the filesystem.

    Args:
        path: Path to the file to delete.
    """
    await ctx.require_approval(f"About to delete {path}. Approve?")
    import os
    os.remove(path)
    return f"Deleted {path}"

agent = Agent(
    name="file_manager",
    tools=[delete_file],
    human_input_handler=ConsoleHandler(),
)
```

### Conditional Approval

The power of `require_approval()` is that the tool controls when it triggers:

```python
@tool
async def run_sql(query: str, ctx: ToolContext) -> str:
    """Execute a SQL query.

    Args:
        query: The SQL query to execute.
    """
    # Only gate writes, not reads
    if not query.strip().upper().startswith("SELECT"):
        await ctx.require_approval(f"Non-SELECT query:\n{query}\nApprove?")
    return execute_query(query)
```

### Behavior

- **Approved:** Execution continues normally
- **Denied:** Raises `ToolError("Tool execution denied by human")` — the LLM sees this and can adjust
- **No handler:** Raises `ToolError` immediately — never silently skips approval

### When to Use Which

| Approach | Best For |
|----------|----------|
| `HumanInputTool` | LLM-directed: agent decides when to ask the human |
| `ctx.require_approval()` | Tool-directed: tool gates itself based on arguments |

## Patterns

### Confirmation Before Dangerous Actions

```python
agent = Agent(
    name="admin_agent",
    instructions="""You manage server infrastructure.
    ALWAYS use human_input to confirm before:
    - Deleting resources
    - Modifying production configs
    - Restarting services
    Use choices ["confirm", "cancel"] for confirmation prompts.""",
    tools=[
        HumanInputTool(),
        delete_resource_tool,
        restart_service_tool,
    ],
)
```

### Gathering Missing Information

```python
agent = Agent(
    name="form_agent",
    instructions="""You help users fill out forms.
    Use human_input to ask for any required information that
    the user hasn't provided yet.""",
    tools=[HumanInputTool(), submit_form_tool],
)
```

### Multi-Step Approval Workflow

```python
agent = Agent(
    name="approval_agent",
    instructions="""You process expense reports.
    1. Summarize the expense report
    2. Ask for manager approval via human_input with choices ["approve", "reject", "revise"]
    3. If approved, submit. If rejected, notify the submitter. If revise, ask for details.""",
    tools=[HumanInputTool(), submit_expense_tool, notify_tool],
)
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `HumanInputTool` | `exo.human` | Tool that pauses for human input (LLM-directed) |
| `HumanInputHandler` | `exo.human` | ABC for custom input mechanisms |
| `ConsoleHandler` | `exo.human` | Built-in stdin/stderr handler |
| `ToolContext.require_approval()` | `exo.tool_context` | On-demand approval gate inside tools (tool-directed) |
| `Agent(human_input_handler=...)` | `exo.agent` | Sets the handler used by `require_approval()` |
