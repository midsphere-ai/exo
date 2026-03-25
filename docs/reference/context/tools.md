# exo.context.tools

Context tools for agent self-management: planning checklists, knowledge retrieval, and file reading.

## Module Path

```python
from exo.context.tools import (
    get_context_tools,
    get_planning_tools,
    get_knowledge_tools,
    get_file_tools,
)
```

---

## Overview

All context tools are `_ContextTool` instances (a `Tool` subclass) that operate on a `Context` passed via `.bind(ctx)`. The `ctx` parameter is excluded from the JSON schema so LLMs only see user-facing parameters.

Tools must be bound to a context before execution:

```python
tools = get_context_tools()
for tool in tools:
    tool.bind(ctx)
```

---

## Factory Functions

### get_context_tools()

```python
def get_context_tools() -> list[Tool]
```

Return all context tools (planning + knowledge + file). Returns 7 tools total.

### get_planning_tools()

```python
def get_planning_tools() -> list[Tool]
```

Return the 3 planning tools: `add_todo`, `complete_todo`, `get_todo`.

### get_knowledge_tools()

```python
def get_knowledge_tools() -> list[Tool]
```

Return the 3 knowledge tools: `get_knowledge`, `grep_knowledge`, `search_knowledge`.

### get_file_tools()

```python
def get_file_tools() -> list[Tool]
```

Return the 1 file tool: `read_file`.

---

## Planning Tools

### add_todo

Add a todo item to the planning checklist.

| Parameter | Type | Description |
|---|---|---|
| `item` | `str` | The todo item text to add |

**Returns:** Confirmation string.

**State effect:** Appends `{"item": item, "done": False}` to the `todos` list in `ctx.state`.

### complete_todo

Mark a todo item as completed by its index.

| Parameter | Type | Description |
|---|---|---|
| `index` | `int` | The 0-based index of the todo item to mark done |

**Returns:** Confirmation or error string.

**State effect:** Sets `done=True` on the todo at the given index.

### get_todo

Retrieve the current todo checklist.

No parameters.

**Returns:** Markdown-formatted checklist, or `"No todos."` if empty.

---

## Knowledge Tools

### get_knowledge

Retrieve a knowledge artifact by name from the workspace attached to context.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | The artifact name to retrieve |

**Returns:** Artifact content, or error message if not found.

**Requires:** `workspace` in `ctx.state`.

### grep_knowledge

Search a knowledge artifact for lines matching a regex pattern.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | The artifact name to search in |
| `pattern` | `str` | A regex pattern to match against each line |

**Returns:** Matching lines with line numbers, or error message.

**Requires:** `workspace` in `ctx.state`.

### search_knowledge

Search across all knowledge artifacts using keyword matching.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | *(required)* | The search query string |
| `top_k` | `int` | `5` | Maximum number of results to return |

**Returns:** Formatted search results with scores, or error message.

**Requires:** `knowledge_store` in `ctx.state`.

---

## File Tools

### read_file

Read a file from the working directory.

| Parameter | Type | Description |
|---|---|---|
| `path` | `str` | Relative path to the file within the working directory |

**Returns:** File content as text, or error message.

**Requires:** `working_dir` in `ctx.state`.

**Security:** Prevents path traversal outside the working directory.

---

## Example

```python
import asyncio
from exo.context import Context
from exo.context.tools import get_planning_tools

async def main():
    ctx = Context("task-1")

    # Get and bind tools
    tools = get_planning_tools()
    for tool in tools:
        tool.bind(ctx)

    # Use the tools
    add_tool = tools[0]      # add_todo
    complete_tool = tools[1]  # complete_todo
    get_tool = tools[2]       # get_todo

    await add_tool.execute(item="Research algorithms")
    await add_tool.execute(item="Write implementation")
    await complete_tool.execute(index=0)

    result = await get_tool.execute()
    print(result)
    # 0. [x] Research algorithms
    # 1. [ ] Write implementation

asyncio.run(main())
```
