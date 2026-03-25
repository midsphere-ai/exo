# Context Tools

Context tools are `Tool` subclasses that agents can call to interact with the context engine during execution. They provide planning (TODO management), knowledge retrieval (search over ingested documents), and file operations (reading workspace artifacts). Each tool is bound to a `Context` instance at runtime.

## Basic Usage

```python
from exo.context import get_context_tools, Context, ContextConfig

ctx = Context(task_id="task-1", config=ContextConfig())

# Get all context tools
tools = get_context_tools()

# Bind each tool to the context
for tool in tools:
    tool.bind(ctx)

# Pass tools to an agent
from exo.agent import Agent

agent = Agent(
    name="assistant",
    model="openai:gpt-4o",
    tools=tools,
)
```

## Tool Categories

Context tools are organized into three groups, each with its own factory function:

### Planning Tools

```python
from exo.context import get_planning_tools

planning_tools = get_planning_tools()
```

| Tool Name | Description |
|-----------|-------------|
| `add_todo` | Add a TODO item to the context state |
| `complete_todo` | Mark a TODO item as completed |
| `get_todo` | List current TODO items |

Planning tools store TODO items in the context state, allowing agents to maintain task lists across conversation turns.

### Knowledge Tools

```python
from exo.context import get_knowledge_tools

knowledge_tools = get_knowledge_tools()
```

| Tool Name | Description |
|-----------|-------------|
| `get_knowledge` | Retrieve a document by its key |
| `grep_knowledge` | Keyword search across all documents |
| `search_knowledge` | TF-IDF ranked search for relevant documents |

Knowledge tools query the `KnowledgeStore` attached to the context, giving agents access to ingested documents and tool outputs.

### File Tools

```python
from exo.context import get_file_tools

file_tools = get_file_tools()
```

| Tool Name | Description |
|-----------|-------------|
| `read_file` | Read a file from the workspace |

File tools provide read access to workspace artifacts.

### All Context Tools

```python
from exo.context import get_context_tools

# Returns all planning + knowledge + file tools
all_tools = get_context_tools()
```

## Binding Tools to Context

Context tools extend the base `Tool` class with a `bind(ctx)` method. This method must be called before the tool can execute:

```python
from exo.context import get_planning_tools, Context, ContextConfig

ctx = Context(task_id="task-1", config=ContextConfig())
tools = get_planning_tools()

# Bind before use
for tool in tools:
    tool.bind(ctx)

# Now tools can be executed
result = await tools[0].execute(item="Research competitor products")
```

If `execute()` is called before `bind()`, the tool will raise an error.

## Planning Tools in Detail

### add_todo

Adds a TODO item to the context's task list:

```python
# Agent calls: add_todo(item="Write unit tests for auth module")
# Stores in ctx.state under "todos" key
```

### complete_todo

Marks a TODO item as done by index or content match:

```python
# Agent calls: complete_todo(index=0)
# Or: complete_todo(item="Write unit tests for auth module")
```

### get_todo

Returns the current list of TODO items with their completion status:

```python
# Agent calls: get_todo()
# Returns: "1. [ ] Research competitor products\n2. [x] Write tests"
```

## Knowledge Tools in Detail

### get_knowledge

Retrieves a specific document by key:

```python
# Agent calls: get_knowledge(key="readme")
# Returns the full content of the document stored under "readme"
```

### grep_knowledge

Simple keyword matching across all documents:

```python
# Agent calls: grep_knowledge(query="authentication")
# Returns matching document excerpts
```

### search_knowledge

TF-IDF ranked search returning the most relevant documents:

```python
# Agent calls: search_knowledge(query="how to configure OAuth")
# Returns ranked results with relevance scores
```

## Advanced Patterns

### Selective Tool Binding

Only give agents the tools they need:

```python
from exo.context import get_planning_tools, get_knowledge_tools

# Research agent gets knowledge tools
research_agent = Agent(
    name="researcher",
    tools=get_knowledge_tools(),
)

# Planning agent gets planning tools
planner_agent = Agent(
    name="planner",
    tools=get_planning_tools(),
)
```

### Combining with Regular Tools

Mix context tools with regular function tools:

```python
from exo.tool import tool
from exo.context import get_context_tools

@tool
def web_search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

all_tools = get_context_tools() + [web_search]
agent = Agent(name="assistant", tools=all_tools)
```

### Pre-Loading Knowledge

Ingest documents before the agent starts so knowledge tools have content to search:

```python
from exo.context._internal.knowledge import KnowledgeStore, chunk_text

knowledge = KnowledgeStore()

# Index project documentation
for doc_path in project_docs:
    content = doc_path.read_text()
    chunks = chunk_text(content, chunk_size=500, overlap=50)
    for i, chunk in enumerate(chunks):
        knowledge.add(f"{doc_path.stem}_{i}", chunk)

# Create context with knowledge store attached to workspace
workspace = Workspace(knowledge_store=knowledge)

# Knowledge tools will search this store
tools = get_knowledge_tools()
for t in tools:
    t.bind(ctx)
```

### Dynamic Tool Registration

Add tools conditionally based on the automation mode:

```python
from exo.context import make_config, AutomationMode, get_planning_tools, get_knowledge_tools

config = make_config(AutomationMode.PILOT)
tools = []

# Pilot mode: minimal tools
tools.extend(get_planning_tools())

if config.enable_retrieval:
    tools.extend(get_knowledge_tools())
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `get_context_tools()` | `exo.context` | Returns all context tools (planning + knowledge + file) |
| `get_planning_tools()` | `exo.context` | Returns `[add_todo, complete_todo, get_todo]` |
| `get_knowledge_tools()` | `exo.context` | Returns `[get_knowledge, grep_knowledge, search_knowledge]` |
| `get_file_tools()` | `exo.context` | Returns `[read_file]` |
| `_ContextTool` | `exo.context.tools` | Base class with `bind(ctx)` method |
| `_ContextTool.bind(ctx)` | | Bind the tool to a `Context` instance |
