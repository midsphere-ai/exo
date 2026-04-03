<div align="center">

<br>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo-light.svg">
  <img alt="Exo" src="docs/assets/logo-light.svg" width="560">
</picture>

<br><br>

<h3>Smart agents by default, not by configuration.</h3>

<br>

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e.svg?style=flat-square)](LICENSE)
[![UV Workspace](https://img.shields.io/badge/uv-workspace-DE5FE9.svg?style=flat-square&logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-E92063.svg?style=flat-square&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)

<br>

[Docs](docs/getting-started/) · [Examples](examples/) · [API Reference](docs/reference/)

<br>

```
                         ✦
              .    ·  ✧       .    ·
          ·        E X O S P H E R E        ·
       .    The outermost layer.    .
            Invisible, but everything     ✧
     ✦      above it depends on it.
          ·    .        ·    .        ·
                    ✦
```

</div>

<br>

---

<br>

```python
from exo import Agent, run

agent = Agent(name="assistant", model="openai:gpt-4o")

result = run.sync(agent, "What should I know about quantum computing?")
```

<div align="center">
<table>
<tr><td>

**This 3-line agent already has:**

</td></tr>
<tr><td>

```
  ╭─ MEMORY ──────────────────────────────────────────────────╮
  │  Short-term history + ChromaDB vector long-term storage   │
  │  Relevant memories auto-injected as <knowledge> blocks    │
  ╰───────────────────────────────────────────────────────────╯
  ╭─ CONTEXT ENGINE ──────────────────────────────────────────╮
  │  9 neurons dynamically compose the system prompt from     │
  │  task state, history, workspace, facts, entities, todos   │
  ╰───────────────────────────────────────────────────────────╯
  ╭─ PRESSURE MANAGEMENT ────────────────────────────────────╮
  │  Processor pipeline compresses tool chains, offloads      │
  │  oversized messages, triggers LLM summarization           │
  ╰───────────────────────────────────────────────────────────╯
  ╭─ BUDGET AWARENESS ───────────────────────────────────────╮
  │  Per-agent token tracking, auto-compression at limits     │
  ╰───────────────────────────────────────────────────────────╯
  ╭─ WORKSPACE ──────────────────────────────────────────────╮
  │  Artifacts auto-chunked & TF-IDF indexed for retrieval    │
  │  Agent gets search_knowledge, add_todo, read_file tools   │
  ╰───────────────────────────────────────────────────────────╯
```

</td></tr>
<tr><td>

*You didn't configure any of that. It's just what an agent should be.*

</td></tr>
</table>
</div>

<br>

## Install

```bash
pip install exo              # everything
pip install exo-core         # just the agent runtime
```

> Requires Python 3.11+

<br>

---

<div align="center">

```
     ·    .    ·        ✧        ·    .    ·
              W H Y   E X O
     .    ·        .        ·    .        ✦
```

</div>

Other frameworks give you a blank agent and a long checklist: wire up memory, manage context windows, handle token budgets, write summarization logic, build knowledge retrieval. You spend weeks on plumbing before your agent does anything interesting.

Exo agents are intelligent on line one. The agent has a cognitive architecture — not just a prompt and a tool loop, but a full system for building context, managing memory, reasoning about its own state, and operating under token pressure.

```python
# Full cognitive architecture: neurons, memory, context processors,
# knowledge retrieval, workspace indexing, budget-aware compression.
agent = Agent(name="agent", model="openai:gpt-4o")

# Don't need all that? Opt out.
bare = Agent(name="bare", model="openai:gpt-4o", memory=None, context=None)
```

<br>

---

<div align="center">

```
     ·    .    ·        ✧        ·    .    ·
      T H E   C O N T E X T   E N G I N E
     .    ·        .        ·    .        ✦
```

*The core of what makes Exo agents deeply smart.*

</div>

<br>

Most frameworks treat context as a list of messages you trim when it gets too long. Exo treats context as a **living, structured system** the agent builds, queries, and manages.

<br>

### Neurons

<div align="center">

*Composable prompt architecture — the agent's system prompt assembles itself.*

</div>

<br>

Agent prompts aren't static strings. They're dynamically assembled from **neurons** — modular, priority-ordered, async prompt fragments that render based on what context exists:

```
 Priority Flow ─────────────────────────────────────────────────
 
  ┌──────────────┐
  │  TaskNeuron   │  P:1    Current task, expected output, subtask plan
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │  TodoNeuron   │  P:2    Checklist with [x]/[ ] progress markers
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │ HistoryNeuron │  P:10   Windowed conversation history
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │KnowledgeNeuron│  P:20   Relevant knowledge base snippets
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │WorkspaceNeuron│  P:30   Available artifacts with type and size
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │  SkillNeuron  │  P:40   Active skills and capabilities
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │  FactNeuron   │  P:50   Established facts from prior reasoning
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │ EntityNeuron  │  P:60   Named entities in the conversation
  └──────┬───────┘
         │
  ┌──────▼───────┐
  │ SystemNeuron  │  P:100  Date/time, platform, environment info
  └──────────────┘

 ────────────────────────────────────── ▸ Assembled System Prompt
```

Each neuron only renders if its data exists in state. A research agent gets task + history + knowledge + workspace. A quick Q&A agent might only get history + system. **The prompt adapts to the work.**

<br>

### Context Processors

<div align="center">

*Automatic pressure management — before every LLM call.*

</div>

<br>

```
  Messages In                                              Messages Out
  ━━━━━━━━━━                                              ━━━━━━━━━━━
       │                                                       ▲
       ▼                                                       │
  ┌─────────────────────────────────────────────────────────────┐
  │                   PROCESSOR PIPELINE                        │
  │                                                             │
  │  ┌─────────────────────┐   5 tool calls → 1 summary msg    │
  │  │ DialogueCompressor  │   "Called search, parse, extract,  │
  │  │                     │    validate, store. Results: ..."   │
  │  └─────────┬───────────┘                                    │
  │            ▼                                                │
  │  ┌─────────────────────┐   >10KB messages → [[OFFLOAD]]    │
  │  │  MessageOffloader   │   originals stored for retrieval   │
  │  │                     │   system messages never offloaded  │
  │  └─────────┬───────────┘                                    │
  │            ▼                                                │
  │  ┌─────────────────────┐   history > threshold →            │
  │  │ SummarizeProcessor  │   LLM-generated summary replaces  │
  │  │                     │   older messages, keeps recent     │
  │  └─────────┬───────────┘                                    │
  │            ▼                                                │
  │  ┌─────────────────────┐   keep last N rounds, respects    │
  │  │ RoundWindowProcessor│   tool-chain boundaries            │
  │  └─────────────────────┘                                    │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

  The agent never sees token limits — it just has the right context.
```

<br>

### Automation Modes

<div align="center">

*Three presets. From manual control to full autonomy.*

</div>

<br>

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │   PILOT          COPILOT           NAVIGATOR                    │
  │   ○ ─ ─ ─ ─ ─ ─ ◉ ─ ─ ─ ─ ─ ─ ─ ● ─ ─ ─ ─ ▸                 │
  │   manual         balanced          autonomous                   │
  │                                                                 │
  │   100 rounds     20 rounds         10 rounds                    │
  │   you manage     auto-summarize    aggressive compression       │
  │   full history   offload at 50     auto-retrieval enabled       │
  │                  ↑ default                                      │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

```python
Agent(name="a", model="openai:gpt-4o", context_mode="pilot")      # you drive
Agent(name="a", model="openai:gpt-4o", context_mode="copilot")    # balanced (default)
Agent(name="a", model="openai:gpt-4o", context_mode="navigator")  # fully autonomous
```

<br>

### Self-Manipulation Tools

<div align="center">

*The agent reasons about what it's done, what it knows, and what's left.*

</div>

<br>

Every agent with an active context engine gets tools to manage its own execution:

```python
# The agent can call these during its tool loop:
add_todo("Research competing approaches")        # track its own progress
complete_todo(0)                                  # mark steps done
search_knowledge("prior findings on X")          # query its workspace index
grep_knowledge("results.csv", r"accuracy.*\d+")  # regex search within artifacts
read_file("data/input.json")                     # read from working directory
```

These aren't user-facing tools. They're the agent's inner monologue made actionable.

<br>

### Workspace & Auto-Indexing

When an agent writes an artifact (code, CSVs, analysis), the content is automatically chunked and TF-IDF indexed:

```
  Agent writes artifact          Auto-indexed
  ━━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━
                                ┌──────────────────┐
  workspace.write("analysis")──▸│ Chunker (512ch)  │
                                │ Overlap: 64ch    │
                                └────────┬─────────┘
                                         ▼
                                ┌──────────────────┐
                                │ TF-IDF Indexer   │
                                │ search("query")  │──▸ Scored chunks
                                └──────────────────┘

  Artifacts are versioned. Revert, explore history, observe changes.
```

<br>

### Context Forking

When an agent spawns subtasks, each child gets a **forked context** with hierarchical state inheritance:

```
  ┌─────────────────────────────────────────────────┐
  │              PARENT CONTEXT                      │
  │  state: { config, knowledge, workspace }         │
  │  tokens: 1,200                                   │
  │                                                  │
  │     fork()          fork()          fork()       │
  │       │               │               │          │
  │  ┌────▼────┐    ┌─────▼────┐    ┌─────▼────┐    │
  │  │ Child 1 │    │ Child 2  │    │ Child 3  │    │
  │  │ reads ↑ │    │ reads ↑  │    │ reads ↑  │    │
  │  │writes ↓ │    │writes ↓  │    │writes ↓  │    │
  │  └────┬────┘    └─────┬────┘    └─────┬────┘    │
  │       │               │               │          │
  │       └───── merge() ─┴─── merge() ───┘          │
  │                                                  │
  │  Only net-new tokens counted. No double-booking. │
  └─────────────────────────────────────────────────┘
```

Children inherit parent config. Writes are local. Merges aggregate results with accurate token accounting. This is how self-spawning agents work under the hood.

<br>

---

<div align="center">

```
     ·    .    ·        ✧        ·    .    ·
         A G E N T   C A P A B I L I T I E S
     .    ·        .        ·    .        ✦
```

</div>

<br>

<table>
<tr>
<td width="50%" valign="top">

### Three Ways to Run

```python
from exo import Agent, run

agent = Agent(
    name="a",
    model="openai:gpt-4o",
    tools=[...],
)

# Async
result = await run(agent, "Do the thing")

# Blocking
result = run.sync(agent, "Do the thing")

# Streaming
async for event in run.stream(agent, "..."):
    match event.type:
        case "text":
            print(event.text, end="")
        case "tool_call":
            print(f"> {event.tool_name}")
```

Events: `Text` `ToolCall` `ToolResult` `Step` `Usage` `Status` `Error` `Reasoning`

</td>
<td width="50%" valign="top">

### Tools

Type hints in, JSON schema out:

```python
from exo import tool

@tool
async def search_db(
    query: str,
    limit: int = 10,
) -> list[dict]:
    """Search the database.

    Args:
        query: The search query.
        limit: Max results to return.
    """
    return await db.search(query, limit=limit)
```

The `@tool` decorator extracts name, description, and full parameter schema from the signature and docstring. 

`@tool(large_output=True)` offloads results to the workspace and auto-indexes for retrieval.

</td>
</tr>
</table>

<br>

<table>
<tr>
<td width="50%" valign="top">

### Planning

Complex tasks fail when agents jump straight into tool calls. The planning system adds a dedicated reasoning phase:

```python
agent = Agent(
    name="architect",
    model="openai:gpt-4o",
    planning_enabled=True,
    planning_model="openai:o3",
)
```

```
  ┌─────────┐     ┌──────────┐
  │ PLANNER │────▸│ EXECUTOR │
  │         │     │          │
  │ reason  │     │ tools    │
  │ plan    │     │ actions  │
  │ decomp. │     │ results  │
  └─────────┘     └──────────┘
  isolated ctx    follows plan
  no side effects full tool access
```

Planning transcript is discarded. Only the final plan survives.

</td>
<td width="50%" valign="top">

### Self-Spawning Agents

Agents recognize when they've hit a ceiling and split themselves into parallel workers:

```python
agent = Agent(
    name="researcher",
    model="openai:gpt-4o",
    allow_self_spawn=True,
    max_spawn_children=5,
    max_spawn_depth=2,
)
```

```
         ┌──────────┐
         │  PARENT  │
         └────┬─────┘
        spawn_self()
      ┌───┬───┼───┬───┐
      ▼   ▼   ▼   ▼   ▼
     C1  C2  C3  C4  C5
      │   │   │   │   │
      └───┴───┼───┴───┘
           merge()
         ┌────▼─────┐
         │  PARENT  │
         │ + results│
         └──────────┘
```

Children: forked context, shared long-term memory. Knowledge accumulates across all spawns.

</td>
</tr>
</table>

<br>

<table>
<tr>
<td width="50%" valign="top">

### Multi-Agent Orchestration

```python
from exo import Agent, Swarm, run

researcher = Agent(
    name="researcher",
    model="openai:gpt-4o",
    instructions="Research thoroughly.",
)
writer = Agent(
    name="writer",
    model="openai:gpt-4o",
    instructions="Write from research.",
)
reviewer = Agent(
    name="reviewer",
    model="openai:gpt-4o",
    instructions="Review for accuracy.",
)

swarm = Swarm(
    agents=[researcher, writer, reviewer],
    flow="researcher >> writer >> reviewer",
)

result = run.sync(swarm, "Fusion energy")
```

</td>
<td width="50%" valign="top">

<br>

**Three orchestration modes:**

```
 WORKFLOW ─────────────────────────
   A ──▸ B ──▸ C
   Sequential. Output flows forward.
   + BranchNode (conditional routing)
   + LoopNode (iteration)

 HANDOFF ──────────────────────────
   A ──?──▸ B
     └──?──▸ C
   Agent-driven. A decides who's next.

 TEAM ─────────────────────────────
   Lead ──▸ tool(Worker₁)
        ──▸ tool(Worker₂)
        ──▸ tool(Worker₃)
   First agent leads. Others become
   callable tools.
```

Mix `ParallelGroup` and `SerialGroup` within any mode.

</td>
</tr>
</table>

<br>

<table>
<tr>
<td width="50%" valign="top">

### Structured Output

No parsing. No regex. No "please respond as JSON":

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float
    key_topics: list[str]

agent = Agent(
    name="analyzer",
    model="openai:gpt-4o-mini",
    output_type=Analysis,
)

result = run.sync(agent, "Analyze: ...")
result.output.sentiment   # "mixed"
result.output.confidence  # 0.85
```

</td>
<td width="50%" valign="top">

### Human-in-the-Loop

Gate specific tools for human approval:

```python
agent = Agent(
    name="ops",
    model="openai:gpt-4o",
    tools=[
        deploy_to_prod,
        run_migration,
        check_status,
    ],
    hitl_tools=[
        "deploy_to_prod",
        "run_migration",
    ],
)
```

```
  agent runs ──▸ wants deploy ──▸ PAUSE
                                    │
                              human approves
                                    │
                              ◂── RESUME
```

</td>
</tr>
</table>

<br>

<table>
<tr>
<td width="50%" valign="top">

### Guardrails

Security rails as composable lifecycle hooks:

```python
from exo.guardrail import (
    PatternGuardrail,
    LLMGuardrail,
)

agent = Agent(
    name="assistant",
    model="openai:gpt-4o-mini",
    rails=[
        PatternGuardrail(
            blocked_patterns=[
                r"\b\d{3}-\d{2}-\d{4}\b",
            ]
        ),
        LLMGuardrail(
            model="openai:gpt-4o-mini",
            policy="No harmful content",
        ),
    ],
)
```

Actions: `CONTINUE` `SKIP` `RETRY` `ABORT`

</td>
<td width="50%" valign="top">

### Lifecycle Hooks

Intercept and mutate at every stage:

```python
from exo.hooks import HookPoint

async def on_start(context):
    agent = context["agent"]
    await agent.add_tool(my_tool)

agent = Agent(
    name="dynamic",
    model="openai:gpt-4o",
    hooks=[
        (HookPoint.START, on_start),
    ],
)
```

```
  START
    ▼
  PRE_LLM_CALL ──▸ POST_LLM_CALL
    ▼                    │
  PRE_TOOL_CALL ◂───────┘
    ▼
  POST_TOOL_CALL ──▸ (loop)
    ▼
  FINISHED ─── or ─── ERROR
```

</td>
</tr>
</table>

<br>

<table>
<tr>
<td valign="top">

### MCP

Connect to any MCP tool server, or expose your agent's tools as one:

```python
from exo.mcp import MCPServerStdio

agent = Agent(
    name="coder",
    model="openai:gpt-4o",
    mcp_servers=[MCPServerStdio(command="npx", args=["-y", "@modelcontextprotocol/server-github"])],
)
```

</td>
</tr>
</table>

<br>

---

<div align="center">

```
     ·    .    ·        ✧        ·    .    ·
             P A C K A G E S
     .    ·        .        ·    .        ✦
```

*18 focused packages. `exo-core` depends only on `pydantic`. Everything else is opt-in.*

</div>

<br>

```
                              ╭──────╮
                              │  exo │  meta-package
                              ╰──┬───╯
            ┌─────────┬──────────┼──────────┬──────────┐
            │         │          │          │          │
         ╭──▼──╮  ╭───▼──╮  ╭───▼──╮  ╭───▼──╮  ╭───▼──╮
         │ cli │  │server│  │train │  │ a2a  │  │ eval │
         ╰──┬──╯  ╰───┬──╯  ╰───┬──╯  ╰───┬──╯  ╰───┬──╯
            └─────────┬──────────┼──────────┴──────────┘
   ┌──────────┬───────┼──────────┼──────────┬──────────┐
   │          │       │          │          │          │
╭──▼───╮ ╭───▼──╮ ╭──▼──╮ ╭────▼───╮ ╭───▼────╮ ╭───▼───╮
│contxt│ │memory│ │ mcp │ │sandbox │ │observe │ │guardr.│
╰──┬───╯ ╰───┬──╯ ╰──┬──╯ ╰────┬───╯ ╰───┬────╯ ╰───┬───╯
   └──────────┴───────┼──────────┴─────────┘          │
                      │                                │
              ╭───────▼───────╮                        │
              │    models     │                        │
              ╰───────┬───────╯                        │
                      │              ╭─────────────────┘
              ╭───────▼───────╮      │
              │     core      │◂─────╯
              │  (pydantic)   │
              ╰───────────────╯
```

<br>

<div align="center">

| | Package | Purpose |
|:--:|:---|:---|
| **core** | `exo-core` | Agent, Tool, Runner, Swarm, Hooks, Events, Config |
| **models** | `exo-models` | LLM providers — OpenAI, Anthropic, Gemini, Vertex AI |
| **context** | `exo-context` | Context engine, neurons, prompt builder, checkpoints |
| **memory** | `exo-memory` | Short/long-term memory, SQLite, Postgres, ChromaDB |
| **mcp** | `exo-mcp` | Model Context Protocol client and server |
| **guardrail** | `exo-guardrail` | Pattern and LLM-based security guardrails |
| **sandbox** | `exo-sandbox` | Local + Kubernetes sandboxed execution |
| **observe** | `exo-observability` | Logging, OpenTelemetry tracing, metrics, cost tracking |
| **eval** | `exo-eval` | Evaluation — scorers, LLM-as-judge, reflection, pass@k |
| **retrieval** | `exo-retrieval` | RAG — embeddings, vector stores, reranking, knowledge graph |
| **search** | `exo-search` | AI search engine with deep research and citations |
| **skills** | `exo-skills` | Dynamic capability packages — load, compose, hot-reload |
| **a2a** | `exo-a2a` | Agent-to-Agent protocol for network delegation |
| **distributed** | `exo-distributed` | Redis task queue, Temporal workflows, event streaming |
| **server** | `exo-server` | FastAPI server, sessions, WebSocket streaming |
| **cli** | `exo-cli` | CLI runner, interactive console, batch processing |
| **train** | `exo-train` | Trajectory collection, data synthesis, RLHF integration |
| **web** | `exo-web` | Platform UI — visual workflows, playground, knowledge bases |

</div>

<br>

---

<div align="center">

```
     ·    .    ·        ✧        ·    .    ·
          D E V E L O P M E N T
     .    ·        .        ·    .        ✦
```

</div>

```bash
git clone https://github.com/hazel-core/exo-ai && cd exo-ai
uv sync                                      # install everything
uv run pytest                                 # ~2,900 tests
uv run ruff check packages/ --fix            # lint
uv run ruff format packages/                 # format
uv run pyright packages/exo-core/            # type-check
```

<br>

---

<div align="center">

```
              ✦         .    ·
     .    ·        ✧              ·    .
          ·    .        ·    .
                   ✦
```

<sub>MIT License — [Hazel Communications Private Limited](https://midsphere.ai), India (a part of Midsphere AI)</sub>

</div>
