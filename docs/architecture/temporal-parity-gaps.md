# Temporal Parity Gaps

Status: Accepted  
Story: US-004  
Date: 2026-03-09

## Goal

Define the current gap between Exo's local execution contract and the current Temporal durable-execution path, then turn that gap into a small-slice closure order. Temporal is allowed to add durability. It is not allowed to silently degrade streaming, tool visibility, progress events, memory/context behavior, planning behavior, or cancellation semantics.

## Current Inventory And Exact Touchpoints

| Surface | Current touchpoints | What exists today |
| --- | --- | --- |
| Core local execution contract | `packages/exo-core/src/exo/runner.py`, `packages/exo-core/src/exo/agent.py`, `packages/exo-core/src/exo/types.py` | `run.stream()` is the richest observable contract today: text deltas, tool calls/results, usage, status, MCP progress, context events, and injected-message events. |
| Distributed local worker path | `packages/exo-distributed/src/exo/distributed/worker.py`, `client.py`, `memory.py`, `events.py` | The local worker rebuilds agents, hydrates task messages and optional memory, then republishes `run.stream()` events through Redis. |
| Temporal path | `packages/exo-distributed/src/exo/distributed/temporal.py` | The Temporal workflow/activity executes the agent internally but returns only final text to the worker. |
| Planning/context primitives | `packages/exo-context/src/exo/context/tools.py` and `packages/exo-core/src/exo/agent.py` | Planning is currently limited to context-bound todo tools. There is no dedicated planner pre-pass yet. |
| Evidence tests | `packages/exo-core/tests/test_runner.py`, `packages/exo-mcp/tests/test_progress.py`, `packages/exo-distributed/tests/test_worker.py`, `packages/exo-distributed/tests/test_temporal.py`, `packages/exo-distributed/tests/test_events.py`, `packages/exo-distributed/tests/test_cancel.py` | The tests document the real event ordering and the current Temporal assumptions better than the guides alone. |

## What The Local Baseline Does Today

### 1. Streaming And Detailed Events

- `run.stream(..., detailed=True)` emits a structured event sequence from `runner._stream()`: `StatusEvent(starting)`, `StepEvent(started)`, text/tool events, `UsageEvent`, `StepEvent(completed)`, and terminal status.
- Tool rounds emit `ToolCallEvent` after the LLM round finishes, then `ToolResultEvent` after tool execution.
- `packages/exo-core/tests/test_runner.py` fixes the detailed ordering contract for both text-only and tool-using turns.

### 2. Tool Execution

- `Agent._execute_tools()` executes tool calls in parallel with `asyncio.TaskGroup()`.
- Tool errors become `ToolResult(error=...)` instead of crashing the whole run.
- Large string results are offloaded through `_offload_large_result()` before they are reintroduced to the model.

### 3. MCP Progress

- MCP tools queue progress notifications during execution.
- `runner._stream()` drains those queues after tool execution and yields `MCPProgressEvent` before `ToolResultEvent`.
- `packages/exo-mcp/tests/test_progress.py` and `tests/integration/test_mcp_progress_stream.py` lock that ordering in.

### 4. Memory, Context, And Planning State

- Core local execution loads persisted history through `_memory_persistence`, stores the new user turn, applies `_apply_context_windowing()`, and injects long-term knowledge before the model call.
- The distributed local worker adds another layer of runtime setup: it deserializes `task.messages`, hydrates `task.metadata["memory"]`, loads prior memory items, and then calls `run.stream()`.
- Planning today is not a separate planner phase. The only current planning surface is the auto-loaded todo tools from `exo-context`, which operate on the live context state.

### 5. Cancellation

- The local worker listens on `exo:cancel:{task_id}` and flips a `CancellationToken`.
- `Worker._run_agent()` checks that token between yielded events, publishes a terminal `StatusEvent(status="cancelled")`, and stops forwarding more events.
- Cancellation is cooperative, not preemptive: a long tool call can still run until it yields control.

## What The Temporal Path Does Today

### 1. Final Text Only

- `TemporalExecutor.execute_task()` starts `AgentExecutionWorkflow` and waits for `handle.result()`.
- `execute_agent_activity()` reconstructs the agent and calls `run.stream()`, but it only appends `TextEvent.text` into `text_parts`.
- No event from the activity is published through `EventPublisher`, and no terminal `StatusEvent` is emitted to Redis.

### 2. Task Setup Is Not Shared With The Local Worker

- The activity ignores `task.messages`.
- The activity ignores `task.metadata["memory"]`, so worker-side memory hydration and scoped history loading do not happen.
- The activity ignores the worker's `provider_factory`.
- The result is that Temporal runs a fresh reconstructed agent, not the same prepared runtime state that the local worker uses.

### 3. Context And Planning Are Internal Only

- Because the activity still calls `run.stream()`, context windowing and context-bound planning tools can still execute inside the activity process.
- Those context actions are not externally visible because the activity drops every non-text event.
- Any worker-prepared history or memory-backed planning state is absent because the Temporal path skips the local worker setup.

### 4. Cancellation Is Not Propagated Into The Workflow

- `TemporalExecutor.execute_task()` checks `token.cancelled` only before starting the workflow.
- After `start_workflow(...)`, it awaits `handle.result()` without polling the token and without cancelling the workflow/activity.
- A user-visible status can become `cancelled` while the workflow keeps running in Temporal until normal completion.

### 5. Heartbeats Depend On Event Count

- The activity heartbeats every tenth streamed event.
- Inference from the implementation: a long provider call or long tool call that emits fewer than ten events for more than 30 seconds can miss the configured heartbeat timeout even though work is still progressing.

## Decision

The parity target is the local core streaming contract as surfaced through distributed task APIs:

- `run.stream()` is the semantic baseline.
- `TaskHandle.stream()` must preserve the same event types, field meanings, and ordering boundaries as that baseline.
- `TaskHandle.result()` must preserve the same completion, failure, and cancellation semantics as the local worker path.
- Temporal may add workflow ids, retries, and heartbeats internally, but those must not change the externally observable behavior of a single run.

If Temporal cannot honor a capability yet, it must fail explicitly. Silent omission is not allowed.

## Externally Observable Parity Contract

### 1. Same Inputs Before The First Model Call

For the same task payload, both local and Temporal execution must apply the same pre-run setup:

- `task.messages`
- memory hydration from `task.metadata`
- long-term memory injection
- context windowing and context-budget compaction
- provider resolution rules
- future planner configuration from US-002 once that story is implemented

### 2. Same Event Surface

The distributed stream contract must carry every event family the core runner can emit today:

- `text`
- `tool_call`
- `tool_result`
- `usage`
- `status`
- `error`
- `mcp_progress`
- `context`
- `message_injected`

Future planner/HITL/runtime-control events should extend this same contract instead of creating a Temporal-only side channel.

### 3. Same Ordering Rules

Parity is about ordering boundaries, not byte-for-byte timestamp equality.

Required ordering rules:

- `StatusEvent(starting)` precedes step 1.
- `StepEvent(started)` precedes all events for that step.
- For tool rounds, `UsageEvent` precedes `ToolCallEvent`, and `ToolCallEvent` precedes `MCPProgressEvent` and `ToolResultEvent`.
- `MCPProgressEvent` precedes the corresponding `ToolResultEvent`.
- Each step ends with `StepEvent(completed)`.
- The run ends with exactly one terminal `StatusEvent`: `completed`, `error`, or `cancelled`.

### 4. Same Cancellation Boundary

After cancellation becomes visible to the caller:

- no further text, tool, progress, or usage events may be emitted
- exactly one terminal `StatusEvent(status="cancelled")` must be emitted
- `TaskHandle.result()` must resolve as cancelled without waiting for the workflow to finish natural execution

### 5. Explicit Failure For Unsupported Features

If a Temporal run cannot satisfy the contract yet, it must fail with a deterministic operator-facing error before silently dropping behavior.

## Canonical Detailed-Event Trace

Example task: the agent makes one MCP tool call that emits two progress notifications and then returns a final text answer.

This is the expected trace for both local execution and a parity-correct Temporal execution:

| Order | Expected local event | Expected Temporal event |
| --- | --- | --- |
| 1 | `StatusEvent(status="starting")` | `StatusEvent(status="starting")` |
| 2 | `StepEvent(step=1, status="started")` | `StepEvent(step=1, status="started")` |
| 3 | `UsageEvent(step=1)` for the tool-call LLM round | `UsageEvent(step=1)` for the tool-call LLM round |
| 4 | `ToolCallEvent(tool_name="mcp__srv__search")` | `ToolCallEvent(tool_name="mcp__srv__search")` |
| 5 | `MCPProgressEvent(progress=1, total=2, message="step 1")` | `MCPProgressEvent(progress=1, total=2, message="step 1")` |
| 6 | `MCPProgressEvent(progress=2, total=2, message="step 2")` | `MCPProgressEvent(progress=2, total=2, message="step 2")` |
| 7 | `ToolResultEvent(tool_name="mcp__srv__search", success=true)` | `ToolResultEvent(tool_name="mcp__srv__search", success=true)` |
| 8 | `StepEvent(step=1, status="completed")` | `StepEvent(step=1, status="completed")` |
| 9 | `StepEvent(step=2, status="started")` | `StepEvent(step=2, status="started")` |
| 10 | `TextEvent("Final answer")` | `TextEvent("Final answer")` |
| 11 | `UsageEvent(step=2)` | `UsageEvent(step=2)` |
| 12 | `StepEvent(step=2, status="completed")` | `StepEvent(step=2, status="completed")` |
| 13 | `StatusEvent(status="completed")` | `StatusEvent(status="completed")` |

Current state:

- Direct local `run.stream()` already follows this trace.
- The distributed transport cannot yet replay `mcp_progress`, `context`, or `message_injected` because `packages/exo-distributed/src/exo/distributed/events.py` does not deserialize those event types.
- The current Temporal path publishes none of the thirteen events externally and returns only final output text.

## Parity Gap Matrix

| Area | Local path today | Temporal path today | Required closure |
| --- | --- | --- | --- |
| Streaming | Core local streaming emits rich ordered events; local worker republishes them. | No Redis event publishing; callers only get final text result. | Temporal must forward the same event stream and terminal status. |
| Tool execution | Tool calls execute in the agent and are externally visible via `ToolCallEvent` and `ToolResultEvent`. | Tools still execute internally, but tool visibility is dropped. | Reuse a shared event-forwarding helper so tool events survive Temporal. |
| MCP progress | Core runner emits progress before tool results. Distributed transport cannot deserialize it yet. | Progress events are dropped entirely because the activity keeps only `TextEvent`. | Fix event transport first, then publish the same progress sequence from Temporal. |
| Memory | Local worker hydrates memory from task metadata and loads prior scoped history. | Memory metadata is ignored; Temporal runs a fresh reconstructed agent. | Move worker memory setup into a shared helper used by both backends. |
| Context | Local core applies context windowing, long-term injection, and emits `ContextEvent`. | Context helpers may run on a fresh auto context, but state/history parity and event visibility are missing. | Share pre-run context setup and forward `ContextEvent`. |
| Planning | Only context todo tools exist today; they operate on live local context. | Same tools may exist on a fresh context, but prior planning state is not hydrated and tool/context events are hidden. | Keep planner behavior identical once US-002 lands; do not invent Temporal-only planning behavior. |
| Cancellation | Cooperative cancel emits terminal cancelled status and stops event forwarding at the next boundary. | Token is not wired to workflow cancellation after start; work can continue after user-visible cancel. | Propagate cancellation into the workflow/activity and emit a single terminal cancelled event promptly. |

## Negative Case That Must Fail Explicitly

The first explicit-failure rule should be:

- A Temporal submission that asks for detailed streaming semantics must not silently degrade to final-text-only behavior.

Concretely, until Temporal publishes the full event stream, the worker should fail the task immediately when any of these are requested with `executor="temporal"`:

- `detailed=True`
- non-empty `task.messages`
- `task.metadata["memory"]`

Why this is the minimum acceptable failure mode:

- `detailed=True` currently implies tool, usage, status, and progress visibility that the Temporal path drops.
- `task.messages` and `metadata["memory"]` are accepted today but ignored, which changes the actual prompt seen by the model without telling the caller.
- `TaskHandle.stream()` can otherwise wait for a terminal event that Temporal never publishes.

## Closure Order In Small Slices

### Slice 1: Make The Surface Truthful

- Add the missing deserializers for `mcp_progress`, `context`, and `message_injected` in the distributed event transport.
- Reject unsupported Temporal capabilities up front instead of silently accepting them.

Why first:

- This creates an honest baseline and prevents hangs while the deeper parity work is still in flight.
- It also fixes the local distributed path so it can serve as the real comparison target.

### Slice 2: Share Pre-Run Setup

- Extract a shared helper for agent reconstruction, message deserialization, provider resolution, and memory hydration.
- Use that helper from both `Worker._run_agent()` and `execute_agent_activity()`.

Why second:

- Parity starts before the first model token. If the prompt inputs differ, event parity later is misleading.

### Slice 3: Ship Text/Tool/Usage/Status Event Parity

- Have the Temporal path forward streamed events through the same Redis publisher contract the local worker uses.
- End every Temporal stream with the same terminal status event rules.

Why third:

- This closes the largest externally visible gap without taking on every advanced event family at once.

### Slice 4: Add MCP Progress And Context Event Parity

- Forward `MCPProgressEvent`, `ContextEvent`, and `MessageInjectedEvent` with the same ordering as local `run.stream()`.
- Add a regression test that compares the canonical trace above across local and Temporal execution.

Why fourth:

- These are the highest-value non-text events still missing after Slice 3.

### Slice 5: Make Cancellation Durable

- Cancel the Temporal workflow/activity when the broker cancel signal arrives.
- Emit exactly one terminal cancelled status event.
- Ensure `TaskHandle.result()` resolves as cancelled without waiting for natural workflow completion.

Why fifth:

- Durable execution is not feature parity if cancelled tasks continue running invisibly in the background.

### Slice 6: Apply Future Planner And Runtime-Control Features

- After US-002 implementation stories land, round-trip planner config and any new runtime-control events through `TaskPayload`, the worker, and Temporal.
- Planner transcripts must remain invisible to executor history on both backends.

Why last:

- The dedicated planner contract does not exist in the code yet, so Temporal should follow the shared implementation once that work exists instead of guessing early.

## Implementation Guidance

- Do not add a separate Temporal-only event model. Reuse `exo.types.StreamEvent`.
- Do not duplicate worker setup logic in both `worker.py` and `temporal.py`; factor it once and test both backends against it.
- Treat "no event published" as a bug, not as an acceptable optimization.
- Keep parity tests at the task-handle boundary, not just inside the Temporal activity, so the distributed transport is covered too.
