# Execution Flow

This document traces what happens when you call `run()` -- from user code through Exo's internals and back.

## Entry Points

Exo provides three entry points, all defined in `exo.runner`:

| Entry Point | Signature | Description |
|-------------|-----------|-------------|
| `run()` | `async def run(agent, input, ...) -> RunResult` | Primary async API |
| `run.sync()` | `def run.sync(agent, input, ...) -> RunResult` | Blocking wrapper via `asyncio.run()` |
| `run.stream()` | `async def run.stream(agent, input, ...) -> AsyncIterator[StreamEvent]` | Streaming via async generator |

## Full Execution Trace

```
User code                    exo internals
---------                    -----------------
run(agent, input)
  |
  +---> resolve provider from agent.model string
  |       (via exo.models.provider.get_provider)
  |
  +---> detect Swarm vs Agent
  |       if Swarm: delegate to swarm.run()
  |       if Agent: continue below
  |
  +---> call_runner(agent, input, state)
  |       |
  |       +---> create RunState (tracks messages, nodes, usage)
  |       +---> state.start()
  |       +---> state.new_node(agent_name)
  |       +---> node.start()
  |       |
  |       +---> agent.run(input, messages, provider)
  |       |       |
  |       |       +---> resolve instructions (str or callable)
  |       |       +---> build initial message list:
  |       |       |       message_builder.build_messages(instructions, history)
  |       |       |       [SystemMessage, ...history, UserMessage(input)]
  |       |       |
  |       |       +---> get tool schemas (OpenAI function-calling format)
  |       |       |
  |       |       +---> TOOL LOOP (up to max_steps):
  |       |       |       |
  |       |       |       +---> _call_llm(msg_list, tool_schemas, provider)
  |       |       |       |       |
  |       |       |       |       +---> hooks: PRE_LLM_CALL
  |       |       |       |       +---> provider.complete(messages, tools)
  |       |       |       |       +---> hooks: POST_LLM_CALL
  |       |       |       |       +---> output_parser.parse_response()
  |       |       |       |       \---> return AgentOutput
  |       |       |       |
  |       |       |       +---> if no tool_calls: return output (done)
  |       |       |       |
  |       |       |       +---> if tool_calls:
  |       |       |       |       +---> parse_tool_arguments(tool_calls)
  |       |       |       |       |       (JSON string -> dict[str, Any])
  |       |       |       |       |
  |       |       |       |       +---> _execute_tools(actions)
  |       |       |       |       |       |
  |       |       |       |       |       +---> for each tool IN PARALLEL:
  |       |       |       |       |       |       +---> hooks: PRE_TOOL_CALL
  |       |       |       |       |       |       +---> tool.execute(**args)
  |       |       |       |       |       |       +---> hooks: POST_TOOL_CALL
  |       |       |       |       |       |       \---> ToolResult (or error)
  |       |       |       |       |       |
  |       |       |       |       |       \---> return [ToolResult, ...]
  |       |       |       |       |
  |       |       |       |       +---> append AssistantMessage + ToolResults
  |       |       |       |       \---> loop back to _call_llm
  |       |       |       |
  |       |       |       \---> max_steps exhausted: return last output
  |       |       |
  |       |       \---> return AgentOutput
  |       |
  |       +---> record usage on state
  |       +---> _check_loop (detect repeated tool-call patterns)
  |       +---> build final messages
  |       +---> state.succeed()
  |       |
  |       \---> return RunResult(output, messages, usage, steps)
  |
  \---> return RunResult
```

## Key Components in the Loop

### RunState (`exo._internal.state`)

`RunState` is the mutable execution tracker for a single run. It holds:

- **messages** -- Full conversation history accumulated during the run
- **nodes** -- List of `RunNode` objects, one per LLM call step
- **iterations** -- Step counter
- **total_usage** -- Aggregated `Usage` (input_tokens, output_tokens, total_tokens)

Each `RunNode` tracks:
- Status transitions: `INIT -> RUNNING -> SUCCESS | FAILED | TIMEOUT`
- Timing: `created_at`, `started_at`, `ended_at`, `duration`
- Token usage for that step
- Metadata (e.g., `tool_signature` for loop detection)

### Message Builder (`exo._internal.message_builder`)

`build_messages()` constructs the correctly ordered message list:

```
[SystemMessage(instructions), ...history, UserMessage(input)]
```

It also provides:
- `validate_message_order()` -- Detects dangling tool calls (tool call without matching result)
- `extract_last_assistant_tool_calls()` -- Checks if conversation is mid-tool-execution
- `merge_usage()` -- Accumulates token counts across multiple LLM calls

### Output Parser (`exo._internal.output_parser`)

Bridges the model layer to the agent layer:

- `parse_response()` -- Maps raw model fields to `AgentOutput`
- `parse_tool_arguments()` -- Decodes JSON-encoded `ToolCall.arguments` into `ActionModel` objects with `dict[str, Any]` arguments
- `parse_structured_output()` -- Validates LLM text against a Pydantic model when `output_type` is set

### Loop Detection (`exo._internal.call_runner`)

The call runner detects endless loops where an agent repeatedly produces identical tool calls. It works by:

1. Computing a deterministic signature from tool call names and arguments (sorted, order-independent)
2. Storing the signature in the `RunNode.metadata`
3. Counting consecutive nodes with the same signature
4. Raising `CallRunnerError` when the count reaches `loop_threshold` (default: 3)

## Swarm Execution Modes

When `run()` receives a `Swarm` instead of a bare `Agent`, it delegates to the Swarm's own `run()` method, which dispatches based on mode:

### Workflow Mode (`mode="workflow"`)

Agents execute sequentially in topological order (from flow DSL). Each agent's output becomes the next agent's input:

```
researcher.run("topic") -> output_1
writer.run(output_1)    -> output_2
editor.run(output_2)    -> final_result
```

### Handoff Mode (`mode="handoff"`)

The first agent runs. If its output matches a declared handoff target name, control transfers to that agent with the full conversation history. This continues until an agent produces output that is not a handoff, or `max_handoffs` is exceeded.

### Team Mode (`mode="team"`)

The first agent is the lead. Auto-generated `delegate_to_{worker_name}` tools are added to the lead. When the lead calls a delegate tool, the worker agent runs and returns its output as the tool result. The lead synthesizes the final answer.

## Streaming Flow

`run.stream()` follows a similar pattern but uses the provider's `stream()` method:

1. Build message list and tool schemas
2. Stream from provider, yielding `TextEvent` for each text delta
3. Accumulate `ToolCallDelta` chunks into complete `ToolCall` objects
4. When tool calls are detected, yield `ToolCallEvent` for each, execute tools, and re-stream with results
5. Loop until a text-only response or `max_steps` is reached

## Retry Logic

`Agent._call_llm()` implements exponential backoff retry:

- Retries up to `max_retries` (default: 3) on transient errors
- Delay between retries: `2^attempt` seconds
- Context-length errors are detected and raised immediately (no retry)
- After all retries exhausted, raises `AgentError` with the last exception chained via `from`
