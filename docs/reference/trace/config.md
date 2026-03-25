# exo.trace.config

Trace configuration and semantic conventions for agent/tool observability.

```python
from exo.trace.config import TraceBackend, TraceConfig
```

---

## TraceBackend

```python
class TraceBackend(StrEnum)
```

Supported trace export backends.

| Value | Description |
|---|---|
| `OTLP = "otlp"` | OpenTelemetry Protocol exporter |
| `MEMORY = "memory"` | In-memory storage (for testing) |
| `CONSOLE = "console"` | Console/stdout exporter |

---

## TraceConfig

```python
class TraceConfig(BaseModel, frozen=True)
```

Immutable configuration for the trace / observability layer. Controls backend selection, sampling, export endpoint, and attribute namespace used for all Exo-specific span attributes.

Built on Pydantic `BaseModel` with `frozen=True`.

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `backend` | `TraceBackend` | `TraceBackend.OTLP` | Trace export backend |
| `endpoint` | `str \| None` | `None` | OTLP collector endpoint (e.g. `http://localhost:4318`) |
| `service_name` | `str` | `"exo"` | Service name reported in exported spans |
| `sample_rate` | `float` | `1.0` | Probability of sampling a trace (0.0 = none, 1.0 = all). Must be in [0.0, 1.0] |
| `enabled` | `bool` | `True` | Global toggle -- when False, tracing is a no-op |
| `headers` | `dict[str, str]` | `{}` | Extra headers sent with each export request |
| `namespace` | `str` | `"exo"` | Attribute namespace prefix (e.g. `exo.agent.name`) |
| `extra` | `dict[str, Any]` | `{}` | Extension point for custom exporter/processor config |

### Example

```python
from exo.trace import TraceConfig, TraceBackend

config = TraceConfig(
    backend=TraceBackend.OTLP,
    endpoint="http://localhost:4318",
    service_name="my-agent-service",
    sample_rate=0.5,
)

# Immutable -- this raises an error:
# config.sample_rate = 0.8  # ValidationError
```

---

## Semantic conventions

The module defines constants for OpenTelemetry semantic conventions used across Exo's trace instrumentation.

### GenAI conventions (`gen_ai.*`)

| Constant | Value | Description |
|---|---|---|
| `GEN_AI_SYSTEM` | `"gen_ai.system"` | LLM system identifier |
| `GEN_AI_REQUEST_MODEL` | `"gen_ai.request.model"` | Requested model name |
| `GEN_AI_REQUEST_MAX_TOKENS` | `"gen_ai.request.max_tokens"` | Max tokens requested |
| `GEN_AI_REQUEST_TEMPERATURE` | `"gen_ai.request.temperature"` | Temperature setting |
| `GEN_AI_REQUEST_TOP_P` | `"gen_ai.request.top_p"` | Top-p sampling |
| `GEN_AI_REQUEST_TOP_K` | `"gen_ai.request.top_k"` | Top-k sampling |
| `GEN_AI_REQUEST_FREQUENCY_PENALTY` | `"gen_ai.request.frequency_penalty"` | Frequency penalty |
| `GEN_AI_REQUEST_PRESENCE_PENALTY` | `"gen_ai.request.presence_penalty"` | Presence penalty |
| `GEN_AI_REQUEST_STOP_SEQUENCES` | `"gen_ai.request.stop_sequences"` | Stop sequences |
| `GEN_AI_REQUEST_STREAMING` | `"gen_ai.request.streaming"` | Whether streaming is enabled |
| `GEN_AI_PROMPT` | `"gen_ai.prompt"` | Prompt content |
| `GEN_AI_COMPLETION` | `"gen_ai.completion"` | Completion content |
| `GEN_AI_DURATION` | `"gen_ai.duration"` | Request duration |
| `GEN_AI_RESPONSE_FINISH_REASONS` | `"gen_ai.response.finish_reasons"` | Finish reasons |
| `GEN_AI_RESPONSE_ID` | `"gen_ai.response.id"` | Response identifier |
| `GEN_AI_RESPONSE_MODEL` | `"gen_ai.response.model"` | Model that produced the response |
| `GEN_AI_USAGE_INPUT_TOKENS` | `"gen_ai.usage.input_tokens"` | Input token count |
| `GEN_AI_USAGE_OUTPUT_TOKENS` | `"gen_ai.usage.output_tokens"` | Output token count |
| `GEN_AI_USAGE_TOTAL_TOKENS` | `"gen_ai.usage.total_tokens"` | Total token count |
| `GEN_AI_OPERATION_NAME` | `"gen_ai.operation.name"` | Operation name |
| `GEN_AI_SERVER_ADDRESS` | `"gen_ai.server.address"` | Server address |

### Agent conventions (`exo.agent.*`)

| Constant | Value | Description |
|---|---|---|
| `AGENT_ID` | `"exo.agent.id"` | Agent identifier |
| `AGENT_NAME` | `"exo.agent.name"` | Agent name |
| `AGENT_TYPE` | `"exo.agent.type"` | Agent type |
| `AGENT_MODEL` | `"exo.agent.model"` | Agent model |
| `AGENT_STEP` | `"exo.agent.step"` | Current step number |
| `AGENT_MAX_STEPS` | `"exo.agent.max_steps"` | Maximum steps allowed |
| `AGENT_RUN_SUCCESS` | `"exo.agent.run.success"` | Whether the run succeeded |

### Tool conventions (`exo.tool.*`)

| Constant | Value | Description |
|---|---|---|
| `TOOL_NAME` | `"exo.tool.name"` | Tool name |
| `TOOL_CALL_ID` | `"exo.tool.call_id"` | Tool call identifier |
| `TOOL_ARGUMENTS` | `"exo.tool.arguments"` | Tool arguments |
| `TOOL_RESULT` | `"exo.tool.result"` | Tool result |
| `TOOL_ERROR` | `"exo.tool.error"` | Tool error |
| `TOOL_DURATION` | `"exo.tool.duration"` | Tool execution duration |
| `TOOL_STEP_SUCCESS` | `"exo.tool.step.success"` | Whether the tool step succeeded |

### Task/session conventions

| Constant | Value | Description |
|---|---|---|
| `TASK_ID` | `"exo.task.id"` | Task identifier |
| `TASK_INPUT` | `"exo.task.input"` | Task input |
| `SESSION_ID` | `"exo.session.id"` | Session identifier |
| `USER_ID` | `"exo.user.id"` | User identifier |
| `TRACE_ID` | `"exo.trace.id"` | Trace identifier |

### Span name prefixes

| Constant | Value | Description |
|---|---|---|
| `SPAN_PREFIX_AGENT` | `"agent."` | Agent span prefix |
| `SPAN_PREFIX_TOOL` | `"tool."` | Tool span prefix |
| `SPAN_PREFIX_LLM` | `"llm."` | LLM call span prefix |
| `SPAN_PREFIX_TASK` | `"task."` | Task span prefix |
