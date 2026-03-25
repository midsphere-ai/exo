# exo.trace

OpenTelemetry-based observability for Exo agents and tools.

## Installation

```bash
pip install exo-trace
```

## Module path

```python
import exo.trace
```

## Package exports

| Export | Module | Description |
|---|---|---|
| `TraceConfig` | `exo.trace.config` | Immutable configuration for the trace layer |
| `TraceBackend` | `exo.trace.config` | Supported trace export backends enum |
| `traced` | `exo.trace.decorator` | Decorator wrapping a function in an OpenTelemetry span |
| `span_sync` | `exo.trace.decorator` | Synchronous span context manager |
| `Timer` | `exo.trace.instrumentation` | Simple timer for measuring durations |
| `build_agent_attributes` | `exo.trace.instrumentation` | Build attribute dict for agent metrics |
| `build_tool_attributes` | `exo.trace.instrumentation` | Build attribute dict for tool metrics |
| `record_agent_run` | `exo.trace.instrumentation` | Record agent run metrics |
| `record_tool_step` | `exo.trace.instrumentation` | Record tool step metrics |
| `PromptLogger` | `exo.trace.prompt_logger` | Structured LLM execution logger |
| `ExecutionLogEntry` | `exo.trace.prompt_logger` | Structured record of a single LLM execution |
| `TokenBreakdown` | `exo.trace.prompt_logger` | Per-role token counts and context window analysis |
| `compute_token_breakdown` | `exo.trace.prompt_logger` | Compute a TokenBreakdown from message dicts |
| `estimate_tokens` | `exo.trace.prompt_logger` | Estimate token count from character length |
| `BaggagePropagator` | `exo.trace.propagation` | Extract and inject W3C Baggage headers |
| `Carrier` | `exo.trace.propagation` | Protocol for reading/writing propagation headers |
| `DictCarrier` | `exo.trace.propagation` | Carrier backed by a plain dict |
| `SpanConsumer` | `exo.trace.propagation` | ABC for span consumers |
| `get_baggage` | `exo.trace.propagation` | Return the current baggage as a dict |
| `get_baggage_value` | `exo.trace.propagation` | Return a single baggage value |
| `set_baggage` | `exo.trace.propagation` | Set a single baggage key-value pair |
| `clear_baggage` | `exo.trace.propagation` | Remove all baggage entries |
| `register_span_consumer` | `exo.trace.propagation` | Register a span consumer |
| `get_span_consumer` | `exo.trace.propagation` | Look up a span consumer by name |
| `list_span_consumers` | `exo.trace.propagation` | List all registered span consumer names |
| `dispatch_spans` | `exo.trace.propagation` | Send spans to all registered consumers |
| `clear_span_consumers` | `exo.trace.propagation` | Remove all registered span consumers |

## Submodules

- [exo.trace.config](config.md) -- TraceConfig, TraceBackend, semantic conventions
- [exo.trace.decorator](decorator.md) -- @traced decorator and span context managers
- [exo.trace.instrumentation](instrumentation.md) -- Metrics recording and Timer
- [exo.trace.prompt_logger](prompt-logger.md) -- PromptLogger with token breakdown
- [exo.trace.propagation](propagation.md) -- W3C Baggage propagation and span consumers
