# exo.trace.propagation

W3C Baggage propagation and span consumer plugin system.

```python
from exo.trace.propagation import (
    BaggagePropagator,
    Carrier,
    DictCarrier,
    SpanConsumer,
    clear_baggage,
    clear_span_consumers,
    dispatch_spans,
    get_baggage,
    get_baggage_value,
    get_span_consumer,
    list_span_consumers,
    register_span_consumer,
    set_baggage,
)
```

---

## Carrier

```python
@runtime_checkable
class Carrier(Protocol)
```

Minimal protocol for reading/writing propagation headers.

### Methods

```python
def get(self, key: str) -> str | None: ...
def set(self, key: str, value: str) -> None: ...
```

---

## DictCarrier

```python
class DictCarrier(headers: dict[str, str] | None = None)
```

Carrier backed by a plain dict.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `headers` | `dict[str, str] \| None` | `None` | Initial headers dict. Creates a new empty dict if None |

### Properties

| Property | Type | Description |
|---|---|---|
| `headers` | `dict[str, str]` | The underlying headers dict |

### Methods

```python
def get(self, key: str) -> str | None
def set(self, key: str, value: str) -> None
```

### Example

```python
carrier = DictCarrier({"traceparent": "00-abc-def-01"})
carrier.set("baggage", "key=value")
print(carrier.headers)
```

---

## Baggage context functions

These functions manipulate async-safe baggage storage via `ContextVar`. Baggage is scoped to the current async context.

### get_baggage

```python
def get_baggage() -> dict[str, str]
```

Return the current baggage as a read-only copy. Returns an empty dict if no baggage is set.

### get_baggage_value

```python
def get_baggage_value(key: str) -> str | None
```

Return a single baggage value, or `None` if not set.

### set_baggage

```python
def set_baggage(key: str, value: str) -> None
```

Set a single baggage key-value pair in the current context.

### clear_baggage

```python
def clear_baggage() -> None
```

Remove all baggage entries from the current context.

### Example

```python
from exo.trace import set_baggage, get_baggage, get_baggage_value, clear_baggage

set_baggage("user_id", "user-123")
set_baggage("session_id", "sess-456")

print(get_baggage())           # {"user_id": "user-123", "session_id": "sess-456"}
print(get_baggage_value("user_id"))  # "user-123"

clear_baggage()
print(get_baggage())           # {}
```

---

## BaggagePropagator

```python
class BaggagePropagator()
```

Extract and inject W3C Baggage headers (RFC 9110). Handles URL-encoding of keys/values and enforces size limits.

### Constants

| Constant | Value | Description |
|---|---|---|
| `BAGGAGE_HEADER` | `"baggage"` | Header name |
| `MAX_HEADER_LENGTH` | `8192` | Maximum header length in bytes |
| `MAX_PAIR_LENGTH` | `4096` | Maximum length per key=value pair |
| `MAX_PAIRS` | `180` | Maximum number of baggage pairs |

### Methods

#### extract

```python
def extract(self, carrier: Carrier) -> dict[str, str]
```

Extract baggage from a carrier into the current context. Sets each extracted pair in the context baggage via `set_baggage()`.

**Returns:** Dict of extracted key-value pairs.

#### inject

```python
def inject(
    self,
    carrier: Carrier,
    baggage: dict[str, str] | None = None,
) -> None
```

Inject baggage into a carrier. Uses the provided *baggage* dict, or falls back to the current context baggage if *baggage* is `None`. Keys and values are URL-encoded.

| Name | Type | Default | Description |
|---|---|---|---|
| `carrier` | `Carrier` | *(required)* | Target carrier for injection |
| `baggage` | `dict[str, str] \| None` | `None` | Explicit baggage to inject. Falls back to context baggage |

### Example

```python
from exo.trace import BaggagePropagator, DictCarrier, set_baggage

propagator = BaggagePropagator()

# Inject into outgoing headers
set_baggage("user_id", "user-123")
carrier = DictCarrier()
propagator.inject(carrier)
print(carrier.headers)  # {"baggage": "user_id=user-123"}

# Extract from incoming headers
incoming = DictCarrier({"baggage": "session_id=sess-456"})
pairs = propagator.extract(incoming)
print(pairs)  # {"session_id": "sess-456"}
```

---

## SpanConsumer

```python
class SpanConsumer(ABC)
```

Abstract base class for span consumers. Span consumers receive completed spans for processing (e.g. logging, exporting to external systems, analytics).

### Abstract properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | Unique name for this consumer |

### Abstract methods

#### consume

```python
def consume(self, spans: Sequence[Any]) -> None
```

Process a batch of completed spans.

---

## Span consumer registry functions

### register_span_consumer

```python
def register_span_consumer(consumer: SpanConsumer | None = None) -> Any
```

Register a span consumer, directly or as a decorator.

**Direct usage:**
```python
register_span_consumer(MyConsumer())
```

**Decorator usage:**
```python
@register_span_consumer
class MyConsumer(SpanConsumer):
    @property
    def name(self) -> str:
        return "my_consumer"

    def consume(self, spans):
        for span in spans:
            print(span)
```

When used as a decorator on a class, the class is instantiated with no arguments and the resulting instance is registered.

### get_span_consumer

```python
def get_span_consumer(name: str) -> SpanConsumer | None
```

Look up a registered span consumer by name. Returns `None` if not found.

### list_span_consumers

```python
def list_span_consumers() -> list[str]
```

Return the names of all registered span consumers.

### dispatch_spans

```python
def dispatch_spans(spans: Sequence[Any]) -> None
```

Send a batch of spans to all registered consumers. Consumers are invoked in registration order.

### clear_span_consumers

```python
def clear_span_consumers() -> None
```

Remove all registered span consumers (useful for testing).

### Example

```python
from exo.trace import (
    SpanConsumer,
    register_span_consumer,
    dispatch_spans,
    list_span_consumers,
)

class LoggingConsumer(SpanConsumer):
    @property
    def name(self) -> str:
        return "logger"

    def consume(self, spans):
        for span in spans:
            print(f"Span: {span}")

register_span_consumer(LoggingConsumer())
print(list_span_consumers())  # ["logger"]

dispatch_spans([{"name": "agent.run", "duration": 1.5}])
```
