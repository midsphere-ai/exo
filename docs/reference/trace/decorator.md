# exo.trace.decorator

`@traced` decorator and span context managers for function-level instrumentation.

```python
from exo.trace.decorator import (
    extract_metadata,
    get_user_frame,
    is_user_code,
    span_async,
    span_sync,
    traced,
)
```

---

## traced

```python
def traced(
    name: str | None = None,
    *,
    attributes: dict[str, Any] | None = None,
    extract_args: bool = False,
) -> Any
```

Decorator that wraps a function in an OpenTelemetry span. Supports sync functions, async functions, sync generators, and async generators. Metadata (qualname, module, line number, parameters) is automatically recorded as span attributes.

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str \| None` | `None` | Span name override. Defaults to `func.__qualname__` |
| `attributes` | `dict[str, Any] \| None` | `None` | Extra attributes merged onto the span |
| `extract_args` | `bool` | `False` | When True, record the function's call arguments as `arg.{name}` attributes |

**Returns:** The decorated function (same type: sync, async, generator, or async generator).

### Automatically recorded attributes

| Attribute | Description |
|---|---|
| `code.function` | Function qualified name |
| `code.module` | Module name |
| `code.lineno` | First line number |
| `code.filepath` | Relative file path |
| `code.parameters` | Parameter names (excluding `self`) |

### Example

```python
from exo.trace import traced

@traced()
async def process_request(query: str) -> str:
    return f"Processed: {query}"

@traced(name="custom.span", attributes={"env": "prod"})
def compute(x: int, y: int) -> int:
    return x + y

@traced(extract_args=True)
async def search(query: str, limit: int = 10) -> list:
    # Span will include arg.query="..." and arg.limit="10"
    return []
```

### Generator support

```python
@traced(name="data.stream")
def generate_items():
    for i in range(10):
        yield i

@traced(name="async.stream")
async def async_generate():
    for i in range(10):
        yield i
```

---

## span_sync

```python
@contextmanager
def span_sync(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Iterator[trace.Span]
```

Synchronous span context manager.

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | Span name |
| `attributes` | `dict[str, Any] \| None` | `None` | Span attributes |

**Yields:** `opentelemetry.trace.Span` -- The active span.

### Example

```python
from exo.trace import span_sync

with span_sync("my.operation", {"key": "value"}) as span:
    result = do_work()
    span.set_attribute("result.size", len(result))
```

---

## span_async

```python
@asynccontextmanager
async def span_async(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> AsyncIterator[trace.Span]
```

Asynchronous span context manager.

| Name | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | Span name |
| `attributes` | `dict[str, Any] \| None` | `None` | Span attributes |

**Yields:** `opentelemetry.trace.Span` -- The active span.

---

## extract_metadata

```python
def extract_metadata(func: Any) -> dict[str, Any]
```

Extract tracing metadata from a callable.

**Returns:** Dict with keys `code.function`, `code.module`, `code.lineno`, `code.filepath`, `code.parameters`.

---

## is_user_code

```python
def is_user_code(filename: str) -> bool
```

Return `True` if *filename* belongs to user code (not stdlib or trace library).

---

## get_user_frame

```python
def get_user_frame() -> inspect.FrameInfo | None
```

Walk the call stack to find the first user-code frame. Returns `None` if no user-code frame is found.
