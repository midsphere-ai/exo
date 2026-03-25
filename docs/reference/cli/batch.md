# exo_cli.batch

Batch execution for Exo CLI. Loads inputs from JSON, CSV, or JSONL files and runs an agent against each input concurrently with configurable parallelism.

```python
from exo_cli.batch import (
    BatchError,
    BatchItem,
    BatchResult,
    InputFormat,
    ItemResult,
    load_batch_items,
    batch_execute,
    results_to_csv,
    results_to_jsonl,
)
```

---

## BatchError

```python
class BatchError(Exception)
```

Raised for batch-level errors (missing file, invalid format, concurrency issues).

---

## InputFormat

```python
class InputFormat(StrEnum)
```

Supported batch input formats.

| Value | Description |
|---|---|
| `JSON = "json"` | JSON array of objects |
| `CSV = "csv"` | CSV file with header row |
| `JSONL = "jsonl"` | JSON Lines (one JSON object per line) |

---

## BatchItem

```python
@dataclass(frozen=True, slots=True)
class BatchItem
```

A single batch input item.

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | *(required)* | Unique item identifier (row number or explicit id) |
| `input` | `str` | *(required)* | The text prompt sent to the agent |
| `metadata` | `dict[str, Any]` | `{}` | Extra columns / fields carried through from the source |

---

## ItemResult

```python
@dataclass(slots=True)
class ItemResult
```

Result for one batch item.

| Field | Type | Default | Description |
|---|---|---|---|
| `item_id` | `str` | *(required)* | Batch item identifier |
| `success` | `bool` | *(required)* | Whether execution succeeded |
| `output` | `str` | *(required)* | Agent output text (or error message) |
| `elapsed` | `float` | `0.0` | Execution time in seconds |
| `error` | `str` | `""` | Error message when `success` is `False` |

---

## BatchResult

```python
@dataclass(slots=True)
class BatchResult
```

Aggregate result for a batch run.

| Field | Type | Default | Description |
|---|---|---|---|
| `results` | `list[ItemResult]` | `[]` | Per-item results |
| `total` | `int` | `0` | Total items processed |
| `succeeded` | `int` | `0` | Count of successful items |
| `failed` | `int` | `0` | Count of failed items |

### Methods

#### summary

```python
def summary(self) -> str
```

Human-readable summary, e.g. `"10 items: 8 succeeded, 2 failed"`.

---

## load_batch_items

```python
def load_batch_items(
    path: str | Path,
    *,
    input_key: str = "input",
    id_key: str = "id",
    fmt: InputFormat | None = None,
) -> list[BatchItem]
```

Load batch items from a file. Auto-detects format from file extension if `fmt` is not specified.

| Name | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| Path` | *(required)* | Path to the input file |
| `input_key` | `str` | `"input"` | Column/field name containing the agent input text |
| `id_key` | `str` | `"id"` | Column/field name for item IDs (falls back to row number) |
| `fmt` | `InputFormat \| None` | `None` | Force a specific format (auto-detected from extension if `None`) |

**Returns:** `list[BatchItem]` -- Loaded batch items.

**Raises:** `BatchError` -- On missing file, invalid format, or missing input key.

### Example

```python
from exo_cli import load_batch_items

# Load from JSONL (auto-detected)
items = load_batch_items("inputs.jsonl")

# Load from CSV with custom key
items = load_batch_items("data.csv", input_key="prompt", id_key="row_id")

# Force format
items = load_batch_items("data.txt", fmt=InputFormat.JSONL)
```

---

## batch_execute

```python
async def batch_execute(
    agent: Any,
    items: Sequence[BatchItem],
    *,
    provider: Any = None,
    concurrency: int = 4,
    timeout: float = 0.0,
) -> BatchResult
```

Execute an agent against multiple inputs concurrently. Uses `asyncio.Semaphore` to limit concurrency.

| Name | Type | Default | Description |
|---|---|---|---|
| `agent` | `Any` | *(required)* | Agent or Swarm instance |
| `items` | `Sequence[BatchItem]` | *(required)* | Batch items to process |
| `provider` | `Any` | `None` | LLM provider (auto-resolved when `None`) |
| `concurrency` | `int` | `4` | Maximum concurrent executions |
| `timeout` | `float` | `0.0` | Per-item timeout in seconds (0 = no timeout) |

**Returns:** `BatchResult` with per-item results.

**Raises:** `BatchError` -- If `concurrency < 1`.

### Example

```python
from exo_cli import load_batch_items, batch_execute

items = load_batch_items("questions.jsonl")
result = await batch_execute(my_agent, items, concurrency=8, timeout=30.0)
print(result.summary())  # "5 items: 4 succeeded, 1 failed"
```

---

## results_to_jsonl

```python
def results_to_jsonl(batch: BatchResult) -> str
```

Serialize batch results to JSONL string. Each line is a JSON object with `id`, `success`, `output`, `elapsed`, and optionally `error`.

| Name | Type | Default | Description |
|---|---|---|---|
| `batch` | `BatchResult` | *(required)* | Batch results to serialize |

**Returns:** `str` -- JSONL-formatted string.

---

## results_to_csv

```python
def results_to_csv(batch: BatchResult) -> str
```

Serialize batch results to CSV string with columns: `id`, `success`, `output`, `elapsed`, `error`.

| Name | Type | Default | Description |
|---|---|---|---|
| `batch` | `BatchResult` | *(required)* | Batch results to serialize |

**Returns:** `str` -- CSV-formatted string with header row.

### Example

```python
from exo_cli import batch_execute, results_to_jsonl, results_to_csv

result = await batch_execute(agent, items)

# Save as JSONL
with open("results.jsonl", "w") as f:
    f.write(results_to_jsonl(result))

# Save as CSV
with open("results.csv", "w") as f:
    f.write(results_to_csv(result))
```
