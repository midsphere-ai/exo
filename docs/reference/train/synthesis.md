# exo.train.synthesis

Data synthesis utilities for generating training data from trajectories.

```python
from exo.train.synthesis import (
    DataSynthesiser,
    SynthesisConfig,
    SynthesisError,
    SynthesisPipeline,
    SynthesisResult,
    SynthesisStrategy,
    TemplateSynthesiser,
    augment_add_noise,
    augment_swap_io,
    deduplicate,
    filter_by_score,
    split_dataset,
)
```

---

## SynthesisError

```python
class SynthesisError(Exception)
```

Error during data synthesis.

---

## SynthesisStrategy

```python
class SynthesisStrategy(StrEnum)
```

Strategy for generating synthetic data.

| Value | Description |
|---|---|
| `LLM = "llm"` | LLM-based generation |
| `TEMPLATE = "template"` | Template-based transforms |
| `AUGMENT = "augment"` | Data augmentation |

---

## SynthesisConfig

```python
@dataclass(frozen=True, slots=True)
class SynthesisConfig
```

Configuration for a synthesis pipeline run.

| Field | Type | Default | Description |
|---|---|---|---|
| `strategy` | `SynthesisStrategy` | `SynthesisStrategy.TEMPLATE` | Synthesis strategy |
| `num_samples` | `int` | `10` | Number of samples to generate (must be >= 1) |
| `train_ratio` | `float` | `0.9` | Fraction for training set (must be in (0, 1]) |
| `min_score` | `float \| None` | `None` | Minimum score threshold for filtering |
| `seed` | `int \| None` | `None` | Random seed for reproducibility |
| `extra` | `dict[str, Any]` | `{}` | Backend-specific settings |

**Raises:** `ValueError` -- If `num_samples < 1` or `train_ratio` not in `(0, 1]`.

---

## SynthesisResult

```python
@dataclass(frozen=True, slots=True)
class SynthesisResult
```

Output of a synthesis pipeline run.

| Field | Type | Default | Description |
|---|---|---|---|
| `items` | `tuple[dict[str, Any], ...]` | `()` | All synthesized items |
| `train_items` | `tuple[dict[str, Any], ...]` | `()` | Training split |
| `test_items` | `tuple[dict[str, Any], ...]` | `()` | Test split |
| `metadata` | `dict[str, Any]` | `{}` | Pipeline metadata |

### Properties

| Property | Type | Description |
|---|---|---|
| `total` | `int` | Total number of items |
| `train_count` | `int` | Number of training items |
| `test_count` | `int` | Number of test items |

### Methods

#### to_json

```python
def to_json(self) -> str
```

Serialise to JSON string.

---

## Standalone functions

### filter_by_score

```python
def filter_by_score(
    items: Sequence[dict[str, Any]],
    min_score: float,
    *,
    score_key: str = "score",
) -> list[dict[str, Any]]
```

Keep only items where `score_key >= min_score`.

| Name | Type | Default | Description |
|---|---|---|---|
| `items` | `Sequence[dict[str, Any]]` | *(required)* | Items to filter |
| `min_score` | `float` | *(required)* | Minimum score threshold |
| `score_key` | `str` | `"score"` | Key containing the score value |

### split_dataset

```python
def split_dataset(
    items: Sequence[dict[str, Any]],
    train_ratio: float = 0.9,
    *,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]
```

Split items into train/test lists by ratio. Shuffles before splitting.

| Name | Type | Default | Description |
|---|---|---|---|
| `items` | `Sequence[dict[str, Any]]` | *(required)* | Items to split |
| `train_ratio` | `float` | `0.9` | Fraction for training set |
| `seed` | `int \| None` | `None` | Random seed |

**Returns:** `(train_items, test_items)` tuple.

**Raises:** `ValueError` -- If `train_ratio` not in `(0, 1]`.

### deduplicate

```python
def deduplicate(
    items: Sequence[dict[str, Any]],
    *,
    key: str = "input",
) -> list[dict[str, Any]]
```

Remove duplicate items based on a key field.

| Name | Type | Default | Description |
|---|---|---|---|
| `items` | `Sequence[dict[str, Any]]` | *(required)* | Items to deduplicate |
| `key` | `str` | `"input"` | Field name to check for duplicates |

### augment_swap_io

```python
def augment_swap_io(item: dict[str, Any]) -> dict[str, Any]
```

Create an augmented item by making the output a new input prompt. The new input becomes `"Given this answer: {output}\nWhat was the original question?"` and the new output becomes the original input. Adds `metadata.augmented = "swap_io"`.

### augment_add_noise

```python
def augment_add_noise(
    item: dict[str, Any],
    *,
    noise_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]
```

Create an augmented item with noise applied to the input.

| Name | Type | Default | Description |
|---|---|---|---|
| `item` | `dict[str, Any]` | *(required)* | Source item |
| `noise_fn` | `Callable[[str], str] \| None` | `None` | Custom noise function. Defaults to appending `" (rephrased)"` |

---

## DataSynthesiser

```python
class DataSynthesiser(ABC)
```

Abstract base for custom data synthesisers.

### Abstract methods

#### synthesise

```python
async def synthesise(
    self,
    source: Sequence[dict[str, Any]],
    config: SynthesisConfig,
) -> list[dict[str, Any]]
```

Generate synthetic items from source data.

---

## TemplateSynthesiser

```python
class TemplateSynthesiser(
    transforms: Sequence[Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
)
```

Generate items from trajectory items via template transforms.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `transforms` | `Sequence[Callable] \| None` | `None` | Transform functions. Defaults to `(augment_swap_io,)` |

---

## SynthesisPipeline

```python
class SynthesisPipeline(
    config: SynthesisConfig | None = None,
    *,
    synthesiser: DataSynthesiser | None = None,
)
```

Orchestrates data synthesis from trajectory items.

Pipeline phases:
1. Filter source items (optional score threshold)
2. Deduplicate
3. Synthesise via a `DataSynthesiser`
4. Split into train/test sets

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `config` | `SynthesisConfig \| None` | `None` | Pipeline configuration. Defaults to `SynthesisConfig()` |
| `synthesiser` | `DataSynthesiser \| None` | `None` | Data synthesiser. Defaults to `TemplateSynthesiser()` |

### Properties

| Property | Type | Description |
|---|---|---|
| `config` | `SynthesisConfig` | Pipeline configuration |
| `synthesiser` | `DataSynthesiser` | The synthesiser instance |

### Methods

#### run

```python
async def run(
    self,
    source: Sequence[dict[str, Any]],
) -> SynthesisResult
```

Execute the full synthesis pipeline.

**Returns:** `SynthesisResult` with items, train/test splits, and metadata.

### Example

```python
from exo.train import SynthesisPipeline, SynthesisConfig

pipeline = SynthesisPipeline(
    SynthesisConfig(num_samples=100, train_ratio=0.8, seed=42),
)

source_data = [
    {"input": "What is Python?", "output": "A programming language.", "score": 0.9},
    {"input": "What is Java?", "output": "A programming language.", "score": 0.8},
]

result = await pipeline.run(source_data)
print(f"Generated {result.total} items: {result.train_count} train, {result.test_count} test")
```
