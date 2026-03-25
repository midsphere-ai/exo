# exo.train.evolution

Agent evolution utilities for iterative improvement via multi-epoch synthesis-training-evaluation cycles.

```python
from exo.train.evolution import (
    EpochResult,
    EvolutionConfig,
    EvolutionError,
    EvolutionPhase,
    EvolutionPipeline,
    EvolutionResult,
    EvolutionState,
    EvolutionStrategy,
)
```

---

## EvolutionError

```python
class EvolutionError(Exception)
```

Error during evolution operations.

---

## EvolutionPhase

```python
class EvolutionPhase(StrEnum)
```

Phases in an evolution epoch.

| Value | Description |
|---|---|
| `SYNTHESIS = "synthesis"` | Data synthesis / augmentation |
| `TRAINING = "training"` | Model fine-tuning |
| `EVALUATION = "evaluation"` | Performance evaluation |

---

## EvolutionState

```python
class EvolutionState(StrEnum)
```

State of the evolution pipeline.

| Value | Description |
|---|---|
| `IDLE = "idle"` | Ready to run |
| `RUNNING = "running"` | Currently executing |
| `COMPLETED = "completed"` | Finished successfully |
| `FAILED = "failed"` | Failed during execution |

---

## EvolutionConfig

```python
@dataclass(frozen=True, slots=True)
class EvolutionConfig
```

Configuration for an evolution run.

| Field | Type | Default | Description |
|---|---|---|---|
| `max_epochs` | `int` | `1` | Number of evolution cycles to run (must be >= 1) |
| `phases` | `tuple[EvolutionPhase, ...]` | `(SYNTHESIS, TRAINING, EVALUATION)` | Which phases to execute each epoch |
| `early_stop_threshold` | `float \| None` | `None` | Stop if evaluation accuracy >= this value (must be in [0, 1]) |
| `extra` | `dict[str, Any]` | `{}` | Backend-specific settings |

**Raises:** `ValueError` -- If `max_epochs < 1` or `early_stop_threshold` not in `[0, 1]`.

---

## EpochResult

```python
@dataclass(slots=True)
class EpochResult
```

Metrics for a single evolution epoch.

| Field | Type | Default | Description |
|---|---|---|---|
| `epoch` | `int` | `0` | Epoch index |
| `synthesis_count` | `int` | `0` | Number of items synthesized |
| `train_loss` | `float` | `0.0` | Training loss |
| `eval_accuracy` | `float` | `0.0` | Evaluation accuracy |
| `extra` | `dict[str, Any]` | `{}` | Backend-specific metrics |

---

## EvolutionResult

```python
@dataclass(slots=True)
class EvolutionResult
```

Aggregate result of a full evolution run.

| Field | Type | Default | Description |
|---|---|---|---|
| `epochs` | `list[EpochResult]` | `[]` | Per-epoch results |
| `final_accuracy` | `float` | `0.0` | Accuracy after the last epoch |
| `early_stopped` | `bool` | `False` | Whether early stopping was triggered |

### Properties

| Property | Type | Description |
|---|---|---|
| `total_epochs` | `int` | Number of epochs completed |
| `best_epoch` | `EpochResult \| None` | Epoch with highest `eval_accuracy` |

---

## EvolutionStrategy

```python
class EvolutionStrategy(ABC)
```

Pluggable strategy for each phase of an evolution epoch. Subclasses implement concrete backends (VeRL, TRL, custom loops, etc.).

### Abstract methods

#### synthesise

```python
async def synthesise(
    self,
    agent: Any,
    data: Sequence[dict[str, Any]],
    epoch: int,
) -> list[dict[str, Any]]
```

Generate or augment training data for this epoch.

**Returns:** New or augmented training items.

#### train

```python
async def train(
    self,
    agent: Any,
    data: Sequence[dict[str, Any]],
    epoch: int,
) -> float
```

Train the agent on *data*.

**Returns:** Training loss.

#### evaluate

```python
async def evaluate(
    self,
    agent: Any,
    data: Sequence[dict[str, Any]],
    epoch: int,
) -> float
```

Evaluate the agent.

**Returns:** Accuracy score in [0, 1].

---

## EvolutionPipeline

```python
class EvolutionPipeline(
    strategy: EvolutionStrategy,
    config: EvolutionConfig | None = None,
)
```

Multi-epoch evolution pipeline. Runs synthesis, training, and evaluation for each epoch, with optional early stopping when evaluation accuracy meets the threshold.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `strategy` | `EvolutionStrategy` | *(required)* | Strategy implementing the phase methods |
| `config` | `EvolutionConfig \| None` | `None` | Configuration. Defaults to `EvolutionConfig()` |

### Properties

| Property | Type | Description |
|---|---|---|
| `config` | `EvolutionConfig` | Pipeline configuration |
| `strategy` | `EvolutionStrategy` | The strategy instance |
| `state` | `EvolutionState` | Current pipeline state |

### Methods

#### run

```python
async def run(
    self,
    agent: Any,
    data: Sequence[dict[str, Any]],
) -> EvolutionResult
```

Execute the full evolution pipeline.

| Name | Type | Default | Description |
|---|---|---|---|
| `agent` | `Any` | *(required)* | The agent to evolve |
| `data` | `Sequence[dict[str, Any]]` | *(required)* | Initial training data |

**Returns:** `EvolutionResult` with per-epoch metrics.

**Raises:** `EvolutionError` -- If the pipeline is not idle or a phase fails.

#### reset

```python
def reset(self) -> None
```

Reset pipeline to `IDLE` so it can be re-run.

### Example

```python
from exo.train import EvolutionPipeline, EvolutionConfig, EvolutionStrategy

class MyStrategy(EvolutionStrategy):
    async def synthesise(self, agent, data, epoch):
        return data  # pass through

    async def train(self, agent, data, epoch):
        return 0.5  # loss

    async def evaluate(self, agent, data, epoch):
        return 0.8 + epoch * 0.05  # improving accuracy

pipeline = EvolutionPipeline(
    MyStrategy(),
    EvolutionConfig(max_epochs=5, early_stop_threshold=0.95),
)

result = await pipeline.run(my_agent, training_data)
print(f"Final accuracy: {result.final_accuracy}")
print(f"Early stopped: {result.early_stopped}")
print(f"Best epoch: {result.best_epoch.epoch}")
```
