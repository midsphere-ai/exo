# exo.train.trainer

Base Trainer ABC for agent fine-tuning with multi-phase lifecycle.

```python
from exo.train.trainer import (
    TrainConfig,
    Trainer,
    TrainerError,
    TrainerState,
    TrainMetrics,
)
```

---

## TrainerError

```python
class TrainerError(Exception)
```

Error during training operations.

---

## TrainerState

```python
class TrainerState(StrEnum)
```

Trainer lifecycle state.

| Value | Description |
|---|---|
| `CREATED = "created"` | Trainer instantiated, not yet validated |
| `VALIDATED = "validated"` | All validation checks passed |
| `TRAINING = "training"` | Training in progress |
| `COMPLETED = "completed"` | Training completed successfully |
| `FAILED = "failed"` | Training failed |

---

## TrainMetrics

```python
@dataclass(slots=True)
class TrainMetrics
```

Metrics produced by training or evaluation.

| Field | Type | Default | Description |
|---|---|---|---|
| `loss` | `float` | `0.0` | Training loss |
| `accuracy` | `float` | `0.0` | Accuracy metric |
| `steps` | `int` | `0` | Number of training steps |
| `extra` | `dict[str, Any]` | `{}` | Backend-specific metrics |

---

## TrainConfig

```python
@dataclass(slots=True)
class TrainConfig
```

Minimal training configuration. Subclass or extend via `extra` for backend-specific settings.

| Field | Type | Default | Description |
|---|---|---|---|
| `epochs` | `int` | `1` | Number of training epochs |
| `batch_size` | `int` | `8` | Batch size |
| `learning_rate` | `float` | `1e-5` | Learning rate |
| `output_dir` | `str` | `""` | Output directory for checkpoints |
| `extra` | `dict[str, Any]` | `{}` | Backend-specific settings |

---

## Trainer

```python
class Trainer(ABC)(config: TrainConfig | None = None)
```

Abstract base class for agent training frameworks.

The lifecycle follows a strict phase ordering:
1. Call `check_agent`, `check_dataset`, `check_reward`, `check_config` (in any order) to validate all inputs.
2. Call `mark_validated()` when all checks pass.
3. Call `train()` to execute training.
4. Call `evaluate()` to run evaluation on a test set.

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `config` | `TrainConfig \| None` | `None` | Training configuration. Defaults to `TrainConfig()` |

### Properties

| Property | Type | Description |
|---|---|---|
| `state` | `TrainerState` | Current lifecycle state |
| `config` | `TrainConfig` | Training configuration |

### Abstract methods

#### check_agent

```python
def check_agent(self, agent: Any) -> None
```

Validate that *agent* meets training requirements.

**Raises:** `TrainerError` -- If validation fails.

#### check_dataset

```python
def check_dataset(
    self,
    train_data: Any,
    test_data: Any | None = None,
) -> None
```

Validate training (and optional test) data.

**Raises:** `TrainerError` -- If validation fails.

#### check_reward

```python
def check_reward(self, reward_fn: Any | None = None) -> None
```

Validate reward function or scoring mechanism.

**Raises:** `TrainerError` -- If validation fails.

#### check_config

```python
def check_config(
    self,
    config: TrainConfig | dict[str, Any] | None = None,
) -> None
```

Validate and optionally update training configuration.

**Raises:** `TrainerError` -- If configuration is invalid.

#### train

```python
async def train(self) -> TrainMetrics
```

Execute the training loop. Must be called after `mark_validated()`.

**Returns:** Training metrics.

**Raises:** `TrainerError` -- If training fails or trainer not validated.

#### evaluate

```python
async def evaluate(self, test_data: Any | None = None) -> TrainMetrics
```

Run evaluation on test data.

| Name | Type | Default | Description |
|---|---|---|---|
| `test_data` | `Any \| None` | `None` | Optional test dataset. If None, use data from `check_dataset` |

**Returns:** Evaluation metrics.

**Raises:** `TrainerError` -- If evaluation fails.

### Concrete methods

#### mark_validated

```python
def mark_validated(self) -> None
```

Transition to `VALIDATED` state after all checks pass.

**Raises:** `TrainerError` -- If not in `CREATED` state.

### Example

```python
from exo.train import TrainConfig

# See VeRLTrainer for a concrete implementation
config = TrainConfig(epochs=3, batch_size=16, learning_rate=2e-5)
```
