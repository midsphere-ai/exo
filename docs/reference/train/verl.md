# exo.train.verl

VeRL integration for reinforcement learning from human feedback (RLHF).

```python
from exo.train.verl import (
    RewardSpec,
    VeRLAlgorithm,
    VeRLConfig,
    VeRLTrainer,
)
```

**Requires:** `pip install exo-train[verl]`

---

## VeRLAlgorithm

```python
class VeRLAlgorithm(StrEnum)
```

Supported VeRL RL algorithms.

| Value | Description |
|---|---|
| `PPO = "ppo"` | Proximal Policy Optimization |
| `GRPO = "grpo"` | Group Relative Policy Optimization |

---

## RewardSpec

```python
@dataclass(frozen=True, slots=True)
class RewardSpec
```

Descriptor for a reward function used during RL training. Either `callable` (an in-process function) or `module_path` + `func_name` (an importable reference) must be provided.

| Field | Type | Default | Description |
|---|---|---|---|
| `callable` | `Callable[..., float] \| None` | `None` | In-process reward function |
| `module_path` | `str` | `""` | Module containing the reward function |
| `func_name` | `str` | `""` | Function name within the module |

**Raises:** `TrainerError` -- If neither `callable` nor both `module_path` and `func_name` are provided.

### Methods

#### resolve

```python
def resolve(self) -> Callable[..., float]
```

Return the concrete callable, importing if necessary.

**Raises:** `TrainerError` -- If the function cannot be found or is not callable.

### Example

```python
# In-process callable
spec = RewardSpec(callable=lambda output, expected: 1.0 if output == expected else 0.0)

# Importable reference
spec = RewardSpec(module_path="my_rewards", func_name="accuracy_reward")
fn = spec.resolve()  # imports and returns the function
```

---

## VeRLConfig

```python
@dataclass(slots=True)
class VeRLConfig(TrainConfig)
```

VeRL-specific training configuration. Extends `TrainConfig` with RL algorithm selection, rollout parameters, and model/tokenizer references.

### Fields (inherited from TrainConfig)

| Field | Type | Default | Description |
|---|---|---|---|
| `epochs` | `int` | `1` | Number of training epochs |
| `batch_size` | `int` | `8` | Batch size |
| `learning_rate` | `float` | `1e-5` | Learning rate |
| `output_dir` | `str` | `""` | Output directory |
| `extra` | `dict[str, Any]` | `{}` | Extra settings |

### Fields (VeRL-specific)

| Field | Type | Default | Description |
|---|---|---|---|
| `algorithm` | `VeRLAlgorithm` | `VeRLAlgorithm.GRPO` | RL algorithm |
| `rollout_batch_size` | `int` | `4` | Rollout batch size (must be >= 1) |
| `ppo_epochs` | `int` | `4` | PPO inner loop epochs (must be >= 1) |
| `kl_coeff` | `float` | `0.1` | KL divergence coefficient |
| `clip_range` | `float` | `0.2` | PPO clip range (must be in [0, 1]) |
| `gamma` | `float` | `1.0` | Discount factor |
| `lam` | `float` | `0.95` | GAE lambda |
| `model_name` | `str` | `""` | Model name/path for VeRL |
| `tokenizer_name` | `str` | `""` | Tokenizer name/path |
| `max_prompt_length` | `int` | `1024` | Maximum prompt length in tokens |
| `max_response_length` | `int` | `512` | Maximum response length in tokens |

**Raises:** `ValueError` -- If `rollout_batch_size < 1`, `ppo_epochs < 1`, or `clip_range` not in `[0, 1]`.

---

## VeRLTrainer

```python
class VeRLTrainer(Trainer)(config: VeRLConfig | None = None)
```

Concrete trainer that integrates with the VeRL framework.

Lifecycle:
1. `check_agent(agent)` -- validate agent compatibility
2. `check_dataset(data)` -- validate dataset format
3. `check_reward(spec)` -- validate reward function
4. `check_config(cfg)` -- validate and merge VeRL config
5. `mark_validated()` -- transition to VALIDATED
6. `train()` -- execute RL training loop
7. `evaluate(test_data)` -- run evaluation

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `config` | `VeRLConfig \| None` | `None` | VeRL configuration. Defaults to `VeRLConfig()` |

### Properties

| Property | Type | Description |
|---|---|---|
| `verl_config` | `VeRLConfig` | Typed access to the VeRL-specific config |
| `state` | `TrainerState` | Current lifecycle state (inherited) |
| `config` | `TrainConfig` | Training configuration (inherited) |

### Methods

#### check_agent

```python
def check_agent(self, agent: Any) -> None
```

Validate that agent is usable for VeRL training. Agent must be non-None and should have an `instructions` attribute.

**Raises:** `TrainerError` -- If agent is None.

#### check_dataset

```python
def check_dataset(
    self,
    train_data: Any,
    test_data: Any | None = None,
) -> None
```

Validate training data format. Expects a sequence of dicts, each containing at least an `input` key.

**Raises:** `TrainerError` -- If data is empty, not a list/tuple, items are not dicts, or items lack `input` key.

#### check_reward

```python
def check_reward(self, reward_fn: Any | None = None) -> None
```

Validate reward function. Accepts a `RewardSpec`, a plain callable, or `None` (uses VeRL built-in reward).

**Raises:** `TrainerError` -- If reward_fn is an unsupported type.

#### check_config

```python
def check_config(
    self,
    config: TrainConfig | dict[str, Any] | None = None,
) -> None
```

Validate and optionally merge VeRL config overrides. Dict values are merged into `extra`. A `VeRLConfig` replaces the current config entirely.

#### train

```python
async def train(self) -> TrainMetrics
```

Execute the VeRL RL training loop. Requires VeRL to be installed.

**Returns:** `TrainMetrics` with training statistics.

**Raises:** `TrainerError` -- If not validated or VeRL is not installed.

#### evaluate

```python
async def evaluate(self, test_data: Any | None = None) -> TrainMetrics
```

Run evaluation on test data. Uses test data from `check_dataset` if not provided.

**Returns:** `TrainMetrics` with evaluation statistics.

### Example

```python
from exo.train import VeRLTrainer, VeRLConfig, VeRLAlgorithm, RewardSpec

config = VeRLConfig(
    algorithm=VeRLAlgorithm.GRPO,
    epochs=3,
    batch_size=16,
    rollout_batch_size=8,
    model_name="Qwen/Qwen2.5-7B",
    output_dir="/tmp/verl_output",
)

trainer = VeRLTrainer(config)

# Validation phase
trainer.check_agent(my_agent)
trainer.check_dataset(
    train_data=[{"input": "What is Python?"}, {"input": "Explain ML"}],
    test_data=[{"input": "What is Java?"}],
)
trainer.check_reward(RewardSpec(callable=my_reward_fn))
trainer.check_config()
trainer.mark_validated()

# Training phase
metrics = await trainer.train()
print(f"Steps: {metrics.steps}, Loss: {metrics.loss}")

# Evaluation phase
eval_metrics = await trainer.evaluate()
print(f"Accuracy: {eval_metrics.accuracy}")
```
