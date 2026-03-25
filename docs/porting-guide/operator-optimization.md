# Operator Pattern & Self-Optimization — agent-core to Orbiter Mapping

**Epic:** 10 — Operator Pattern with Self-Optimization
**Date:** 2026-03-11

This document maps agent-core's (openJiuwen) `agent_evolving/` system to
Orbiter's operator pattern in the `orbiter-train` package, helping contributors
familiar with either framework navigate both.

---

## 1. Agent-Core Overview

Agent-core's self-optimization system lives in `openjiuwen/agent_evolving/` and
enables iterative improvement of agent parameters (prompts, tool descriptions,
memory configs) through textual gradients.

### Key Components

**`Operator` ABC** — The atomic unit of execution and optimization. Each
operator wraps a single agent capability and exposes tunable parameters via
`get_tunables()`, state snapshots via `get_state()`/`load_state()`, and
parameter mutation via `set_parameter()`. Concrete implementations:

| Operator | Domain | Tunable Parameters |
|----------|--------|--------------------|
| `LLMCallOperator` | LLM calls | `system_prompt`, `user_prompt` |
| `ToolCallOperator` | Tool invocations | `tool_description`, `tool_filter` |
| `MemoryCallOperator` | Memory retrieval | `enabled`, `max_retries` |

**`TunableSpec`** — Declares what an optimizer can modify on an operator.
Each spec has a `name`, `kind` (one of `PROMPT`, `CONTINUOUS`, `DISCRETE`,
`TOOL_SELECTOR`, `MEMORY_SELECTOR`), optional `path`, and `constraint` string.
Analogous to `nn.Module.parameters()` in PyTorch.

**`Trainer`** — Orchestrates the full optimization loop: forward pass →
trajectory extraction → update generation → candidate selection → checkpointing.
Manages an `EvolveCheckpoint` for operator state persistence and resume across
training runs.

**`TracerTrajectoryExtractor`** — Builds DAG-linked `TrajectoryStep` sequences
from agent-core's `Session.tracer()` spans. Each step is typed (`LLM`, `TOOL`,
`MEMORY`, `WORKFLOW`, `AGENT`) and links to its originating operator via
`operator_id`.

**`InstructionOptimizer`** — Textual gradient-based prompt optimization.
Two-phase loop: `backward()` analyzes failures to generate natural-language
gradients describing *what went wrong*; `step()` rewrites prompts to address
those issues.

**`ToolOptimizerBase`** — Multi-stage beam search for tool description
optimization: generate → evaluate → select → refine.

**`MemoryOptimizerBase`** — Optimizes memory retrieval configuration
(enable/disable, retry counts) based on trajectory analysis.

**`SingleDimUpdater` / `MultiDimUpdater`** — Updaters compose optimizers.
`SingleDimUpdater` wraps one optimizer; `MultiDimUpdater` composes multiple
domain-specific optimizers with attribution (attributes failures to the
responsible domain before running that domain's optimizer).

### 3-Dimension Evolution

Agent-core's evolution operates across three independent dimensions, each
with its own optimization strategy:

1. **Prompt** — Textual gradients rewrite system/user prompts (via `InstructionOptimizer`)
2. **Tool** — Beam search improves tool descriptions (via `ToolOptimizerBase`)
3. **Memory** — Configuration tuning for memory retrieval (via `MemoryOptimizerBase`)

These can run independently (`SingleDimUpdater`) or jointly with failure
attribution (`MultiDimUpdater`).

### Checkpoint/Resume

Agent-core persists `EvolveCheckpoint` objects containing operator states,
optimizer states, and training metadata. Training can resume from any saved
checkpoint, restoring all operator parameters and optimizer gradient history.

---

## 2. Orbiter Equivalent

Orbiter's operator pattern lives in `orbiter-train` (`packages/orbiter-train/`)
and was added **alongside** the existing `EvolutionPipeline`/`SynthesisPipeline`
rather than replacing them — the two paradigms serve different purposes.

### Architecture Difference

Where agent-core uses a monolithic `Trainer` that owns the entire optimization
loop, Orbiter separates concerns into composable pieces that integrate with
the existing `Trainer` ABC lifecycle:

```python
# Agent-core: monolithic trainer
trainer = AgentEvolvingTrainer(agent, dataset, optimizer)
trainer.train(epochs=5)

# Orbiter: composable trainer + updater + optimizer
optimizer = InstructionOptimizer(operators, llm_fn=my_llm)
updater = SingleDimUpdater(optimizer)
trainer = OperatorTrainer(updater=updater, evaluator=my_eval_fn)
trainer.check_agent(agent)
trainer.check_dataset(train_data, test_data)
trainer.check_config(OperatorTrainConfig(epochs=5))
trainer.mark_validated()
metrics = await trainer.train()
```

The key difference: Orbiter's `OperatorTrainer` inherits from the `Trainer` ABC,
gaining the lifecycle state machine (CREATED → VALIDATED → TRAINING → COMPLETED)
and validation guards for free.

### Component Mapping

| Agent-Core Component | Orbiter Equivalent | Notes |
|---------------------|-------------------|-------|
| `Operator` ABC | `Operator` ABC (`operator/base.py`) | Same interface: `get_tunables()`, `get_state()`/`load_state()`, `invoke()` |
| `LLMCallOperator` | `LLMCallOperator` (`operator/llm_call.py`) | Adds `LLMCallTrace` recording |
| `ToolCallOperator` | `ToolCallOperator` (`operator/tool_call.py`) | Adds `ToolCallTrace` recording |
| `MemoryCallOperator` | `MemoryCallOperator` (`operator/memory_call.py`) | Adds `MemoryCallTrace` with retry logic |
| `TunableSpec` | `TunableSpec` (`operator/base.py`) | Frozen dataclass; adds `path` and `constraint` fields |
| `TunableKind` | `TunableKind` (`operator/base.py`) | StrEnum; adds `TOOL_SELECTOR` and `MEMORY_SELECTOR` kinds |
| `InstructionOptimizer` | `InstructionOptimizer` (`optimizer.py`) | Two-phase backward/step; preserves `{{...}}` template variables |
| `ToolOptimizerBase` | `ToolOptimizer` (`optimizer.py`) | Four-stage beam search pipeline |
| `MemoryOptimizerBase` | *(handled by MemoryCallOperator tunables)* | Memory optimization via operator tunables rather than separate optimizer |
| `SingleDimUpdater` | `SingleDimUpdater` (`updater/`) | Wraps single `BaseOptimizer` |
| `MultiDimUpdater` | `MultiDimUpdater` (`updater/`) | Domain-specific composition with attribution |
| `Trainer` (agent_evolving) | `OperatorTrainer` (`operator_trainer.py`) | Extends Orbiter's `Trainer` ABC with operator lifecycle |
| `TracerTrajectoryExtractor` | `DefaultTrajectoryExtractor` (`trajectory/extractor.py`) | Dict-based instead of tracer-span-based; `TrajectoryExtractor` ABC for custom implementations |
| `TrajectoryStep` | `TrajectoryStep` (`trajectory/types.py`) | Adds `StepKind` enum, `ExecutionSpec`, `Trajectory` container |
| `EvolveCheckpoint` | `OperatorCheckpoint` + `CheckpointManager` | Protocol-based; `FileCheckpointStore` for JSON persistence |

### Key Orbiter Additions Beyond Agent-Core

**`BaseOptimizer` ABC** — Agent-core's optimizers are standalone classes.
Orbiter introduces a formal `BaseOptimizer` ABC with `bind()`, `backward()`,
`step()`, `add_trajectory()`, and `requires_forward_data()` — giving all
optimizers a uniform interface.

**`TextualParameter`** — Explicit container for optimizer gradients, keyed by
`(operator_id, target)`. Agent-core stores gradients implicitly in optimizer
state.

**`Updater` Protocol** — Formal protocol separating update logic from training.
Supports both single-domain and multi-domain optimization with the same interface.

**`CheckpointManager` Protocol** — Pluggable checkpoint policy (`should_save`,
`build`, `restore`) with `DefaultCheckpointManager` implementation supporting
periodic and improvement-triggered saves.

**Lifecycle State Machine** — `OperatorTrainer` inherits Orbiter's `Trainer`
validation phase (`check_agent`, `check_dataset`, `check_reward`, `check_config`,
`mark_validated`), preventing training on invalid configurations.

---

## 3. Code Comparison

### Defining Operators

```python
# Agent-core
from openjiuwen.agent_evolving import LLMCallOperator

op = LLMCallOperator(
    operator_id="summarizer",
    system_prompt="Summarize the following text.",
)
tunables = op.get_tunables()  # dict of TunableSpec

# Orbiter
from orbiter.train.operator import LLMCallOperator

op = LLMCallOperator(
    name="summarizer",
    system_prompt="Summarize the following text.",
    llm_fn=my_llm_fn,
)
tunables = op.get_tunables()  # list[TunableSpec]
```

### Running an Optimization Loop

```python
# Agent-core
from openjiuwen.agent_evolving import (
    InstructionOptimizer, SingleDimUpdater, Trainer
)

optimizer = InstructionOptimizer(llm=meta_llm)
updater = SingleDimUpdater(optimizer)
trainer = Trainer(agent, train_data, updater)
trainer.train(epochs=3)
# Checkpoint saved implicitly

# Orbiter
from orbiter.train.operator_trainer import OperatorTrainer, OperatorTrainConfig
from orbiter.train.optimizer import InstructionOptimizer

optimizer = InstructionOptimizer(operators=agent.operators, llm_fn=meta_llm)
updater = SingleDimUpdater(optimizer)
trainer = OperatorTrainer(updater=updater, evaluator=eval_fn)

# Validation phase (required)
trainer.check_agent(agent)
trainer.check_dataset(train_data, test_data)
trainer.check_config(OperatorTrainConfig(epochs=3, checkpoint_dir="./ckpts"))
trainer.mark_validated()

# Training phase
metrics = await trainer.train()  # async, returns TrainMetrics
# Checkpoint managed via CheckpointManager protocol
```

### Textual Gradient Flow

```python
# Agent-core: implicit gradient flow
optimizer.backward(failing_cases)  # writes gradients internally
updates = optimizer.step()         # returns new parameter values
agent.apply_updates(updates)

# Orbiter: explicit TextualParameter gradients
optimizer.backward(evaluated_cases)  # writes TextualParameter.gradients
updates = optimizer.step()           # Updates = dict[(op_id, target), value]
for (op_id, target), value in updates.items():
    operators[op_id].set_parameter(target, value)
```

### Multi-Domain Optimization

```python
# Agent-core
multi = MultiDimUpdater({
    "llm": InstructionOptimizer(llm),
    "tool": ToolOptimizer(llm),
    "memory": MemoryOptimizer(llm),
})
updates = multi.update(trajectories, cases)

# Orbiter — same pattern
from orbiter.train.optimizer import InstructionOptimizer, ToolOptimizer

multi = MultiDimUpdater({
    "llm": InstructionOptimizer(operators, llm_fn=meta_llm),
    "tool": ToolOptimizer(operators, llm_fn=meta_llm),
})
updates = multi.update(trajectories, evaluated_cases)
```

---

## 4. How EvolutionPipeline/SynthesisPipeline Coexist

The operator pattern and existing evolution system serve **different paradigms**:

| Aspect | EvolutionPipeline | Operator Pattern |
|--------|-------------------|-----------------|
| **Optimizes** | Training data (synthesis + augmentation) | Agent parameters (prompts, tool descriptions) |
| **Strategy** | `EvolutionStrategy` ABC (synthesise/train/evaluate) | `BaseOptimizer` ABC (backward/step) |
| **Trainer** | Pluggable (VeRLTrainer, custom) | `OperatorTrainer` (textual gradients) |
| **Data flow** | `SynthesisPipeline` → `TrajectoryDataset` → training | Trajectories → optimizers → parameter updates |
| **Use case** | Fine-tuning, RL, data augmentation | Prompt engineering, tool description tuning |

### Composition Points

The two systems compose naturally:

1. **EvolutionStrategy using operators** — An `EvolutionStrategy.train()` method
   can internally use `OperatorTrainer` to optimize agent parameters as part of
   a broader evolution loop.

2. **Operator optimization using SynthesisPipeline** — `OperatorTrainer` can use
   `SynthesisPipeline` to augment its training cases before running optimization.

3. **Shared trajectory infrastructure** — Both systems use `TrajectoryDataset`
   for data capture. The operator system adds finer-grained `TrajectoryStep`
   for attribution, but these coexist with message-level `TrajectoryItem`.

```python
# Example: EvolutionStrategy that uses operator optimization internally
class OperatorEvolutionStrategy(EvolutionStrategy):
    async def train(self, agent, data, epoch):
        optimizer = InstructionOptimizer(agent.operators, llm_fn=self.llm)
        updater = SingleDimUpdater(optimizer)
        trainer = OperatorTrainer(updater=updater, evaluator=self.eval_fn)
        trainer.check_agent(agent)
        trainer.check_dataset(data)
        trainer.check_config(OperatorTrainConfig(epochs=1))
        trainer.mark_validated()
        await trainer.train()
```

---

## 5. Migration Table

| Agent-Core Path | Orbiter Import | Symbol |
|----------------|----------------|--------|
| `openjiuwen.agent_evolving.Operator` | `orbiter.train.operator.Operator` | ABC with `get_tunables()`, `invoke()`, `get_state()`/`load_state()` |
| `openjiuwen.agent_evolving.LLMCallOperator` | `orbiter.train.operator.LLMCallOperator` | Wraps LLM calls; tunables: `system_prompt`, `user_prompt` |
| `openjiuwen.agent_evolving.ToolCallOperator` | `orbiter.train.operator.ToolCallOperator` | Wraps tool invocations; tunables: `tool_description` |
| `openjiuwen.agent_evolving.MemoryCallOperator` | `orbiter.train.operator.MemoryCallOperator` | Wraps memory retrieval; tunables: `enabled`, `max_retries` |
| `openjiuwen.agent_evolving.TunableSpec` | `orbiter.train.operator.TunableSpec` | Frozen dataclass declaring tunable parameters |
| `openjiuwen.agent_evolving.TunableKind` | `orbiter.train.operator.TunableKind` | StrEnum: `PROMPT`, `CONTINUOUS`, `DISCRETE`, `TOOL_SELECTOR`, `MEMORY_SELECTOR` |
| `openjiuwen.agent_evolving.InstructionOptimizer` | `orbiter.train.optimizer.InstructionOptimizer` | Textual gradient prompt optimization (backward/step) |
| `openjiuwen.agent_evolving.ToolOptimizerBase` | `orbiter.train.optimizer.ToolOptimizer` | Beam search tool description optimization |
| `openjiuwen.agent_evolving.MemoryOptimizerBase` | *(via MemoryCallOperator tunables)* | Memory config optimization through operator tunables |
| `openjiuwen.agent_evolving.SingleDimUpdater` | `orbiter.train.updater.SingleDimUpdater` | Single-optimizer wrapper |
| `openjiuwen.agent_evolving.MultiDimUpdater` | `orbiter.train.updater.MultiDimUpdater` | Multi-domain composition with attribution |
| `openjiuwen.agent_evolving.Trainer` | `orbiter.train.operator_trainer.OperatorTrainer` | Extends Orbiter `Trainer` ABC with operator optimization loop |
| `openjiuwen.agent_evolving.TracerTrajectoryExtractor` | `orbiter.train.trajectory.DefaultTrajectoryExtractor` | Dict-based extraction (replaces tracer-span-based) |
| `openjiuwen.agent_evolving.TrajectoryStep` | `orbiter.train.trajectory.TrajectoryStep` | Frozen dataclass with `StepKind`, `operator_id`, `timing` |
| `openjiuwen.agent_evolving.EvolveCheckpoint` | `orbiter.train.checkpointing.OperatorCheckpoint` | Checkpoint with `operators_state`, `updater_state`, `best_score` |
| *(no equivalent)* | `orbiter.train.operator.base.TunableKind.TOOL_SELECTOR` | New kind for tool selection parameters |
| *(no equivalent)* | `orbiter.train.operator.base.TunableKind.MEMORY_SELECTOR` | New kind for memory selection parameters |
| *(no equivalent)* | `orbiter.train.optimizer.BaseOptimizer` | Formal ABC for all optimizers |
| *(no equivalent)* | `orbiter.train.optimizer.TextualParameter` | Explicit gradient container per operator |
| *(no equivalent)* | `orbiter.train.updater.Updater` | Protocol for update strategies |
| *(no equivalent)* | `orbiter.train.checkpointing.CheckpointManager` | Protocol for checkpoint policy |
| *(no equivalent)* | `orbiter.train.checkpointing.DefaultCheckpointManager` | Periodic + improvement-triggered saves |
| *(no equivalent)* | `orbiter.train.checkpointing.FileCheckpointStore` | JSON file persistence for checkpoints |
| *(no equivalent)* | `orbiter.train.trajectory.StepKind` | StrEnum: `LLM`, `TOOL`, `MEMORY`, `WORKFLOW`, `AGENT` |
| *(no equivalent)* | `orbiter.train.trajectory.ExecutionSpec` | Execution metadata (case_id, execution_id, seed, tags) |
| *(no equivalent)* | `orbiter.train.trajectory.Trajectory` | Container with steps + optional DAG edges |
