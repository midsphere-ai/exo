# Operator Pattern & Self-Optimization

**Status:** Proposed
**Epic:** 10 — Operator Pattern with Self-Optimization
**Date:** 2026-03-10
**Package:** `exo-train` (new `operator/`, `optimizer/`, `updater/` subpackages)

---

## 1. Motivation

Exo's `exo-train` package provides a solid training foundation:

- **Trainer ABC** — Lifecycle state machine (CREATED → VALIDATED → TRAINING → COMPLETED)
  with abstract `check_agent`, `check_dataset`, `check_reward`, `check_config`, `train`,
  `evaluate` methods.
- **EvolutionPipeline** — Multi-epoch loop with pluggable `EvolutionStrategy` driving
  synthesis → training → evaluation per epoch, with early stopping.
- **SynthesisPipeline** — Data augmentation via filter → dedup → synthesise → split,
  with `DataSynthesiser` ABC and `TemplateSynthesiser` concrete implementation.
- **TrajectoryDataset** — SAR (state-action-reward) capture with `TrajectoryItem`,
  `TrajectoryStrategy` ABC, and `DefaultStrategy` for message-based extraction.
- **VeRLTrainer** — Concrete `Trainer` for VeRL-based RL (PPO/GRPO) with `VeRLConfig`,
  `RewardSpec`, and lazy VeRL import.

However, it lacks **operator-centric self-optimization** — the ability to decompose
an agent into individually optimizable units (operators) and iteratively improve their
parameters (prompts, tool descriptions, memory configs) using textual gradients.

Agent-core (`openjiuwen/agent_evolving/`) provides a production-grade implementation:

1. **Operator ABC** — Atomic execution and optimization unit with `get_tunables()`,
   `set_parameter()`, `get_state()`/`load_state()`, and `invoke()`.
2. **Domain Optimizers** — `InstructionOptimizer` (LLM prompts), `ToolOptimizerBase`
   (tool descriptions), `MemoryOptimizerBase` (memory configs) with textual gradient
   backward/step pipeline.
3. **Updaters** — `SingleDimUpdater` (one optimizer) and `MultiDimUpdater` (multi-domain
   composition with attribution).
4. **Trainer** — Orchestrates forward pass → trajectory extraction → update generation →
   candidate selection → checkpointing.
5. **Trajectory Extraction** — `TracerTrajectoryExtractor` builds DAG-linked
   `TrajectoryStep` sequences from session tracer spans.
6. **Checkpointing** — `EvolveCheckpoint` with operator state persistence and resume.

This document designs how the operator pattern and self-optimization integrate with
Exo's existing training architecture.

---

## 2. Key Decision: Operator Pattern Adds Alongside Existing EvolutionStrategy

### Option A — Replace EvolutionStrategy with Operator/Updater pattern (rejected)

Replace the existing `EvolutionStrategy` ABC with the operator-centric optimizer
pattern. This would break all existing strategy implementations and remove the
simpler synthesis → training → evaluation workflow that works well for straightforward
fine-tuning scenarios.

### Option B — Add operator pattern as new modules alongside existing abstractions (chosen)

Add `operator/`, `optimizer/`, `updater/` subpackages inside `exo-train`. The
existing `Trainer` ABC, `EvolutionPipeline`, `EvolutionStrategy`, `SynthesisPipeline`,
`VeRLTrainer`, and `TrajectoryDataset` remain fully functional and unchanged.

**Why Option B:**

- Operator-centric optimization is a *different paradigm* from evolution strategies —
  it optimizes *agent parameters* (prompts, configs) rather than *training data*.
- Existing users of `EvolutionPipeline` and `VeRLTrainer` experience zero disruption.
- The two paradigms compose: an `EvolutionStrategy` could internally use operators
  for its `train()` phase, or an operator optimizer could use `SynthesisPipeline`
  to augment its training cases.
- Agent-core's `Trainer` maps cleanly to a new `OperatorTrainer` concrete subclass
  of Exo's `Trainer` ABC — it inherits the lifecycle state machine for free.

---

## 3. Component Design

### 3.1 Operator ABC (`operator/base.py`)

The fundamental unit of execution and optimization. Each operator wraps a single
agent capability (LLM call, tool invocation, memory retrieval) and exposes its
tunable parameters.

```python
class TunableKind(StrEnum):
    PROMPT = "prompt"
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    TOOL_SELECTOR = "tool_selector"
    MEMORY_SELECTOR = "memory_selector"

@dataclass(frozen=True, slots=True)
class TunableSpec:
    name: str
    kind: TunableKind
    path: str = ""
    constraint: str = ""

class Operator(ABC):
    @property
    @abstractmethod
    def operator_id(self) -> str: ...

    @abstractmethod
    def get_tunables(self) -> dict[str, TunableSpec]: ...

    @abstractmethod
    def set_parameter(self, target: str, value: Any) -> None: ...

    @abstractmethod
    def get_state(self) -> dict[str, Any]: ...

    @abstractmethod
    def load_state(self, state: dict[str, Any]) -> None: ...

    @abstractmethod
    async def invoke(self, inputs: dict[str, Any], **kwargs: Any) -> Any: ...
```

**Design notes:**
- `operator_id` links trajectory steps to operators for attribution.
- `get_tunables()` declares what the optimizer can modify — analogous to
  `nn.Module.parameters()` in PyTorch.
- `get_state()`/`load_state()` enables snapshot/rollback for candidate selection
  and checkpointing, following the same pattern as Exo's existing
  `EvolutionState` state machine.
- `invoke()` takes `**kwargs` rather than a typed `Session` to avoid coupling
  to agent-core's session model. Exo agents can pass context as needed.

### 3.2 Concrete Operators (`operator/`)

```python
class LLMCallOperator(Operator):
    """Wraps an LLM call with optimizable system/user prompts."""
    # Tunables: system_prompt, user_prompt
    # get_state() returns {"system_prompt": ..., "user_prompt": ...}

class ToolCallOperator(Operator):
    """Wraps a tool invocation with optimizable description/filter."""
    # Tunables: tool_description, tool_filter
    # get_state() returns {"tool_description": ..., "tool_filter": ...}

class MemoryCallOperator(Operator):
    """Wraps memory retrieval with optimizable config."""
    # Tunables: enabled, max_retries
    # get_state() returns {"enabled": ..., "max_retries": ...}
```

### 3.3 Trajectory Types (`trajectory/types.py`)

Extend the existing `TrajectoryItem` with operator-aware step tracking:

```python
class StepKind(StrEnum):
    LLM = "llm"
    TOOL = "tool"
    MEMORY = "memory"
    WORKFLOW = "workflow"
    AGENT = "agent"

@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    kind: StepKind
    operator_id: str | None = None
    agent_id: str = ""
    inputs: Any = None
    outputs: Any = None
    error: str | None = None
    timing: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    case_id: str
    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    seed: int | None = None
    tags: dict[str, str] = field(default_factory=dict)

@dataclass(slots=True)
class Trajectory:
    case_id: str = ""
    execution_id: str = ""
    steps: list[TrajectoryStep] = field(default_factory=list)
    edges: list[tuple[int, int]] | None = None

# Type alias for optimizer updates
Updates = dict[tuple[str, str], Any]  # (operator_id, target) → new_value
```

**Relationship to existing TrajectoryDataset:** `TrajectoryStep` is a finer-grained,
operator-aware step type. The existing `TrajectoryItem` captures message-level
state-action-reward triples. Both coexist — `TrajectoryItem` for dataset export,
`TrajectoryStep`/`Trajectory` for optimizer attribution.

### 3.4 Trajectory Extractor (`trajectory/extractor.py`)

```python
class TrajectoryExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        execution_data: Any,
        spec: ExecutionSpec,
    ) -> Trajectory: ...

class DefaultTrajectoryExtractor(TrajectoryExtractor):
    """Builds Trajectory from message history and tool call records."""
    def extract(self, execution_data: Any, spec: ExecutionSpec) -> Trajectory:
        # Walk execution_data (messages, tool results, etc.)
        # Build TrajectoryStep per operation
        # Return Trajectory with steps and optional DAG edges
        ...
```

**Design note:** Agent-core's `TracerTrajectoryExtractor` relies on its `Session.tracer()`
spans. Exo does not have an identical tracer, so we define `TrajectoryExtractor`
as an ABC. The `DefaultTrajectoryExtractor` works with dict-based execution records.
Custom extractors can integrate with Exo's hook system.

### 3.5 Optimizer Framework (`optimizer/`)

#### Base Optimizer (`optimizer/base.py`)

```python
@dataclass
class TextualParameter:
    operator_id: str
    gradients: dict[str, str] = field(default_factory=dict)  # target → gradient text
    description: str = ""

    def set_gradient(self, target: str, gradient: str) -> None:
        self.gradients[target] = gradient

class BaseOptimizer(ABC):
    domain: str  # "llm", "tool", "memory"

    def bind(
        self,
        operators: dict[str, Operator],
        targets: Sequence[str] | None = None,
        **config: Any,
    ) -> int:
        """Filter operators by domain, store bound set. Returns count."""
        ...

    @staticmethod
    def requires_forward_data() -> bool:
        """True if optimizer needs framework-driven forward pass."""
        return True

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Cache trajectory for backward pass."""
        ...

    def backward(self, evaluated_cases: Sequence[Any]) -> None:
        """Analyze failures, write textual gradients."""
        self._backward(evaluated_cases)

    def step(self) -> Updates:
        """Generate parameter updates from gradients."""
        return self._step()

    @abstractmethod
    def _backward(self, evaluated_cases: Sequence[Any]) -> None: ...

    @abstractmethod
    def _step(self) -> Updates: ...
```

**Design note:** The `backward()`/`step()` split mirrors PyTorch's gradient
accumulation pattern. Textual gradients are strings describing *what went wrong*
and *how to fix it* — the optimizer LLM rewrites parameters based on these.

#### Instruction Optimizer (`optimizer/instruction.py`)

```python
class InstructionOptimizer(BaseOptimizer):
    """LLM-based prompt optimizer using textual gradients."""
    domain = "llm"

    def _backward(self, evaluated_cases: Sequence[Any]) -> None:
        # For each failing case:
        #   Ask LLM: "Why did this prompt produce a wrong answer?"
        #   Write gradient to TextualParameter
        ...

    def _step(self) -> Updates:
        # For each operator with gradients:
        #   Ask LLM: "Rewrite the prompt to address these issues"
        #   Return {(operator_id, "system_prompt"): new_prompt}
        ...
```

### 3.6 Updater Protocol (`updater/`)

```python
class Updater(Protocol):
    def bind(
        self,
        operators: dict[str, Operator],
        targets: Sequence[str] | None = None,
        **config: Any,
    ) -> int: ...

    def requires_forward_data(self) -> bool: ...

    def update(
        self,
        trajectories: Sequence[Trajectory],
        evaluated_cases: Sequence[Any],
        config: dict[str, Any] | None = None,
    ) -> Updates | list[Updates]: ...

    def get_state(self) -> dict[str, Any]: ...
    def load_state(self, state: dict[str, Any]) -> None: ...

class SingleDimUpdater:
    """Wraps a single BaseOptimizer."""
    def __init__(self, optimizer: BaseOptimizer) -> None: ...

    def update(self, trajectories, evaluated_cases, config=None) -> Updates:
        for traj in trajectories:
            self._optimizer.add_trajectory(traj)
        self._optimizer.backward(evaluated_cases)
        return self._optimizer.step()

class MultiDimUpdater:
    """Composes domain-specific optimizers with attribution."""
    def __init__(self, domain_optimizers: dict[str, BaseOptimizer]) -> None: ...

    def update(self, trajectories, evaluated_cases, config=None) -> Updates:
        # 1. Attribute failing cases to domains (llm/tool/memory)
        # 2. Run each domain optimizer's backward + step
        # 3. Merge all Updates dicts
        ...
```

### 3.7 OperatorTrainer (`trainer/operator_trainer.py`)

A concrete `Trainer` subclass that orchestrates the operator optimization loop:

```python
class OperatorTrainer(Trainer):
    """Trainer that optimizes agent operators via textual gradients."""

    def __init__(
        self,
        updater: Updater,
        evaluator: Any,  # BaseEvaluator or callable
        extractor: TrajectoryExtractor | None = None,
        config: OperatorTrainConfig | None = None,
    ) -> None: ...

    # --- Trainer ABC validation phase ---
    def check_agent(self, agent: Any) -> None:
        # Validate agent has get_operators() -> dict[str, Operator]
        ...

    def check_dataset(self, train_data: Any, test_data: Any = None) -> None:
        # Validate list of Case-like dicts with inputs/label
        ...

    def check_reward(self, reward_fn: Any = None) -> None:
        # Optional — evaluator handles scoring
        ...

    def check_config(self, config: Any = None) -> None:
        # Merge OperatorTrainConfig
        ...

    # --- Trainer ABC training phase ---
    async def train(self) -> TrainMetrics:
        self._require_validated()
        operators = self._agent.get_operators()
        bound = self._updater.bind(operators)
        if bound == 0:
            return TrainMetrics()  # Nothing to optimize

        for epoch in range(self._config.epochs):
            # 1. Forward: predict + evaluate + extract trajectories
            predictions, exec_data = await self._predict(self._train_data)
            evaluated = self._evaluate_cases(predictions)
            trajectories = self._extract_trajectories(exec_data, self._train_data)

            # 2. Update: optimizer generates updates
            updates = self._updater.update(trajectories, evaluated)

            # 3. Apply: write updates to operators (with candidate selection)
            if isinstance(updates, list):
                self._select_best_candidate(operators, updates, self._val_data)
            else:
                self._apply_updates(operators, updates)

            # 4. Validate: evaluate on validation set
            val_score = await self._validate(self._val_data)

            # 5. Checkpoint if improved
            if val_score > self._best_score:
                self._save_checkpoint(operators)
                self._best_score = val_score

            # 6. Early stopping
            if val_score >= self._config.early_stop_score:
                break

        return TrainMetrics(accuracy=self._best_score, steps=epoch + 1)

    async def evaluate(self, test_data: Any = None) -> TrainMetrics: ...

    # --- Internal helpers ---
    def _apply_updates(self, operators, updates: Updates) -> None:
        for (op_id, target), value in updates.items():
            if op_id in operators:
                operators[op_id].set_parameter(target, value)

    def _snapshot_state(self, operators) -> dict[str, dict]: ...
    def _restore_state(self, operators, state) -> None: ...
    def _select_best_candidate(self, operators, candidates, val_data) -> None: ...
```

**Integration with Trainer lifecycle:**
- Inherits `Trainer`'s state machine (CREATED → VALIDATED → TRAINING → COMPLETED).
- Reuses `_require_validated()` guard.
- `TrainMetrics` returned with accuracy from best validation score.
- `TrainerError` raised on failures.

### 3.8 Checkpointing (`checkpointing/`)

```python
@dataclass(slots=True)
class OperatorCheckpoint:
    version: str = "1"
    run_id: str = ""
    epoch: int = 0
    best_score: float = 0.0
    operators_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    updater_state: dict[str, Any] = field(default_factory=dict)

class CheckpointManager(Protocol):
    def should_save(self, epoch: int, improved: bool) -> bool: ...
    def build(self, operators: dict[str, Operator], epoch: int, score: float) -> OperatorCheckpoint: ...
    def restore(self, operators: dict[str, Operator], checkpoint: OperatorCheckpoint) -> None: ...

class DefaultCheckpointManager:
    def __init__(self, save_every: int = 1, save_on_improve: bool = True) -> None: ...

class FileCheckpointStore:
    def __init__(self, directory: str) -> None: ...
    def save(self, checkpoint: OperatorCheckpoint, filename: str = "latest.json") -> str: ...
    def load(self, path: str) -> OperatorCheckpoint: ...
```

---

## 4. File Layout

```
packages/exo-train/src/exo/train/
├── __init__.py                       # Existing + new exports
├── trainer.py                        # Existing — Trainer ABC (unchanged)
├── evolution.py                      # Existing — EvolutionPipeline (unchanged)
├── synthesis.py                      # Existing — SynthesisPipeline (unchanged)
├── trajectory.py                     # Existing — TrajectoryDataset (unchanged)
├── verl.py                           # Existing — VeRLTrainer (unchanged)
│
├── operator/
│   ├── __init__.py                   # Operator, TunableSpec, TunableKind
│   ├── base.py                       # Operator ABC, TunableSpec, TunableKind
│   ├── llm_call.py                   # LLMCallOperator
│   ├── tool_call.py                  # ToolCallOperator
│   └── memory_call.py               # MemoryCallOperator
│
├── optimizer/
│   ├── __init__.py                   # BaseOptimizer, TextualParameter
│   ├── base.py                       # BaseOptimizer ABC, TextualParameter
│   └── instruction.py               # InstructionOptimizer
│
├── updater/
│   ├── __init__.py                   # Updater protocol, SingleDimUpdater, MultiDimUpdater
│   ├── protocol.py                   # Updater Protocol
│   ├── single_dim.py                # SingleDimUpdater
│   └── multi_dim.py                 # MultiDimUpdater
│
├── trajectory/
│   ├── __init__.py                   # TrajectoryStep, Trajectory, ExecutionSpec, Updates
│   ├── types.py                      # StepKind, TrajectoryStep, ExecutionSpec, Trajectory, Updates
│   └── extractor.py                  # TrajectoryExtractor ABC, DefaultTrajectoryExtractor
│
├── checkpointing/
│   ├── __init__.py                   # OperatorCheckpoint, CheckpointManager, FileCheckpointStore
│   ├── types.py                      # OperatorCheckpoint
│   ├── manager.py                    # CheckpointManager protocol, DefaultCheckpointManager
│   └── store.py                      # FileCheckpointStore
│
└── operator_trainer.py               # OperatorTrainer (concrete Trainer subclass)
```

---

## 5. Integration Flow

### 5.1 Agent → Operators Registration

```python
# Agent exposes operators (duck-type convention)
class MyAgent:
    def get_operators(self) -> dict[str, Operator]:
        return {
            "main_llm": LLMCallOperator(
                operator_id="main_llm",
                system_prompt=self.instructions,
                user_prompt=self.user_template,
            ),
            "search_tool": ToolCallOperator(
                operator_id="search_tool",
                tool_description=self.tools[0].description,
            ),
        }
```

### 5.2 Full Self-Optimization Flow

```
┌────────────────────────────────────────────────────────────────┐
│                    OperatorTrainer.train()                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. operators = agent.get_operators()                          │
│  2. updater.bind(operators)                                    │
│                                                                │
│  for epoch in range(N):                                        │
│    ┌──────────────────────────────────────────────────────┐    │
│    │ FORWARD: predict(agent, train_cases)                 │    │
│    │   → predictions, execution_data                      │    │
│    │                                                      │    │
│    │ EVALUATE: evaluate(cases, predictions)               │    │
│    │   → evaluated_cases (with scores)                    │    │
│    │                                                      │    │
│    │ EXTRACT: extractor.extract(exec_data, spec)          │    │
│    │   → trajectories (TrajectoryStep sequences)          │    │
│    └──────────────────────────────────────────────────────┘    │
│                          ↓                                     │
│    ┌──────────────────────────────────────────────────────┐    │
│    │ UPDATE: updater.update(trajectories, evaluated)      │    │
│    │                                                      │    │
│    │   SingleDimUpdater:                                  │    │
│    │     optimizer.add_trajectory(traj)                   │    │
│    │     optimizer.backward(evaluated)  → gradients       │    │
│    │     optimizer.step()               → Updates         │    │
│    │                                                      │    │
│    │   MultiDimUpdater:                                   │    │
│    │     attribute bad cases → domains                    │    │
│    │     per domain: backward + step → Updates            │    │
│    │     merge all updates                                │    │
│    └──────────────────────────────────────────────────────┘    │
│                          ↓                                     │
│    ┌──────────────────────────────────────────────────────┐    │
│    │ APPLY: operator.set_parameter(target, value)         │    │
│    │   or: candidate selection (snapshot/rollback/best)    │    │
│    │                                                      │    │
│    │ VALIDATE: evaluate(agent, val_cases)                 │    │
│    │   → val_score                                        │    │
│    │                                                      │    │
│    │ CHECKPOINT: if improved → save operator states       │    │
│    │ EARLY STOP: if val_score ≥ threshold → break         │    │
│    └──────────────────────────────────────────────────────┘    │
│                                                                │
│  return TrainMetrics(accuracy=best_score, steps=epochs)        │
└────────────────────────────────────────────────────────────────┘
```

### 5.3 Composition with Existing Pipelines

The operator pattern composes with existing exo-train features:

- **EvolutionStrategy + Operators:** A custom `EvolutionStrategy.train()` could
  internally create an `OperatorTrainer` for its training phase, combining
  data synthesis with operator optimization.
- **SynthesisPipeline + Operators:** Use `SynthesisPipeline` to augment training
  cases before feeding them to `OperatorTrainer`.
- **TrajectoryDataset + TrajectoryStep:** Existing `TrajectoryDataset.from_messages()`
  can still capture message-level data; `TrajectoryStep` adds operator-level detail
  for attribution.

---

## 6. Mapping: Agent-Core Trainer → Exo Trainer ABC

| Agent-Core (`Trainer`)               | Exo (`OperatorTrainer extends Trainer`)  |
|--------------------------------------|----------------------------------------------|
| `train(agent, train_cases, val_cases, num_iterations)` | `check_agent()` + `check_dataset()` + `mark_validated()` + `train()` |
| `forward(agent, cases)` → score, evaluated, trajectories | `_predict()` + `_evaluate_cases()` + `_extract_trajectories()` |
| `predict(agent, cases)` → predictions, sessions | `_predict()` (async, uses agent.invoke or similar) |
| `evaluate(agent, cases)` → score, evaluated | `evaluate(test_data)` (Trainer ABC method) |
| `apply_updates(operators, updates)` | `_apply_updates(operators, updates)` |
| `_snapshot_operators_state(operators)` | `_snapshot_state(operators)` |
| `_restore_operators_state(operators, state)` | `_restore_state(operators, state)` |
| `_select_best_candidate_on_val(...)` | `_select_best_candidate(...)` |
| `_save_checkpoint_if_needed(...)` | `_save_checkpoint(...)` via `CheckpointManager` |
| `Progress` (epoch tracker) | Exo's `TrainerState` + internal epoch counter |
| `Callbacks` (lifecycle hooks) | Future: HookManager integration |

**Key differences:**
- Exo splits agent-core's monolithic `train()` into validation + training phases.
- Exo uses `Trainer.state` (StrEnum) instead of agent-core's `Progress` class.
- Exo's `TrainMetrics` replaces agent-core's inline score tracking.
- Exo does not require `Session` — trajectory extraction is pluggable.

---

## 7. Existing Functionality Remains Unchanged

The following existing classes are **not modified** by this work:

| Class | Module | Status |
|-------|--------|--------|
| `Trainer` (ABC) | `trainer.py` | Unchanged — `OperatorTrainer` extends it |
| `TrainConfig` | `trainer.py` | Unchanged — `OperatorTrainConfig` extends it |
| `TrainMetrics` | `trainer.py` | Unchanged — reused by `OperatorTrainer` |
| `TrainerState` | `trainer.py` | Unchanged — inherited by `OperatorTrainer` |
| `EvolutionStrategy` (ABC) | `evolution.py` | Unchanged |
| `EvolutionPipeline` | `evolution.py` | Unchanged |
| `EvolutionConfig` | `evolution.py` | Unchanged |
| `SynthesisPipeline` | `synthesis.py` | Unchanged |
| `DataSynthesiser` (ABC) | `synthesis.py` | Unchanged |
| `TemplateSynthesiser` | `synthesis.py` | Unchanged |
| `TrajectoryDataset` | `trajectory.py` | Unchanged |
| `TrajectoryItem` | `trajectory.py` | Unchanged |
| `TrajectoryStrategy` (ABC) | `trajectory.py` | Unchanged |
| `VeRLTrainer` | `verl.py` | Unchanged |
| `VeRLConfig` | `verl.py` | Unchanged |

All existing `hook_manager.add(HookPoint.X, my_func)` calls continue to work.
All existing tests (~2,900) remain unaffected.

---

## 8. Open Questions

1. **Evaluator reuse:** Should `OperatorTrainer` accept Exo's own evaluator
   interface, or define a new `BaseEvaluator` ABC matching agent-core's pattern?
   Recommendation: Define a lightweight `CaseEvaluator` protocol in the optimizer
   module, with an adapter for existing evaluation functions.

2. **Agent contract:** The `get_operators()` duck-type convention works for
   flexibility. Should we formalize it as a `TrainableAgent` protocol?
   Recommendation: Yes, add a `TrainableAgent` protocol for type safety.

3. **Multi-candidate updaters:** Agent-core supports `List[Updates]` for candidate
   selection. This is powerful but adds complexity. Implement in first iteration
   or defer? Recommendation: Implement — it's the core value of snapshot/rollback.

4. **HookManager integration:** Should optimizer lifecycle events (backward, step,
   checkpoint) emit hooks? Recommendation: Defer to a follow-up story; keep the
   first implementation focused.

---

## 9. Summary

The operator pattern adds a new self-optimization paradigm to `exo-train`:

- **Operator ABC** decomposes agents into optimizable units with tunable parameters.
- **BaseOptimizer** framework uses textual gradients (backward/step) to improve
  operator parameters iteratively.
- **Updater** protocol provides single-dimension and multi-dimension composition.
- **OperatorTrainer** extends the existing `Trainer` ABC, inheriting its lifecycle
  state machine and validation phases.
- **TrajectoryStep/Trajectory** add operator-aware execution tracing alongside
  the existing `TrajectoryItem`/`TrajectoryDataset`.
- **Checkpointing** enables save/resume with operator state persistence.

All existing `exo-train` functionality — `EvolutionPipeline`, `SynthesisPipeline`,
`VeRLTrainer`, `TrajectoryDataset` — remains fully functional and unchanged.
