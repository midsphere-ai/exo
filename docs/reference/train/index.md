# exo.train

Training framework with trajectory capture, data synthesis, evolution pipelines, and VeRL integration.

## Installation

```bash
pip install exo-train

# For VeRL integration:
pip install exo-train[verl]
```

## Module path

```python
import exo.train
```

## Package exports

| Export | Module | Description |
|---|---|---|
| `TrajectoryItem` | `exo.train.trajectory` | A single step in an agent execution trajectory |
| `TrajectoryDataset` | `exo.train.trajectory` | Dataset of trajectory items with capture and export |
| `TrajectoryStrategy` | `exo.train.trajectory` | ABC for building trajectory items from messages |
| `DefaultStrategy` | `exo.train.trajectory` | Default strategy extracting input/output/tool_calls |
| `TrajectoryError` | `exo.train.trajectory` | Error during trajectory operations |
| `Trainer` | `exo.train.trainer` | Abstract base class for agent training |
| `TrainConfig` | `exo.train.trainer` | Minimal training configuration |
| `TrainerState` | `exo.train.trainer` | Trainer lifecycle state enum |
| `TrainMetrics` | `exo.train.trainer` | Metrics produced by training or evaluation |
| `TrainerError` | `exo.train.trainer` | Error during training operations |
| `DataSynthesiser` | `exo.train.synthesis` | ABC for custom data synthesisers |
| `TemplateSynthesiser` | `exo.train.synthesis` | Generate items via template transforms |
| `SynthesisPipeline` | `exo.train.synthesis` | Orchestrates data synthesis from trajectory items |
| `SynthesisConfig` | `exo.train.synthesis` | Configuration for a synthesis pipeline run |
| `SynthesisResult` | `exo.train.synthesis` | Output of a synthesis pipeline run |
| `SynthesisStrategy` | `exo.train.synthesis` | Strategy enum (llm, template, augment) |
| `SynthesisError` | `exo.train.synthesis` | Error during data synthesis |
| `augment_swap_io` | `exo.train.synthesis` | Augment by swapping input/output |
| `augment_add_noise` | `exo.train.synthesis` | Augment by adding noise to input |
| `filter_by_score` | `exo.train.synthesis` | Filter items by minimum score |
| `deduplicate` | `exo.train.synthesis` | Remove duplicate items |
| `split_dataset` | `exo.train.synthesis` | Split items into train/test sets |
| `EvolutionPipeline` | `exo.train.evolution` | Multi-epoch evolution pipeline |
| `EvolutionConfig` | `exo.train.evolution` | Configuration for an evolution run |
| `EvolutionStrategy` | `exo.train.evolution` | ABC for evolution phase strategies |
| `EvolutionState` | `exo.train.evolution` | Pipeline state enum |
| `EvolutionPhase` | `exo.train.evolution` | Evolution epoch phases enum |
| `EpochResult` | `exo.train.evolution` | Metrics for a single evolution epoch |
| `EvolutionResult` | `exo.train.evolution` | Aggregate result of a full evolution run |
| `EvolutionError` | `exo.train.evolution` | Error during evolution operations |
| `VeRLTrainer` | `exo.train.verl` | Concrete trainer integrating with VeRL |
| `VeRLConfig` | `exo.train.verl` | VeRL-specific training configuration |
| `VeRLAlgorithm` | `exo.train.verl` | Supported VeRL RL algorithms enum |
| `RewardSpec` | `exo.train.verl` | Descriptor for a reward function |

## Submodules

- [exo.train.trajectory](trajectory.md) -- TrajectoryDataset, TrajectoryItem, strategies
- [exo.train.trainer](trainer.md) -- Trainer ABC, TrainConfig, TrainMetrics
- [exo.train.synthesis](synthesis.md) -- DataSynthesiser, SynthesisPipeline, augment helpers
- [exo.train.evolution](evolution.md) -- EvolutionPipeline, EvolutionConfig, strategies
- [exo.train.verl](verl.md) -- VeRLTrainer, VeRLConfig, RewardSpec
