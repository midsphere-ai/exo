# exo-train

Training framework for the [Exo](../../README.md) multi-agent framework. Collect trajectories, synthesize training data, and integrate with RLHF pipelines.

## Installation

```bash
pip install exo-train

# With VeRL integration
pip install exo-train[verl]
```

Requires Python 3.11+, `exo-core`, and `exo-models`.

## What's Included

- **Trajectory collection** -- capture agent execution traces for training data.
- **Data synthesis** -- generate training examples from agent interactions.
- **Evolution** -- mutate and evolve training data for diversity.
- **VeRL integration** -- connect to VeRL for RLHF/GRPO training loops.
- **Dataset management** -- load, filter, and batch trajectory datasets.

## Quick Example

```python
from exo.train import TrajectoryCollector

collector = TrajectoryCollector()
result = await collector.run(agent, "Solve this problem step by step...")

# Export trajectories for training
dataset = collector.to_dataset()
dataset.save("trajectories.jsonl")
```

## Documentation

- [Training Guide](../../docs/guides/training.md)
- [API Reference](../../docs/reference/train/)
