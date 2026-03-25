"""Tests for the base Trainer ABC and supporting types."""

from __future__ import annotations

from typing import Any

import pytest

from exo.train.trainer import (  # pyright: ignore[reportMissingImports]
    TrainConfig,
    Trainer,
    TrainerError,
    TrainerState,
    TrainMetrics,
)

# ---------------------------------------------------------------------------
# Concrete test implementation
# ---------------------------------------------------------------------------


class _StubTrainer(Trainer):
    """Minimal concrete trainer for testing the ABC lifecycle."""

    def __init__(self, config: TrainConfig | None = None) -> None:
        super().__init__(config)
        self.agent_checked = False
        self.dataset_checked = False
        self.reward_checked = False
        self.config_checked = False
        self.train_called = False
        self.evaluate_called = False

    def check_agent(self, agent: Any) -> None:
        if agent is None:
            msg = "agent required"
            raise TrainerError(msg)
        self.agent_checked = True

    def check_dataset(self, train_data: Any, test_data: Any | None = None) -> None:
        if not train_data:
            msg = "train_data required"
            raise TrainerError(msg)
        self.dataset_checked = True

    def check_reward(self, reward_fn: Any | None = None) -> None:
        self.reward_checked = True

    def check_config(self, config: TrainConfig | dict[str, Any] | None = None) -> None:
        self.config_checked = True

    async def train(self) -> TrainMetrics:
        self._require_validated()
        self._state = TrainerState.TRAINING
        self.train_called = True
        self._state = TrainerState.COMPLETED
        return TrainMetrics(loss=0.1, accuracy=0.95, steps=100)

    async def evaluate(self, test_data: Any | None = None) -> TrainMetrics:
        self.evaluate_called = True
        return TrainMetrics(loss=0.2, accuracy=0.9, steps=50)


# ---------------------------------------------------------------------------
# TrainerError
# ---------------------------------------------------------------------------


class TestTrainerError:
    def test_is_exception(self) -> None:
        err = TrainerError("boom")
        assert isinstance(err, Exception)
        assert str(err) == "boom"


# ---------------------------------------------------------------------------
# TrainerState
# ---------------------------------------------------------------------------


class TestTrainerState:
    def test_values(self) -> None:
        assert TrainerState.CREATED == "created"
        assert TrainerState.VALIDATED == "validated"
        assert TrainerState.TRAINING == "training"
        assert TrainerState.COMPLETED == "completed"
        assert TrainerState.FAILED == "failed"

    def test_is_str(self) -> None:
        assert isinstance(TrainerState.CREATED, str)


# ---------------------------------------------------------------------------
# TrainMetrics
# ---------------------------------------------------------------------------


class TestTrainMetrics:
    def test_defaults(self) -> None:
        m = TrainMetrics()
        assert m.loss == 0.0
        assert m.accuracy == 0.0
        assert m.steps == 0
        assert m.extra == {}

    def test_custom(self) -> None:
        m = TrainMetrics(loss=0.5, accuracy=0.8, steps=10, extra={"lr": 1e-4})
        assert m.loss == 0.5
        assert m.accuracy == 0.8
        assert m.steps == 10
        assert m.extra["lr"] == 1e-4


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------


class TestTrainConfig:
    def test_defaults(self) -> None:
        c = TrainConfig()
        assert c.epochs == 1
        assert c.batch_size == 8
        assert c.learning_rate == 1e-5
        assert c.output_dir == ""
        assert c.extra == {}

    def test_custom(self) -> None:
        c = TrainConfig(epochs=3, batch_size=16, learning_rate=3e-4, output_dir="/tmp/run")
        assert c.epochs == 3
        assert c.batch_size == 16
        assert c.learning_rate == 3e-4
        assert c.output_dir == "/tmp/run"


# ---------------------------------------------------------------------------
# Trainer lifecycle
# ---------------------------------------------------------------------------


class TestTrainerInit:
    def test_default_state(self) -> None:
        t = _StubTrainer()
        assert t.state == TrainerState.CREATED

    def test_default_config(self) -> None:
        t = _StubTrainer()
        assert isinstance(t.config, TrainConfig)
        assert t.config.epochs == 1

    def test_custom_config(self) -> None:
        cfg = TrainConfig(epochs=5)
        t = _StubTrainer(cfg)
        assert t.config.epochs == 5


class TestTrainerValidation:
    def test_check_agent_success(self) -> None:
        t = _StubTrainer()
        t.check_agent("my-agent")
        assert t.agent_checked

    def test_check_agent_failure(self) -> None:
        t = _StubTrainer()
        with pytest.raises(TrainerError, match="agent required"):
            t.check_agent(None)

    def test_check_dataset_success(self) -> None:
        t = _StubTrainer()
        t.check_dataset(["item1", "item2"])
        assert t.dataset_checked

    def test_check_dataset_failure(self) -> None:
        t = _StubTrainer()
        with pytest.raises(TrainerError, match="train_data required"):
            t.check_dataset([])

    def test_check_reward(self) -> None:
        t = _StubTrainer()
        t.check_reward(lambda x: 1.0)
        assert t.reward_checked

    def test_check_config(self) -> None:
        t = _StubTrainer()
        t.check_config({"epochs": 2})
        assert t.config_checked

    def test_mark_validated(self) -> None:
        t = _StubTrainer()
        t.mark_validated()
        assert t.state == TrainerState.VALIDATED

    def test_mark_validated_twice_fails(self) -> None:
        t = _StubTrainer()
        t.mark_validated()
        with pytest.raises(TrainerError, match="Cannot validate"):
            t.mark_validated()


class TestTrainerTrain:
    async def test_train_after_validation(self) -> None:
        t = _StubTrainer()
        t.check_agent("agent")
        t.check_dataset(["data"])
        t.check_reward()
        t.check_config()
        t.mark_validated()
        metrics = await t.train()
        assert t.train_called
        assert t.state == TrainerState.COMPLETED
        assert metrics.loss == 0.1
        assert metrics.accuracy == 0.95
        assert metrics.steps == 100

    async def test_train_without_validation_fails(self) -> None:
        t = _StubTrainer()
        with pytest.raises(TrainerError, match="must be validated"):
            await t.train()


class TestTrainerEvaluate:
    async def test_evaluate(self) -> None:
        t = _StubTrainer()
        metrics = await t.evaluate()
        assert t.evaluate_called
        assert metrics.loss == 0.2
        assert metrics.accuracy == 0.9

    async def test_evaluate_with_data(self) -> None:
        t = _StubTrainer()
        metrics = await t.evaluate(test_data=["test-item"])
        assert t.evaluate_called
        assert metrics.steps == 50


# ---------------------------------------------------------------------------
# Full lifecycle integration
# ---------------------------------------------------------------------------


class TestTrainerLifecycle:
    async def test_full_lifecycle(self) -> None:
        cfg = TrainConfig(epochs=2, batch_size=4)
        t = _StubTrainer(cfg)

        # Phase 1: Validation
        assert t.state == TrainerState.CREATED
        t.check_agent("agent-1")
        t.check_dataset(["d1", "d2"], test_data=["t1"])
        t.check_reward(lambda x: x)
        t.check_config()
        t.mark_validated()
        assert t.state == TrainerState.VALIDATED

        # Phase 2: Training
        train_metrics = await t.train()
        assert t.state == TrainerState.COMPLETED
        assert train_metrics.loss == 0.1

        # Phase 3: Evaluation
        eval_metrics = await t.evaluate()
        assert eval_metrics.accuracy == 0.9

    async def test_cannot_train_after_completed(self) -> None:
        t = _StubTrainer()
        t.mark_validated()
        await t.train()
        assert t.state == TrainerState.COMPLETED
        # Training again should fail (not in VALIDATED state)
        with pytest.raises(TrainerError, match="must be validated"):
            await t.train()
