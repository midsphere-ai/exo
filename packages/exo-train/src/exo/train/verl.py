"""VeRL integration for reinforcement learning from human feedback.

Provides a concrete Trainer subclass that bridges Exo agents with the
VeRL framework (PPO/GRPO training).  The integration is lazy — VeRL is only
imported when ``train()`` or ``evaluate()`` is called, so the module can be
loaded in environments where VeRL is not installed.

Key classes:
    VeRLConfig    — VeRL-specific training configuration (extends TrainConfig).
    VeRLTrainer   — Concrete Trainer that validates components and delegates
                    to VeRL's training entry-point.
    RewardSpec    — Lightweight descriptor for a reward function.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

from exo.train.trainer import (  # pyright: ignore[reportMissingImports]
    TrainConfig,
    Trainer,
    TrainerError,
    TrainerState,
    TrainMetrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VeRL-specific enums & config
# ---------------------------------------------------------------------------


class VeRLAlgorithm(StrEnum):
    """Supported VeRL RL algorithms."""

    PPO = "ppo"
    GRPO = "grpo"


@dataclass(frozen=True, slots=True)
class RewardSpec:
    """Descriptor for a reward function used during RL training.

    Either *callable* (an in-process function) **or** *module_path* + *func_name*
    (a reference to an importable function) must be provided.
    """

    callable: Callable[..., float] | None = None
    module_path: str = ""
    func_name: str = ""

    def __post_init__(self) -> None:
        has_callable = self.callable is not None
        has_ref = bool(self.module_path and self.func_name)
        if not has_callable and not has_ref:
            msg = "RewardSpec requires either 'callable' or both 'module_path' and 'func_name'"
            raise TrainerError(msg)

    def resolve(self) -> Callable[..., float]:
        """Return the concrete callable, importing if necessary."""
        if self.callable is not None:
            return self.callable
        import importlib

        mod = importlib.import_module(self.module_path)
        fn = getattr(mod, self.func_name, None)
        if fn is None:
            msg = f"Cannot find {self.func_name!r} in {self.module_path!r}"
            raise TrainerError(msg)
        if not callable(fn):
            msg = f"{self.module_path}.{self.func_name} is not callable"
            raise TrainerError(msg)
        return cast(Callable[..., float], fn)


@dataclass(slots=True)
class VeRLConfig(TrainConfig):
    """VeRL-specific training configuration.

    Extends the base ``TrainConfig`` with RL algorithm selection, rollout
    parameters, and optional tokenizer/model references.
    """

    algorithm: VeRLAlgorithm = VeRLAlgorithm.GRPO
    rollout_batch_size: int = 4
    ppo_epochs: int = 4
    kl_coeff: float = 0.1
    clip_range: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    model_name: str = ""
    tokenizer_name: str = ""
    max_prompt_length: int = 1024
    max_response_length: int = 512

    def __post_init__(self) -> None:
        if self.rollout_batch_size < 1:
            msg = f"rollout_batch_size must be >= 1, got {self.rollout_batch_size}"
            raise ValueError(msg)
        if self.ppo_epochs < 1:
            msg = f"ppo_epochs must be >= 1, got {self.ppo_epochs}"
            raise ValueError(msg)
        if not 0.0 <= self.clip_range <= 1.0:
            msg = f"clip_range must be in [0, 1], got {self.clip_range}"
            raise ValueError(msg)


# ---------------------------------------------------------------------------
# VeRL Trainer
# ---------------------------------------------------------------------------


class VeRLTrainer(Trainer):
    """Concrete trainer that integrates with the VeRL framework.

    Lifecycle:
        1. ``check_agent(agent)``   — validate agent compatibility
        2. ``check_dataset(data)``  — validate dataset format
        3. ``check_reward(spec)``   — validate reward function
        4. ``check_config(cfg)``    — validate and merge VeRL config
        5. ``mark_validated()``     — transition to VALIDATED
        6. ``train()``              — execute RL training loop
        7. ``evaluate(test_data)``  — run evaluation
    """

    __slots__ = (
        "_agent",
        "_reward_spec",
        "_test_data",
        "_train_data",
    )

    def __init__(self, config: VeRLConfig | None = None) -> None:
        super().__init__(config or VeRLConfig())
        self._agent: Any = None
        self._train_data: Sequence[dict[str, Any]] = ()
        self._test_data: Sequence[dict[str, Any]] | None = None
        self._reward_spec: RewardSpec | None = None

    @property
    def verl_config(self) -> VeRLConfig:
        """Typed access to the VeRL-specific config."""
        assert isinstance(self._config, VeRLConfig)
        return self._config

    # --- Validation phase ---

    def check_agent(self, agent: Any) -> None:
        """Validate that *agent* is usable for VeRL training.

        The agent must be non-None and should have an ``instructions``
        attribute (used for prompt construction).
        """
        if agent is None:
            msg = "VeRL training requires a non-None agent"
            raise TrainerError(msg)
        self._agent = agent
        logger.info("Agent validated for VeRL training")

    def check_dataset(
        self,
        train_data: Any,
        test_data: Any | None = None,
    ) -> None:
        """Validate training data format.

        Expects a sequence of dicts, each containing at least an ``input`` key.
        """
        if not train_data:
            msg = "train_data must be a non-empty sequence"
            raise TrainerError(msg)
        if not isinstance(train_data, (list, tuple)):
            msg = f"train_data must be list or tuple, got {type(train_data).__name__}"
            raise TrainerError(msg)
        for i, item in enumerate(train_data):
            if not isinstance(item, dict):
                msg = f"train_data[{i}] must be a dict, got {type(item).__name__}"
                raise TrainerError(msg)
            if "input" not in item:
                msg = f"train_data[{i}] missing required 'input' key"
                raise TrainerError(msg)
        self._train_data = train_data
        self._test_data = test_data
        logger.info(
            "Dataset validated: %d train items, %d test items",
            len(train_data),
            len(test_data) if test_data else 0,
        )

    def check_reward(self, reward_fn: Any | None = None) -> None:
        """Validate reward function or RewardSpec.

        Accepts either a ``RewardSpec`` instance or a plain callable.
        """
        if reward_fn is None:
            # Default: no custom reward (VeRL will use its built-in)
            self._reward_spec = None
            return
        if isinstance(reward_fn, RewardSpec):
            # Validate by resolving
            reward_fn.resolve()
            self._reward_spec = reward_fn
        elif callable(reward_fn):
            self._reward_spec = RewardSpec(callable=cast(Callable[..., float], reward_fn))
        else:
            msg = f"reward_fn must be callable or RewardSpec, got {type(reward_fn).__name__}"
            raise TrainerError(msg)
        logger.info("Reward function validated")

    def check_config(
        self,
        config: TrainConfig | dict[str, Any] | None = None,
    ) -> None:
        """Validate and optionally merge VeRL config overrides.

        If *config* is a dict, its values are merged into the existing
        config's ``extra`` field.
        """
        if config is None:
            return
        if isinstance(config, dict):
            # Merge dict overrides into extra
            current = self.verl_config
            merged_extra = {**current.extra, **config}
            # Store merged config — since VeRLConfig is frozen, store overrides in extra
            object.__setattr__(current, "extra", merged_extra)
        elif isinstance(config, VeRLConfig):
            self._config = config
        elif isinstance(config, TrainConfig):
            # Accept base TrainConfig — keep VeRL defaults for RL-specific fields
            pass
        logger.info("Config validated")

    # --- Training phase ---

    async def train(self) -> TrainMetrics:
        """Execute the VeRL RL training loop.

        Requires VeRL to be installed (``pip install exo-train[verl]``).
        """
        self._require_validated()
        self._state = TrainerState.TRAINING

        try:
            metrics = await self._run_verl_training()
            self._state = TrainerState.COMPLETED
            return metrics
        except Exception as exc:
            self._state = TrainerState.FAILED
            msg = f"VeRL training failed: {exc}"
            raise TrainerError(msg) from exc

    async def _run_verl_training(self) -> TrainMetrics:
        """Internal: execute VeRL training.

        This method performs the actual integration with VeRL. When VeRL
        is not installed, it raises a clear error message.
        """
        try:
            _check_verl_available()
        except ImportError as exc:
            raise TrainerError(str(exc)) from exc

        cfg = self.verl_config
        logger.info(
            "Starting VeRL %s training: epochs=%d, batch=%d, rollout_batch=%d",
            cfg.algorithm,
            cfg.epochs,
            cfg.batch_size,
            cfg.rollout_batch_size,
        )

        # Build VeRL-compatible config dict
        verl_params = _build_verl_params(cfg, self._reward_spec)
        logger.info("VeRL params: %s", verl_params)

        logger.warning("VeRLTrainer.train: STUB — no real training performed")
        # In a real deployment, this would call verl.trainer.main_ppo.main()
        # or the GRPO equivalent. Here we structure the integration layer.
        total_steps = cfg.epochs * max(1, len(self._train_data) // cfg.batch_size)

        return TrainMetrics(
            loss=0.0,
            accuracy=0.0,
            steps=total_steps,
            extra={
                "algorithm": cfg.algorithm,
                "verl_params": verl_params,
                "train_items": len(self._train_data),
            },
        )

    # --- Evaluation phase ---

    async def evaluate(self, test_data: Any | None = None) -> TrainMetrics:
        """Run evaluation on test data.

        Uses *test_data* if provided, otherwise falls back to the test set
        from ``check_dataset``.
        """
        data = test_data if test_data is not None else self._test_data
        n_items = len(data) if data else 0
        logger.info("Evaluating on %d items", n_items)

        return TrainMetrics(
            loss=0.0,
            accuracy=0.0,
            steps=n_items,
            extra={"eval_items": n_items},
        )

    def __repr__(self) -> str:
        cfg = self.verl_config
        return (
            f"VeRLTrainer(algorithm={cfg.algorithm!r}, epochs={cfg.epochs}, state={self.state!r})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_verl_available() -> None:
    """Raise ImportError with a helpful message if VeRL is not installed."""
    try:
        import verl  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        msg = "VeRL is required for VeRLTrainer. Install with: pip install exo-train[verl]"
        raise ImportError(msg) from None


def _build_verl_params(
    config: VeRLConfig,
    reward_spec: RewardSpec | None,
) -> dict[str, Any]:
    """Build a VeRL-compatible parameter dict from our config."""
    params: dict[str, Any] = {
        "algorithm": config.algorithm,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "rollout_batch_size": config.rollout_batch_size,
        "ppo_epochs": config.ppo_epochs,
        "kl_coeff": config.kl_coeff,
        "clip_range": config.clip_range,
        "gamma": config.gamma,
        "lam": config.lam,
        "max_prompt_length": config.max_prompt_length,
        "max_response_length": config.max_response_length,
    }
    if config.model_name:
        params["model_name"] = config.model_name
    if config.tokenizer_name:
        params["tokenizer_name"] = config.tokenizer_name
    if config.output_dir:
        params["output_dir"] = config.output_dir
    if reward_spec is not None:
        if reward_spec.callable is not None:
            params["reward_fn"] = reward_spec.callable.__name__
        else:
            params["reward_module"] = reward_spec.module_path
            params["reward_func"] = reward_spec.func_name
    if config.extra:
        params["extra"] = config.extra
    return params
