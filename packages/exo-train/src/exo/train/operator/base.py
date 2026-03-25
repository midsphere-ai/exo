"""Operator ABC and tunable parameter specification.

An Operator is the fundamental unit of execution and optimization.  Each
operator wraps a single agent capability (LLM call, tool invocation, memory
retrieval) and exposes its tunable parameters so that an optimizer can
iteratively improve them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TunableKind(StrEnum):
    """The kind of a tunable parameter."""

    PROMPT = "prompt"
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class TunableSpec:
    """Specification for a single tunable parameter.

    Args:
        name: Unique parameter name within the operator.
        kind: The category of tunability.
        current_value: Current value of the parameter (any serialisable type).
        constraints: Arbitrary constraints the optimizer should respect
            (e.g. ``{"min": 0, "max": 1}`` for continuous, ``{"choices": [...]}``
            for discrete).
    """

    name: str
    kind: TunableKind
    current_value: Any = None
    constraints: dict[str, Any] = field(default_factory=dict)


class Operator(ABC):
    """Abstract base class for atomic execution units with tunable parameters.

    Each operator wraps a single agent capability and exposes its tunable
    parameters for optimization.  The ``name`` property links trajectory
    steps to operators for attribution.

    Subclasses must implement all abstract members.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this operator."""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """Execute the operator with the given inputs.

        Args:
            **kwargs: Operator-specific inputs.

        Returns:
            Operator-specific output.
        """

    @abstractmethod
    def get_tunables(self) -> list[TunableSpec]:
        """Return the list of tunable parameter specifications.

        The returned specs describe *what* an optimizer may modify and the
        current value of each parameter.
        """

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Snapshot current parameter values for checkpointing."""

    @abstractmethod
    def load_state(self, state: dict[str, Any]) -> None:
        """Restore parameter values from a previously saved snapshot.

        Args:
            state: A dict previously returned by ``get_state()``.
        """
