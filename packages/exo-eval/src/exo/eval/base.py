"""Evaluation framework: targets, scorers, criteria, and evaluator."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from exo.types import ExoError


class EvalError(ExoError):
    """Raised when an evaluation fails."""


class EvalStatus(StrEnum):
    """Outcome status for a single metric evaluation."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScorerResult:
    """Output from a single scorer applied to one case."""

    scorer_name: str
    score: float
    status: EvalStatus = EvalStatus.NOT_EVALUATED
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalCaseResult:
    """Result for one evaluation case (one input/output pair)."""

    case_id: str
    input: Any
    output: Any
    scores: dict[str, ScorerResult] = field(default_factory=dict)


@dataclass(slots=True)
class EvalResult:
    """Aggregated result across all cases."""

    case_results: list[EvalCaseResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    pass_at_k: dict[int, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# EvalCriteria
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvalCriteria:
    """Threshold-based pass/fail criteria for a metric."""

    metric_name: str
    threshold: float = 0.5

    def judge(self, value: float) -> EvalStatus:
        """Return PASSED if *value* meets or exceeds *threshold*."""
        return EvalStatus.PASSED if value >= self.threshold else EvalStatus.FAILED


# ---------------------------------------------------------------------------
# ABCs
# ---------------------------------------------------------------------------


class EvalTarget(ABC):
    """Callable evaluation subject — wraps the system under test."""

    @abstractmethod
    async def predict(self, case_id: str, input: Any) -> Any:
        """Run the system under test and return its output."""


class Scorer(ABC):
    """Abstract scorer that evaluates one (input, output) pair."""

    @abstractmethod
    async def score(self, case_id: str, input: Any, output: Any) -> ScorerResult:
        """Score a single case and return a ScorerResult."""


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Runs an EvalTarget over a dataset and scores results.

    Supports parallel execution via semaphore and *repeat_times* for
    pass@k metric computation.
    """

    __slots__ = ("_criteria", "_parallel", "_repeat_times", "_scorers")

    def __init__(
        self,
        scorers: list[Scorer],
        *,
        criteria: list[EvalCriteria] | None = None,
        parallel: int = 4,
        repeat_times: int = 1,
    ) -> None:
        if parallel < 1:
            msg = "parallel must be >= 1"
            raise EvalError(msg)
        if repeat_times < 1:
            msg = "repeat_times must be >= 1"
            raise EvalError(msg)
        self._scorers = list(scorers)
        self._criteria = {c.metric_name: c for c in (criteria or [])}
        self._parallel = parallel
        self._repeat_times = repeat_times

    # ---- public API -------------------------------------------------------

    async def evaluate(
        self,
        target: EvalTarget,
        dataset: list[dict[str, Any]],
    ) -> EvalResult:
        """Run *target* over *dataset*, score, and return aggregated result."""
        sem = asyncio.Semaphore(self._parallel)
        case_results: list[EvalCaseResult] = []

        async def _run(case: dict[str, Any], repeat: int) -> EvalCaseResult:
            async with sem:
                case_id = str(case.get("id", f"case-{id(case)}-r{repeat}"))
                inp = case.get("input")
                output = await target.predict(case_id, inp)
                scores = {}
                for scorer in self._scorers:
                    sr = await scorer.score(case_id, inp, output)
                    criterion = self._criteria.get(sr.scorer_name)
                    if criterion is not None:
                        status = criterion.judge(sr.score)
                        sr = ScorerResult(
                            scorer_name=sr.scorer_name,
                            score=sr.score,
                            status=status,
                            details=sr.details,
                        )
                    scores[sr.scorer_name] = sr
                return EvalCaseResult(case_id=case_id, input=inp, output=output, scores=scores)

        tasks = [_run(case, r) for case in dataset for r in range(self._repeat_times)]
        case_results = list(await asyncio.gather(*tasks))

        summary = self._summarize(case_results)
        pass_at_k = self._compute_pass_at_k(case_results, dataset)
        return EvalResult(case_results=case_results, summary=summary, pass_at_k=pass_at_k)

    # ---- internal ---------------------------------------------------------

    def _summarize(self, results: list[EvalCaseResult]) -> dict[str, Any]:
        """Compute mean score per scorer across all cases."""
        totals: dict[str, list[float]] = defaultdict(list)
        for cr in results:
            for name, sr in cr.scores.items():
                totals[name].append(sr.score)
        return {name: sum(vals) / len(vals) for name, vals in totals.items()}

    def _compute_pass_at_k(
        self,
        results: list[EvalCaseResult],
        dataset: list[dict[str, Any]],
    ) -> dict[int, float]:
        """Compute pass@k for k=1..repeat_times."""
        if self._repeat_times <= 1 or not self._criteria:
            return {}

        # Group results by base case index (dataset order, repeats consecutive)
        groups: dict[int, list[EvalCaseResult]] = defaultdict(list)
        for idx, cr in enumerate(results):
            case_idx = idx // self._repeat_times
            groups[case_idx].append(cr)

        n_cases = len(dataset)
        if n_cases == 0:
            return {}

        pass_at: dict[int, float] = {}
        for k in range(1, self._repeat_times + 1):
            passed = 0
            for case_idx in range(n_cases):
                group = groups.get(case_idx, [])
                first_k = group[:k]
                if any(
                    sr.status == EvalStatus.PASSED for cr in first_k for sr in cr.scores.values()
                ):
                    passed += 1
            pass_at[k] = passed / n_cases
        return pass_at

    def __repr__(self) -> str:
        s = len(self._scorers)
        return (
            f"Evaluator(scorers={s}, parallel={self._parallel}, repeat_times={self._repeat_times})"
        )
