"""Service Level Objective (SLO) tracking.

Define SLOs for agent metrics and track compliance over sliding time
windows. Provides budget calculations and compliance reports.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SLO:
    """A Service Level Objective definition.

    *comparator* determines how the target is evaluated:
      - ``"lt"``: actual average must be less than *target* (e.g. latency < 2s)
      - ``"gt"``: actual average must be greater than *target* (e.g. success rate > 0.99)
    """

    name: str
    metric_name: str
    target: float
    window_seconds: int = 3600
    comparator: str = "gt"  # "lt" or "gt"


@dataclass(frozen=True)
class SLOReport:
    """Result of evaluating an SLO against recorded samples."""

    slo_name: str
    target: float
    actual: float
    budget_remaining: float
    compliant: bool
    window_start: datetime
    total_samples: int
    violating_samples: int


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class SLOTracker:
    """Tracks metric samples and evaluates SLO compliance over sliding windows."""

    def __init__(self) -> None:
        self._slos: dict[str, SLO] = {}
        # metric_name -> list of (monotonic_timestamp, value)
        self._samples: dict[str, list[tuple[float, float]]] = {}

    # -- SLO management -------------------------------------------------------

    def register(self, slo: SLO) -> None:
        """Register an SLO. Overwrites any existing SLO with the same name."""
        self._slos[slo.name] = slo
        logger.debug(
            "registered SLO %r (metric=%s, target=%s, comparator=%s)",
            slo.name,
            slo.metric_name,
            slo.target,
            slo.comparator,
        )

    def unregister(self, name: str) -> None:
        """Remove a registered SLO by name."""
        self._slos.pop(name, None)

    def list_slos(self) -> list[SLO]:
        """Return all registered SLOs."""
        return list(self._slos.values())

    # -- Recording ------------------------------------------------------------

    def record(self, metric_name: str, value: float) -> None:
        """Record a sample for a metric."""
        ts = time.monotonic()
        self._samples.setdefault(metric_name, []).append((ts, value))

    # -- Reporting ------------------------------------------------------------

    def report(self, slo_name: str) -> SLOReport | None:
        """Generate a compliance report for a single SLO.

        Returns ``None`` if the SLO is not registered.
        """
        slo = self._slos.get(slo_name)
        if slo is None:
            return None

        now = time.monotonic()
        cutoff = now - slo.window_seconds
        window_start = datetime.now(UTC)

        raw = self._samples.get(slo.metric_name, [])
        samples = [(ts, v) for ts, v in raw if ts >= cutoff]

        if not samples:
            return SLOReport(
                slo_name=slo.name,
                target=slo.target,
                actual=0.0,
                budget_remaining=1.0,
                compliant=True,
                window_start=window_start,
                total_samples=0,
                violating_samples=0,
            )

        values = [v for _, v in samples]
        actual = sum(values) / len(values)

        if slo.comparator == "lt":
            violating = sum(1 for v in values if v >= slo.target)
            compliant = actual < slo.target
        else:  # "gt"
            violating = sum(1 for v in values if v < slo.target)
            compliant = actual >= slo.target

        # Budget: fraction of samples that are still within tolerance.
        # 1.0 = all good, 0.0 = all violating.
        budget_remaining = max(0.0, 1.0 - (violating / len(values)))

        report = SLOReport(
            slo_name=slo.name,
            target=slo.target,
            actual=actual,
            budget_remaining=budget_remaining,
            compliant=compliant,
            window_start=window_start,
            total_samples=len(values),
            violating_samples=violating,
        )
        if not compliant:
            logger.warning(
                "SLO %r is non-compliant: actual=%.4f target=%.4f budget=%.2f%%",
                slo.name,
                actual,
                slo.target,
                budget_remaining * 100,
            )
        else:
            logger.debug(
                "SLO %r compliant: actual=%.4f target=%.4f budget=%.2f%%",
                slo.name,
                actual,
                slo.target,
                budget_remaining * 100,
            )
        return report

    def report_all(self) -> list[SLOReport]:
        """Generate compliance reports for all registered SLOs."""
        reports: list[SLOReport] = []
        for name in self._slos:
            r = self.report(name)
            if r is not None:
                reports.append(r)
        return reports

    # -- Management -----------------------------------------------------------

    def reset(self) -> None:
        """Clear all samples (SLO definitions are preserved)."""
        self._samples.clear()

    def clear(self) -> None:
        """Clear everything — SLOs and samples."""
        self._slos.clear()
        self._samples.clear()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_global_tracker = SLOTracker()


def get_tracker() -> SLOTracker:
    """Return the global SLO tracker."""
    return _global_tracker


def reset() -> None:
    """Reset global SLO tracker — for testing only."""
    _global_tracker.reset()
    _global_tracker.clear()
