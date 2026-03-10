"""Core guardrail types: risk levels, assessments, and errors."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from orbiter.types import OrbiterError


class RiskLevel(StrEnum):
    """Severity level of a detected risk."""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAssessment(BaseModel, frozen=True):
    """Result of a backend's risk analysis.

    Attributes:
        has_risk: Whether any risk was detected.
        risk_level: Severity of the detected risk.
        risk_type: Category of risk (e.g., "prompt_injection", "pii_leak").
        confidence: Backend's confidence in the assessment (0.0-1.0).
        details: Additional metadata for logging and auditing.
    """

    has_risk: bool
    risk_level: RiskLevel
    risk_type: str | None = None
    confidence: float = 1.0
    details: dict[str, Any] = Field(default_factory=dict)


class GuardrailError(OrbiterError):
    """Raised when a guardrail blocks an operation.

    Attributes:
        risk_level: The risk level that triggered the block.
        risk_type: Category of the detected risk.
        details: Additional context from the risk assessment.
    """

    def __init__(
        self,
        message: str,
        *,
        risk_level: RiskLevel,
        risk_type: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.risk_level = risk_level
        self.risk_type = risk_type
        self.details = details or {}
