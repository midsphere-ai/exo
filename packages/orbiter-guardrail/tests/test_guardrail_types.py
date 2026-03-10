"""Tests for guardrail core types: RiskLevel, RiskAssessment, GuardrailError."""

from __future__ import annotations

import pytest

from orbiter.guardrail.types import GuardrailError, RiskAssessment, RiskLevel
from orbiter.types import OrbiterError


class TestRiskLevel:
    def test_values(self) -> None:
        assert RiskLevel.SAFE == "safe"
        assert RiskLevel.LOW == "low"
        assert RiskLevel.MEDIUM == "medium"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.CRITICAL == "critical"

    def test_all_members(self) -> None:
        assert set(RiskLevel) == {
            RiskLevel.SAFE,
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        }

    def test_is_str(self) -> None:
        assert isinstance(RiskLevel.SAFE, str)


class TestRiskAssessment:
    def test_create_safe(self) -> None:
        assessment = RiskAssessment(has_risk=False, risk_level=RiskLevel.SAFE)
        assert assessment.has_risk is False
        assert assessment.risk_level == RiskLevel.SAFE
        assert assessment.risk_type is None
        assert assessment.confidence == 1.0
        assert assessment.details == {}

    def test_create_risky(self) -> None:
        assessment = RiskAssessment(
            has_risk=True,
            risk_level=RiskLevel.HIGH,
            risk_type="prompt_injection",
            confidence=0.95,
            details={"pattern": "ignore previous"},
        )
        assert assessment.has_risk is True
        assert assessment.risk_level == RiskLevel.HIGH
        assert assessment.risk_type == "prompt_injection"
        assert assessment.confidence == 0.95
        assert assessment.details == {"pattern": "ignore previous"}

    def test_frozen(self) -> None:
        assessment = RiskAssessment(has_risk=False, risk_level=RiskLevel.SAFE)
        with pytest.raises(Exception):  # noqa: B017
            assessment.has_risk = True  # type: ignore[misc]

    def test_defaults(self) -> None:
        assessment = RiskAssessment(has_risk=False, risk_level=RiskLevel.SAFE)
        assert assessment.risk_type is None
        assert assessment.confidence == 1.0
        assert assessment.details == {}

    def test_validation_risk_level(self) -> None:
        """RiskLevel field validates against the enum."""
        assessment = RiskAssessment(has_risk=True, risk_level=RiskLevel.CRITICAL)
        assert assessment.risk_level == RiskLevel.CRITICAL

    def test_details_default_factory(self) -> None:
        """Each instance gets its own details dict."""
        a = RiskAssessment(has_risk=False, risk_level=RiskLevel.SAFE)
        b = RiskAssessment(has_risk=False, risk_level=RiskLevel.SAFE)
        assert a.details is not b.details


class TestGuardrailError:
    def test_inherits_orbiter_error(self) -> None:
        err = GuardrailError("blocked", risk_level=RiskLevel.HIGH)
        assert isinstance(err, OrbiterError)
        assert isinstance(err, Exception)

    def test_message(self) -> None:
        err = GuardrailError("injection detected", risk_level=RiskLevel.CRITICAL)
        assert str(err) == "injection detected"

    def test_attributes(self) -> None:
        err = GuardrailError(
            "blocked",
            risk_level=RiskLevel.HIGH,
            risk_type="prompt_injection",
            details={"pattern": "ignore"},
        )
        assert err.risk_level == RiskLevel.HIGH
        assert err.risk_type == "prompt_injection"
        assert err.details == {"pattern": "ignore"}

    def test_defaults(self) -> None:
        err = GuardrailError("blocked", risk_level=RiskLevel.HIGH)
        assert err.risk_type is None
        assert err.details == {}

    def test_raises(self) -> None:
        with pytest.raises(GuardrailError, match="blocked"):
            raise GuardrailError("blocked", risk_level=RiskLevel.HIGH)
