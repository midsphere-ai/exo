# exo-guardrail

Pluggable security detection framework for Exo agents.

## What's Included

- **RiskLevel** — severity enum (SAFE, LOW, MEDIUM, HIGH, CRITICAL)
- **RiskAssessment** — immutable result of a backend's risk analysis
- **GuardrailError** — exception raised when a guardrail blocks an operation
