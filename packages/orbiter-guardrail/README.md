# orbiter-guardrail

Pluggable security detection framework for Orbiter agents.

## What's Included

- **RiskLevel** — severity enum (SAFE, LOW, MEDIUM, HIGH, CRITICAL)
- **RiskAssessment** — immutable result of a backend's risk analysis
- **GuardrailError** — exception raised when a guardrail blocks an operation
