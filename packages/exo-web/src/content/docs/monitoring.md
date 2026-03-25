---
title: Monitoring
description: Track agent runs, costs, and performance
section: guides
order: 5
---

# Monitoring

Exo provides real-time monitoring for all agent and workflow executions. Track performance, costs, and errors from a single dashboard.

## Runs Dashboard

Navigate to **Monitoring > Runs** to see all executions:

- **Status** — Running, completed, failed, or cancelled
- **Duration** — Total execution time
- **Token Usage** — Input and output tokens consumed
- **Cost** — Estimated cost based on provider pricing

Click any run to see the full trace with individual steps.

## Cost Tracking

The **Costs** page shows aggregated spending by:

- Time period (daily, weekly, monthly)
- Provider and model
- Agent or workflow
- Project

Set budget alerts to get notified when spending exceeds thresholds.

## Alerts

Configure alerts for:

- **Error rate** — Triggers when failure percentage exceeds a threshold
- **Latency** — Triggers when response time exceeds a limit
- **Budget** — Triggers when estimated cost exceeds a limit
- **Availability** — Triggers on consecutive failures

Alerts can notify via the in-app notification system.

## Logs

The **Logs** page provides searchable, structured logs from all executions. Filter by:

- Severity level (info, warning, error)
- Agent or workflow name
- Time range
- Search text
