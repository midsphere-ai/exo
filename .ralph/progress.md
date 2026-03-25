# Progress Log
Started: Sun Mar  8 11:47:39 PM IST 2026

## Codebase Patterns
- (add reusable patterns here)

---
## [2026-03-09 00:55:24 IST] - US-017: Extend serializable agent-definition schema
Thread: 
Run: 20260308-234806-741576 (iteration 4)
Run log: /home/atg/Github/exo-ai/.ralph/runs/run-20260308-234806-741576-iter-4.log
Run summary: /home/atg/Github/exo-ai/.ralph/runs/run-20260308-234806-741576-iter-4.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: d243982d feat(agent-config): extend runtime control schema
- Post-commit status: remaining pre-existing/unrelated files in worktree: `.ralph/activity.log`, `packages/exo-memory/*`, `packages/exo-models/src/exo/models/context_windows.py`, `packages/exo-web/.astro/data-store.json`, `tasks/prd-exo-framework.md`, `tasks/prd-exo-observability.md`, `tasks/prd-exo-web.md`, `tasks/prd-sukuna-compat.md`, `tasks/prd-agents-roadmap.json`, `AGENTS.md`, `assets/`, `audit.md`, `examples/distributed/mcp_sse_workers.py`, `uv.lock`, `.ralph/guardrails.md`, `.ralph/runs/`, `site/`
- Verification:
  - Command: `uv run pytest packages/exo-core/tests/ -q` -> PASS
  - Command: `uv run ruff check packages/exo-core` -> FAIL (pre-existing repo lint issues in untouched files)
  - Command: `uv run pyright packages/exo-core` -> FAIL (pre-existing repo type issues in untouched files)
  - Command: `uv run ruff check packages/exo-core/src/exo/config.py packages/exo-core/src/exo/loader.py packages/exo-core/tests/test_config.py packages/exo-core/tests/test_loader.py packages/exo-core/tests/test_serialization.py` -> PASS
  - Command: `uv run pyright packages/exo-core/src/exo/config.py packages/exo-core/src/exo/loader.py` -> PASS
  - Command: `uv run mkdocs build` -> FAIL (`mkdocs` is not installed in the workspace tool env)
- Files changed:
  - packages/exo-core/src/exo/agent.py
  - packages/exo-core/src/exo/config.py
  - packages/exo-core/src/exo/loader.py
  - packages/exo-core/tests/test_config.py
  - packages/exo-core/tests/test_loader.py
  - packages/exo-core/tests/test_serialization.py
  - docs/reference/core/config.md
  - docs/guides/agents.md
  - .ralph/activity.log
  - .ralph/progress.md
- What was implemented
  - Added the new runtime-control fields to `Agent` and `AgentConfig`, with defaults that preserve current behavior and validation for planner overrides, budget-awareness strings, injected tool args, HITL tool names, and bounded parallel-subagent counts.
  - Extended `Agent.to_dict()` and `Agent.from_dict()` so the runtime-control contract round-trips cleanly, including a full-field example and the required negative deserialization cases.
  - Updated the YAML loader plus the agent/config docs so serialized configs and documented examples expose the same field set.
- **Learnings for future iterations:**
  - Patterns discovered
    - The cleanest way to keep constructor and serialized-config behavior aligned is to centralize validation helpers in `exo.config` and reuse them from `Agent`.
  - Gotchas encountered
    - `hitl_tools` validation has to run after auto-registration of built-in/context tools, otherwise valid tool names can fail construction.
  - Useful context
    - Package-wide `ruff` and `pyright` still report older backlog in untouched `exo-core` files, so focused file-level checks are useful for isolating this story's signal.
---
## [2026-03-08 23:59:43 IST] - US-002: Research agent-runtime control contracts
Thread: 
Run: 20260308-234806-741576 (iteration 1)
Run log: /home/atg/Github/exo-ai/.ralph/runs/run-20260308-234806-741576-iter-1.log
Run summary: /home/atg/Github/exo-ai/.ralph/runs/run-20260308-234806-741576-iter-1.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 08212e8c docs(agent-runtime): add control contract memo
- Post-commit status: remaining pre-existing/unrelated files in worktree: `packages/exo-memory/*`, `packages/exo-models/src/exo/models/context_windows.py`, `packages/exo-web/.astro/data-store.json`, `tasks/prd-exo-framework.md`, `tasks/prd-exo-observability.md`, `tasks/prd-exo-web.md`, `tasks/prd-sukuna-compat.md`, `tasks/prd-agents-roadmap.json`, `AGENTS.md`, `assets/`, `audit.md`, `examples/distributed/mcp_sse_workers.py`, `uv.lock`, `.ralph/`, `site/`
- Verification:
  - Command: `uv run pytest packages/exo-core/tests/test_serialization.py packages/exo-web/tests/test_agent_runtime.py packages/exo-distributed/tests/test_models.py -q` -> PASS
  - Command: `uv run ruff check packages/exo-core packages/exo-distributed packages/exo-web` -> FAIL (pre-existing repo issues in untouched files)
  - Command: `uv run pyright packages/exo-core packages/exo-distributed packages/exo-web` -> FAIL (pre-existing repo issues in untouched files)
  - Command: `uv run mkdocs build --strict` -> FAIL (`mkdocs` not installed in the workspace tool env)
  - Command: `uv run --with mkdocs-material mkdocs build --strict` -> FAIL (pre-existing strict-mode doc warnings outside US-002)
  - Command: `uv run --with mkdocs-material mkdocs build` -> PASS
- Files changed:
  - docs/architecture/agent-runtime-control-contracts.md
  - docs/architecture/index.md
  - mkdocs.yml
  - .ralph/progress.md
  - .ralph/activity.log
- What was implemented
  - Added a decision memo that fixes the contract for planning, budget awareness, HITL tools, MCP progress emission, injected tool args, parallel sub-agents, child tool subsets, child `output_schema`, and tool-result metadata.
  - Mapped the exact implementation touchpoints across `Agent`, runner, serialization/parsing, distributed worker payloads, Temporal/event transport gaps, and Exo Web storage/runtime.
  - Included a serialized agent-config JSON example plus explicit rejections for planner-history mixing, injected-arg kwarg pollution, parallel counts above 7, and unknown HITL tool names.
- **Learnings for future iterations:**
  - Patterns discovered
    - `Agent.get_tool_schemas()`, `Agent._execute_tools()`, `Agent._make_spawn_self_tool()`, and `runner._stream()` are the core seams for these runtime controls.
  - Gotchas encountered
    - Exo Web already has planner and approval tables, but they are user-managed or workflow-specific and should not be reused directly for runtime planner transcripts or agent HITL approvals.
  - Useful context
    - Distributed `events.py` currently omits `mcp_progress`, and Temporal currently drops non-text events, so parity work must include transport updates rather than only agent serialization.
---
## [2026-03-09 00:28:03 IST] - US-003: Research context-management and infinite-context stages
Thread: 
Run: 20260308-234806-741576 (iteration 2)
Run log: /home/atg/Github/exo-ai/.ralph/runs/run-20260308-234806-741576-iter-2.log
Run summary: /home/atg/Github/exo-ai/.ralph/runs/run-20260308-234806-741576-iter-2.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: e5a236ad docs(context): add infinite context memo
- Post-commit status: remaining pre-existing/unrelated files in worktree: `packages/exo-memory/*`, `packages/exo-models/src/exo/models/context_windows.py`, `packages/exo-web/.astro/data-store.json`, `tasks/prd-exo-framework.md`, `tasks/prd-exo-observability.md`, `tasks/prd-exo-web.md`, `tasks/prd-sukuna-compat.md`, `tasks/prd-agents-roadmap.json`, `AGENTS.md`, `assets/`, `audit.md`, `examples/distributed/mcp_sse_workers.py`, `uv.lock`, `.ralph/`, `site/`
- Verification:
  - Command: `uv run pytest packages/exo-context/tests/test_context_integration.py packages/exo-memory/tests/test_summary.py packages/exo-memory/tests/test_short_term.py packages/exo-memory/tests/test_long_term.py -q` -> PASS
  - Command: `uv run ruff check packages/exo-core packages/exo-context packages/exo-memory packages/exo-web` -> FAIL (pre-existing repo issues in untouched files)
  - Command: `uv run pyright packages/exo-core packages/exo-context packages/exo-memory packages/exo-web` -> FAIL (pre-existing repo issues in untouched files)
  - Command: `uv run mkdocs build --strict` -> FAIL (`mkdocs` not installed in the workspace tool env)
  - Command: `uv run --with mkdocs-material --with pymdown-extensions mkdocs build --strict` -> FAIL (pre-existing strict-mode doc warnings outside US-003)
  - Command: `uv run --with mkdocs-material --with pymdown-extensions mkdocs build` -> PASS
- Files changed:
  - docs/architecture/context-management-infinite-context-stages.md
  - docs/architecture/index.md
  - mkdocs.yml
  - .ralph/progress.md
  - .ralph/activity.log
- What was implemented
  - Added a decision memo inventorying Exo's current summarization, trimming, vector injection, branch isolation, memory integration, and workspace/artifact behavior from the actual core, context, memory, and web code paths.
  - Summarized current vendor and research approaches for long-context handling, including prompt caching, server-managed compaction/state, subagent isolation, retrieval-backed memory, prompt compression, and long-conversation benchmark evidence.
  - Selected a staged Exo path centered on persisted branch-scoped summaries and checkpoints, retrieval-aware prompt assembly, later workspace alignment, and explicit out-of-scope "true infinite context" claims.
  - Included a concrete long-running conversation flow showing when summaries, checkpoints, retrieval, and compaction should occur.
- **Learnings for future iterations:**
  - Patterns discovered
    - The live runtime path is concentrated in `exo-core/src/exo/agent.py` and `exo-core/src/exo/runner.py`; `exo-context` has richer primitives, but the agent loop still uses bespoke helpers.
  - Gotchas encountered
    - Exo Web currently has parallel memory/checkpoint concepts that are not the same as core `Context.snapshot()` and `MemoryPersistence`, so follow-on stories need explicit unification work rather than assuming one shared model already exists.
  - Useful context
    - Current summaries are transient system messages, current tool-result offloads are process-local workspace artifacts, and current branch isolation stops short of long-term-memory and workspace isolation.
---
## [2026-03-09 00:41:08 IST] - US-004: Research Temporal parity gaps
Thread: 
Run: 20260308-234806-741576 (iteration 3)
Run log: /home/atg/Github/exo-ai/.ralph/runs/run-20260308-234806-741576-iter-3.log
Run summary: /home/atg/Github/exo-ai/.ralph/runs/run-20260308-234806-741576-iter-3.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 3f36e0e4 docs(temporal): add parity gap memo
- Post-commit status: remaining pre-existing/unrelated files in worktree: `packages/exo-memory/*`, `packages/exo-models/src/exo/models/context_windows.py`, `packages/exo-web/.astro/data-store.json`, `tasks/prd-exo-framework.md`, `tasks/prd-exo-observability.md`, `tasks/prd-exo-web.md`, `tasks/prd-sukuna-compat.md`, `tasks/prd-agents-roadmap.json`, `AGENTS.md`, `assets/`, `audit.md`, `examples/distributed/mcp_sse_workers.py`, `uv.lock`, `.ralph/guardrails.md`, `.ralph/runs/`, `site/`
- Verification:
  - Command: `uv run pytest packages/exo-core/tests/test_runner.py packages/exo-mcp/tests/test_progress.py packages/exo-distributed/tests/test_worker.py packages/exo-distributed/tests/test_events.py packages/exo-distributed/tests/test_temporal.py packages/exo-distributed/tests/test_cancel.py -q` -> PASS
  - Command: `uv run ruff check packages/exo-core packages/exo-distributed packages/exo-mcp packages/exo-context` -> FAIL (pre-existing repo issues in untouched files)
  - Command: `uv run pyright packages/exo-core packages/exo-distributed packages/exo-mcp packages/exo-context` -> FAIL (pre-existing repo issues in untouched files)
  - Command: `uv run mkdocs build --strict` -> FAIL (`mkdocs` is not installed in the workspace tool env)
  - Command: `uv run --with mkdocs-material --with pymdown-extensions mkdocs build --strict` -> FAIL (pre-existing docs warnings outside US-004)
  - Command: `uv run --with mkdocs-material --with pymdown-extensions mkdocs build` -> PASS
- Files changed:
  - docs/architecture/temporal-parity-gaps.md
  - docs/architecture/index.md
  - mkdocs.yml
  - .ralph/activity.log
  - .ralph/progress.md
- What was implemented
  - Added a Temporal parity decision memo that compares the current local execution contract against the current Temporal path for streaming, tool execution, MCP progress, memory, context, planning, and cancellation.
  - Defined the externally observable parity contract, including event-surface and ordering rules, cancellation boundaries, and explicit-failure requirements for unsupported Temporal capabilities.
  - Included a canonical detailed-event trace for the same task under local and Temporal execution, plus a small-slice closure order for fixing the current gaps.
- **Learnings for future iterations:**
  - Patterns discovered
    - The right parity baseline is `exo.runner.run.stream()` plus the distributed worker setup that prepares messages and memory before the first model call.
  - Gotchas encountered
    - Temporal currently executes the agent internally but only returns final text, while `TaskHandle.stream()` depends on Redis terminal events that the Temporal path never publishes.
  - Useful context
    - `packages/exo-distributed/src/exo/distributed/events.py` still cannot deserialize `mcp_progress`, `context`, or `message_injected`, so the distributed transport needs a prerequisite fix even before full Temporal parity lands.
---
