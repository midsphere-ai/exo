# Progress Log
Started: Sun Mar  8 11:47:39 PM IST 2026

## Codebase Patterns
- (add reusable patterns here)

---
## [2026-03-08 23:59:43 IST] - US-002: Research agent-runtime control contracts
Thread: 
Run: 20260308-234806-741576 (iteration 1)
Run log: /home/atg/Github/orbiter-ai/.ralph/runs/run-20260308-234806-741576-iter-1.log
Run summary: /home/atg/Github/orbiter-ai/.ralph/runs/run-20260308-234806-741576-iter-1.md
- Guardrails reviewed: yes
- No-commit run: false
- Commit: 08212e8c docs(agent-runtime): add control contract memo
- Post-commit status: remaining pre-existing/unrelated files in worktree: `packages/orbiter-memory/*`, `packages/orbiter-models/src/orbiter/models/context_windows.py`, `packages/orbiter-web/.astro/data-store.json`, `tasks/prd-orbiter-framework.md`, `tasks/prd-orbiter-observability.md`, `tasks/prd-orbiter-web.md`, `tasks/prd-sukuna-compat.md`, `tasks/prd-agents-roadmap.json`, `AGENTS.md`, `assets/`, `audit.md`, `examples/distributed/mcp_sse_workers.py`, `uv.lock`, `.ralph/`, `site/`
- Verification:
  - Command: `uv run pytest packages/orbiter-core/tests/test_serialization.py packages/orbiter-web/tests/test_agent_runtime.py packages/orbiter-distributed/tests/test_models.py -q` -> PASS
  - Command: `uv run ruff check packages/orbiter-core packages/orbiter-distributed packages/orbiter-web` -> FAIL (pre-existing repo issues in untouched files)
  - Command: `uv run pyright packages/orbiter-core packages/orbiter-distributed packages/orbiter-web` -> FAIL (pre-existing repo issues in untouched files)
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
  - Mapped the exact implementation touchpoints across `Agent`, runner, serialization/parsing, distributed worker payloads, Temporal/event transport gaps, and Orbiter Web storage/runtime.
  - Included a serialized agent-config JSON example plus explicit rejections for planner-history mixing, injected-arg kwarg pollution, parallel counts above 7, and unknown HITL tool names.
- **Learnings for future iterations:**
  - Patterns discovered
    - `Agent.get_tool_schemas()`, `Agent._execute_tools()`, `Agent._make_spawn_self_tool()`, and `runner._stream()` are the core seams for these runtime controls.
  - Gotchas encountered
    - Orbiter Web already has planner and approval tables, but they are user-managed or workflow-specific and should not be reused directly for runtime planner transcripts or agent HITL approvals.
  - Useful context
    - Distributed `events.py` currently omits `mcp_progress`, and Temporal currently drops non-text events, so parity work must include transport updates rather than only agent serialization.
---
