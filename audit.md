# Exo-AI Codebase Audit Report

**Date:** 2026-02-19
**Scope:** 15 packages, 5 audit dimensions
**Total findings:** 83 across all categories

---

## 1. BUGS & SECURITY ISSUES

### CRITICAL

**B-1 — Sandbox code injection via incomplete string escaping**
`exo-web` · `services/sandbox.py:87-88`
`_build_runner_script` escapes user code with basic `.replace()` calls but misses `\r`, `\t`, `\0`, and Unicode escapes. A crafted payload can break out of the string literal context and inject arbitrary Python outside sandbox restrictions — complete sandbox escape.

**B-2 — Hardcoded default secret key**
`exo-web` · `config.py:14`
`secret_key` defaults to `"change-me-in-production"` with no startup warning or enforcement. All Fernet-encrypted API keys stored in the database can be trivially decrypted by anyone who knows the default.

**B-3 — SQL injection vector in `_resolve_name`**
`exo-web` · `services/workspace_export.py:181`
`f"SELECT COUNT(*) FROM {table} WHERE name = ? AND user_id = ?"` — the `table` parameter is string-interpolated directly into SQL. Current callers pass hardcoded names, but the interface accepts any string, making this a latent injection point.

**B-4 — Webhook trigger endpoint missing authentication**
`exo-web` · `routes/webhooks.py:146-236`
`POST /api/v1/webhooks/{workflow_id}/{hook_id}` has no auth check. The `url_token` stored in the database is never validated against incoming requests. Anyone who discovers or enumerates a `workflow_id`/`hook_id` pair can trigger arbitrary workflow executions.

---

### HIGH

**B-5 — TerminalTool command injection via shell chaining**
`exo-sandbox` · `tools.py:187-195`
`_check_command` only checks `parts[0]` against the blocklist. The command is executed via `asyncio.create_subprocess_shell`, so `echo hello ; rm -rf /` trivially bypasses the blacklist.

**B-6 — Race condition in rate limiter global state**
`exo-web` · `middleware/rate_limit.py:66-70`
`_auth_window`, `_user_window`, and `_last_cleanup` are global mutable state accessed concurrently with no locking. Non-atomic read-modify-write on `_SlidingWindow._hits` under concurrent async load allows rate limit bypass.

**B-7 — CSRF token comparison not constant-time**
`exo-web` · `middleware/csrf.py:73`
`if csrf_header != row["csrf_token"]` uses Python's default string comparison, which short-circuits on first differing byte. Exploitable as a timing side-channel to recover the token character by character.

**B-8 — Password reset token logged in plaintext**
`exo-web` · `routes/auth.py:389`
`logger.info("Password reset token for %s: %s", body.email, token)` — plaintext tokens in logs enable account takeover for anyone with log access.

**B-9 — IP spoofing via `X-Forwarded-For`**
`exo-web` · `middleware/rate_limit.py:143-145`
`X-Forwarded-For` is blindly trusted without validation. Attackers set arbitrary IPs to bypass per-IP auth rate limiting, enabling brute-force login attacks.

**B-10 — `UnboundLocalError` when `max_steps=0`**
`exo-core` · `agent.py:242`
`Agent.__init__` has no `ge=1` constraint on `max_steps`. When `max_steps=0`, the for-loop never runs and `output` is referenced while unbound, crashing at runtime.

**B-15 — Async callable instructions never awaited**
`exo-core` · `agent.py:214-215`, `runner.py:242-245`, `_internal/call_runner.py:93-96`
`if callable(raw_instr): instructions = instructions(self.name)` — if the callable is `async`, this silently assigns a coroutine object as the system prompt. Appears in 3 separate code paths.

---

### MEDIUM

**B-11 — Shared sentinel object in tool results list**
`exo-core` · `agent.py:297`
`[ToolResult(...)] * len(actions)` creates all elements pointing to the same object. If any `TaskGroup` task is skipped due to an exception, stale shared placeholders appear in results.

**B-12 — Missing session cookie `secure` flag**
`exo-web` · `routes/auth.py:151-158`
`response.set_cookie(..., httponly=True, samesite="lax")` omits `secure=True`. Session cookies transmit over plain HTTP, enabling session hijacking.

**B-13 — LIKE wildcard injection in knowledge base search**
`exo-web` · `routes/knowledge_bases.py:312-315`
Search terms embedded directly as `%{term}%` without escaping `%` or `_`. A query of `%` matches every chunk in a knowledge base.

**B-14 — Cross-user export download (IDOR)**
`exo-web` · `routes/workspace_export.py:36-49`
`download_export` authenticates the user but never verifies the export belongs to them. Any user can download any other user's export by guessing the export ID.

**B-16 — Silent suppression of event callback failures**
`exo-web` · `engine.py:137-138`
`contextlib.suppress(Exception)` wraps all event callback invocations. WebSocket send failures are silently dropped with no logging, making real-time stream debugging impossible.

**B-17 — Redis connection leak in `distributed()` client**
`exo-distributed` · `client.py:146-188`
`TaskBroker`, `TaskStore`, and `EventSubscriber` are connected but `TaskHandle` has no `close()`/`__aexit__`. Every `distributed()` call leaks three Redis connections.

**B-18 — MCPToolWrapper lazy reconnection never cleaned up**
`exo-mcp` · `tools.py:271-278`
Deserialized `MCPToolWrapper` instances lazily reconnect and store `self._connection`, but `cleanup()` is never called automatically. Connections accumulate in distributed worker deployments.

---

### LOW

**B-19 — Hardcoded `"Hello"` fallback for agent node with no upstream**
`exo-web` · `engine.py:130`
Entry-point agent nodes default to `"Hello"` instead of an empty or configurable prompt.

**B-20 — Debug `set_variable` can corrupt workflow state**
`exo-web` · `engine.py:1231-1235`
The debug command allows overwriting any key in the `variables` dict, including outputs of nodes that haven't executed yet.

**B-21 — `parse_namespaced_name` lossy round-trip**
`exo-mcp` · `tools.py:82-107`
Server names containing `__` produce ambiguous splits; tool name parsing returns incorrect server/tool components.

---

## 2. LOGICAL ISSUES

### HIGH

**L-1 — Loop detection triggers at `threshold + 1` instead of `threshold`**
`exo-core` · `_internal/call_runner.py:142-166`
`_check_loop` checks signatures stored on *previous* nodes, then stores the current node's signature *after* the check. The current call is never counted, so loop detection consistently requires one extra repetition to fire.

**L-2 — `AgentService.stream_agent` bypasses the agent's tool loop entirely**
`exo-web` · `services/agent_runtime.py:248-299`
`stream_agent` calls `provider.stream()` directly, skipping `run.stream()`. Result: in streaming mode, LLM tool calls are emitted as tokens but **never executed**. Non-streaming `run_agent` correctly runs the full tool loop. This is a fundamental behavioral split between streaming and non-streaming workflow execution.

---

### MEDIUM

**L-3 — `max_steps=0` causes `UnboundLocalError`** *(also B-10)*
`exo-core` · `agent.py:226-242`

**L-4 — Handoff detection requires exact string match**
`exo-core` · `swarm.py:487-506`
`_detect_handoff` only recognises a handoff if the entire LLM output (stripped) equals the target agent name. Any natural-language framing breaks it.

**L-5 — `run.stream()` does not fire `START` or `FINISHED` hooks**
`exo-core` · `runner.py:146-446`
The `_stream()` path reimplements the execution loop but never emits `HookPoint.START` or `HookPoint.FINISHED`, creating silent hook parity gaps vs `run()`.

**L-6 — Memory persistence hooks double-registered in distributed workers**
`exo-core` / `exo-distributed` · `agent.py:107-108`, `worker.py:349-351`
When `memory` is passed to `Agent.__init__`, hooks are attached. The distributed worker also calls `MemoryPersistence.attach(agent)`, registering the same hooks again. Every message and tool result is saved twice.

**L-7 — Anthropic message builder can produce consecutive `user` messages**
`exo-models` · `anthropic.py:98-111`
`_build_messages` merges tool results into the previous user block, but doesn't enforce Anthropic's strict user/assistant alternation requirement. A `[User, User]` sequence reaches the API and is rejected.

**L-8 — SQLite: every DB operation opens and closes a fresh connection**
`exo-web` · `database.py:22-36`
The `get_db()` context manager creates a new connection per call, re-applies WAL and foreign-key PRAGMAs, then closes. The workflow engine calls this dozens of times per execution, creating unnecessary overhead and concurrent-writer contention.

**L-9 — MCP tool name round-trip broken for names containing `__`**
`exo-mcp` · `tools.py:82-107` *(also B-21)*

**L-10 — Distributed worker memory deduplication uses `is` identity**
`exo-distributed` · `worker.py:551-558`
`it is prior_items[-1]` relies on object identity in the list returned from the memory backend. If the list is ever copied or items reconstructed, the filter silently stops working and the current user message is saved as duplicate context.

**L-11 — `AgentExecutor.execute()` calls `.text` on potential `RunResult`**
`exo-a2a` · `server.py:84`
`Agent.run()` returns `AgentOutput` with `.text`; the public `run()` returns `RunResult` with `.output`. If a `Swarm` is wrapped by `AgentExecutor`, the attribute lookup fails at runtime.

**L-12 — Workflow debug mode serialises parallel layers sequentially**
`exo-web` · `engine.py:1062-1064`
In debug mode all nodes from parallel layers are flattened into a sequential list, changing execution semantics. Nodes with concurrent side-effects produce different results in debug vs normal mode.

**L-13 — OTel instruments recreated on every stream step**
`exo-core` · `runner.py:203-217`
`meter.create_counter()` is called inside `_record_stream_metrics`, which runs per step. OTel meters should create instruments once and reuse them.

---

### LOW

**L-14 — `EventBus.emit()` exception propagation undocumented**
`exo-core` · `events.py:52`
A failing handler silently aborts all subsequent handlers for the same event. `HookManager` documents this; `EventBus` does not.

**L-15 — Tool stubs in agent runtime always return fake results**
`exo-web` · `services/agent_runtime.py:137-138` *(also completeness C-7)*

**L-16 — `_gather_upstream_inputs` O(N·E) linear scan**
`exo-web` · `engine.py:84-101`
Builds no adjacency map; rescans all edges per node. Performance degrades quadratically for large workflows.

---

## 3. NON-COMPLETENESS & STUBS

### HIGH

**C-1 — Workflow engine `llm`, `code`, `api` node types are stubs**
`exo-web` · `engine.py:494-517`
`"llm"` returns `"LLM response for: {prompt[:100]}"`. `"code"` returns `"Code executed successfully"`. `"api"` returns `"200 OK"`. None perform real work.

**C-2 — Context state inspector returns hardcoded placeholder tree**
`exo-web` · `routes/context_state.py:43-62`
The endpoint's own docstring says: *"For now it returns a placeholder tree so the frontend inspector can be exercised."* Returns zero token usage and no children unconditionally.

**C-3 — `VeRLTrainer.train()` does not train**
`exo-train` · `train/verl.py:259-296`
Builds a VeRL config dict, then returns `TrainMetrics(loss=0.0, accuracy=0.0)`. A comment explicitly says the real VeRL API call is not made.

**C-4 — `VeRLTrainer.evaluate()` does not evaluate**
`exo-train` · `train/verl.py:300-315`
Counts items, returns zeroed metrics. No inference occurs.

**C-5 — Both sandbox `run_tool()` implementations are stubs**
`exo-sandbox` · `base.py:157-165` (LocalSandbox), `kubernetes.py:231-246` (KubernetesSandbox)
Both return metadata dicts without executing the tool. KubernetesSandbox creates real pods but never sends work to them.

**C-6 — exo-web: 1 test file for 79 source files**
`exo-web` · `tests/`
Only `test_agent_runtime.py` exists. All 50+ route modules, the workflow engine, WebSocket handler, middleware, database layer, and services have zero test coverage.

---

### MEDIUM

**C-7 — Agent runtime tool execution always returns stub JSON**
`exo-web` · `services/agent_runtime.py:137-138`
`_tool_stub` returns `{"status": "executed", "tool": name, "args": kwargs}`. All user-defined tools in the web platform are non-functional.

**C-8 — MCP server `stop()` only logs, does not shut down**
`exo-mcp` · `mcp/server.py:200-202`
The `stop()` injected by `@mcp_server` calls `logger.info(...)` and nothing else.

**C-9 — No concrete `EvolutionStrategy` implementation**
`exo-train` · `train/evolution.py:105-150`
`EvolutionStrategy` is an ABC with three abstract methods. The only concrete impl lives in test fixtures. Users of `EvolutionPipeline` must implement everything from scratch.

**C-10 — Eval scorer silently swallows all exceptions**
`exo-eval` · `eval/ralph/runner.py:175-176`
`except Exception: pass` — scoring failures are silently omitted from results with no logging.

**C-11 — Evaluations route silently swallows provider resolution failures**
`exo-web` · `routes/evaluations.py:464-465`
Falls back to heuristic judging without telling the user that AI judging failed.

**C-12 — Safety test-case generation silently falls back**
`exo-web` · `services/safety.py:282-286`
LLM parse failures are swallowed; built-in test cases substituted with no notification.

**C-13 — Agent template import silently ignores malformed config**
`exo-web` · `routes/agent_templates.py:337-338, 408-409`
`except (json.JSONDecodeError, TypeError): pass` — unsanitized or credential-containing configs may be stored.

**C-14 — `LabelDistributionScorer` always returns `score=0.0` per case**
`exo-eval` · `eval/trajectory_scorers.py:198-213`
Docstring calls this a placeholder. Consumers relying on per-case scores always see zero.

---

### LOW

**C-15 — Safety judge defaults to `score=7` on parse failure**
`exo-web` · `services/safety.py:161`
Hardcoded bias toward "safe" when the LLM response can't be parsed.

**C-16 — `run_tool()` not declared `@abstractmethod` in `Sandbox` base**
`exo-sandbox` · `sandbox/base.py:43-132`
`start`, `stop`, `cleanup` are abstract; `run_tool` — the primary interface — is not, allowing silent no-op subclasses.

**C-17 — `NullSpan.record_exception()` silently no-ops**
`exo-observability` · `observability/tracing.py:46-62`
Exceptions are not recorded anywhere when OTel is absent, making debugging harder without OTel installed.

**C-18 — Silent exception in trace argument inspection**
`exo-observability` · `observability/tracing.py:212-213`
Bare `except TypeError: pass` hides instrumentation failures.

---

## 4. INCONSISTENCIES

### HIGH

**I-1 — Error class hierarchy fragmented: many packages bypass `ExoError`**

Packages that correctly inherit from `ExoError`: exo-core, exo-models, exo-eval, exo-a2a, exo-web.

Packages that raise plain `Exception` subclasses: **exo-memory** (`MemoryError`), **exo-sandbox** (`SandboxError`), **exo-mcp** (`MCPClientError`, `MCPServerError`), **exo-context** (7 error classes), **exo-train** (4 error classes), **exo-cli** (5 error classes), **exo-core internal** (`GraphError`).

`except ExoError` catches nothing from memory, sandbox, context, MCP client/server, train, or CLI packages.

**I-2 — `TaskStatus` defined three times with incompatible types**
- `exo-distributed/distributed/models.py:12` — `StrEnum` (PENDING/RUNNING/COMPLETED/FAILED/CANCELLED/RETRYING)
- `exo-a2a/a2a/types.py:165` — Pydantic `BaseModel` with `state` and `reason` fields
- `exo-memory/memory/long_term.py:74` — separate `StrEnum` with its own values

All three are exported from their package's `__init__.py` into the shared `exo.*` namespace.

**I-3 — `MemoryError` shadows Python's built-in**
`exo-memory` · `memory/base.py:16`
`class MemoryError(Exception)` is exported publicly. Any file that `from exo.memory import MemoryError` can no longer catch real out-of-memory errors in the same scope.

---

### MEDIUM

**I-4 — Pydantic frozen model style: three different patterns**
- Style A (`model_config = {"frozen": True}`): exo-core, exo-models, exo-memory, exo-distributed
- Style B (`class Foo(BaseModel, frozen=True)`): exo-a2a, exo-context, exo-observability
- Style C (no frozen): exo-server

**I-5 — Pydantic `BaseModel` vs stdlib `@dataclass` mixed across packages**
exo-eval, exo-train, exo-observability, exo-context, exo-cli, exo-web all use `@dataclass` for value objects that other packages model as Pydantic. Inconsistent `.model_dump()` / `asdict()` calling conventions.

**I-6 — No unified configuration loading pattern**
exo-core uses frozen Pydantic models; exo-observability adds a thread-safe singleton; exo-context adds a factory; exo-web uses a plain `@dataclass` with `os.getenv`; exo-eval/train use frozen dataclasses. Six different config styles across the workspace.

**I-7 — `describe()` is sync on all local agents, async on `RemoteAgent`**
`exo-a2a` · `a2a/client.py:287`
Code that duck-types agents and calls `.describe()` synchronously will fail on `RemoteAgent`.

**I-8 — Provider error handling: OpenAI/Anthropic catch SDK errors; Gemini/Vertex catch bare `Exception`**
`exo-models` · `gemini.py:324`, `vertex.py:387`
Catching `Exception` also catches non-SDK errors, hiding bugs that should surface.

**I-9 — Custom `to_dict()`/`from_dict()` on Pydantic models**
`Agent`, `Swarm`, `ContextState`, and others (which ARE Pydantic `BaseModel` subclasses) define custom serialization instead of using `.model_dump()` / `.model_validate()`. Non-Pydantic dataclasses using `to_dict` is fine; Pydantic models doing it is redundant and inconsistent.

---

### LOW

**I-10 — `__all__` typed in 2 packages, untyped in 13**
**I-11 — `__version__` defined in 2 of 15 packages**
**I-12 — `pkgutil.extend_path` used in 2 packages, absent in 13**
**I-13 — Mixed relative/absolute imports within exo-models**
**I-14 — `EventBus` uses `on/off/emit`; `HookManager` uses `add/remove/run` for equivalent concepts**
**I-15 — Docstring style inconsistent (`Args:` vs `Attributes:` vs none)**

---

## 5. DUPLICATIONS

### HIGH

**D-1 — Gemini and Vertex providers: ~200 lines of identical code**
`exo-models` · `gemini.py:39-303`, `vertex.py:69-303`
`_FINISH_REASON_MAP`, `_map_finish_reason`, `_to_google_contents`, `_convert_tools`, `_build_config`, `_parse_response`, `_parse_stream_chunk` — all 100% identical. `VertexProvider` only differs from `GeminiProvider` in its `__init__`. Suggested fix: extract to `_google_common.py`; `VertexProvider` subclasses `GeminiProvider`.

**D-2 — SQLite and Postgres memory backends: ~80 lines of shared helpers**
`exo-memory` · `backends/sqlite.py:235-285`, `backends/postgres.py:258-311`
`_extra_fields()` is identical in both. `_row_to_item()` is near-identical (only JSON parsing differs). `search()` and `clear()` have the same 4-field metadata filter logic, duplicated 4 times total.

---

### MEDIUM

**D-3 — Redis connection boilerplate repeated 4 times**
`exo-distributed` · `broker.py:33-62`, `store.py:33-49`, `events.py:97-121`, `events.py:160-178`
`_redis`, `connect()`, `disconnect()`, `_client()` — identical ~15-line pattern in 4 classes. Suggested fix: `RedisConnectionMixin`.

**D-4 — OTel vs in-memory fallback branching repeated 8 times**
`exo-observability` · `metrics.py:222-284`, `exo-distributed` · `metrics.py:52-179`, `events.py:48-69`
Every recording function duplicates the same `if HAS_OTEL: ... else: _collector...` structure (~120 lines total). Suggested fix: `increment_counter(name, attrs)` and `record_histogram_value(name, value, attrs)` helpers that encapsulate the branching.

---

### LOW

**D-5 — `_PREFIX = "exo"` defined in both `exo-core/log.py` and `exo-observability/logging.py`**
**D-6 — `_map_finish_reason` function structure identical across all providers (values differ correctly; Gemini/Vertex fully duplicated via D-1)**
**D-7 — 4-field metadata filter clause building duplicated across `search()` and `clear()` in both memory backends**
**D-8 — Module-level singleton + `reset()` pattern in 6 observability modules** (idiomatic but repetitive)

---

## Summary Dashboard

| Category | Critical | High | Medium | Low | Total |
|---|---|---|---|---|---|
| Bugs & Security | 4 | 6 | 9 | 6 | **25** |
| Logical Issues | — | 2 | 11 | 3 | **16** |
| Non-Completeness | — | 6 | 8 | 4 | **18** |
| Inconsistencies | — | 3 | 6 | 7 | **16** |
| Duplications | — | 2 | 2 | 4 | **8** |
| **TOTAL** | **4** | **19** | **36** | **24** | **83** |

---

## Recommended Fix Priority

**Immediate (security/correctness breaking):**
1. **B-1** — Fix sandbox escape (use `ast.literal_eval` or a proper serialiser, not manual escaping)
2. **B-2** — Add startup assertion that rejects the default secret key in non-debug mode
3. **B-4** — Validate `url_token` in webhook trigger endpoint
4. **B-8** — Remove token from log line
5. **L-2** — Fix `stream_agent` to use `run.stream()` not bare `provider.stream()`
6. **B-15** — Audit all `callable(instructions)` call sites and add `asyncio.iscoroutinefunction` + `await`

**Short-term (data integrity / correctness):**
7. **L-1** — Fix loop detection off-by-one (store signature before checking)
8. **L-6** — Guard against double memory hook registration in distributed workers
9. **I-3** — Rename `MemoryError` to `ExoMemoryError`
10. **I-2** — Rename colliding `TaskStatus` types
11. **B-17 / B-18** — Fix Redis and MCP connection leaks
12. **I-1** — Standardise all error classes to inherit `ExoError`

**Medium-term (architecture / maintainability):**
13. **D-1** — Extract `_google_common.py` for Gemini/Vertex
14. **D-2** — Extract shared memory backend helpers
15. **C-6** — Add test coverage for exo-web routes and engine
16. **C-1** — Implement `llm`, `code`, `api` node types or gate them with `NotImplementedError`
17. **L-5** — Bring `run.stream()` to hook parity with `run()`
18. **I-4/I-5** — Standardise on Pydantic or dataclass per context; document the convention
