# PRD: Port Agent-Core Features to Orbiter

## Introduction

Port all significant features from the agent-core (openJiuwen) framework into Orbiter, achieving full feature parity while preserving Orbiter's modular architecture, code style, and every existing capability. Agent-core is a Chinese-origin single-package AI agent SDK with sophisticated RAG pipelines, self-optimizing agents, Pregel-based graph execution, context engineering, security guardrails, and a 5-type memory system. Orbiter is a 15-package UV workspace monorepo that excels in provider breadth, observability, distributed execution, and web platform — but lacks these features.

**Package Strategy (Consolidation):** Only two new packages will be created — `orbiter-guardrail` and `orbiter-retrieval`. All other features fold into existing packages (`orbiter-core`, `orbiter-context`, `orbiter-memory`, `orbiter-train`).

**Backward Compatibility Guarantee:** Every existing Orbiter API, class, function, test, hook, event, config, and behavior MUST remain fully functional. No breaking changes. New features are purely additive or extend existing abstractions via new optional parameters, new subclasses, or new modules. All ~2,900 existing tests must continue to pass after every story.

---

## Goals

- Achieve full feature parity with agent-core across all 10 capability areas
- Preserve Orbiter's modular monorepo architecture and coding conventions
- Zero regressions — all existing tests pass, all existing APIs stable
- Each story produces ≤200 lines of code with extensive documentation and tests
- Research-first approach: every epic begins with a design doc story
- Production-grade code quality: Google docstrings, Pydantic v2, async-first, pyright strict, ruff clean

---

## Epic 1: Security Guardrail Framework

**Package:** New `orbiter-guardrail` (depends on `orbiter-core`)
**Source Reference:** `agent-core/openjiuwen/core/security/guardrail/`

### US-101: Research guardrail architecture
**Description:** As a developer, I want a design document for the guardrail framework so that implementation follows a clear plan that integrates with Orbiter's existing HookManager.

**Acceptance Criteria:**
- [ ] Read agent-core's `guardrail.py`, `backends.py`, `models.py`, `enums.py`, `builtin.py`
- [ ] Read Orbiter's `hooks.py` (HookPoint enum, HookManager class, 7 lifecycle points)
- [ ] Read Orbiter's `events.py` (EventBus pattern)
- [ ] Produce `docs/design/guardrail-design.md` covering: type hierarchy, integration points with HookManager, backend protocol, risk model, event flow diagram
- [ ] Identify which HookPoints guardrails attach to (PRE_LLM_CALL, PRE_TOOL_CALL minimum)
- [ ] Document how guardrails interact with existing hooks without breaking sequential execution order

### US-102: Package scaffold + RiskLevel + RiskAssessment
**Description:** As a developer, I want the `orbiter-guardrail` package structure with core risk types so that guardrail implementations have a foundation.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-guardrail/` with standard layout (pyproject.toml, src/orbiter/guardrail/, tests/)
- [ ] `RiskLevel` StrEnum: SAFE, LOW, MEDIUM, HIGH, CRITICAL
- [ ] `RiskAssessment` frozen Pydantic model: `has_risk: bool`, `risk_level: RiskLevel`, `risk_type: str | None`, `confidence: float`, `details: dict[str, Any]`
- [ ] `GuardrailError` exception inheriting `OrbiterError`
- [ ] `__init__.py` with `extend_path` and public exports
- [ ] Add to workspace root `pyproject.toml` members
- [ ] `uv sync` succeeds
- [ ] Ruff + pyright clean

### US-103: GuardrailBackend ABC + GuardrailResult
**Description:** As a developer, I want pluggable guardrail backend abstraction so that detection logic is swappable.

**Acceptance Criteria:**
- [ ] `GuardrailBackend` ABC with single method: `async def analyze(self, data: dict[str, Any]) -> RiskAssessment`
- [ ] `GuardrailResult` frozen Pydantic model: `is_safe: bool`, `risk_level: RiskLevel`, `risk_type: str | None`, `details: dict[str, Any]`, `modified_data: dict[str, Any] | None`
- [ ] Class methods `GuardrailResult.safe()` and `GuardrailResult.block(risk_level, risk_type, details)`
- [ ] Tests for both classes
- [ ] Google docstrings on all public APIs

### US-104: BaseGuardrail with HookManager integration
**Description:** As a developer, I want a base guardrail class that registers itself as hooks on an Agent's HookManager so that guardrails run automatically during agent execution.

**Acceptance Criteria:**
- [ ] `BaseGuardrail` class with: `backend: GuardrailBackend | None`, `events: list[str]` (which HookPoints to attach to)
- [ ] `attach(agent)` method: registers async hooks on agent's `hook_manager` for each event
- [ ] `detach(agent)` method: removes hooks
- [ ] `async detect(event: str, **data) -> GuardrailResult` — calls backend.analyze() if backend set, else returns safe
- [ ] When `GuardrailResult.is_safe` is False and risk_level >= HIGH, raises `GuardrailError`
- [ ] Existing hooks on the agent are NOT disturbed — guardrail hooks append to the list
- [ ] Tests with mock agent + mock backend verifying hook registration and detection flow
- [ ] All ~2,900 existing orbiter tests still pass

### US-105: UserInputGuardrail for prompt injection detection
**Description:** As a developer, I want a built-in guardrail that detects prompt injection and jailbreak attempts in user input.

**Acceptance Criteria:**
- [ ] `UserInputGuardrail(BaseGuardrail)` with default events `[HookPoint.PRE_LLM_CALL]`
- [ ] Built-in `PatternBackend(GuardrailBackend)` using regex patterns for common injection strings (e.g., "ignore previous instructions", "system prompt", role-play injection patterns)
- [ ] Pattern list is configurable via constructor
- [ ] `async analyze(data)` checks `data["messages"]` for the latest user message content against patterns
- [ ] Returns `RiskAssessment` with appropriate risk_level based on match confidence
- [ ] Tests with injection examples (positive detection) and benign input (no false positive)

### US-106: LLM-based guardrail backend
**Description:** As a developer, I want an LLM-powered guardrail backend for sophisticated content analysis beyond pattern matching.

**Acceptance Criteria:**
- [ ] `LLMGuardrailBackend(GuardrailBackend)` that uses an LLM to assess risk
- [ ] Constructor takes `model: str` (provider:model format) and `prompt_template: str`
- [ ] `analyze()` formats data into prompt, calls LLM, parses structured response into `RiskAssessment`
- [ ] Default prompt template for content safety analysis included
- [ ] Tests using MockProvider (no real API calls)

### US-107: Guardrail integration tests + documentation
**Description:** As a developer, I want end-to-end tests showing guardrails working with Agent.run() and documentation for the package.

**Acceptance Criteria:**
- [ ] Integration test: Agent with UserInputGuardrail attached, run with injection input → raises GuardrailError
- [ ] Integration test: Agent with guardrail, run with safe input → succeeds normally
- [ ] Integration test: Multiple guardrails on same agent → all execute in order
- [ ] Verify all existing orbiter-core tests still pass
- [ ] Google docstrings on all public classes and methods

---

## Epic 2: Advanced Workflow Engine

**Package:** Extend `orbiter-core` (swarm.py, _internal/graph.py)
**Source Reference:** `agent-core/openjiuwen/core/graph/`, `openjiuwen/core/workflow/components/`

### US-201: Research workflow extension architecture
**Description:** As a developer, I want a design document for extending Swarm with loops, branches, and sub-workflows so that implementation preserves existing Swarm modes.

**Acceptance Criteria:**
- [ ] Read agent-core's `LoopComponent`, `BranchComponent`, `SubWorkflowComponent`, expression evaluator
- [ ] Read Orbiter's `swarm.py` (703 lines), `_internal/graph.py` (165 lines), `_internal/nested.py` (107 lines)
- [ ] Read Orbiter's `_internal/agent_group.py` (ParallelGroup, SerialGroup)
- [ ] Produce `docs/design/workflow-extensions-design.md` covering: how LoopNode/BranchNode integrate with existing topological sort, how expression evaluator is isolated, backward compatibility with existing `"a >> b >> c"` DSL
- [ ] Verify existing Swarm modes (workflow, handoff, team) are NOT modified
- [ ] Document new DSL extensions (e.g., `"a >> branch(cond, b, c) >> d"`)

### US-202: Safe AST expression evaluator
**Description:** As a developer, I want a security-hardened expression evaluator for workflow branch conditions so that user-provided expressions cannot execute arbitrary code.

**Acceptance Criteria:**
- [ ] New file `_internal/expression.py` (max 200 lines)
- [ ] `evaluate_expression(expr: str, variables: dict[str, Any]) -> Any` function
- [ ] AST-based parsing via `ast.parse(mode='eval')`
- [ ] Blocks: dunder access (`__class__`, `__bases__`), imports, function calls (except allowlisted), attribute access on dangerous types
- [ ] Limits: MAX_EXPRESSION_LENGTH=500, MAX_AST_DEPTH=10, MAX_COLLECTION_SIZE=1000
- [ ] Normalizes: `&&` → `and`, `||` → `or`, `true` → `True`, `false` → `False`
- [ ] Supports: comparisons, boolean ops, arithmetic, string ops, dict/list indexing
- [ ] Tests with valid expressions and attack vectors (injection, DOS, dunder access)

### US-203: BranchNode for conditional routing
**Description:** As a developer, I want a BranchNode that routes execution to different agents based on runtime conditions so that workflows can have dynamic paths.

**Acceptance Criteria:**
- [ ] `BranchNode` class in new file `_internal/branch_node.py`
- [ ] Constructor: `name: str`, `condition: str | Callable`, `if_true: str` (agent name), `if_false: str` (agent name)
- [ ] If condition is a string, evaluates via `evaluate_expression()` with current workflow state
- [ ] If condition is a callable, calls it with `(state: dict) -> bool`
- [ ] Duck-type marker: `is_branch: bool = True` (like `is_group` pattern)
- [ ] Integrates with Swarm's workflow mode execution (detected in `_run_workflow`)
- [ ] Existing Swarm tests all pass unchanged
- [ ] Tests for BranchNode with string conditions, callable conditions, true/false paths

### US-204: LoopNode for iteration
**Description:** As a developer, I want a LoopNode that repeats an agent or group of agents based on count, array, or expression so that workflows can iterate.

**Acceptance Criteria:**
- [ ] `LoopNode` class in new file `_internal/loop_node.py`
- [ ] Three modes: `count: int` (repeat N times), `items: str` (state key for array iteration), `condition: str` (expression-based while loop)
- [ ] `max_iterations: int = 100` safety limit
- [ ] `body: str | list[str]` — agent name(s) to execute in the loop
- [ ] Break support: if body agent returns output containing `"[BREAK]"`, loop terminates
- [ ] Duck-type marker: `is_loop: bool = True`
- [ ] Tests for count-based, array-based, expression-based, and break scenarios

### US-205: Workflow state context for nodes
**Description:** As a developer, I want workflow nodes to read/write a shared state dict so that BranchNode conditions and LoopNode iterations can access runtime data.

**Acceptance Criteria:**
- [ ] `WorkflowState` class (Pydantic model or dataclass): `data: dict[str, Any]`, `set(key, value)`, `get(key, default)`, `to_dict()`
- [ ] Passed through workflow execution; each agent's output stored as `state.set(agent_name, output)`
- [ ] BranchNode and LoopNode read from this state for condition evaluation
- [ ] Existing Swarm workflow mode updated to create and pass WorkflowState (backward-compatible — state is optional, old code works without it)
- [ ] Tests verifying state propagation through linear, branching, and looping flows

### US-206: Workflow state checkpointing
**Description:** As a developer, I want workflow execution to checkpoint state before each node so that failed workflows can resume from the last successful node.

**Acceptance Criteria:**
- [ ] `WorkflowCheckpoint` frozen dataclass: `node_name: str`, `state: dict`, `completed_nodes: list[str]`, `timestamp: float`
- [ ] `WorkflowCheckpointStore` (in-memory): `save(checkpoint)`, `latest() -> Checkpoint | None`, `list_all() -> list[Checkpoint]`
- [ ] Swarm's workflow mode optionally creates checkpoints before each node (enabled via `checkpoint: bool = False` constructor param)
- [ ] `Swarm.resume(checkpoint_store)` method: resumes from latest checkpoint, skipping completed nodes
- [ ] Existing Swarm behavior unchanged when `checkpoint=False` (default)
- [ ] Tests for checkpoint creation, resume, and partial execution

### US-207: Mermaid visualization generator
**Description:** As a developer, I want to auto-generate Mermaid flowcharts from Swarm flows so that I can visualize workflow topology.

**Acceptance Criteria:**
- [ ] `to_mermaid(swarm: Swarm) -> str` function in new file `_internal/visualization.py`
- [ ] Generates valid Mermaid `graph TD` syntax from the Swarm's internal Graph
- [ ] Handles linear chains, parallel groups, BranchNodes (diamond shape), LoopNodes (hexagon shape), SwarmNodes (subroutine shape)
- [ ] Optional `Swarm.to_mermaid() -> str` convenience method
- [ ] Tests comparing output against expected Mermaid strings for various topologies

### US-208: Workflow extension tests + backward compatibility verification
**Description:** As a developer, I want comprehensive tests proving new workflow features work and existing features are unbroken.

**Acceptance Criteria:**
- [ ] Test: existing `"a >> b >> c"` DSL still works identically
- [ ] Test: existing workflow, handoff, and team modes pass all original tests
- [ ] Test: BranchNode + LoopNode combined flow
- [ ] Test: LoopNode with nested SwarmNode
- [ ] Test: Mermaid output for complex topology
- [ ] Run full `uv run pytest packages/orbiter-core/tests/` — all pass

---

## Epic 3: RAG/Retrieval Pipeline

**Package:** New `orbiter-retrieval` (depends on `orbiter-core`)
**Source Reference:** `agent-core/openjiuwen/core/retrieval/`

### US-301: Research retrieval architecture
**Description:** As a developer, I want a design document for the retrieval pipeline so that the package has a clear, modular architecture.

**Acceptance Criteria:**
- [ ] Read agent-core's full retrieval directory: embedding/, vector_store/, retriever/, reranker/, query_rewriter/, indexing/
- [ ] Read Orbiter's `orbiter-memory/backends/vector.py` (existing Embeddings ABC, VectorMemoryStore)
- [ ] Read Orbiter's `orbiter-context/_internal/knowledge.py` (existing KnowledgeStore with chunking)
- [ ] Produce `docs/design/retrieval-design.md` covering: package layout, type hierarchy, embedding abstraction (reuse orbiter-memory's Embeddings or create new?), vector store abstraction, retriever types, indexing pipeline, integration with Agent tools
- [ ] Decision: reuse `orbiter-memory.Embeddings` ABC or create independent embedding abstraction (recommendation: create independent, import if both installed)
- [ ] Dependency analysis: what optional deps needed (tiktoken, chromadb, pgvector, milvus)

### US-302: Package scaffold + core types
**Description:** As a developer, I want the `orbiter-retrieval` package with core data types so that retrievers have shared types.

**Acceptance Criteria:**
- [ ] Create `packages/orbiter-retrieval/` with standard layout
- [ ] `Document` Pydantic model: `id: str`, `content: str`, `metadata: dict[str, Any]`, `embedding: list[float] | None`
- [ ] `Chunk` frozen dataclass: `document_id: str`, `index: int`, `content: str`, `start: int`, `end: int`, `metadata: dict[str, Any]`
- [ ] `RetrievalResult` frozen dataclass: `chunk: Chunk`, `score: float`, `metadata: dict[str, Any]`
- [ ] `RetrievalError` exception
- [ ] `__init__.py` with `extend_path` and exports
- [ ] Add to workspace, `uv sync`, ruff + pyright clean

### US-303: Embedding ABC + OpenAI provider
**Description:** As a developer, I want an embedding abstraction with an OpenAI implementation so that I can generate vector representations of text.

**Acceptance Criteria:**
- [ ] `Embeddings` ABC: `async def embed(self, text: str) -> list[float]`, `async def embed_batch(self, texts: list[str]) -> list[list[float]]`, `@property def dimension(self) -> int`
- [ ] `OpenAIEmbeddings(Embeddings)`: wraps OpenAI API, configurable model + dimension + api_key + base_url
- [ ] Batch embedding via single API call for efficiency
- [ ] Tests with mock HTTP responses (no real API calls)

### US-304: Vertex AI + generic HTTP embedding providers
**Description:** As a developer, I want Vertex AI and generic HTTP embedding providers for flexibility.

**Acceptance Criteria:**
- [ ] `VertexEmbeddings(Embeddings)`: wraps Google Vertex AI, configurable model + dimension + project + location
- [ ] `HTTPEmbeddings(Embeddings)`: generic HTTP POST to any embedding endpoint, configurable URL + headers + request/response field paths
- [ ] Both implement `embed()` and `embed_batch()`
- [ ] Tests with mocked HTTP calls

### US-305: VectorStore ABC + in-memory implementation
**Description:** As a developer, I want a vector store abstraction with an in-memory implementation for development and testing.

**Acceptance Criteria:**
- [ ] `VectorStore` ABC: `async def add(self, chunks: list[Chunk], embeddings: list[list[float]])`, `async def search(self, query_embedding: list[float], *, top_k: int = 5, filter: dict | None = None) -> list[RetrievalResult]`, `async def delete(self, document_id: str)`, `async def clear()`
- [ ] `InMemoryVectorStore(VectorStore)`: stores chunks + embeddings in dicts, cosine similarity search
- [ ] Tests for add, search (ranked by similarity), delete, clear

### US-306: pgvector VectorStore backend
**Description:** As a developer, I want a PostgreSQL/pgvector backend so that embeddings persist in production.

**Acceptance Criteria:**
- [ ] `PgVectorStore(VectorStore)` in `backends/pgvector.py`
- [ ] Uses `asyncpg` for async PostgreSQL access
- [ ] Creates table with `vector` column type (pgvector extension)
- [ ] `search()` uses `<=>` cosine distance operator
- [ ] Optional `[pgvector]` extra dependency
- [ ] Tests with mocked asyncpg (no real database)

### US-307: ChromaDB VectorStore backend
**Description:** As a developer, I want a ChromaDB backend for local persistent vector search.

**Acceptance Criteria:**
- [ ] `ChromaVectorStore(VectorStore)` in `backends/chroma.py`
- [ ] Wraps chromadb Collection API
- [ ] Configurable `collection_name` and `path` (persistent) or ephemeral
- [ ] Optional `[chroma]` extra dependency
- [ ] Tests with mocked ChromaDB client

### US-308: VectorRetriever
**Description:** As a developer, I want a pure dense vector retriever so that I can search by semantic similarity.

**Acceptance Criteria:**
- [ ] `Retriever` ABC: `async def retrieve(self, query: str, *, top_k: int = 5, **kwargs) -> list[RetrievalResult]`
- [ ] `VectorRetriever(Retriever)`: takes `embeddings: Embeddings` + `store: VectorStore`, embeds query, searches store
- [ ] Optional `score_threshold: float` filter
- [ ] Tests with InMemoryVectorStore + mock embeddings

### US-309: SparseRetriever (BM25)
**Description:** As a developer, I want a BM25 sparse retriever for keyword-based search.

**Acceptance Criteria:**
- [ ] `SparseRetriever(Retriever)`: BM25 scoring over indexed chunks
- [ ] `index(chunks: list[Chunk])` — builds inverted index with term frequencies
- [ ] `retrieve(query)` — tokenizes query, computes BM25 scores, returns top-k
- [ ] Pure Python implementation (no external deps beyond stdlib)
- [ ] Tests verifying keyword relevance ranking

### US-310: HybridRetriever with RRF fusion
**Description:** As a developer, I want a hybrid retriever that fuses dense and sparse results via Reciprocal Rank Fusion.

**Acceptance Criteria:**
- [ ] `HybridRetriever(Retriever)`: takes `vector_retriever: VectorRetriever` + `sparse_retriever: SparseRetriever`
- [ ] `retrieve()` calls both retrievers, merges results via RRF: `score = sum(1 / (k + rank))` for each result across both lists
- [ ] Configurable `k: int = 60` (RRF constant)
- [ ] Configurable `vector_weight: float = 0.5` for weighted fusion
- [ ] Tests verifying fusion produces expected ranking

### US-311: Reranker ABC + LLM reranker
**Description:** As a developer, I want a reranking abstraction with an LLM-based implementation for result refinement.

**Acceptance Criteria:**
- [ ] `Reranker` ABC: `async def rerank(self, query: str, results: list[RetrievalResult], *, top_k: int = 5) -> list[RetrievalResult]`
- [ ] `LLMReranker(Reranker)`: sends query + result texts to LLM, asks for relevance ranking, reorders
- [ ] Configurable `model: str` and `prompt_template: str`
- [ ] Tests with MockProvider

### US-312: Text chunking strategies
**Description:** As a developer, I want configurable text chunking so that documents are split into retrieval-friendly chunks.

**Acceptance Criteria:**
- [ ] `Chunker` ABC: `def chunk(self, document: Document) -> list[Chunk]`
- [ ] `CharacterChunker(Chunker)`: splits by character count with configurable `chunk_size` and `chunk_overlap`
- [ ] `ParagraphChunker(Chunker)`: splits at paragraph boundaries (`\n\n`), respects chunk_size limit
- [ ] `TokenChunker(Chunker)`: splits by token count (optional tiktoken dependency)
- [ ] All chunkers preserve chunk start/end offsets
- [ ] Tests for each strategy with edge cases (empty doc, single paragraph, overlap)

### US-313: Document parsers (PDF, Markdown, JSON)
**Description:** As a developer, I want document parsers that extract text from common formats.

**Acceptance Criteria:**
- [ ] `Parser` ABC: `def parse(self, source: str | bytes | Path) -> Document`
- [ ] `MarkdownParser(Parser)`: strips formatting, preserves structure
- [ ] `JSONParser(Parser)`: flattens JSON to readable text with key paths
- [ ] `TextParser(Parser)`: passthrough for plain text
- [ ] `PDFParser(Parser)`: extracts text from PDF (optional `pymupdf` dependency)
- [ ] Tests with sample inputs for each format

### US-314: Query rewriting via LLM
**Description:** As a developer, I want LLM-based query rewriting to improve retrieval quality for ambiguous queries.

**Acceptance Criteria:**
- [ ] `QueryRewriter`: takes `model: str`, `prompt_template: str`
- [ ] `async rewrite(query: str, *, history: list[str] | None = None) -> str`
- [ ] Default template expands the query for better retrieval (adds synonyms, disambiguates)
- [ ] Optional history compression: if history provided, incorporates conversation context
- [ ] Tests with MockProvider

### US-315: AgenticRetriever (multi-round LLM-driven)
**Description:** As a developer, I want an agentic retriever that iteratively refines queries and retrieves until satisfied.

**Acceptance Criteria:**
- [ ] `AgenticRetriever(Retriever)`: takes `base_retriever: Retriever`, `rewriter: QueryRewriter`, `model: str`
- [ ] `retrieve()` loop (max 3 rounds): rewrite query → retrieve → LLM judges if results are sufficient → if not, rewrite and retry
- [ ] Returns best results across all rounds (deduplicated by chunk ID)
- [ ] Configurable `max_rounds: int = 3`, `sufficiency_threshold: float = 0.7`
- [ ] Tests with MockProvider + mock retriever

### US-316: Knowledge graph — triple extraction
**Description:** As a developer, I want to extract subject-predicate-object triples from text for knowledge graph construction.

**Acceptance Criteria:**
- [ ] `Triple` frozen dataclass: `subject: str`, `predicate: str`, `object: str`, `confidence: float`, `source_chunk_id: str`
- [ ] `TripleExtractor`: takes `model: str`, uses LLM to extract triples from text chunks
- [ ] `async extract(chunks: list[Chunk]) -> list[Triple]`
- [ ] Default prompt template for triple extraction
- [ ] Tests with MockProvider and sample text

### US-317: Knowledge graph — graph retriever
**Description:** As a developer, I want a graph-based retriever that expands search results via knowledge graph traversal.

**Acceptance Criteria:**
- [ ] `GraphRetriever(Retriever)`: takes `base_retriever: Retriever`, `triples: list[Triple]`
- [ ] After initial retrieval, expands results by finding triples where retrieved entities appear as subject or object
- [ ] Beam search expansion: configurable `beam_width: int = 3`, `max_hops: int = 2`
- [ ] Returns expanded results with graph-traversal metadata
- [ ] Tests with mock triples and base retriever

### US-318: Retrieval tools for Agent integration
**Description:** As a developer, I want retrieval tools that can be added to an Agent so that agents can search knowledge bases.

**Acceptance Criteria:**
- [ ] `retrieve_tool(retriever: Retriever) -> Tool`: creates a `@tool`-decorated function wrapping `retriever.retrieve()`
- [ ] Tool schema: `query: str`, `top_k: int = 5`
- [ ] Tool output: formatted list of results with content and scores
- [ ] `index_tool(chunker: Chunker, store: VectorStore, embeddings: Embeddings) -> Tool`: tool for indexing new documents
- [ ] Tests verifying tools work with Agent via MockProvider

---

## Epic 4: Context Engine Enhancement

**Package:** Extend `orbiter-context`
**Source Reference:** `agent-core/openjiuwen/core/context_engine/`

### US-401: Research context enhancement architecture
**Description:** As a developer, I want a design document for context engine enhancements so that new processors integrate cleanly with existing ProcessorPipeline.

**Acceptance Criteria:**
- [ ] Read agent-core's `context_engine.py`, `processor/offloader/`, `processor/compressor/`, `token/`
- [ ] Read Orbiter's `processor.py` (ContextProcessor ABC, ProcessorPipeline, SummarizeProcessor, ToolResultOffloader)
- [ ] Read Orbiter's `token_tracker.py`, `config.py` (ContextConfig with thresholds)
- [ ] Produce `docs/design/context-enhancement-design.md` covering: offloading markers, reload tool, compression processor, round windowing, token budgeting
- [ ] Confirm new processors plug into existing `ProcessorPipeline.register()` without changes
- [ ] Document interaction with existing SummarizeProcessor and ToolResultOffloader

### US-402: Message offloading processor
**Description:** As a developer, I want a processor that replaces oversized messages with `[[OFFLOAD: handle=<id>]]` markers so that context window stays within budget.

**Acceptance Criteria:**
- [ ] `MessageOffloader(ContextProcessor)` with event `"pre_llm_call"`
- [ ] Constructor: `max_message_size: int = 10000` (character limit per message)
- [ ] `process()`: scans `ctx.state["history"]`, replaces messages exceeding limit with marker text containing a unique handle ID
- [ ] Stores original content in `ctx.state["offloaded_messages"]` dict keyed by handle ID
- [ ] Does NOT modify system messages (only user/assistant/tool messages)
- [ ] Tests: message within limit untouched, oversized message replaced, original stored

### US-403: Reload tool for offloaded content
**Description:** As a developer, I want a context tool that lets the LLM request reload of offloaded content so that important information is recoverable.

**Acceptance Criteria:**
- [ ] `reload_offloaded(handle: str) -> str` tool function in `tools.py`
- [ ] Looks up handle in context state's `offloaded_messages` dict
- [ ] Returns full original content if found, error message if not
- [ ] Added to `get_context_tools()` return list (backward compatible — existing tools still returned)
- [ ] Tests verifying reload of offloaded content and error for unknown handle

### US-404: Dialogue compression processor
**Description:** As a developer, I want an LLM-based processor that compresses long tool call chains into summaries so that context stays readable.

**Acceptance Criteria:**
- [ ] `DialogueCompressor(ContextProcessor)` with event `"pre_llm_call"`
- [ ] Constructor: `min_tool_chain_length: int = 3`, `model: str | None = None`
- [ ] `process()`: identifies consecutive tool-call/tool-result message sequences in history
- [ ] If a chain exceeds `min_tool_chain_length`, replaces it with a `SystemMessage` summary
- [ ] Summary generated by calling the `Summarizer` protocol (same as orbiter-memory's)
- [ ] Falls back to simple concatenation if no summarizer provided
- [ ] Tests with mock summarizer, verifying chain detection and replacement

### US-405: Round-level windowing processor
**Description:** As a developer, I want a processor that understands conversation round structure (user→assistant = 1 round) for smarter windowing.

**Acceptance Criteria:**
- [ ] `RoundWindowProcessor(ContextProcessor)` with event `"pre_llm_call"`
- [ ] Constructor: `max_rounds: int = 20`
- [ ] `process()`: parses history into rounds (user message → assistant response, including any tool calls in between)
- [ ] Keeps last `max_rounds` complete rounds plus all system messages
- [ ] Complements existing `ContextConfig.history_rounds` — can be used as alternative
- [ ] Tests with mixed message sequences, verifying round boundary detection

### US-406: Token budgeting with tiktoken
**Description:** As a developer, I want accurate token counting via tiktoken so that context budgeting is precise.

**Acceptance Criteria:**
- [ ] `TiktokenCounter` class in new file `token_counter.py`
- [ ] Constructor: `model: str = "gpt-4o"` — auto-selects encoding (cl100k_base, o200k_base)
- [ ] `count(text: str) -> int` — exact token count
- [ ] `count_messages(messages: list[dict]) -> int` — token count for message list (with per-message overhead)
- [ ] Optional dependency: `tiktoken` declared as extra `[tiktoken]`
- [ ] Fallback: if tiktoken not installed, uses char-based estimation (existing behavior)
- [ ] Tests with known token counts for various strings

---

## Epic 5: Enhanced Memory System

**Package:** Extend `orbiter-memory`
**Source Reference:** `agent-core/openjiuwen/core/memory/`

### US-501: Research memory enhancement architecture
**Description:** As a developer, I want a design document for memory enhancements so that new features integrate with existing ShortTermMemory, LongTermMemory, and persistence system.

**Acceptance Criteria:**
- [ ] Read agent-core's memory types, encryption, MemUpdateChecker, migration system
- [ ] Read Orbiter's `base.py` (MemoryItem, MemoryStore protocol), `short_term.py`, `long_term.py`, `persistence.py`
- [ ] Produce `docs/design/memory-enhancement-design.md`
- [ ] Key decision: add memory_type taxonomy as StrEnum on MemoryItem (backward-compatible via default value)
- [ ] Key decision: encryption wraps MemoryStore (decorator pattern) vs. built into each backend
- [ ] Confirm all changes are additive — existing MemoryItem, ShortTermMemory, LongTermMemory APIs unchanged

### US-502: Memory type taxonomy
**Description:** As a developer, I want a typed memory taxonomy so that different kinds of knowledge can be stored and queried distinctly.

**Acceptance Criteria:**
- [ ] `MemoryCategory` StrEnum: `USER_PROFILE`, `SEMANTIC`, `EPISODIC`, `VARIABLE`, `SUMMARY`, `CONVERSATION` (new)
- [ ] New optional field on `MemoryItem`: `category: MemoryCategory | None = None` (None = backward compatible)
- [ ] `ShortTermMemory.search()` and `LongTermMemory.search()` support optional `category` filter parameter
- [ ] Existing `memory_type` field (str: "human", "ai", "tool", "system") is unchanged — `category` is orthogonal
- [ ] All existing memory tests pass unchanged
- [ ] Tests for category-based filtering

### US-503: AES-256 encryption at rest
**Description:** As a developer, I want optional AES-256 encryption for memory items so that sensitive data is protected.

**Acceptance Criteria:**
- [ ] `EncryptedMemoryStore` wrapper class implementing `MemoryStore` protocol
- [ ] Constructor: `store: MemoryStore`, `key: bytes` (32-byte AES key)
- [ ] `add()`: encrypts `item.content` before delegating to inner store (AES-256-GCM: nonce + tag + ciphertext)
- [ ] `get()` and `search()`: decrypt content after retrieval from inner store
- [ ] `clear()`: delegates directly
- [ ] Key derivation helper: `derive_key(password: str, salt: bytes) -> bytes` using PBKDF2
- [ ] Uses stdlib `cryptography` or built-in (optional dependency `[encryption]`)
- [ ] Tests: encrypt → store → retrieve → decrypt roundtrip, wrong key fails

### US-504: LLM-based memory deduplication
**Description:** As a developer, I want LLM-powered deduplication that detects semantically similar memories and decides ADD, DELETE, or MERGE.

**Acceptance Criteria:**
- [ ] `MemUpdateChecker` class: takes `model: str` (optional, for LLM calls)
- [ ] `async check(new_item: MemoryItem, existing: list[MemoryItem]) -> UpdateDecision`
- [ ] `UpdateDecision` StrEnum: `ADD`, `DELETE`, `MERGE`, `SKIP`
- [ ] If model provided: sends new + top-5 similar existing items to LLM for comparison
- [ ] If no model: falls back to exact content match (simple dedup, matching existing LongTermMemory behavior)
- [ ] `MergeResult` dataclass: `decision: UpdateDecision`, `merged_content: str | None`, `delete_ids: list[str]`
- [ ] Integration with `LongTermMemory.add()` — optional checker parameter
- [ ] Tests with MockProvider for LLM-based dedup, tests for fallback behavior

### US-505: Memory migration system
**Description:** As a developer, I want a versioned migration system for memory store schemas so that backends can be upgraded safely.

**Acceptance Criteria:**
- [ ] `Migration` frozen dataclass: `version: int`, `description: str`, `up: Callable`, `down: Callable | None`
- [ ] `MigrationRegistry`: `register(migration: Migration)`, `list_pending(current_version: int) -> list[Migration]`
- [ ] `async run_migrations(store: MemoryStore, registry: MigrationRegistry)` — applies pending migrations in order, tracks version in store metadata
- [ ] SQLite and Postgres backends store version in `_migrations` metadata table
- [ ] Tests for migration ordering, skip-already-applied, error handling

### US-506: Unified SearchManager
**Description:** As a developer, I want a SearchManager that queries across all memory types and stores in a single call.

**Acceptance Criteria:**
- [ ] `SearchManager` class: takes `stores: list[MemoryStore]`
- [ ] `async search(query: str, *, limit: int = 10, category: MemoryCategory | None = None) -> list[MemoryItem]`
- [ ] Queries all stores in parallel via `asyncio.gather()`
- [ ] Merges results, deduplicates by item ID, sorts by relevance (if scores available) or created_at
- [ ] Respects per-store limit to prevent one store dominating results
- [ ] Tests with multiple mock stores returning overlapping results

---

## Epic 6: Agent Rails — Typed Lifecycle Guards

**Package:** Extend `orbiter-core` (hooks.py, agent.py)
**Source Reference:** `agent-core/openjiuwen/core/single_agent/rail/`

### US-601: Research rails architecture
**Description:** As a developer, I want a design document for the typed rails system so that it extends HookManager without breaking existing hook behavior.

**Acceptance Criteria:**
- [ ] Read agent-core's `rail/base.py` — AgentCallbackEvent, typed inputs, AgentRail ABC, `@rail` decorator
- [ ] Read Orbiter's `hooks.py` (HookPoint, HookManager, Hook type alias)
- [ ] Read how hooks are called in `agent.py` (PRE_LLM_CALL with messages, POST_LLM_CALL with response+usage, etc.)
- [ ] Produce `docs/design/rails-design.md`
- [ ] Key decision: Rails extend hooks (new optional typed interface) vs. parallel system
- [ ] Ensure all existing `hook_manager.add(HookPoint.X, my_func)` calls continue to work

### US-602: Typed event input models
**Description:** As a developer, I want Pydantic models for each hook event's data so that hooks have structured, validated input.

**Acceptance Criteria:**
- [ ] New file `rail_types.py` in orbiter-core
- [ ] `InvokeInputs`: `input: str`, `messages: list[Any] | None`, `result: Any | None`
- [ ] `ModelCallInputs`: `messages: list[Any]`, `tools: list[dict] | None`, `response: Any | None`, `usage: Any | None`
- [ ] `ToolCallInputs`: `tool_name: str`, `arguments: dict[str, Any]`, `result: Any | None`, `metadata: Any | None`
- [ ] `RailContext`: `agent: Any`, `event: HookPoint`, `inputs: InvokeInputs | ModelCallInputs | ToolCallInputs`, `extra: dict[str, Any]`
- [ ] All are Pydantic models (not frozen — hooks may need to mutate inputs)
- [ ] Tests for model creation and validation

### US-603: Rail ABC with priority and retry
**Description:** As a developer, I want a Rail abstract class with priority ordering and retry capability so that lifecycle guards are structured.

**Acceptance Criteria:**
- [ ] `Rail` ABC in new file `rail.py`: `name: str`, `priority: int = 50`
- [ ] Abstract method: `async def handle(self, ctx: RailContext) -> RailAction | None`
- [ ] `RailAction` StrEnum: `CONTINUE`, `SKIP`, `RETRY`, `ABORT`
- [ ] `RetryRequest` dataclass: `delay: float = 0.0`, `max_retries: int = 1`, `reason: str = ""`
- [ ] If `handle()` returns `RETRY`, the caller retries the operation up to `max_retries` with `delay`
- [ ] If `handle()` returns `ABORT`, raises `RailAbortError`
- [ ] If `handle()` returns `None` or `CONTINUE`, execution proceeds normally
- [ ] Tests for each action type

### US-604: RailManager with priority ordering + cross-rail state
**Description:** As a developer, I want a RailManager that runs rails in priority order with a shared extra dict so that rails can coordinate.

**Acceptance Criteria:**
- [ ] `RailManager` class: `add(rail: Rail)`, `remove(rail: Rail)`, `clear()`
- [ ] `async run(event: HookPoint, **data) -> RailAction`: builds `RailContext` with typed inputs, runs all rails sorted by priority (ascending), returns first non-CONTINUE action (or CONTINUE if all pass)
- [ ] `extra: dict[str, Any]` shared across all rails in same invocation (set on RailContext)
- [ ] Compatible with `HookManager`: `RailManager` can be registered as a single hook on HookManager, wrapping the rail execution
- [ ] Existing hooks registered directly on HookManager are unaffected
- [ ] Tests for priority ordering, cross-rail extra dict, abort propagation

### US-605: Agent integration + backward compatibility tests
**Description:** As a developer, I want rails integrated into Agent with full backward compatibility verification.

**Acceptance Criteria:**
- [ ] `Agent` constructor gains optional `rails: list[Rail] | None = None` parameter
- [ ] If rails provided, creates `RailManager` and registers it as hooks on the agent's `hook_manager`
- [ ] Agent with no rails behaves exactly as before (no performance or behavior change)
- [ ] Agent with both traditional hooks and rails: both execute (traditional hooks + rail hooks)
- [ ] All existing orbiter-core tests pass unchanged
- [ ] Integration test: Rail that aborts on specific tool name
- [ ] Integration test: Rail that retries on model error

---

## Epic 7: Task Management Controller

**Package:** Extend `orbiter-core` (new `_internal/task_controller/` subpackage)
**Source Reference:** `agent-core/openjiuwen/core/controller/`

### US-701: Research task controller architecture
**Description:** As a developer, I want a design document for task management so that it integrates with Orbiter's agent execution model.

**Acceptance Criteria:**
- [ ] Read agent-core's `controller/modules/` — TaskManager, TaskScheduler, EventQueue, IntentRecognizer
- [ ] Read Orbiter's `agent.py` (Agent.run loop), `_internal/state.py` (RunState, RunNode)
- [ ] Read Orbiter's `_internal/background.py` (BackgroundTask, PendingQueue)
- [ ] Produce `docs/design/task-controller-design.md`
- [ ] Key decision: task controller as optional module in orbiter-core vs. separate package
- [ ] Document how tasks relate to agent runs and swarm workflows

### US-702: Task model + TaskStatus + TaskManager
**Description:** As a developer, I want a Task model with lifecycle management so that complex workflows can track sub-tasks.

**Acceptance Criteria:**
- [ ] `TaskStatus` StrEnum: SUBMITTED, WORKING, PAUSED, INPUT_REQUIRED, COMPLETED, CANCELED, FAILED, WAITING
- [ ] `Task` Pydantic model: `id: str`, `name: str`, `description: str`, `status: TaskStatus`, `priority: int = 0`, `parent_id: str | None`, `metadata: dict[str, Any]`, `created_at: str`, `updated_at: str`
- [ ] Valid status transitions enforced (e.g., COMPLETED cannot go back to WORKING)
- [ ] `TaskManager`: `create(task: Task) -> str`, `get(task_id: str) -> Task | None`, `update(task_id: str, **fields)`, `delete(task_id: str)`, `list(status: TaskStatus | None = None) -> list[Task]`
- [ ] In-memory store (dict-based, suitable for single-process)
- [ ] Tests for CRUD and status transitions

### US-703: Priority indexing + parent-child hierarchy
**Description:** As a developer, I want task priority ordering and parent-child relationships so that tasks can be decomposed and prioritized.

**Acceptance Criteria:**
- [ ] `TaskManager.list()` returns tasks sorted by priority (descending) then created_at
- [ ] `TaskManager.get_children(parent_id: str) -> list[Task]`
- [ ] `TaskManager.get_subtree(task_id: str) -> list[Task]` — recursive children
- [ ] When parent task is CANCELED, all children are also CANCELED
- [ ] When all children are COMPLETED, parent auto-transitions to COMPLETED (optional, configurable)
- [ ] Tests for priority sorting, hierarchy traversal, cascading status

### US-704: TaskScheduler with concurrent execution
**Description:** As a developer, I want a scheduler that runs tasks concurrently up to a configurable limit.

**Acceptance Criteria:**
- [ ] `TaskScheduler`: takes `task_manager: TaskManager`, `max_concurrent: int = 3`
- [ ] `async schedule(executor: Callable[[Task], Coroutine]) -> list[Task]` — runs next eligible tasks (SUBMITTED status, priority order) up to `max_concurrent` via `asyncio.Semaphore`
- [ ] `pause(task_id: str)` — transitions to PAUSED
- [ ] `resume(task_id: str)` — transitions back to SUBMITTED (re-schedulable)
- [ ] `cancel(task_id: str)` — transitions to CANCELED
- [ ] Tests for concurrent execution limit, pause/resume, cancel

### US-705: Intent-based task routing
**Description:** As a developer, I want LLM-powered intent recognition for routing user input to task actions.

**Acceptance Criteria:**
- [ ] `IntentRecognizer`: takes `model: str`
- [ ] `Intent` dataclass: `action: str` (create_task, pause_task, resume_task, cancel_task, query_task, general), `task_id: str | None`, `confidence: float`, `details: dict[str, Any]`
- [ ] `async recognize(input: str, *, available_tasks: list[Task] | None = None) -> Intent`
- [ ] Uses LLM with structured prompt to classify user intent
- [ ] Tests with MockProvider for various intent patterns

### US-706: Event queue with pub/sub
**Description:** As a developer, I want an in-memory event queue for task lifecycle events so that components can react to task state changes.

**Acceptance Criteria:**
- [ ] `TaskEvent` dataclass: `event_type: str`, `task_id: str`, `data: dict[str, Any]`, `timestamp: float`
- [ ] `TaskEventBus`: `subscribe(event_type: str, handler: Callable)`, `unsubscribe(...)`, `async emit(event: TaskEvent)`
- [ ] Event types: `task.created`, `task.started`, `task.completed`, `task.failed`, `task.paused`, `task.canceled`
- [ ] `TaskManager` emits events on state transitions (via optional event_bus parameter)
- [ ] Tests for subscribe/emit/unsubscribe flow

### US-707: Task controller integration tests
**Description:** As a developer, I want end-to-end tests for the task controller working with agents.

**Acceptance Criteria:**
- [ ] Test: create task → schedule → agent runs → task completes
- [ ] Test: parent task with 3 children → all children complete → parent auto-completes
- [ ] Test: concurrent scheduling respects limit
- [ ] Test: pause/resume/cancel lifecycle
- [ ] All existing orbiter-core tests still pass

---

## Epic 8: DeepAgent Toolkit

**Package:** Extend `orbiter-context` (tools.py) + `orbiter-sandbox`
**Source Reference:** `agent-core/openjiuwen/deepagents/`

### US-801: Research deep agent toolkit architecture
**Description:** As a developer, I want a design document for the built-in agent tools so that they extend existing orbiter-context tools safely.

**Acceptance Criteria:**
- [ ] Read agent-core's `deepagents/tools/` — CodeTool, ShellTool, FileSystemTool, TodoTool
- [ ] Read Orbiter's `orbiter-context/tools.py` (existing planning, knowledge, file tools)
- [ ] Read Orbiter's `orbiter-sandbox/` (Sandbox ABC, LocalSandbox, SandboxBuilder)
- [ ] Produce `docs/design/deep-agent-tools-design.md`
- [ ] Key decision: extend orbiter-context/tools.py for file tools, use orbiter-sandbox for shell/code
- [ ] Document security model (sandboxing, command allowlists)

### US-802: Enhanced file I/O tools
**Description:** As a developer, I want comprehensive file tools (read, write, edit, glob, grep) so that agents can manipulate files autonomously.

**Acceptance Criteria:**
- [ ] `write_file(path: str, content: str) -> str` tool — writes content, returns confirmation
- [ ] `edit_file(path: str, old_text: str, new_text: str) -> str` tool — find-and-replace
- [ ] `glob_files(pattern: str) -> str` tool — glob search, returns matched paths
- [ ] `grep_files(pattern: str, path: str) -> str` tool — regex search in files
- [ ] All tools enforce path safety via `is_relative_to()` (like existing `read_file`)
- [ ] Added to `get_file_tools()` return list (backward compatible — existing `read_file` unchanged)
- [ ] Tests for each tool with temp directories

### US-803: Shell execution tool with allowlist
**Description:** As a developer, I want a shell tool that executes commands from a configurable allowlist so that agents can run safe system commands.

**Acceptance Criteria:**
- [ ] `shell_tool(allowed_commands: list[str] | None = None) -> Tool` factory function
- [ ] Default allowlist: `["ls", "cat", "grep", "find", "echo", "wc", "sort", "head", "tail", "diff"]`
- [ ] Validates command starts with an allowed command before execution
- [ ] Uses `asyncio.create_subprocess_exec()` for execution
- [ ] Configurable `timeout: float = 30.0` (seconds)
- [ ] Returns stdout + stderr, truncated at 10000 chars
- [ ] Tests with allowed and disallowed commands

### US-804: Code execution tool (sandboxed)
**Description:** As a developer, I want a sandboxed code execution tool so that agents can run Python code safely.

**Acceptance Criteria:**
- [ ] `code_tool(sandbox: Sandbox | None = None) -> Tool` factory function
- [ ] If sandbox provided, delegates to `sandbox.run_tool("execute", {"code": code, "language": language})`
- [ ] If no sandbox, uses `exec()` in a restricted namespace with timeout via `asyncio.wait_for()`
- [ ] Restricted namespace: no `__import__`, no `open`, no `os`, no `sys` (configurable)
- [ ] Returns execution output (stdout capture) or error message
- [ ] Tests with safe code and blocked code

### US-805: Enhanced todo management tools
**Description:** As a developer, I want richer todo tools with status tracking so that agents can manage multi-step plans.

**Acceptance Criteria:**
- [ ] Existing `add_todo`, `complete_todo`, `get_todo` kept unchanged
- [ ] New `update_todo(index: int, item: str) -> str` — edit todo text
- [ ] New `remove_todo(index: int) -> str` — delete todo item
- [ ] New `set_todo_status(index: int, status: str) -> str` — PENDING, IN_PROGRESS, COMPLETED
- [ ] Status tracking via `ctx.state["todos"]` entries gaining `"status"` key (backward compatible — defaults to `"done": bool` existing pattern)
- [ ] Tests for new tools, existing todo tests unchanged

### US-806: Task loop architecture tool
**Description:** As a developer, I want a priority event queue tool for managing agent task iterations so that agents can handle ABORT, STEER, and FOLLOWUP events.

**Acceptance Criteria:**
- [ ] `TaskLoopEvent` dataclass: `type: str` (ABORT, STEER, FOLLOWUP), `priority: int`, `content: str`, `metadata: dict`
- [ ] `TaskLoopQueue`: priority queue (ABORT=0 > STEER=1 > FOLLOWUP=2)
- [ ] `push(event: TaskLoopEvent)`, `pop() -> TaskLoopEvent | None`, `peek() -> TaskLoopEvent | None`
- [ ] `steer_agent(content: str) -> str` tool — pushes STEER event to queue
- [ ] `abort_agent(reason: str) -> str` tool — pushes ABORT event
- [ ] Integration: Agent checks queue between tool loop iterations (optional, via rail or hook)
- [ ] Tests for priority ordering and event handling

---

## Epic 9: Context Evolver / Memory Evolution

**Package:** Extend `orbiter-memory`
**Source Reference:** `agent-core/openjiuwen/extensions/context_evolver/`

### US-901: Research context evolver architecture
**Description:** As a developer, I want a design document for memory evolution algorithms so that ACE, ReasoningBank, and ReMe integrate with existing orbiter-memory.

**Acceptance Criteria:**
- [ ] Read agent-core's `context_evolver/` — ACE, ReasoningBank, ReMe implementations
- [ ] Read Orbiter's `long_term.py` (LongTermMemory, ExtractionType, MemoryOrchestrator)
- [ ] Read Orbiter's `base.py` (MemoryItem, MemoryStore protocol)
- [ ] Produce `docs/design/memory-evolution-design.md`
- [ ] Key decision: evolution strategies as new ExtractionType values or separate module
- [ ] Document composable pipeline operator pattern (>> and |)

### US-902: Memory evolution strategy ABC
**Description:** As a developer, I want a base class for memory evolution strategies so that different algorithms share a common interface.

**Acceptance Criteria:**
- [ ] `EvolutionStrategy` ABC: `name: str`, `async def evolve(self, items: list[MemoryItem], *, context: dict[str, Any] | None = None) -> list[MemoryItem]`
- [ ] `EvolutionPipeline`: composes strategies sequentially or in parallel
- [ ] `>>` operator overload for sequential composition: `strategy_a >> strategy_b`
- [ ] `|` operator overload for parallel composition: `strategy_a | strategy_b` (runs both, merges results)
- [ ] Tests for composition and execution

### US-903: ACE (Adaptive Context Engine)
**Description:** As a developer, I want the ACE memory evolution algorithm that uses playbook-based scoring with helpful/harmful/neutral counters.

**Acceptance Criteria:**
- [ ] `ACEStrategy(EvolutionStrategy)`: maintains per-memory counters (`helpful: int`, `harmful: int`, `neutral: int`)
- [ ] `evolve()`: for each memory, scores based on counter ratios, prunes memories with high harmful counts
- [ ] `async reflect(items: list[MemoryItem], feedback: str, model: str | None = None)` — uses LLM to classify each memory as helpful/harmful/neutral
- [ ] `async curate(threshold: float = 0.3) -> list[MemoryItem]` — removes memories below threshold
- [ ] File-based persistence for counters (JSON)
- [ ] Tests with mock reflection and curation

### US-904: ReasoningBank strategy
**Description:** As a developer, I want the ReasoningBank evolution strategy that stores title/description/content memories with query-based recall.

**Acceptance Criteria:**
- [ ] `ReasoningBankStrategy(EvolutionStrategy)`: stores memories as `ReasoningEntry(title, description, content)`
- [ ] `evolve()`: deduplicates by semantic similarity, summarizes redundant entries
- [ ] `async recall(query: str, *, top_k: int = 5) -> list[ReasoningEntry]` — semantic search over entries
- [ ] Uses embeddings for similarity (optional dependency)
- [ ] Falls back to keyword matching if no embeddings provider
- [ ] Tests with mock embeddings

### US-905: ReMe (Relevant Memory) strategy
**Description:** As a developer, I want the ReMe evolution strategy that extracts success/failure patterns with when-to-use metadata.

**Acceptance Criteria:**
- [ ] `ReMeStrategy(EvolutionStrategy)`: stores memories with `when_to_use: str` metadata
- [ ] `evolve()`: extracts success and failure patterns from items, deduplicates
- [ ] `async extract_patterns(items: list[MemoryItem], model: str) -> list[MemoryItem]` — LLM extracts when-to-use metadata
- [ ] Deduplication via content similarity
- [ ] Tests with MockProvider for pattern extraction

---

## Epic 10: Operator Pattern & Self-Optimization

**Package:** Replace `orbiter-train` approach (extend, do not delete existing)
**Source Reference:** `agent-core/openjiuwen/agent_evolving/`

### US-1001: Research operator & self-optimization architecture
**Description:** As a developer, I want a design document for the operator pattern and self-optimization so that agent-core's approach coexists with existing orbiter-train features.

**Acceptance Criteria:**
- [ ] Read agent-core's `agent_evolving/` — Trainer, optimizers, operators, trajectory extraction, checkpointing
- [ ] Read Orbiter's `orbiter-train/` — EvolutionPipeline, SynthesisPipeline, VeRLTrainer, TrajectoryDataset
- [ ] Produce `docs/design/operator-optimization-design.md`
- [ ] Key decision: operator pattern replaces EvolutionStrategy or adds alongside it
- [ ] Document: existing EvolutionPipeline, SynthesisPipeline, VeRLTrainer, TrajectoryDataset MUST remain functional
- [ ] Map agent-core's Trainer lifecycle to Orbiter's Trainer ABC lifecycle

### US-1002: Operator ABC + TunableSpec
**Description:** As a developer, I want an operator abstraction that wraps atomic execution units with tunable parameters.

**Acceptance Criteria:**
- [ ] `Operator` ABC: `name: str`, `async def execute(self, **kwargs) -> Any`, `def get_tunables(self) -> list[TunableSpec]`, `def get_state(self) -> dict`, `def load_state(self, state: dict)`
- [ ] `TunableSpec` frozen dataclass: `name: str`, `kind: str` (prompt, continuous, discrete, text), `current_value: Any`, `constraints: dict[str, Any] | None`
- [ ] Tests for abstract enforcement and tunable spec creation

### US-1003: LLMCallOperator + ToolCallOperator + MemoryCallOperator
**Description:** As a developer, I want concrete operator implementations wrapping LLM calls, tool calls, and memory operations.

**Acceptance Criteria:**
- [ ] `LLMCallOperator(Operator)`: wraps an LLM call, exposes `system_prompt` and `user_prompt` as tunables
- [ ] `ToolCallOperator(Operator)`: wraps a tool, exposes `tool_description` as tunable
- [ ] `MemoryCallOperator(Operator)`: wraps memory ops, exposes `enabled: bool` and `max_retries: int` as tunables
- [ ] Each operator records execution trace for trajectory analysis
- [ ] Tests for each operator type

### US-1004: Trajectory extractor from session traces
**Description:** As a developer, I want to extract complete execution DAGs from agent session traces for optimization analysis.

**Acceptance Criteria:**
- [ ] `TrajectoryExtractor`: takes `TrajectoryDataset` (existing orbiter-train class)
- [ ] `extract(messages: list[dict], *, include_tool_calls: bool = True) -> list[TrajectoryItem]` — builds complete execution trace
- [ ] Extracts: each LLM call → operator invocation → tool call chain → final output
- [ ] Pairs with existing `DefaultStrategy.build_item()` for compatibility
- [ ] Tests with sample message histories

### US-1005: InstructionOptimizer (textual gradients)
**Description:** As a developer, I want an optimizer that improves system prompts via LLM-generated textual gradients from failure analysis.

**Acceptance Criteria:**
- [ ] `InstructionOptimizer`: takes `model: str`, `operators: list[LLMCallOperator]`
- [ ] `async backward(evaluated_cases: list[dict]) -> list[str]` — generates "textual gradients" (LLM analysis of what went wrong)
- [ ] `async step() -> dict[str, str]` — generates improved prompts from gradients
- [ ] Preserves template variables (e.g., `{{query}}`) across optimization
- [ ] Integrates with existing `EvolutionStrategy` ABC — can be used as a strategy
- [ ] Tests with MockProvider

### US-1006: ToolOptimizer (description beam search)
**Description:** As a developer, I want an optimizer that improves tool descriptions via beam search over candidates.

**Acceptance Criteria:**
- [ ] `ToolOptimizer`: takes `model: str`, `operators: list[ToolCallOperator]`
- [ ] `async optimize(eval_cases: list[dict], *, beam_width: int = 3) -> dict[str, str]` — returns optimized tool descriptions
- [ ] Multi-stage: generate candidates → evaluate → select best → refine
- [ ] Returns mapping of `tool_name → optimized_description`
- [ ] Tests with MockProvider

### US-1007: Training loop with checkpoint/resume
**Description:** As a developer, I want a training loop that orchestrates operators + optimizers with checkpoint support.

**Acceptance Criteria:**
- [ ] `OperatorTrainer(Trainer)` — extends existing Trainer ABC from orbiter-train
- [ ] `check_agent()`: validates agent has operators
- [ ] `check_dataset()`: validates train/test data
- [ ] `async train() -> TrainMetrics`: epoch loop — evaluate → backward (textual gradients) → step (generate updates) → apply → validate
- [ ] Checkpoint: after each epoch, saves operator states + optimizer states to `FileCheckpointStore`
- [ ] Resume: `resume_from: str | None` constructor param loads checkpoint
- [ ] Existing `Trainer` ABC interface unchanged — `OperatorTrainer` is a new concrete subclass
- [ ] Existing `VeRLTrainer` and `EvolutionPipeline` tests still pass
- [ ] Tests for full training loop with mock operators and providers

---

## Functional Requirements

### New Packages
- FR-1: `orbiter-guardrail` package with `RiskLevel`, `RiskAssessment`, `GuardrailBackend`, `BaseGuardrail`, `UserInputGuardrail`, `LLMGuardrailBackend`
- FR-2: `orbiter-retrieval` package with `Embeddings`, `VectorStore`, `Retriever`, `Reranker`, `Chunker`, `Parser`, `QueryRewriter`, and all concrete implementations

### Extensions to Existing Packages
- FR-3: `orbiter-core` gains `BranchNode`, `LoopNode`, `WorkflowState`, `WorkflowCheckpoint`, `to_mermaid()`, `Rail`, `RailManager`, `TaskManager`, `TaskScheduler`, `IntentRecognizer`
- FR-4: `orbiter-context` gains `MessageOffloader`, `DialogueCompressor`, `RoundWindowProcessor`, `TiktokenCounter`, `reload_offloaded` tool, enhanced file/shell/code/todo tools
- FR-5: `orbiter-memory` gains `MemoryCategory`, `EncryptedMemoryStore`, `MemUpdateChecker`, `MigrationRegistry`, `SearchManager`, `ACEStrategy`, `ReasoningBankStrategy`, `ReMeStrategy`
- FR-6: `orbiter-train` gains `Operator`, `TunableSpec`, `LLMCallOperator`, `ToolCallOperator`, `InstructionOptimizer`, `ToolOptimizer`, `OperatorTrainer`

### Backward Compatibility
- FR-7: All ~2,900 existing tests pass after every story
- FR-8: No existing public API signature changes — all new features are additive
- FR-9: New optional parameters on existing classes default to `None` or backward-compatible values
- FR-10: Existing `Agent`, `Swarm`, `Tool`, `run()`, `HookManager`, `EventBus` behavior unchanged when new features are not used

---

## Non-Goals (Out of Scope)

- No distributed message queue (Pulsar) — Orbiter already has Redis-based distributed execution
- No Chinese-language prompts or Chinese-first documentation
- No Card/metadata pattern (AgentCard, ToolCard) — Orbiter uses direct configuration
- No Runner singleton pattern — Orbiter uses dependency injection via package hierarchy
- No vLLM inference affinity (GPU session caching) — out of scope for framework
- No Studio UI (agent-core's Studio is a separate product)
- No migration of agent-core's specific LLM providers (DashScope, SiliconFlow) — Orbiter already has 11 providers
- No Milvus vector store initially (pgvector + ChromaDB cover most use cases)
- No replacement of existing orbiter-train classes — only additive
- No frontend changes to orbiter-web (all changes are backend/library packages)

---

## Technical Considerations

### Dependency Management
- New packages declare only `orbiter-core` as required dependency
- Heavy optional deps via extras: `[tiktoken]`, `[pgvector]`, `[chroma]`, `[encryption]`, `[pdf]`
- `orbiter-retrieval` has ZERO required deps beyond `orbiter-core` and `pydantic`

### File Size Discipline
- Max 200 lines per source file (Orbiter convention)
- If a module grows beyond 200 lines, split into `_internal/` submodules
- Each story targets ≤200 lines of code changes

### Testing Strategy
- All tests use MockProvider — no real LLM or database calls
- Test file names prefixed to avoid pytest collection conflicts
- Each story includes tests in its acceptance criteria
- Integration tests run at epic completion

### Performance Considerations
- Vector search uses `asyncio.gather()` for parallel queries
- Task scheduler uses `asyncio.Semaphore` for concurrency control
- Token counting falls back to char estimation when tiktoken unavailable
- Memory encryption adds ~1ms overhead per item (AES-GCM is fast)

### Security Considerations
- Expression evaluator blocks all dangerous AST nodes (audit hardened)
- Shell tool requires explicit allowlist (deny-by-default)
- File tools enforce path traversal protection via `is_relative_to()`
- Encryption uses AES-256-GCM (authenticated encryption)
- Guardrails run before LLM calls (prevent injection before it reaches the model)

---

## Success Metrics

- All 10 epics implemented with no regressions
- All ~2,900+ existing tests pass continuously throughout implementation
- Each new module has ≥90% test coverage
- Zero pyright errors in strict mode
- Zero ruff lint violations
- All new public APIs have Google-style docstrings
- New packages installable independently via `pip install orbiter-guardrail` / `pip install orbiter-retrieval`

---

## Open Questions

1. **Embedding sharing:** Should `orbiter-retrieval` reuse `orbiter-memory`'s `Embeddings` ABC or define its own? (Recommendation: define own to avoid cross-dependency, but make them protocol-compatible)
2. **Task controller location:** Should task management live in `orbiter-core/_internal/task_controller/` or as a new lightweight `orbiter-tasks` package? (Current plan: orbiter-core, revisit if it grows large)
3. **Rail vs Hook naming:** Should the typed guard system be called "Rails" (agent-core terminology) or "Guards" (more intuitive in English)? (Current plan: Rails)
4. **Memory encryption key management:** Should encryption keys be managed per-store or globally? (Current plan: per-store, passed at construction time)
5. **Graph retriever scope:** Is knowledge graph retrieval (triple extraction + beam search) worth the complexity for v1? (Current plan: include, mark as experimental)

---

## Story Dependency Graph

```
Epic 6 (Rails) ──→ Epic 1 (Guardrails)  [guardrails use rail/hook integration]
Epic 4 (Context) ──→ Epic 3 (RAG)       [retrieval uses context tools]
Epic 5 (Memory)  ──→ Epic 9 (Evolver)   [evolver extends memory]
Epic 5 (Memory)  ──→ Epic 10 (Operators) [operators use memory]
Epic 7 (Tasks)   ──→ Epic 8 (DeepAgent)  [deep agent uses task loop]
Epic 2 (Workflow) — independent
```

**Recommended implementation order:**
1. Epic 6 (Rails) — foundational hook extension
2. Epic 1 (Guardrails) — security, depends on rails
3. Epic 4 (Context Enhancement) — core context improvements
4. Epic 5 (Memory Enhancement) — core memory improvements
5. Epic 2 (Workflow Engine) — independent, high value
6. Epic 3 (RAG/Retrieval) — largest epic, benefits from context/memory work
7. Epic 7 (Task Controller) — needed for deep agent
8. Epic 8 (DeepAgent Toolkit) — depends on task controller
9. Epic 9 (Context Evolver) — depends on memory enhancements
10. Epic 10 (Operator/Self-Opt) — depends on memory + train existing
11. Epic 11 (Porting Guide Documentation) — after all implementation epics

---

## Epic 11: Porting Guide Documentation

**Location:** `docs/porting-guide/`
**Purpose:** Create dedicated reference documentation that maps every agent-core (openJiuwen) feature to its Orbiter counterpart, serving as both a historical record and a migration guide.

Each epic gets its own documentation story. These stories produce `docs/porting-guide/<topic>.md` files with a consistent 4-section structure:
1. **Agent-core overview** — what the Chinese framework's implementation looks like, directory structure, key classes
2. **Orbiter equivalent** — the corresponding Orbiter classes, modules, imports
3. **Side-by-side code examples** — the same operation written in both frameworks
4. **Migration table** — two-column table mapping every agent-core public symbol to its Orbiter import path

### US-1101: Agent Rails porting guide (Epic 6)
**Description:** As a developer, I want reference docs mapping agent-core's rail system to Orbiter's typed rails.

**Acceptance Criteria:**
- [ ] Create `docs/porting-guide/rails.md`
- [ ] Section 1: Agent-core's `rail/base.py` — AgentCallbackEvent, AgentRail ABC, @rail decorator, 10 lifecycle events, priority ordering, cross-rail state dict
- [ ] Section 2: Orbiter's Rail ABC, @rail decorator, RailContext, RailEvent models, HookManager integration
- [ ] Section 3: Same guard written in agent-core vs Orbiter (side-by-side)
- [ ] Section 4: Migration table — every agent-core rail symbol → Orbiter import path
- [ ] Code examples typecheck

### US-1102: Guardrails porting guide (Epic 1)
**Description:** As a developer, I want reference docs mapping agent-core's security guardrails to orbiter-guardrail.

**Acceptance Criteria:**
- [ ] Create `docs/porting-guide/guardrails.md`
- [ ] Section 1: Agent-core's `security/guardrail/` — GuardrailBackend ABC, RiskAssessment, RiskLevel, UserInputGuardrail, event monitoring
- [ ] Section 2: Orbiter's GuardrailBackend, RiskAssessment, RiskLevel, UserInputGuardrail, HookManager integration
- [ ] Section 3: Custom guardrail backend in agent-core vs Orbiter (side-by-side)
- [ ] Section 4: Migration table
- [ ] Code examples typecheck

### US-1103: Context Engine porting guide (Epic 4)
**Description:** As a developer, I want reference docs mapping agent-core's context engine to Orbiter's enhanced orbiter-context.

**Acceptance Criteria:**
- [ ] Create `docs/porting-guide/context-engine.md`
- [ ] Section 1: Agent-core's `context/` — ContextManager, OffloadManager, CompressManager, RoundManager, TokenBudgetManager
- [ ] Section 2: Orbiter's ToolResultOffloader, DialogueCompressor, RoundWindow, TokenBudget
- [ ] Section 3: Configuration comparison — agent-core ContextConfig vs Orbiter ContextConfig
- [ ] Section 4: Migration table
- [ ] Code examples typecheck

### US-1104: Memory System porting guide (Epic 5)
**Description:** As a developer, I want reference docs mapping agent-core's 5-type memory system to Orbiter's enhanced orbiter-memory.

**Acceptance Criteria:**
- [ ] Create `docs/porting-guide/memory-system.md`
- [ ] Section 1: Agent-core's `memory/` — MemoryType taxonomy, AES-256 encryption, MemUpdateChecker, migration system, SearchManager
- [ ] Section 2: Orbiter's MemoryCategory, EncryptedMemoryStore, MemoryDeduplicator, MigrationRunner, UnifiedSearchManager
- [ ] Section 3: Side-by-side examples — encrypted memory storage, dedup in both
- [ ] Section 4: Migration table
- [ ] Code examples typecheck

### US-1105: Workflow Engine porting guide (Epic 2)
**Description:** As a developer, I want reference docs mapping agent-core's Pregel engine to Orbiter's extended Swarm.

**Acceptance Criteria:**
- [ ] Create `docs/porting-guide/workflow-engine.md`
- [ ] Section 1: Agent-core's Pregel — PregelNode, channels, LoopNode, BranchNode, SubWorkflowNode, checkpoint, Mermaid viz
- [ ] Section 2: Orbiter's LoopNode, BranchNode, SubSwarmNode, SwarmCheckpoint, MermaidRenderer
- [ ] Section 3: Same branching workflow in agent-core vs Orbiter
- [ ] Section 4: Key differences — channel-based messaging vs Swarm transfer functions
- [ ] Section 5: Migration table

### US-1106: RAG/Retrieval Pipeline porting guide (Epic 3)
**Description:** As a developer, I want reference docs mapping agent-core's RAG pipeline to orbiter-retrieval.

**Acceptance Criteria:**
- [ ] Create `docs/porting-guide/rag-retrieval.md`
- [ ] Section 1: Agent-core's `rag/` — EmbeddingProvider, VectorStore, 5 retriever types, reranker, document pipeline, query rewriter
- [ ] Section 2: Orbiter's equivalents in orbiter-retrieval
- [ ] Section 3: End-to-end hybrid retrieval pipeline in both frameworks
- [ ] Section 4: Migration table

### US-1107: Task Management porting guide (Epic 7)
**Description:** As a developer, I want reference docs mapping agent-core's controller/task system to Orbiter's task management.

**Acceptance Criteria:**
- [ ] Create `docs/porting-guide/task-management.md`
- [ ] Section 1: Agent-core's `controller/` — TaskManager, TaskScheduler, TaskState, IntentRecognizer, EventQueue
- [ ] Section 2: Orbiter's TaskManager, TaskScheduler, TaskState, IntentRouter, TaskEventBus
- [ ] Section 3: Task hierarchy + scheduling in both frameworks
- [ ] Section 4: Migration table

### US-1108: DeepAgent Toolkit porting guide (Epic 8)
**Description:** As a developer, I want reference docs mapping agent-core's DeepAgent tools to Orbiter's built-in toolkit.

**Acceptance Criteria:**
- [ ] Create `docs/porting-guide/deep-agent-toolkit.md`
- [ ] Section 1: Agent-core's `deep_agent/` — file tools, shell execution, code execution, todo system, priority event queue
- [ ] Section 2: Orbiter's FileTools, ShellTool, CodeExecutionTool, TodoManager, PriorityEventQueue
- [ ] Section 3: Tool registration + shell command in both frameworks
- [ ] Section 4: Migration table

### US-1109: Context Evolver porting guide (Epic 9)
**Description:** As a developer, I want reference docs mapping agent-core's evolution algorithms to Orbiter.

**Acceptance Criteria:**
- [ ] Create `docs/porting-guide/context-evolver.md`
- [ ] Section 1: Agent-core's `context_evolver/` — ACE, ReasoningBank, ReMe, composable pipeline operators
- [ ] Section 2: Orbiter's ACEStrategy, ReasoningBankStrategy, ReMeStrategy, pipeline composition
- [ ] Section 3: Multi-strategy evolution pipeline in both frameworks
- [ ] Section 4: Migration table

### US-1110: Operator Pattern porting guide (Epic 10)
**Description:** As a developer, I want reference docs mapping agent-core's agent_evolving system to Orbiter's operator pattern.

**Acceptance Criteria:**
- [ ] Create `docs/porting-guide/operator-optimization.md`
- [ ] Section 1: Agent-core's `agent_evolving/` — Trainer, Operator, TunableSpec, trajectory extraction, 3-dimension evolution, checkpoint/resume
- [ ] Section 2: Orbiter's Operator ABC, operators, optimizers, TrajectoryExtractor, OperatorTrainer
- [ ] Section 3: Optimization loop in both frameworks
- [ ] Section 4: How existing EvolutionPipeline/SynthesisPipeline coexist with operator pattern
- [ ] Section 5: Migration table

### US-1111: Comprehensive porting guide index
**Description:** As a developer, I want a top-level index that provides a complete architectural comparison and links to all per-epic guides.

**Acceptance Criteria:**
- [ ] Create `docs/porting-guide/index.md`
- [ ] Section 1: Introduction — what agent-core (openJiuwen) is, why features were ported
- [ ] Section 2: Architecture comparison — single-package vs 15+ package monorepo, diagram of directory-to-package mapping
- [ ] Section 3: Design philosophy differences — inheritance vs composition, channels vs transfers
- [ ] Section 4: Package mapping table — every agent-core directory → Orbiter package(s)
- [ ] Section 5: Per-epic guide links table with summaries
- [ ] Section 6: Acknowledgments — credit to the openJiuwen project
