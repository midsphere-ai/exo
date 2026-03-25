# API Reference

Complete API reference for the Exo framework.

## Packages

### [exo-core](core/index.md)

The foundational package providing agents, tools, runners, swarms, hooks, events, configuration, and the type system. Everything you need to build and run LLM-powered agents.

**Public modules:**

| Module | Description |
|--------|-------------|
| [types](core/types.md) | Message types, agent I/O, run results, streaming events, base error |
| [config](core/config.md) | Configuration dataclasses: ModelConfig, AgentConfig, TaskConfig, RunConfig |
| [registry](core/registry.md) | Generic named registry with duplicate detection |
| [events](core/events.md) | Async event bus for decoupled communication |
| [hooks](core/hooks.md) | Lifecycle hook system for agent execution interception |
| [tool](core/tool.md) | Tool ABC, FunctionTool, @tool decorator, schema generation |
| [agent](core/agent.md) | Agent class -- the core autonomous LLM-powered unit |
| [runner](core/runner.md) | `run()`, `run.sync()`, `run.stream()` -- the primary execution API |
| [swarm](core/swarm.md) | Multi-agent orchestration with flow DSL |
| [human](core/human.md) | Human-in-the-loop tool and input handlers |
| [loader](core/loader.md) | YAML agent and swarm loader with variable substitution |
| [skills](core/skills.md) | Multi-source skill registry for loading skills from local paths and GitHub |

### [exo-core Internal](core-internal/index.md)

Internal modules that power the public API. These are implementation details subject to change without notice.

| Module | Description |
|--------|-------------|
| [message_builder](core-internal/message-builder.md) | Build ordered message lists for LLM provider calls |
| [output_parser](core-internal/output-parser.md) | Parse LLM responses into agent-level output types |
| [state](core-internal/state.md) | Run state tracking with execution nodes and lifecycle |
| [call_runner](core-internal/call-runner.md) | Core execution loop with loop detection |
| [handlers](core-internal/handlers.md) | Handler abstractions for composable agent execution |
| [agent_group](core-internal/agent-group.md) | ParallelGroup and SerialGroup execution primitives |
| [nested](core-internal/nested.md) | SwarmNode for nesting swarms within swarms |
| [graph](core-internal/graph.md) | Directed graph, topological sort, flow DSL parser |
| [background](core-internal/background.md) | Background task handler with hot-merge and wake-up-merge |
