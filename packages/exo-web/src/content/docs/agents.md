---
title: Agents
description: Create and configure AI agents
section: guides
order: 2
---

# Agents

Agents are the core building block in Exo. Each agent combines a language model with a system prompt, tools, and configuration to perform tasks.

## Creating an Agent

Navigate to **Agents > New Agent** and configure:

- **Name** — A descriptive identifier for the agent
- **Model** — The LLM provider and model (e.g., `openai:gpt-4o`)
- **System Prompt** — Instructions that define the agent's behavior
- **Temperature** — Controls randomness (0 = deterministic, 1 = creative)
- **Max Tokens** — Maximum response length

## Tools

Tools extend what an agent can do beyond generating text. Exo supports:

- **Built-in tools** — Code execution, web search, file operations
- **Custom tools** — Define your own tools with Python functions
- **Workflow tools** — Run workflows as tool calls

## Multi-Agent Patterns

### Supervisor

A supervisor agent delegates tasks to specialized sub-agents and synthesizes their results.

### Handoff

Agents can hand off conversations to other agents based on the user's needs — for example, routing from a general assistant to a billing specialist.

## Agent Settings

| Setting | Description |
|---------|-------------|
| `max_turns` | Maximum tool-call loops before stopping |
| `temperature` | Model creativity (0–1) |
| `top_p` | Nucleus sampling threshold |
| `parallel_tool_calls` | Allow multiple tool calls per turn |
| `tool_choice` | Force specific tool usage or let model decide |
