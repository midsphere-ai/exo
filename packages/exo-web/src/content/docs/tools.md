---
title: Tools
description: Extend agents with custom tools
section: guides
order: 7
---

# Tools

Tools give agents the ability to take actions beyond text generation. Exo ships with built-in tools and lets you create custom ones.

## Built-in Tools

- **Code Execution** — Run Python code in a sandboxed environment
- **Web Search** — Search the web and return results
- **File Operations** — Read, write, and manage files
- **HTTP Request** — Make API calls to external services

## Custom Tools

Create tools with Python functions:

1. Navigate to **Tools > New Tool**
2. Write your tool function with type-annotated parameters
3. Exo automatically generates the JSON schema for the LLM
4. Test the tool and assign it to agents

## Tool Schemas

Tools use OpenAI-compatible function calling format:

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "City name"
        }
      },
      "required": ["location"]
    }
  }
}
```

## Workflow Tool Nodes

Nodes in a workflow can be exposed as tools for agents. Enable **Tool Mode** on a node to make it callable by agents during conversations.
