---
title: Workflows
description: Build visual automation workflows
section: guides
order: 3
---

# Workflows

Workflows let you build multi-step automation using a visual node-based canvas. Connect nodes together to create pipelines that process data, call APIs, run agents, and more.

## Canvas Editor

The workflow canvas uses drag-and-drop nodes connected by edges. Open any workflow to see the visual editor.

### Node Types

- **Start** — Entry point for the workflow
- **Agent** — Runs an AI agent with a prompt
- **LLM** — Direct model call without agent tool loop
- **Code** — Execute Python code in a sandbox
- **Condition** — Branch based on a condition
- **Loop** — Repeat a section of nodes
- **HTTP** — Make external API calls
- **Transform** — Reshape data between nodes
- **End** — Terminal node that outputs the final result

### Connecting Nodes

Click and drag from a node's output handle to another node's input handle to create a connection. Data flows along these edges during execution.

## Running Workflows

1. Click the **Run** button in the canvas toolbar
2. Provide any required input variables
3. Watch execution progress in real-time — each node highlights as it runs
4. View results in the output panel

## Debug Mode

Enable **Step-through debugging** to pause at each node:

- Inspect variables and intermediate values
- Resume execution node-by-node
- Modify values mid-execution

Use the keyboard shortcut **F5** to start debug mode, **F10** to step, and **F8** to continue.

## Scheduling

Workflows can run on a schedule using cron expressions:

1. Open the workflow's settings
2. Enable **Schedule**
3. Set a cron expression (e.g., `0 9 * * 1-5` for weekdays at 9 AM)
4. Save — the scheduler picks it up automatically
