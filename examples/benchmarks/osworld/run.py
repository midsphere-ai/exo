"""OSWorld benchmark runner for Exo agents.

OSWorld evaluates AI agents on real-world computer tasks involving desktop OS
interactions (file management, web browsing, office apps, system settings).
This runner loads OSWorld task configs, runs them through an Exo agent,
and scores based on task completion.

Usage:
    export OPENAI_API_KEY=sk-...
    export OSWORLD_DATASET_PATH=/path/to/osworld  # directory with tasks/*.json
    uv run python examples/benchmarks/osworld/run.py
    uv run python examples/benchmarks/osworld/run.py --domain os --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from exo import Agent, run, tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tools available to the OSWorld agent
# ---------------------------------------------------------------------------


@tool
async def screenshot() -> str:
    """Take a screenshot of the current desktop. Returns a description."""
    return "[Mock] Screenshot captured: Desktop with file manager open."


@tool
async def click(x: int, y: int, button: str = "left") -> str:
    """Click at screen coordinates (x, y) with the specified mouse button."""
    return f"[Mock] Clicked ({x}, {y}) with {button} button."


@tool
async def type_text(text: str) -> str:
    """Type the given text using the keyboard."""
    return f"[Mock] Typed: {text}"


@tool
async def hotkey(keys: str) -> str:
    """Press a keyboard shortcut (e.g., 'ctrl+c', 'alt+tab')."""
    return f"[Mock] Pressed hotkey: {keys}"


@tool
async def shell_command(command: str) -> str:
    """Execute a shell command and return stdout/stderr."""
    return f"[Mock] Executed: {command}\n(no live shell configured)"


@tool
async def wait(seconds: float = 1.0) -> str:
    """Wait for a specified number of seconds."""
    return f"[Mock] Waited {seconds}s."


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an AI agent operating a Linux desktop to complete user tasks.
You can interact with the desktop using the available tools:
- screenshot: observe the current state of the screen
- click: click at screen coordinates
- type_text: type text on the keyboard
- hotkey: press keyboard shortcuts (e.g., 'ctrl+s')
- shell_command: run terminal commands
- wait: pause between actions

Strategy:
1. Start by taking a screenshot to observe the current state.
2. Plan your actions step by step.
3. Execute actions one at a time, taking screenshots to verify.
4. When the task is complete, state "TASK_COMPLETE" with a summary."""


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


@dataclass
class OSWorldTask:
    """A single OSWorld benchmark task."""

    task_id: str
    instruction: str
    domain: str = "general"
    difficulty: str = "medium"
    setup: str = ""


def load_dataset(dataset_path: str, *, domain: str | None = None) -> list[OSWorldTask]:
    """Load OSWorld tasks from JSON files in the dataset directory."""
    tasks: list[OSWorldTask] = []
    base = Path(dataset_path)

    # Try tasks/ subdirectory first, then root
    task_dir = base / "tasks" if (base / "tasks").exists() else base

    for json_file in sorted(task_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
            items = data if isinstance(data, list) else [data]
            for item in items:
                task = OSWorldTask(
                    task_id=item.get("id", json_file.stem),
                    instruction=item.get("instruction", item.get("task", "")),
                    domain=item.get("domain", "general"),
                    difficulty=item.get("difficulty", "medium"),
                    setup=item.get("setup", ""),
                )
                if domain is None or task.domain == domain:
                    tasks.append(task)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Skipping invalid task file: %s", json_file)

    # Also check for JSONL format
    for jsonl_file in sorted(task_dir.glob("*.jsonl")):
        tasks.extend(_load_jsonl(jsonl_file, domain=domain))

    return tasks


def _load_jsonl(path: Path, *, domain: str | None = None) -> list[OSWorldTask]:
    """Parse a JSONL file into OSWorldTask objects."""
    items: list[OSWorldTask] = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        task = OSWorldTask(
            task_id=data.get("id", ""),
            instruction=data.get("instruction", data.get("task", "")),
            domain=data.get("domain", "general"),
            difficulty=data.get("difficulty", "medium"),
            setup=data.get("setup", ""),
        )
        if domain is None or task.domain == domain:
            items.append(task)
    return items


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def check_completion(agent_output: str) -> bool:
    """Check if the agent claims task completion."""
    return bool(re.search(r"TASK_COMPLETE", agent_output))


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""

    total: int = 0
    completed: int = 0
    errors: int = 0
    results: list[dict[str, object]] = field(default_factory=list)

    @property
    def completion_rate(self) -> float:
        return self.completed / self.total if self.total > 0 else 0.0


async def run_benchmark(
    dataset_path: str,
    *,
    domain: str | None = None,
    model: str = "openai:gpt-4o",
    limit: int | None = None,
    max_steps: int = 30,
) -> BenchmarkResult:
    """Run the OSWorld benchmark and return results."""
    tasks = load_dataset(dataset_path, domain=domain)
    if limit is not None:
        tasks = tasks[:limit]

    agent = Agent(
        name="osworld-agent",
        model=model,
        instructions=SYSTEM_PROMPT,
        tools=[screenshot, click, type_text, hotkey, shell_command, wait],
        max_steps=max_steps,
    )

    bench = BenchmarkResult()
    for i, task in enumerate(tasks):
        logger.info(
            "Task %d/%d: %s [%s/%s]",
            i + 1,
            len(tasks),
            task.task_id,
            task.domain,
            task.difficulty,
        )
        try:
            result = await run(agent, task.instruction)
            completed = check_completion(result.output)

            bench.total += 1
            if completed:
                bench.completed += 1

            status = "DONE" if completed else "INCOMPLETE"
            logger.info("[%s] %s", status, task.task_id)

            bench.results.append(
                {
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "difficulty": task.difficulty,
                    "completed": completed,
                }
            )

        except Exception:
            logger.exception("Error on task %s", task.task_id)
            bench.total += 1
            bench.errors += 1
            bench.results.append(
                {
                    "task_id": task.task_id,
                    "domain": task.domain,
                    "difficulty": task.difficulty,
                    "completed": False,
                    "error": True,
                }
            )

    return bench


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OSWorld benchmark with Exo")
    parser.add_argument(
        "--dataset",
        default=os.environ.get("OSWORLD_DATASET_PATH", ""),
        help="Path to OSWorld dataset directory",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Filter by domain (os, web, office, etc.)",
    )
    parser.add_argument("--model", default="openai:gpt-4o", help="Model string")
    parser.add_argument("--limit", type=int, default=None, help="Max tasks to run")
    parser.add_argument("--max-steps", type=int, default=30, help="Max agent steps per task")
    args = parser.parse_args()

    if not args.dataset:
        print("Set OSWORLD_DATASET_PATH or pass --dataset /path/to/osworld")
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = asyncio.run(
        run_benchmark(
            args.dataset,
            domain=args.domain,
            model=args.model,
            limit=args.limit,
            max_steps=args.max_steps,
        )
    )

    print(f"\n{'=' * 50}")
    print("OSWorld Benchmark Results")
    print(f"{'=' * 50}")
    print(f"Total:           {result.total}")
    print(f"Completed:       {result.completed}")
    print(f"Errors:          {result.errors}")
    print(f"Completion Rate: {result.completion_rate:.1%}")
    print(f"{'=' * 50}")

    out_path = Path("osworld_results.json")
    out_path.write_text(json.dumps(result.results, indent=2))
    print(f"Detailed results written to {out_path}")


if __name__ == "__main__":
    main()
