"""VisualWebArena benchmark runner for Exo agents.

VisualWebArena evaluates multimodal web agents on visually-grounded tasks
that require understanding web page screenshots, interpreting visual elements,
and performing multi-step browser actions. It extends WebArena with tasks
that require visual reasoning (e.g., "click the red button", "find the image
matching this description").

Usage:
    export OPENAI_API_KEY=sk-...
    export VISUALWEBARENA_DATASET_PATH=/path/to/visualwebarena  # directory with *.json
    uv run python examples/benchmarks/visualwebarena/run.py
    uv run python examples/benchmarks/visualwebarena/run.py --site shopping --limit 10
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
# Tools available to the VisualWebArena agent
# ---------------------------------------------------------------------------


@tool
async def screenshot() -> str:
    """Take a screenshot of the current browser page. Returns a description."""
    return "[Mock] Screenshot captured: Web page with navigation bar and content area."


@tool
async def click(selector: str) -> str:
    """Click on a web element identified by CSS selector or description."""
    return f"[Mock] Clicked element: {selector}"


@tool
async def type_text(selector: str, text: str) -> str:
    """Type text into a web form element identified by selector."""
    return f"[Mock] Typed '{text}' into {selector}"


@tool
async def navigate(url: str) -> str:
    """Navigate the browser to a URL."""
    return f"[Mock] Navigated to: {url}"


@tool
async def scroll(direction: str = "down", amount: int = 3) -> str:
    """Scroll the page up or down by the specified number of viewport heights."""
    return f"[Mock] Scrolled {direction} by {amount} units."


@tool
async def select_option(selector: str, value: str) -> str:
    """Select an option from a dropdown menu."""
    return f"[Mock] Selected '{value}' from {selector}"


@tool
async def go_back() -> str:
    """Navigate back to the previous page in browser history."""
    return "[Mock] Navigated back to previous page."


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a web agent navigating real websites to complete user tasks.
You can see screenshots of the current page and interact using browser tools:
- screenshot: observe the current page state
- click: click on elements (use CSS selectors or descriptive text)
- type_text: type into form fields
- navigate: go to a specific URL
- scroll: scroll the page up or down
- select_option: choose from dropdown menus
- go_back: return to the previous page

Strategy:
1. Start by taking a screenshot to understand the current page.
2. Identify the relevant visual elements for your task.
3. Interact step by step, taking screenshots to verify each action.
4. When the task is complete, state "TASK_COMPLETE" with a brief summary.
5. Pay attention to visual cues: colors, icons, images, and layout."""


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


@dataclass
class VWATask:
    """A single VisualWebArena benchmark task."""

    task_id: str
    intent: str
    site: str = "shopping"
    require_login: bool = False
    start_url: str = ""
    eval_type: str = "string_match"
    reference_answer: str = ""


def load_dataset(dataset_path: str, *, site: str | None = None) -> list[VWATask]:
    """Load VisualWebArena tasks from JSON files in the dataset directory."""
    tasks: list[VWATask] = []
    base = Path(dataset_path)

    # Try test_*.json pattern first, then any JSON files
    json_files = sorted(base.glob("test_*.json"))
    if not json_files:
        json_files = sorted(base.glob("*.json"))

    for json_file in json_files:
        try:
            data = json.loads(json_file.read_text())
            items = data if isinstance(data, list) else [data]
            for item in items:
                task = _parse_task(item, json_file.stem)
                if site is None or task.site == site:
                    tasks.append(task)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Skipping invalid task file: %s", json_file)

    # Also check for JSONL format
    for jsonl_file in sorted(base.glob("*.jsonl")):
        tasks.extend(_load_jsonl(jsonl_file, site=site))

    return tasks


def _parse_task(data: dict[str, object], fallback_id: str) -> VWATask:
    """Parse a single task dict into a VWATask object."""
    eval_data = data.get("eval")
    eval_type = "string_match"
    reference_answer = ""
    if isinstance(eval_data, dict):
        types = eval_data.get("eval_types", ["string_match"])
        eval_type = str(types[0]) if isinstance(types, list) and types else "string_match"
        refs = eval_data.get("reference_answers")
        if isinstance(refs, dict):
            reference_answer = str(refs.get("exact_match", ""))

    return VWATask(
        task_id=str(data.get("task_id", data.get("id", fallback_id))),
        intent=str(data.get("intent", data.get("task", ""))),
        site=str(data.get("sites", data.get("site", "shopping"))),
        require_login=bool(data.get("require_login", False)),
        start_url=str(data.get("start_url", "")),
        eval_type=eval_type,
        reference_answer=reference_answer,
    )


def _load_jsonl(path: Path, *, site: str | None = None) -> list[VWATask]:
    """Parse a JSONL file into VWATask objects."""
    items: list[VWATask] = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        task = _parse_task(data, "")
        if site is None or task.site == site:
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
    site: str | None = None,
    model: str = "openai:gpt-4o",
    limit: int | None = None,
    max_steps: int = 30,
) -> BenchmarkResult:
    """Run the VisualWebArena benchmark and return results."""
    tasks = load_dataset(dataset_path, site=site)
    if limit is not None:
        tasks = tasks[:limit]

    agent = Agent(
        name="visualwebarena-agent",
        model=model,
        instructions=SYSTEM_PROMPT,
        tools=[screenshot, click, type_text, navigate, scroll, select_option, go_back],
        max_steps=max_steps,
    )

    bench = BenchmarkResult()
    for i, task in enumerate(tasks):
        logger.info(
            "Task %d/%d: %s [site=%s, eval=%s]",
            i + 1,
            len(tasks),
            task.task_id,
            task.site,
            task.eval_type,
        )
        try:
            prompt = task.intent
            if task.start_url:
                prompt = f"Starting URL: {task.start_url}\n\nTask: {task.intent}"

            result = await run(agent, prompt)
            completed = check_completion(result.output)

            bench.total += 1
            if completed:
                bench.completed += 1

            status = "DONE" if completed else "INCOMPLETE"
            logger.info("[%s] %s", status, task.task_id)

            bench.results.append(
                {
                    "task_id": task.task_id,
                    "site": task.site,
                    "eval_type": task.eval_type,
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
                    "site": task.site,
                    "eval_type": task.eval_type,
                    "completed": False,
                    "error": True,
                }
            )

    return bench


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VisualWebArena benchmark with Exo")
    parser.add_argument(
        "--dataset",
        default=os.environ.get("VISUALWEBARENA_DATASET_PATH", ""),
        help="Path to VisualWebArena dataset directory",
    )
    parser.add_argument(
        "--site",
        default=None,
        help="Filter by site (shopping, reddit, classifieds, etc.)",
    )
    parser.add_argument("--model", default="openai:gpt-4o", help="Model string")
    parser.add_argument("--limit", type=int, default=None, help="Max tasks to run")
    parser.add_argument("--max-steps", type=int, default=30, help="Max agent steps per task")
    args = parser.parse_args()

    if not args.dataset:
        print("Set VISUALWEBARENA_DATASET_PATH or pass --dataset /path/to/visualwebarena")
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = asyncio.run(
        run_benchmark(
            args.dataset,
            site=args.site,
            model=args.model,
            limit=args.limit,
            max_steps=args.max_steps,
        )
    )

    print(f"\n{'=' * 50}")
    print("VisualWebArena Benchmark Results")
    print(f"{'=' * 50}")
    print(f"Total:           {result.total}")
    print(f"Completed:       {result.completed}")
    print(f"Errors:          {result.errors}")
    print(f"Completion Rate: {result.completion_rate:.1%}")
    print(f"{'=' * 50}")

    out_path = Path("visualwebarena_results.json")
    out_path.write_text(json.dumps(result.results, indent=2))
    print(f"Detailed results written to {out_path}")


if __name__ == "__main__":
    main()
