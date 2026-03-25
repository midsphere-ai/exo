"""XBench benchmark runner for Exo agents.

XBench is a cross-domain benchmark evaluating AI agents across diverse task
categories: coding, math reasoning, web navigation, knowledge retrieval, and
general instruction following. Each task has a category, difficulty level,
and expected answer for automated scoring.

Usage:
    export OPENAI_API_KEY=sk-...
    export XBENCH_DATASET_PATH=/path/to/xbench  # directory with tasks.jsonl
    uv run python examples/benchmarks/xbench/run.py
    uv run python examples/benchmarks/xbench/run.py --category coding --limit 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import string
from dataclasses import dataclass, field
from pathlib import Path

from exo import Agent, run, tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tools available to the XBench agent
# ---------------------------------------------------------------------------


@tool
async def web_search(query: str) -> str:
    """Search the web for information. Returns a summary of search results."""
    return f"[Mock] Search results for '{query}': No live search configured."


@tool
async def python_exec(code: str) -> str:
    """Execute Python code and return stdout. Use for calculations and coding tasks."""
    import contextlib
    import io

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": __builtins__})
    except Exception as e:
        return f"Error: {e}"
    return buf.getvalue() or "(no output)"


@tool
async def read_file(path: str) -> str:
    """Read a file and return its contents (up to 8000 chars)."""
    p = Path(path)
    if not p.exists():
        return f"File not found: {path}"
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:8000]
    except Exception as e:
        return f"Error reading {path}: {e}"


@tool
async def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if needed."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


# ---------------------------------------------------------------------------
# Prompts and scoring
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an AI agent being evaluated on the XBench cross-domain benchmark.
You will receive tasks from various domains: coding, math, web, knowledge, and
general instruction following.

Strategy:
1. Carefully read the task and identify the domain.
2. Use the appropriate tools for the task type.
3. For coding: write and execute code to verify your solution.
4. For math: show your reasoning step by step.
5. For web/knowledge: search for information as needed.
6. When you have the final answer, wrap it in <answer>...</answer> tags.
The answer should be concise and directly address the question."""


def normalize(text: str) -> str:
    """Normalize text for comparison: lowercase, strip punctuation/whitespace."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def score_answer(predicted: str, expected: str) -> bool:
    """Check if the predicted answer matches the expected answer."""
    return normalize(predicted) == normalize(expected)


def extract_answer(text: str) -> str:
    """Extract the answer from <answer>...</answer> tags, or return full text."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

VALID_CATEGORIES = frozenset({"coding", "math", "web", "knowledge", "general"})


@dataclass
class XBenchTask:
    """A single XBench benchmark task."""

    task_id: str
    question: str
    expected_answer: str
    category: str = "general"
    difficulty: str = "medium"
    metadata: dict[str, object] = field(default_factory=dict)


def load_dataset(dataset_path: str, *, category: str | None = None) -> list[XBenchTask]:
    """Load XBench tasks from JSON or JSONL files in the dataset directory."""
    tasks: list[XBenchTask] = []
    base = Path(dataset_path)

    # Try tasks/ subdirectory first, then root
    task_dir = base / "tasks" if (base / "tasks").exists() else base

    for jsonl_file in sorted(task_dir.glob("*.jsonl")):
        tasks.extend(_load_jsonl(jsonl_file, category=category))

    for json_file in sorted(task_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
            items = data if isinstance(data, list) else [data]
            for item in items:
                task = _parse_task(item, fallback_id=json_file.stem)
                if category is None or task.category == category:
                    tasks.append(task)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Skipping invalid task file: %s", json_file)

    return tasks


def _load_jsonl(path: Path, *, category: str | None = None) -> list[XBenchTask]:
    """Parse a JSONL file into XBenchTask objects."""
    items: list[XBenchTask] = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        task = _parse_task(data)
        if category is None or task.category == category:
            items.append(task)
    return items


def _parse_task(data: dict[str, object], fallback_id: str = "") -> XBenchTask:
    """Parse a dict into an XBenchTask."""
    return XBenchTask(
        task_id=str(data.get("task_id", data.get("id", fallback_id))),
        question=str(data.get("question", data.get("input", ""))),
        expected_answer=str(data.get("expected_answer", data.get("answer", ""))),
        category=str(data.get("category", "general")),
        difficulty=str(data.get("difficulty", "medium")),
    )


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""

    total: int = 0
    correct: int = 0
    errors: int = 0
    results: list[dict[str, object]] = field(default_factory=list)
    per_category: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


def _update_category_stats(
    stats: dict[str, dict[str, int]], category: str, *, correct: bool
) -> None:
    """Track per-category accuracy."""
    if category not in stats:
        stats[category] = {"total": 0, "correct": 0}
    stats[category]["total"] += 1
    if correct:
        stats[category]["correct"] += 1


async def run_benchmark(
    dataset_path: str,
    *,
    category: str | None = None,
    model: str = "openai:gpt-4o-mini",
    limit: int | None = None,
    max_steps: int = 20,
) -> BenchmarkResult:
    """Run the XBench benchmark and return results."""
    tasks = load_dataset(dataset_path, category=category)
    if limit is not None:
        tasks = tasks[:limit]

    agent = Agent(
        name="xbench-agent",
        model=model,
        instructions=SYSTEM_PROMPT,
        tools=[web_search, python_exec, read_file, write_file],
        max_steps=max_steps,
    )

    bench = BenchmarkResult()
    for i, task in enumerate(tasks):
        logger.info(
            "Task %d/%d: %s [%s/%s]",
            i + 1,
            len(tasks),
            task.task_id,
            task.category,
            task.difficulty,
        )
        try:
            result = await run(agent, task.question)
            predicted = extract_answer(result.output)
            correct = score_answer(predicted, task.expected_answer)

            bench.total += 1
            if correct:
                bench.correct += 1

            _update_category_stats(bench.per_category, task.category, correct=correct)

            status = "PASS" if correct else "FAIL"
            logger.info(
                "[%s] %s — predicted=%r expected=%r",
                status,
                task.task_id,
                predicted,
                task.expected_answer,
            )

            bench.results.append(
                {
                    "task_id": task.task_id,
                    "category": task.category,
                    "difficulty": task.difficulty,
                    "predicted": predicted,
                    "expected": task.expected_answer,
                    "correct": correct,
                }
            )

        except Exception:
            logger.exception("Error on task %s", task.task_id)
            bench.total += 1
            bench.errors += 1
            _update_category_stats(bench.per_category, task.category, correct=False)
            bench.results.append(
                {
                    "task_id": task.task_id,
                    "category": task.category,
                    "difficulty": task.difficulty,
                    "predicted": "",
                    "expected": task.expected_answer,
                    "correct": False,
                    "error": True,
                }
            )

    return bench


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XBench benchmark with Exo")
    parser.add_argument(
        "--dataset",
        default=os.environ.get("XBENCH_DATASET_PATH", ""),
        help="Path to XBench dataset directory",
    )
    parser.add_argument(
        "--category",
        default=None,
        choices=sorted(VALID_CATEGORIES),
        help="Filter by task category",
    )
    parser.add_argument("--model", default="openai:gpt-4o-mini", help="Model string")
    parser.add_argument("--limit", type=int, default=None, help="Max tasks to run")
    parser.add_argument("--max-steps", type=int, default=20, help="Max agent steps per task")
    args = parser.parse_args()

    if not args.dataset:
        print("Set XBENCH_DATASET_PATH or pass --dataset /path/to/xbench")
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = asyncio.run(
        run_benchmark(
            args.dataset,
            category=args.category,
            model=args.model,
            limit=args.limit,
            max_steps=args.max_steps,
        )
    )

    print(f"\n{'=' * 50}")
    print("XBench Benchmark Results")
    print(f"{'=' * 50}")
    print(f"Total:    {result.total}")
    print(f"Correct:  {result.correct}")
    print(f"Errors:   {result.errors}")
    print(f"Accuracy: {result.accuracy:.1%}")

    if result.per_category:
        print(f"\n{'─' * 50}")
        print("Per-Category Breakdown:")
        for cat in sorted(result.per_category):
            stats = result.per_category[cat]
            cat_acc = stats["correct"] / stats["total"] if stats["total"] else 0.0
            print(f"  {cat:12s}: {stats['correct']}/{stats['total']} ({cat_acc:.1%})")

    print(f"{'=' * 50}")

    out_path = Path("xbench_results.json")
    out_path.write_text(json.dumps(result.results, indent=2))
    print(f"Detailed results written to {out_path}")


if __name__ == "__main__":
    main()
