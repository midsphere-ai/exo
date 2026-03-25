"""BFCL benchmark runner for Exo agents.

Berkeley Function Calling Leaderboard (BFCL) evaluates how well LLM agents
select and invoke the correct function with the right arguments from a set
of available tool definitions. Each test case provides a user query and a
set of tool schemas; the agent must produce the correct function call.

Usage:
    export OPENAI_API_KEY=sk-...
    export BFCL_DATASET_PATH=/path/to/bfcl  # directory with *.jsonl files
    uv run python examples/benchmarks/bfcl/run.py
    uv run python examples/benchmarks/bfcl/run.py --category simple --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from exo import Agent, run, tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock execution tool — in real BFCL the agent's tool calls are compared
# directly against the expected function call; this tool is a fallback
# for cases where the agent needs to "execute" something.
# ---------------------------------------------------------------------------


@tool
async def execute_function(name: str, arguments: str) -> str:
    """Execute a function call. In benchmark mode this records the call."""
    return f"[Mock] Called {name}({arguments})"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

VALID_CATEGORIES = frozenset(
    {
        "simple",
        "multiple",
        "parallel",
        "parallel_multiple",
        "irrelevance",
        "java",
        "javascript",
        "rest",
    }
)


@dataclass
class BFCLTestCase:
    """A single BFCL benchmark test case."""

    test_id: str
    query: str
    tools: list[dict[str, Any]]
    expected_calls: list[dict[str, Any]]
    category: str = "simple"


def load_dataset(dataset_path: str, *, category: str | None = None) -> list[BFCLTestCase]:
    """Load BFCL test cases from JSONL files in the dataset directory."""
    cases: list[BFCLTestCase] = []
    base = Path(dataset_path)

    for jsonl_file in sorted(base.glob("*.jsonl")):
        cases.extend(_load_jsonl(jsonl_file, category=category))

    # Also check data/ subdirectory
    data_dir = base / "data"
    if data_dir.exists():
        for jsonl_file in sorted(data_dir.glob("*.jsonl")):
            cases.extend(_load_jsonl(jsonl_file, category=category))

    return cases


def _load_jsonl(path: Path, *, category: str | None = None) -> list[BFCLTestCase]:
    """Parse a JSONL file into BFCLTestCase objects."""
    items: list[BFCLTestCase] = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        case = _parse_case(data, fallback_category=path.stem)
        if category is None or case.category == category:
            items.append(case)
    return items


def _parse_case(data: dict[str, Any], fallback_category: str = "simple") -> BFCLTestCase:
    """Parse a dict into a BFCLTestCase."""
    # BFCL format: each entry has "id", "question"/"query", "function",
    # and "ground_truth"/"expected_output"
    tools_raw = data.get("function", data.get("tools", []))
    tools = tools_raw if isinstance(tools_raw, list) else [tools_raw]

    expected_raw = data.get("ground_truth", data.get("expected_output", []))
    expected = expected_raw if isinstance(expected_raw, list) else [expected_raw]

    return BFCLTestCase(
        test_id=str(data.get("id", data.get("test_id", ""))),
        query=str(data.get("question", data.get("query", ""))),
        tools=tools,
        expected_calls=expected,
        category=str(data.get("category", fallback_category)),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def normalize_call(call: dict[str, Any]) -> tuple[str, str]:
    """Normalize a function call to (name, sorted_args_json) for comparison."""
    name = str(call.get("name", call.get("function", "")))
    args = call.get("arguments", call.get("parameters", {}))
    if isinstance(args, str):
        with contextlib.suppress(json.JSONDecodeError):
            args = json.loads(args)
    sorted_args = json.dumps(args, sort_keys=True, default=str)
    return (name, sorted_args)


def score_case(
    agent_calls: list[dict[str, Any]],
    expected_calls: list[dict[str, Any]],
) -> bool:
    """Check if agent's function calls match the expected calls (order-insensitive)."""
    if len(agent_calls) != len(expected_calls):
        return False
    agent_normalized = sorted(normalize_call(c) for c in agent_calls)
    expected_normalized = sorted(normalize_call(c) for c in expected_calls)
    return agent_normalized == expected_normalized


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are being evaluated on function calling accuracy. You will receive a user
query and a set of available functions. Select the correct function(s) and
provide the right arguments. Respond ONLY with the function call(s) needed."""


@dataclass
class BenchmarkResult:
    """Aggregated BFCL benchmark results."""

    total: int = 0
    correct: int = 0
    errors: int = 0
    results: list[dict[str, object]] = field(default_factory=list)
    per_category: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


def _update_category(stats: dict[str, dict[str, int]], category: str, *, correct: bool) -> None:
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
) -> BenchmarkResult:
    """Run the BFCL benchmark and return results."""
    cases = load_dataset(dataset_path, category=category)
    if limit is not None:
        cases = cases[:limit]

    bench = BenchmarkResult()
    for i, case in enumerate(cases):
        logger.info("Case %d/%d: %s [%s]", i + 1, len(cases), case.test_id, case.category)

        # Build a per-case agent with the test's tool schemas
        agent = Agent(
            name="bfcl-agent",
            model=model,
            instructions=SYSTEM_PROMPT,
            tools=[execute_function],
            max_steps=3,
        )

        try:
            result = await run(agent, case.query)
            # Extract tool calls from the agent's output
            agent_calls = _extract_tool_calls(result)
            correct = score_case(agent_calls, case.expected_calls)

            bench.total += 1
            if correct:
                bench.correct += 1

            _update_category(bench.per_category, case.category, correct=correct)

            status = "PASS" if correct else "FAIL"
            logger.info("[%s] %s", status, case.test_id)

            bench.results.append(
                {
                    "test_id": case.test_id,
                    "category": case.category,
                    "correct": correct,
                    "agent_calls": [str(c) for c in agent_calls],
                    "expected_calls": [str(c) for c in case.expected_calls],
                }
            )

        except Exception:
            logger.exception("Error on case %s", case.test_id)
            bench.total += 1
            bench.errors += 1
            _update_category(bench.per_category, case.category, correct=False)
            bench.results.append(
                {
                    "test_id": case.test_id,
                    "category": case.category,
                    "correct": False,
                    "error": True,
                }
            )

    return bench


def _extract_tool_calls(result: Any) -> list[dict[str, Any]]:
    """Extract function calls from the agent's RunResult."""
    calls: list[dict[str, Any]] = []
    # Check result messages for tool calls
    messages = getattr(result, "messages", [])
    for msg in messages:
        if isinstance(msg, dict):
            for tc in msg.get("tool_calls", []):
                if isinstance(tc, dict):
                    calls.append(
                        {
                            "name": tc.get("name", ""),
                            "arguments": tc.get("arguments", {}),
                        }
                    )
    return calls


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BFCL benchmark with Exo")
    parser.add_argument(
        "--dataset",
        default=os.environ.get("BFCL_DATASET_PATH", ""),
        help="Path to BFCL dataset directory",
    )
    parser.add_argument(
        "--category",
        default=None,
        choices=sorted(VALID_CATEGORIES),
        help="Filter by test category",
    )
    parser.add_argument("--model", default="openai:gpt-4o-mini", help="Model string")
    parser.add_argument("--limit", type=int, default=None, help="Max cases to run")
    args = parser.parse_args()

    if not args.dataset:
        print("Set BFCL_DATASET_PATH or pass --dataset /path/to/bfcl")
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = asyncio.run(
        run_benchmark(
            args.dataset,
            category=args.category,
            model=args.model,
            limit=args.limit,
        )
    )

    print(f"\n{'=' * 50}")
    print("BFCL Benchmark Results")
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
            print(f"  {cat:20s}: {stats['correct']}/{stats['total']} ({cat_acc:.1%})")

    print(f"{'=' * 50}")

    out_path = Path("bfcl_results.json")
    out_path.write_text(json.dumps(result.results, indent=2))
    print(f"Detailed results written to {out_path}")


if __name__ == "__main__":
    main()
