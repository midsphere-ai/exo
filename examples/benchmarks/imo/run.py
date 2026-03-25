"""IMO benchmark runner for Exo agents.

IMO (International Mathematical Olympiad) evaluates advanced mathematical
reasoning and proof generation. This runner loads IMO problems from a JSONL
dataset, runs them through an Exo agent with a guard/review loop, and
scores the agent's responses.

Usage:
    export OPENAI_API_KEY=sk-...
    export IMO_DATASET_PATH=/path/to/imo_dataset  # directory with metadata.jsonl
    uv run python examples/benchmarks/imo/run.py
    uv run python examples/benchmarks/imo/run.py --limit 3 --max-rounds 5
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
# Tools available to the IMO agent
# ---------------------------------------------------------------------------


@tool
async def python_exec(code: str) -> str:
    """Execute Python code and return stdout. Use for calculations and verification."""
    import contextlib
    import io

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": __builtins__})
    except Exception as e:
        return f"Error: {e}"
    return buf.getvalue() or "(no output)"


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SOLVER_PROMPT = """\
You are an expert mathematician competing in the International Mathematical Olympiad.
Solve the problem rigorously, step by step.

Requirements:
- Use precise mathematical reasoning with clear justifications.
- Present a structured solution: Summary, then Detailed Solution.
- When you have the final answer, wrap it in <answer>...</answer> tags.
- For proof problems, state "QED" or "proved" inside the answer tags.
- You may use the python_exec tool for numerical verification."""

REVIEWER_PROMPT = """\
You are a rigorous IMO grader reviewing a proposed solution.
Identify any logical gaps, incorrect steps, or unjustified claims.
If the solution is correct and complete, say "SOLUTION_ACCEPTED".
Otherwise, list the specific issues that need to be addressed."""


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


@dataclass
class IMOProblem:
    """A single IMO benchmark problem."""

    task_id: str
    question: str


def load_dataset(dataset_path: str) -> list[IMOProblem]:
    """Load IMO problems from a metadata.jsonl file."""
    problems: list[IMOProblem] = []
    meta_file = Path(dataset_path) / "metadata.jsonl"

    if not meta_file.exists():
        # Try any jsonl files in directory
        for jsonl in Path(dataset_path).glob("*.jsonl"):
            problems.extend(_load_jsonl(jsonl))
        return problems

    problems.extend(_load_jsonl(meta_file))
    return problems


def _load_jsonl(path: Path) -> list[IMOProblem]:
    """Parse a JSONL file into IMOProblem objects."""
    items: list[IMOProblem] = []
    for line in path.read_text().strip().splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        data = json.loads(line)
        task_id = data.get("task_id", "")
        if task_id == "0-0-0-0-0":
            continue
        items.append(IMOProblem(task_id=task_id, question=data.get("Question", "")))
    return items


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def extract_answer(text: str) -> str:
    """Extract the answer from <answer>...</answer> tags, or return full text."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""

    total: int = 0
    completed: int = 0
    accepted: int = 0
    results: list[dict[str, object]] = field(default_factory=list)

    @property
    def completion_rate(self) -> float:
        return self.completed / self.total if self.total > 0 else 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.total if self.total > 0 else 0.0


async def run_benchmark(
    dataset_path: str,
    *,
    model: str = "openai:gpt-4o",
    limit: int | None = None,
    max_steps: int = 20,
    max_rounds: int = 3,
) -> BenchmarkResult:
    """Run the IMO benchmark with a solver + reviewer loop."""
    problems = load_dataset(dataset_path)
    if limit is not None:
        problems = problems[:limit]

    solver = Agent(
        name="imo-solver",
        model=model,
        instructions=SOLVER_PROMPT,
        tools=[python_exec],
        max_steps=max_steps,
    )
    reviewer = Agent(
        name="imo-reviewer",
        model=model,
        instructions=REVIEWER_PROMPT,
        max_steps=5,
    )

    bench = BenchmarkResult()
    for i, prob in enumerate(problems):
        logger.info("Problem %d/%d: %s", i + 1, len(problems), prob.task_id)
        try:
            solution_text = ""
            review_text = ""
            accepted = False

            for round_num in range(max_rounds):
                if round_num == 0:
                    prompt = prob.question
                else:
                    prompt = (
                        f"Original problem:\n{prob.question}\n\n"
                        f"Your previous solution:\n{solution_text}\n\n"
                        f"Reviewer feedback:\n{review_text}\n\n"
                        "Please revise your solution addressing all issues."
                    )

                result = await run(solver, prompt)
                solution_text = result.output

                # Review the solution
                review_result = await run(
                    reviewer,
                    f"Problem:\n{prob.question}\n\nProposed solution:\n{solution_text}",
                )
                review_text = review_result.output

                if "SOLUTION_ACCEPTED" in review_text:
                    accepted = True
                    break

            bench.total += 1
            bench.completed += 1
            if accepted:
                bench.accepted += 1

            answer = extract_answer(solution_text)
            status = "ACCEPTED" if accepted else "REVIEWED"
            logger.info(
                "[%s] %s (rounds=%d) answer=%s",
                status,
                prob.task_id,
                round_num + 1,
                answer[:80],
            )

            bench.results.append(
                {
                    "task_id": prob.task_id,
                    "answer": answer,
                    "accepted": accepted,
                    "rounds": round_num + 1,
                }
            )

        except Exception:
            logger.exception("Error on problem %s", prob.task_id)
            bench.total += 1
            bench.results.append(
                {
                    "task_id": prob.task_id,
                    "answer": "",
                    "accepted": False,
                    "rounds": 0,
                }
            )

    return bench


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IMO benchmark with Exo")
    parser.add_argument(
        "--dataset",
        default=os.environ.get("IMO_DATASET_PATH", ""),
        help="Path to IMO dataset directory",
    )
    parser.add_argument("--model", default="openai:gpt-4o", help="Model string")
    parser.add_argument("--limit", type=int, default=None, help="Max problems to run")
    parser.add_argument("--max-steps", type=int, default=20, help="Max agent steps")
    parser.add_argument("--max-rounds", type=int, default=3, help="Max review rounds")
    args = parser.parse_args()

    if not args.dataset:
        print("Set IMO_DATASET_PATH or pass --dataset /path/to/imo_dataset")
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = asyncio.run(
        run_benchmark(
            args.dataset,
            model=args.model,
            limit=args.limit,
            max_steps=args.max_steps,
            max_rounds=args.max_rounds,
        )
    )

    print(f"\n{'=' * 50}")
    print("IMO Benchmark Results")
    print(f"{'=' * 50}")
    print(f"Total:           {result.total}")
    print(f"Completed:       {result.completed}")
    print(f"Accepted:        {result.accepted}")
    print(f"Completion Rate: {result.completion_rate:.1%}")
    print(f"Acceptance Rate: {result.acceptance_rate:.1%}")
    print(f"{'=' * 50}")

    out_path = Path("imo_results.json")
    out_path.write_text(json.dumps(result.results, indent=2))
    print(f"Detailed results written to {out_path}")


if __name__ == "__main__":
    main()
