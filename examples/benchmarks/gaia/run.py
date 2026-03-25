"""GAIA benchmark runner for Exo agents.

GAIA (General AI Assistants) evaluates multi-step reasoning with tool use.
This runner loads GAIA dataset questions, runs them through an Exo agent,
and scores the agent's answers against ground truth.

Usage:
    export OPENAI_API_KEY=sk-...
    export GAIA_DATASET_PATH=/path/to/GAIA/2023  # directory with validation/*.jsonl
    uv run python examples/benchmarks/gaia/run.py
    uv run python examples/benchmarks/gaia/run.py --split validation --limit 5
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
# Tools available to the GAIA agent
# ---------------------------------------------------------------------------


@tool
async def web_search(query: str) -> str:
    """Search the web for information. Returns a summary of results."""
    return f"[Mock] Search results for '{query}': No live search configured."


@tool
async def read_file(path: str) -> str:
    """Read a file from the GAIA dataset attachments directory."""
    p = Path(path)
    if not p.exists():
        return f"File not found: {path}"
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:8000]
    except Exception as e:
        return f"Error reading {path}: {e}"


@tool
async def python_exec(code: str) -> str:
    """Execute Python code and return stdout. Use for calculations."""
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
# Scoring
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a general-purpose AI assistant being evaluated on the GAIA benchmark.
Answer the question using the tools available to you.
Think step-by-step. When you have the final answer, wrap it in <answer>...</answer> tags.
The answer should be concise — typically a single word, number, or short phrase."""


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


@dataclass
class GaiaQuestion:
    """A single GAIA benchmark question."""

    task_id: str
    question: str
    expected_answer: str
    level: int = 1
    file_name: str = ""


def load_dataset(dataset_path: str, split: str = "validation") -> list[GaiaQuestion]:
    """Load GAIA questions from JSONL files in the dataset directory."""
    questions: list[GaiaQuestion] = []
    split_dir = Path(dataset_path) / split

    if not split_dir.exists():
        # Try loading from flat directory with jsonl files
        for jsonl in Path(dataset_path).glob("*.jsonl"):
            questions.extend(_load_jsonl(jsonl))
        return questions

    for jsonl in split_dir.glob("*.jsonl"):
        questions.extend(_load_jsonl(jsonl))

    return questions


def _load_jsonl(path: Path) -> list[GaiaQuestion]:
    """Parse a JSONL file into GaiaQuestion objects."""
    items: list[GaiaQuestion] = []
    for line in path.read_text().strip().splitlines():
        data = json.loads(line)
        items.append(
            GaiaQuestion(
                task_id=data.get("task_id", ""),
                question=data.get("Question", ""),
                expected_answer=data.get("Final answer", ""),
                level=data.get("Level", 1),
                file_name=data.get("file_name", ""),
            )
        )
    return items


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""

    total: int = 0
    correct: int = 0
    results: list[dict[str, object]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


async def run_benchmark(
    dataset_path: str,
    *,
    split: str = "validation",
    model: str = "openai:gpt-4o-mini",
    limit: int | None = None,
    max_steps: int = 15,
) -> BenchmarkResult:
    """Run the GAIA benchmark and return results."""
    questions = load_dataset(dataset_path, split)
    if limit is not None:
        questions = questions[:limit]

    agent = Agent(
        name="gaia-agent",
        model=model,
        instructions=SYSTEM_PROMPT,
        tools=[web_search, read_file, python_exec],
        max_steps=max_steps,
    )

    bench = BenchmarkResult()
    for i, q in enumerate(questions):
        logger.info("Running question %d/%d: %s", i + 1, len(questions), q.task_id)
        try:
            result = await run(agent, q.question)
            predicted = extract_answer(result.output)
            correct = score_answer(predicted, q.expected_answer)

            bench.total += 1
            if correct:
                bench.correct += 1

            bench.results.append(
                {
                    "task_id": q.task_id,
                    "level": q.level,
                    "predicted": predicted,
                    "expected": q.expected_answer,
                    "correct": correct,
                }
            )
            status = "PASS" if correct else "FAIL"
            logger.info(
                "[%s] %s — predicted=%r expected=%r",
                status,
                q.task_id,
                predicted,
                q.expected_answer,
            )

        except Exception:
            logger.exception("Error on question %s", q.task_id)
            bench.total += 1
            bench.results.append(
                {
                    "task_id": q.task_id,
                    "level": q.level,
                    "predicted": "",
                    "expected": q.expected_answer,
                    "correct": False,
                }
            )

    return bench


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GAIA benchmark with Exo")
    parser.add_argument(
        "--dataset",
        default=os.environ.get("GAIA_DATASET_PATH", ""),
        help="Path to GAIA dataset directory",
    )
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--model", default="openai:gpt-4o-mini", help="Model string")
    parser.add_argument("--limit", type=int, default=None, help="Max questions to run")
    parser.add_argument("--max-steps", type=int, default=15, help="Max agent steps")
    args = parser.parse_args()

    if not args.dataset:
        print("Set GAIA_DATASET_PATH or pass --dataset /path/to/GAIA/2023")
        return

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = asyncio.run(
        run_benchmark(
            args.dataset,
            split=args.split,
            model=args.model,
            limit=args.limit,
            max_steps=args.max_steps,
        )
    )

    print(f"\n{'=' * 50}")
    print("GAIA Benchmark Results")
    print(f"{'=' * 50}")
    print(f"Total:    {result.total}")
    print(f"Correct:  {result.correct}")
    print(f"Accuracy: {result.accuracy:.1%}")
    print(f"{'=' * 50}")

    # Write detailed results
    out_path = Path("gaia_results.json")
    out_path.write_text(json.dumps(result.results, indent=2))
    print(f"Detailed results written to {out_path}")


if __name__ == "__main__":
    main()
