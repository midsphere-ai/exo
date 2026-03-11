"""CLI entry point for the DeepAgent research system.

Usage:
    python -m deepsearch "What are the effects of climate change on coral reefs?"
    python -m deepsearch --model openai:gpt-4o --search brave --stream "query"
    python -m deepsearch --help
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .agents import build_deep_agent
from .config import DeepAgentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="deepsearch",
        description="DeepAgent — Multi-round deep research agent",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Research query to investigate.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help='Lead model (default: openai:gpt-4o). Format: "provider:model_name".',
    )
    parser.add_argument(
        "--researcher-model",
        default=None,
        help="Researcher/tool model (default: openai:gpt-4o-mini).",
    )
    parser.add_argument(
        "--search",
        default=None,
        choices=["duckduckgo", "brave", "serper", "jina"],
        help="Search provider (default: duckduckgo).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximum search rounds for deep research (default: 3).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output events.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for saving reports (default: deepagent_output).",
    )
    return parser.parse_args()


async def run_query(query: str, args: argparse.Namespace) -> None:
    """Run a research query through the DeepAgent swarm."""
    # Build config from env + CLI overrides
    overrides: dict = {}
    if args.model:
        overrides["lead_model"] = args.model
    if args.researcher_model:
        overrides["researcher_model"] = args.researcher_model
        overrides["tool_model"] = args.researcher_model
    if args.search:
        overrides["search_provider"] = args.search
    if args.max_rounds:
        overrides["search_max_rounds"] = args.max_rounds
    if args.output_dir:
        overrides["output_dir"] = args.output_dir

    config = DeepAgentConfig.from_env(**overrides)
    swarm = build_deep_agent(config)

    if args.stream:
        print(f"\n--- DeepAgent Research (streaming) ---")
        print(f"Query: {query}\n")
        async for event in swarm.stream(query):
            if hasattr(event, "text") and event.text:
                print(event.text, end="", flush=True)
            elif hasattr(event, "status"):
                agent_name = getattr(event, "agent_name", "")
                message = getattr(event, "message", "")
                if message:
                    print(f"\n[{agent_name}] {message}", flush=True)
        print()
    else:
        print(f"\n--- DeepAgent Research ---")
        print(f"Query: {query}\n")
        result = await swarm.run(query)
        print(result.output)


def main() -> None:
    args = parse_args()

    if not args.query:
        print("Error: Please provide a research query.", file=sys.stderr)
        print("Usage: python -m deepsearch 'your research question'", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_query(args.query, args))


if __name__ == "__main__":
    main()
