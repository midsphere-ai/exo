"""CLI entry point for DeepSearch."""
from __future__ import annotations
import argparse
import asyncio
import logging
import sys

from .config import DeepSearchConfig


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def run_research(config: DeepSearchConfig, question: str) -> None:
    from .engine import DeepSearchEngine

    engine = DeepSearchEngine(config)
    print(f"\n🔍 Researching: {question}\n")

    result = await engine.research(question)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result.md_answer or result.answer)

    if result.references:
        print("\n" + "-" * 60)
        print("REFERENCES")
        print("-" * 60)
        for i, ref in enumerate(result.references):
            print(f"[{i+1}] {ref.title or ref.url}")
            print(f"    {ref.url}")

    print(f"\nURLs visited: {len(result.read_urls)}")
    print(f"Token usage: {result.usage}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="deepsearch",
        description="DeepSearch - Iterative AI Research Agent",
    )
    parser.add_argument("question", nargs="?", help="Research question")
    parser.add_argument("--budget", type=int, default=1_000_000, help="Token budget")
    parser.add_argument("--provider", default="gemini", help="LLM provider (gemini|openai|anthropic)")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--search", default="auto", help="Search provider (auto|jina|brave|duck|serper)")
    parser.add_argument("--server", action="store_true", help="Start API server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=3000, help="Server port")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.server:
        import uvicorn
        uvicorn.run(
            "examples.advanced.deepsearch.server:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
        return

    if not args.question:
        parser.print_help()
        print("\nError: question is required (unless --server is used)")
        sys.exit(1)

    config = DeepSearchConfig(
        llm_provider=args.provider,
        model_name=args.model or ("gemini-2.0-flash" if args.provider == "gemini" else "gpt-4o"),
        search_provider=args.search,
        token_budget=args.budget,
    )

    asyncio.run(run_research(config, args.question))


if __name__ == "__main__":
    main()
