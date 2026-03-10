"""CLI entry point for Perplexica — AI-powered search engine.

Usage:
    python -m examples.advanced.perplexica "What is quantum computing?"
    python -m examples.advanced.perplexica --quality balanced "CRISPR gene editing"
    python -m examples.advanced.perplexica --stream "latest AI news"
    python -m examples.advanced.perplexica --chat
    python -m examples.advanced.perplexica --serve
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .config import PerplexicaConfig
from .conversation import ConversationManager


async def run_search(
    query: str,
    quality: str,
    config: PerplexicaConfig,
    stream: bool = False,
    conversation: ConversationManager | None = None,
) -> None:
    """Execute a single search query."""
    history = conversation.turns if conversation else []

    if stream:
        # For streaming, we run the pipeline and print progressively
        # Since the new pipeline is programmatic (not swarm-based),
        # we print status updates at each stage
        print(f"\nSearching ({quality} mode)...\n")

        from .pipeline import run_search_pipeline
        from .agents.classifier import classify
        from .agents.researcher import research
        from .agents.writer import write_answer
        from .agents.suggestion_generator import generate_suggestions
        from .types import Source

        print("  [classifier] Classifying query...")
        classification = await classify(query, history, config)
        effective_query = classification.standalone_follow_up or query
        print(f"  [classifier] Done. Skip search: {classification.classification.skip_search}")

        search_results = []
        if not classification.classification.skip_search:
            print("  [researcher] Researching...")
            search_results = await research(
                query=effective_query,
                classification=classification,
                chat_history=history,
                mode=quality,
                config=config,
            )
            print(f"  [researcher] Found {len(search_results)} results.")

        print("  [writer] Generating answer...")
        answer = await write_answer(
            query=effective_query,
            search_results=search_results,
            chat_history=history,
            system_instructions=config.system_instructions,
            mode=quality,
            config=config,
        )
        print()
        print(answer)

        print("\n  [suggestions] Generating suggestions...")
        updated_history = history + [(query, answer)]
        suggestions = await generate_suggestions(updated_history, config)

        if suggestions:
            print("\nSuggested follow-ups:")
            for i, s in enumerate(suggestions, 1):
                print(f"  {i}. {s}")

        if conversation:
            conversation.add_turn(query, answer)
    else:
        from . import search_with_details

        print(f"\nSearching ({quality} mode)...\n")
        result = await search_with_details(
            query, mode=quality, chat_history=history, config=config
        )
        print(result.answer)

        if result.sources:
            print("\n---\nSources:")
            for i, s in enumerate(result.sources, 1):
                print(f"  [{i}] {s.title} ({s.url})")

        if result.suggestions:
            print("\nSuggested follow-ups:")
            for i, s in enumerate(result.suggestions, 1):
                print(f"  {i}. {s}")

        if conversation:
            conversation.add_turn(query, result.answer)


async def run_chat(
    quality: str,
    config: PerplexicaConfig,
    stream: bool = False,
) -> None:
    """Interactive multi-turn chat mode."""
    conversation = ConversationManager()

    print("Perplexica Interactive Chat")
    print(f"Quality: {quality} | Type 'quit' to exit, 'clear' to reset\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if query.lower() == "clear":
            conversation.clear()
            print("Conversation cleared.\n")
            continue

        await run_search(query, quality, config, stream=stream, conversation=conversation)
        print()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="perplexica",
        description="Perplexica - AI-Powered Search Engine with Citations",
    )
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument(
        "--quality", "-q",
        type=str,
        default="balanced",
        choices=["speed", "balanced", "quality"],
        help="Research quality mode (default: balanced)",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="web",
        help="Comma-separated sources: web,academic,discussions (default: web)",
    )
    parser.add_argument(
        "--stream", "-s", action="store_true", help="Show progress in real-time"
    )
    parser.add_argument(
        "--chat", "-c", action="store_true", help="Interactive multi-turn chat"
    )
    parser.add_argument(
        "--serve", action="store_true", help="Start FastAPI server"
    )
    parser.add_argument("--model", default=None, help="Override LLM model")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )

    args = parser.parse_args()

    # Build config
    config = PerplexicaConfig()
    if args.model:
        config.model = args.model
    config.sources = [s.strip() for s in args.sources.split(",")]
    config.research_mode = args.quality

    # Server mode
    if args.serve:
        import uvicorn
        uvicorn.run(
            "examples.advanced.perplexica.server:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
        return

    # Chat mode
    if args.chat:
        asyncio.run(run_chat(args.quality, config, stream=args.stream))
        return

    # Single query mode
    if not args.query:
        parser.print_help()
        print("\nError: query is required (unless --chat or --serve is used)")
        sys.exit(1)

    asyncio.run(run_search(args.query, args.quality, config, stream=args.stream))


if __name__ == "__main__":
    main()
