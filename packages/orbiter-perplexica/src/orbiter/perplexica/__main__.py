"""CLI entry point for Perplexica — AI-powered search engine.

Usage:
    python -m orbiter.perplexica "What is quantum computing?"
    python -m orbiter.perplexica --quality balanced "CRISPR gene editing"
    python -m orbiter.perplexica --stream "latest AI news"
    python -m orbiter.perplexica --chat
    python -m orbiter.perplexica --serve
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from orbiter.observability.logging import (  # pyright: ignore[reportMissingImports]
    configure_logging,
    get_logger,
)

from .config import PerplexicaConfig
from .conversation import ConversationManager

_log = get_logger(__name__)


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
        from .pipeline import stream_search_pipeline
        from .types import PipelineEvent, PerplexicaResponse

        print(f"\nSearching ({quality} mode)...\n")

        result = None
        async for event in stream_search_pipeline(
            query=query,
            chat_history=history,
            mode=quality,
            config=config,
        ):
            if isinstance(event, PipelineEvent):
                if event.status == "started":
                    print(f"  [{event.stage}] Starting...", flush=True)
                else:
                    msg = f" ({event.message})" if event.message else ""
                    print(f"  [{event.stage}] Done{msg}", flush=True)
                    if event.stage == "writer":
                        print()  # newline after writer completes
            elif isinstance(event, PerplexicaResponse):
                result = event
            else:
                # StreamEvent — print text tokens in real-time
                from orbiter.types import TextEvent, ToolCallEvent
                if isinstance(event, TextEvent) and event.agent_name == "writer":
                    print(event.text, end="", flush=True)
                elif isinstance(event, ToolCallEvent) and event.agent_name == "researcher":
                    print(f"    -> {event.tool_name}()", flush=True)

        if result:
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
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    parser.add_argument("--model", default=None, help="Override LLM model")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )

    args = parser.parse_args()

    configure_logging(level="DEBUG" if args.verbose else "WARNING")

    # Build config
    config = PerplexicaConfig()
    if args.model:
        config.model = args.model
    config.sources = [s.strip() for s in args.sources.split(",")]
    config.research_mode = args.quality
    _log.info("config model=%s fast=%s mode=%s", config.model, config.fast_model, args.quality)

    # Server mode
    if args.serve:
        import uvicorn
        uvicorn.run(
            "orbiter.perplexica.server:app",
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
