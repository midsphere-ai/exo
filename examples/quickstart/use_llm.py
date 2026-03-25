"""Use LLM providers directly — minimal Exo example.

Shows how to use the ``get_provider`` factory to create
a provider and call ``complete()`` directly, and also how
agents auto-resolve providers from their model string.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python examples/quickstart/use_llm.py
"""

import asyncio

from exo import Agent, run
from exo.models import get_provider  # pyright: ignore[reportMissingImports]
from exo.types import SystemMessage, UserMessage


async def direct_provider_call() -> None:
    """Call an LLM provider directly with message objects."""
    provider = get_provider("openai:gpt-4o-mini")
    response = await provider.complete(
        [
            SystemMessage(content="You are a concise assistant."),
            UserMessage(content="What is 2 + 2?"),
        ],
        temperature=0.0,
    )
    print("Direct:", response.content)
    print("Tokens:", response.usage)


async def agent_auto_resolve() -> None:
    """Run an agent — the provider is auto-resolved from its model string."""
    agent = Agent(name="math-bot", model="openai:gpt-4o-mini")
    result = await run(agent, "Explain the Pythagorean theorem in one sentence.")
    print("Agent:", result.output)


async def main() -> None:
    await direct_provider_call()
    await agent_auto_resolve()


if __name__ == "__main__":
    asyncio.run(main())
