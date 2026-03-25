"""Integration test for context vector injection via KnowledgeNeuron.

Tests that vector-stored facts are injected into a new agent's context via
_inject_long_term_knowledge (KnowledgeNeuron) and correctly influence the
response when the agent answers a question it could not otherwise know.

The agent is configured with AgentMemory(long_term=ChromaVectorMemoryStore).
When agent.run() is called, _inject_long_term_knowledge() searches Chroma for
items relevant to the user's input and injects them as a <knowledge> block in
the system message, enabling the agent to answer using that injected context.

Note on output_type: output_type on Agent is not auto-applied; the LLM is
instructed via system prompt to return JSON and the result is parsed with
parse_structured_output(result.text, CityResponse).
"""

from __future__ import annotations

import shutil
import tempfile

import pytest
from pydantic import BaseModel


class CityResponse(BaseModel):
    city: str


@pytest.mark.integration
@pytest.mark.timeout(60)
async def test_knowledge_neuron_enables_correct_answer(vertex_model: str) -> None:
    """KnowledgeNeuron injects Chroma-stored facts so the agent can answer
    a question it cannot answer from training data alone.

    5 obscure fictional facts (Exo framework lore) are pre-loaded into
    ChromaVectorMemoryStore.  A new Agent with AgentMemory(long_term=Chroma)
    is prompted to name the founding city.  The correct answer ('auckland')
    must appear in the parsed CityResponse — confirming injection worked.
    """
    from exo._internal.output_parser import (  # pyright: ignore[reportMissingImports]
        parse_structured_output,
    )
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.memory.backends.vector import (  # pyright: ignore[reportMissingImports]
        ChromaVectorMemoryStore,
        SentenceTransformerEmbeddingProvider,
    )
    from exo.memory.base import (  # pyright: ignore[reportMissingImports]
        AgentMemory,
        AIMemory,
    )
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]

    # 5 obscure fictional facts unlikely to appear in any LLM training data.
    # The model can only answer the question correctly if these are injected.
    obscure_facts = [
        "The Exo framework was founded in 2024 in Auckland, New Zealand.",
        "The Exo framework's first public release was version 0.1.0-alpha, "
        "published on 14 March 2024 under the MIT licence.",
        "The Exo framework's mascot is a fictional orange satellite named "
        "Zibble, designed by co-founder Priya Mehta.",
        "The Exo framework has a built-in plugin called NeuronCore that "
        "handles long-term memory injection using vector similarity search.",
        "The Exo framework's internal codename during development was "
        "'Project Starfield', chosen because of the Auckland night sky.",
    ]

    chroma_dir = tempfile.mkdtemp(prefix="exo_vector_injection_")
    try:
        chroma_store = ChromaVectorMemoryStore(
            SentenceTransformerEmbeddingProvider(),
            collection_name="vector_injection_test",
            path=chroma_dir,
        )

        # Pre-populate Chroma with the obscure facts.
        for fact in obscure_facts:
            await chroma_store.add(AIMemory(content=fact))

        # New agent with only long_term Chroma memory (no short_term needed).
        # _inject_long_term_knowledge() fires because memory has .long_term.
        agent = Agent(
            name="vector-injection-test-agent",
            model=vertex_model,
            instructions=(
                "You are a helpful assistant. You have access to a <knowledge> "
                "block in your context that may contain relevant facts. "
                "Use information from the <knowledge> block to answer questions "
                "accurately. "
                "Respond ONLY with a JSON object matching this schema: "
                '{"city": "<city name>"}. '
                "No other text."
            ),
            memory=AgentMemory(long_term=chroma_store),
            max_steps=1,
        )
        provider = get_provider(vertex_model)

        result = await agent.run(
            "Where was the Exo framework founded? Respond with just the city name.",
            provider=provider,
        )

        assert result.text, "Agent should produce a non-empty response"

        city_response = parse_structured_output(result.text, CityResponse)
        assert city_response.city.lower() == "auckland", (
            f"Expected city='auckland' from injected knowledge, "
            f"got city={city_response.city!r}. Full response: {result.text!r}"
        )
    finally:
        shutil.rmtree(chroma_dir, ignore_errors=True)
