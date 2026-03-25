"""Integration marathon test: learning agent across two runs.

US-INT-031: Verifies an agent learns facts in Run 1 via a learn_fact tool that
embeds each fact into ChromaVectorMemoryStore, and a fresh agent in Run 2
retrieves and correctly uses those facts via KnowledgeNeuron injection.

Run 1: Agent with learn_fact tool is given 5 obscure fictional facts and stores
       each in ChromaVectorMemoryStore via sequential tool calls.
Run 2: New Agent with AgentMemory(long_term=Chroma), no tools, verifies the 5
       facts. _inject_long_term_knowledge() searches Chroma with the query and
       injects retrieved facts as a <knowledge> block so the agent can confirm
       each fact. Output is parsed as FactCheckResult.

The 5 facts are entirely fictional (invented names, specs, dates) and cannot
appear in any LLM training data, so correct answers in Run 2 can only come
from the injected knowledge block.
"""

from __future__ import annotations

import shutil
import tempfile

import pytest
from pydantic import BaseModel


class FactCheckResult(BaseModel):
    fact_1_correct: bool
    fact_2_correct: bool
    fact_3_correct: bool
    fact_4_correct: bool
    fact_5_correct: bool


@pytest.mark.integration
@pytest.mark.marathon
@pytest.mark.timeout(180)
async def test_agent_learns_in_run_1_recalls_in_run_2(vertex_model: str) -> None:
    """Marathon: agent stores 5 facts in Run 1; fresh agent recalls 4+ in Run 2.

    Run 1: Agent with learn_fact tool stores 5 obscure fictional facts into
    ChromaVectorMemoryStore, one at a time via sequential tool calls.

    Run 2: New Agent with AgentMemory(long_term=Chroma), no tools. When run(),
    _inject_long_term_knowledge() searches Chroma using the prompt as the query
    and injects all retrieved facts into the system message as a <knowledge>
    block. The agent verifies each of the 5 statements and outputs FactCheckResult.

    Asserts at least 4 of 5 facts are correctly recalled (fact_N_correct == True).
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
    from exo.tool import tool  # pyright: ignore[reportMissingImports]

    # 5 entirely fictional facts — invented names, institutions, specs, and dates.
    # These cannot appear in any LLM training corpus, so correct recall in Run 2
    # proves the facts were retrieved and injected from ChromaVectorMemoryStore.
    facts = [
        (
            "The Zylvanthos engine, invented in 2019 by Dr. Oriana Fetch at the Kemberly "
            "Institute, uses exactly 17 processing cores arranged in a hexagonal lattice."
        ),
        (
            "The Prixium-9 alloy was first synthesized on March 7, 2021 by chemist Tadashi "
            "Wrenford at NovaMetal Labs; its melting point is exactly 4127 degrees Celsius."
        ),
        (
            "The fictional city of Quarvenheim, established in 1847 in the canton of Veluria, "
            "is locally known as the City of Blue Bridges."
        ),
        (
            "Exo version 3.7 introduced the MemNeuron subsystem on September 12, 2025, "
            "reducing context token usage by exactly 34 percent in internal benchmark tests."
        ),
        (
            "The Flaxenburg Protocol, ratified in 2018, requires all Flaxenburg-compliant "
            "devices to include a 256-bit Qondra hash module as standard equipment."
        ),
    ]

    chroma_dir = tempfile.mkdtemp(prefix="exo_learning_chroma_")
    try:
        chroma_store = ChromaVectorMemoryStore(
            SentenceTransformerEmbeddingProvider(),
            collection_name="learning_test",
            path=chroma_dir,
        )

        # -----------------------------------------------------------------
        # RUN 1: Agent with learn_fact tool stores 5 facts into Chroma.
        # learn_fact is async so it can await chroma_store.add().
        # Closure captures chroma_store from the enclosing test function scope.
        # -----------------------------------------------------------------

        @tool
        async def learn_fact(fact: str) -> str:
            """Store a fact in the knowledge base for later retrieval.

            Call this once per fact. Do not combine multiple facts in a single call.

            Args:
                fact: The complete text of the fact to store in the knowledge base.
            """
            await chroma_store.add(AIMemory(content=fact))
            return f"Stored: {fact[:60]}..."

        provider = get_provider(vertex_model)

        run1_agent = Agent(
            name="learner-agent",
            model=vertex_model,
            instructions=(
                "You are a fact-storage agent. Your ONLY task is to call the learn_fact "
                "tool exactly 5 times — ONCE per fact, in the order given, ONE AT A TIME. "
                "Wait for each call to complete before making the next one. "
                "After all 5 facts are stored, reply: 'Stored 5 facts successfully.'"
            ),
            tools=[learn_fact],
            max_steps=10,
        )

        run1_result = await run1_agent.run(
            "Store these 5 facts using the learn_fact tool, ONE AT A TIME in this order:\n\n"
            f"Fact 1: {facts[0]}\n\n"
            f"Fact 2: {facts[1]}\n\n"
            f"Fact 3: {facts[2]}\n\n"
            f"Fact 4: {facts[3]}\n\n"
            f"Fact 5: {facts[4]}\n\n"
            "Call learn_fact with Fact 1 first, wait for confirmation, then Fact 2, "
            "and so on. Do NOT call more than one learn_fact at the same time.",
            provider=provider,
        )

        # Confirm all 5 facts were stored via tool calls.
        learn_calls = [tc for tc in run1_result.tool_calls if tc.name == "learn_fact"]
        assert len(learn_calls) == 5, (
            f"Expected 5 learn_fact calls in Run 1, got {len(learn_calls)}: "
            f"{[tc.name for tc in run1_result.tool_calls]}"
        )

        # -----------------------------------------------------------------
        # RUN 2: Fresh agent with AgentMemory(long_term=Chroma), no tools.
        # _inject_long_term_knowledge() searches Chroma with the recall_prompt
        # as the query (limit=5) and injects all 5 facts as a <knowledge> block.
        # The agent verifies each statement and outputs FactCheckResult JSON.
        # -----------------------------------------------------------------

        run2_agent = Agent(
            name="recall-agent",
            model=vertex_model,
            instructions=(
                "You are a fact-verification agent. Your system context contains a "
                "<knowledge> block with stored facts. Use ONLY the <knowledge> block "
                "to verify whether each of the 5 statements below is supported. "
                "For each statement, output true if the <knowledge> block confirms it, "
                "or false if it does not appear in the knowledge. "
                "Respond ONLY with a valid JSON object — no markdown fences, no extra text:\n"
                '{"fact_1_correct": <true or false>, '
                '"fact_2_correct": <true or false>, '
                '"fact_3_correct": <true or false>, '
                '"fact_4_correct": <true or false>, '
                '"fact_5_correct": <true or false>}'
            ),
            memory=AgentMemory(long_term=chroma_store),
            max_steps=1,
        )

        # The recall prompt explicitly names the key terms from each stored fact
        # so the vector search query is semantically close to all stored content,
        # maximising the chance that all 5 facts are retrieved (limit=5, 5 stored).
        recall_prompt = (
            "Check your <knowledge> block and verify each of these 5 statements:\n\n"
            "Statement 1: The Zylvanthos engine uses 17 processing cores in a "
            "hexagonal lattice.\n"
            "Statement 2: The Prixium-9 alloy has a melting point of 4127 degrees "
            "Celsius.\n"
            "Statement 3: Quarvenheim is known as the City of Blue Bridges.\n"
            "Statement 4: Exo version 3.7 introduced the MemNeuron subsystem.\n"
            "Statement 5: Flaxenburg-compliant devices must include a Qondra hash "
            "module.\n\n"
            "Output ONLY the JSON object with fact_1_correct through fact_5_correct."
        )

        run2_result = await run2_agent.run(recall_prompt, provider=provider)

        assert run2_result.text, "Run 2 agent should produce a non-empty response"

        fact_check = parse_structured_output(run2_result.text, FactCheckResult)

        correct_count = sum(
            [
                fact_check.fact_1_correct,
                fact_check.fact_2_correct,
                fact_check.fact_3_correct,
                fact_check.fact_4_correct,
                fact_check.fact_5_correct,
            ]
        )

        assert correct_count >= 4, (
            f"Expected at least 4/5 facts recalled correctly, got {correct_count}/5. "
            f"FactCheckResult: {fact_check}. "
            f"Run 2 full response: {run2_result.text!r}"
        )

    finally:
        shutil.rmtree(chroma_dir, ignore_errors=True)
