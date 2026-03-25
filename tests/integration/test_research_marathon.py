"""Integration marathon test: research agent multi-step.

US-INT-028: Verifies a research agent makes 5+ tool calls across multiple steps,
persists conversation to SQLiteMemoryStore, uses context windowing, and produces
a structured ResearchReport with all required fields populated.

Memory item count per run:
  - 1 HumanMemory (user prompt, saved before first LLM call)
  - N AIMemory (one per LLM response, including each step that requests a tool)
  - N ToolMemory (one per tool execution via POST_TOOL_CALL hook)
With 6 tool calls: 1 + 6 + 6 + 1 = 14 items > 10.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel


class ResearchReport(BaseModel):
    topic: str
    key_findings: list[str]
    word_count: int
    sources_consulted: int


@pytest.mark.integration
@pytest.mark.marathon
@pytest.mark.timeout(180)
async def test_research_agent_produces_structured_report(
    vertex_model: str, tmp_sqlite_db: str
) -> None:
    """Marathon: research agent uses 3 tools across 5+ steps to produce ResearchReport.

    The agent is instructed to research Python language history using:
    - search_topic: broad information lookup (called 3+ times)
    - fetch_detail: specific subtopic deep-dive (called 2+ times)
    - count_words: word count utility (called 1+ times)

    Tools are chained sequentially to avoid Gemini parallel-tool-call errors.

    We assert:
    - All 4 ResearchReport fields populated with valid values
    - At least 5 tool calls in result.tool_calls
    - Memory store has > 10 items after the run
    """
    from exo._internal.output_parser import (  # pyright: ignore[reportMissingImports]
        parse_structured_output,
    )
    from exo.agent import Agent  # pyright: ignore[reportMissingImports]
    from exo.context.config import ContextConfig  # pyright: ignore[reportMissingImports]
    from exo.memory.backends.sqlite import (  # pyright: ignore[reportMissingImports]
        SQLiteMemoryStore,
    )
    from exo.memory.base import MemoryMetadata  # pyright: ignore[reportMissingImports]
    from exo.models import get_provider  # pyright: ignore[reportMissingImports]
    from exo.tool import tool  # pyright: ignore[reportMissingImports]

    # -------------------------------------------------------------------------
    # Tool definitions — return pre-set data so no external calls are needed.
    # Each tool result hints at the next tool to call, enabling natural chaining.
    # -------------------------------------------------------------------------

    @tool
    def search_topic(query: str) -> str:
        """Search for general information about a topic and return a summary.

        Returns an overview with specific subtopics to investigate further.

        Args:
            query: The topic or question to search for.
        """
        database: dict[str, str] = {
            "python history origin": (
                "Python was invented by Guido van Rossum and development started in December 1989 "
                "at Centrum Wiskunde & Informatica (CWI) in the Netherlands. "
                "Python 0.9.0 was first released in February 1991. Python 1.0 followed in January 1994. "
                "Suggested next topics: 'guido van rossum biography', 'python 1.0 features'."
            ),
            "python version history": (
                "Python 2.0 was released October 16, 2000, adding list comprehensions, garbage collection, "
                "and Unicode support. Python 3.0 was released December 3, 2008, breaking backward compatibility "
                "to fix design flaws. Python 2 reached end-of-life January 1, 2020. "
                "Suggested next topics: 'python 3 improvements', 'python 2 end of life'."
            ),
            "python applications uses": (
                "Python is widely used across many domains: web development (Django, Flask, FastAPI), "
                "data science and analytics (pandas, NumPy), machine learning and AI (TensorFlow, PyTorch, "
                "scikit-learn), automation and scripting, scientific computing, and DevOps tooling. "
                "By 2021-2022, Python was consistently ranked as the world's most popular programming language. "
                "Suggested next topics: 'python data science ecosystem', 'python web frameworks'."
            ),
            "python design philosophy": (
                "Python's design philosophy prioritizes code readability and simplicity. The Zen of Python "
                "(PEP 20, authored 2004) encodes 19 aphorisms: 'Beautiful is better than ugly', "
                "'Explicit is better than implicit', 'Simple is better than complex'. "
                "Python uses significant whitespace (indentation) to delimit code blocks. "
                "Suggested next topics: 'pep process history', 'python community governance'."
            ),
        }
        q = query.lower()
        for key, value in database.items():
            if any(word in q for word in key.split()):
                return value
        return (
            f"General information on '{query}': Python is a high-level, interpreted programming "
            "language created in 1989. Known for clear syntax, extensive standard library, and "
            "strong community. Now used in data science, web, AI, and automation."
        )

    @tool
    def fetch_detail(topic: str) -> str:
        """Fetch detailed information on a specific subtopic for the research report.

        Use this after search_topic reveals subtopics worth investigating.

        Args:
            topic: A specific person, feature, or subtopic to get detail on.
        """
        details: dict[str, str] = {
            "guido van rossum": (
                "Guido van Rossum (born January 31, 1956) is a Dutch programmer and creator of Python. "
                "He named Python after Monty Python's Flying Circus. Worked at CWI, CNRI, BeOpen, "
                "Zope, Google (2005-2012), and Dropbox (2013-2019). He was Python's BDFL (Benevolent "
                "Dictator For Life) until 2018 when he retired from the role following a governance debate "
                "over PEP 572. He joined Microsoft in 2020. Python now governed by the elected Steering Council."
            ),
            "python 3 improvements": (
                "Python 3.0 (December 2008) fixed fundamental design flaws: print became a function, "
                "integer division (/) returns a float, strings are Unicode by default, raw_input() merged "
                "into input(), xrange() merged into range(). Later versions added: asyncio (3.4), "
                "type hints via PEP 484 (3.5), f-strings (3.6), walrus operator ':=' (3.8), "
                "structural pattern matching 'match' statement (3.10), self-type and tomllib (3.11)."
            ),
            "python data science ecosystem": (
                "The Python data science ecosystem grew rapidly: NumPy (2006) for arrays, "
                "pandas (2008) for data manipulation, matplotlib (2003) for visualization, "
                "scikit-learn (2007) for machine learning. Jupyter notebooks (evolved from IPython, 2014) "
                "became the dominant interface for interactive data analysis. By 2016 Python surpassed R "
                "as the leading data science language in most professional surveys."
            ),
            "pep process history": (
                "PEP (Python Enhancement Proposal) is Python's formal RFC process. PEP 1 (2000) defines "
                "the process itself. PEP 8 (2001) is the style guide. PEP 20 is the Zen of Python. "
                "PEP 257 covers docstring conventions. PEP 484 (2014) introduced type hints. "
                "PEPs are discussed publicly on mailing lists, then accepted or rejected by the BDFL "
                "(now the Steering Council). Over 700 PEPs have been submitted since 2000."
            ),
            "python 2 end of life": (
                "Python 2.7 (released July 3, 2010) was the final major Python 2 release. "
                "Originally set for 2015, the EOL was extended to January 1, 2020 due to the large "
                "existing Python 2 codebase. The migration required significant community effort; tools "
                "like 2to3 and six helped. Projects like Django, NumPy, and pip all completed migration "
                "well before 2020. The decade-long transition is a cautionary tale of backward compatibility."
            ),
        }
        t = topic.lower()
        for key, value in details.items():
            if any(word in t for word in key.split()):
                return value
        return (
            f"Detail on '{topic}': Python's rich ecosystem evolved over 30+ years from a scripting "
            "language for the Amoeba OS into a globally dominant general-purpose language, supported by "
            "hundreds of thousands of open-source packages on PyPI."
        )

    @tool
    def count_words(text: str) -> int:
        """Count the number of words in the provided research summary text.

        Call this after assembling your research summary to determine the word count
        for the ResearchReport.

        Args:
            text: The research summary text to count words in.
        """
        return len(text.split())

    # -------------------------------------------------------------------------
    # Agent setup
    # -------------------------------------------------------------------------

    conv_id = "research-marathon-001"
    agent_name = "research-agent"

    sqlite_store = SQLiteMemoryStore(db_path=tmp_sqlite_db)
    await sqlite_store.init()
    provider = get_provider(vertex_model)

    try:
        agent = Agent(
            name=agent_name,
            model=vertex_model,
            instructions=(
                "You are a systematic research assistant. "
                "When asked to research a topic, follow this EXACT sequence of tool calls "
                "and call them ONE AT A TIME (never call two tools simultaneously): "
                "1. Call search_topic to get an overview of the topic. "
                "2. Call search_topic again with a different query angle. "
                "3. Call fetch_detail on the first specific subtopic mentioned in your search results. "
                "4. Call fetch_detail on a second specific subtopic from the results. "
                "5. Call search_topic a third time to find application or usage information. "
                "6. Assemble a concise 30-50 word summary of your research, then call count_words "
                "   with that exact summary text. "
                "7. After ALL 6 tool calls complete, reply with ONLY a valid JSON object: "
                '   {"topic": "<research topic string>", '
                '    "key_findings": ["<finding 1>", "<finding 2>", "<finding 3>"], '
                '    "word_count": <integer from count_words result>, '
                '    "sources_consulted": <integer count of total tool calls made>} '
                "The key_findings list MUST have at least 3 distinct, specific findings. "
                "Output ONLY the JSON object — no other text, no markdown code fences."
            ),
            tools=[count_words, fetch_detail, search_topic],
            memory=sqlite_store,
            context=ContextConfig(
                mode="copilot",
                history_rounds=10,
                summary_threshold=50,
                offload_threshold=100,
            ),
            max_steps=12,
        )

        result = await agent.run(
            "Research the history and evolution of the Python programming language. "
            "Follow your instructions EXACTLY in this order: "
            "Step 1: Call search_topic with query='python history origin'. "
            "Step 2: Call search_topic with query='python version history'. "
            "Step 3: Call fetch_detail with topic='guido van rossum'. "
            "Step 4: Call fetch_detail with topic='python 3 improvements'. "
            "Step 5: Call search_topic with query='python applications uses'. "
            "Step 6: Write a 30-50 word summary of your findings and call count_words with it. "
            "Step 7: Output ONLY the ResearchReport JSON object.",
            provider=provider,
            conversation_id=conv_id,
        )
    finally:
        await sqlite_store.close()

    # -------------------------------------------------------------------------
    # Assert structured output
    # -------------------------------------------------------------------------
    report = parse_structured_output(result.text, ResearchReport)
    assert isinstance(report, ResearchReport), (
        f"Expected ResearchReport instance, got {type(report)}: {result.text!r}"
    )
    assert report.topic, f"Expected non-empty topic, got: {report.topic!r}"
    assert len(report.key_findings) >= 3, (
        f"Expected >= 3 key_findings, got {len(report.key_findings)}: {report.key_findings}"
    )
    assert report.word_count > 0, (
        f"Expected word_count > 0, got: {report.word_count}"
    )
    assert report.sources_consulted >= 2, (
        f"Expected sources_consulted >= 2, got: {report.sources_consulted}"
    )

    # -------------------------------------------------------------------------
    # Assert at least 5 tool calls
    # -------------------------------------------------------------------------
    assert len(result.tool_calls) >= 5, (
        f"Expected >= 5 tool calls, got {len(result.tool_calls)}: "
        f"{[tc.name for tc in result.tool_calls]}"
    )

    # -------------------------------------------------------------------------
    # Assert memory store has > 10 items after the run.
    # A single agent.run() with N tool calls persists:
    #   1 HumanMemory + N AIMemory (per LLM step) + N ToolMemory (per tool exec) + 1 AIMemory (final)
    # With 6 tool calls: 1 + 6 + 6 + 1 = 14 items.
    # -------------------------------------------------------------------------
    verify_store = SQLiteMemoryStore(db_path=tmp_sqlite_db)
    await verify_store.init()
    try:
        items = await verify_store.search(
            metadata=MemoryMetadata(agent_id=agent_name, task_id=conv_id),
            limit=100,
        )
    finally:
        await verify_store.close()

    assert len(items) > 10, (
        f"Expected > 10 memory items after marathon run, got {len(items)}"
    )
