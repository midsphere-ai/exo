"""Skill-based agent — load skills from a registry and build agents dynamically.

Demonstrates ``SkillRegistry`` for loading skill definitions from local
markdown files, filtering by type/query, and wiring skill metadata into
an Agent's instructions and tool configuration.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python examples/advanced/skill_agent.py
"""

from pathlib import Path
from textwrap import dedent

from exo import Agent, run, tool
from exo.skills import Skill, SkillRegistry

# --- Define tools that correspond to skills ----------------------------------


@tool
async def web_search(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query string.
    """
    return f"Search results for '{query}': [result 1, result 2, result 3]"


@tool
async def summarize_text(text: str) -> str:
    """Summarize a block of text.

    Args:
        text: The text to summarize.
    """
    return f"Summary: {text[:80]}..."


# --- Create a local skills directory with skill.md files ---------------------

SKILLS_DIR = Path(__file__).parent / "_skills"


def _ensure_skills_dir() -> None:
    """Create sample skill.md files for demonstration."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)

    search_dir = SKILLS_DIR / "search"
    search_dir.mkdir(exist_ok=True)
    (search_dir / "skill.md").write_text(
        dedent("""\
        ---
        name: web-search
        description: Search the web for current information
        type: agent
        active: true
        ---
        Use the web_search tool to find information online.
        Always cite your sources and provide relevant links.
        """)
    )

    summary_dir = SKILLS_DIR / "summarizer"
    summary_dir.mkdir(exist_ok=True)
    (summary_dir / "skill.md").write_text(
        dedent("""\
        ---
        name: summarizer
        description: Summarize long documents or text
        type: agent
        active: true
        ---
        Use the summarize_text tool to create concise summaries.
        Focus on key points and main ideas.
        """)
    )


# --- Load skills and build an agent -----------------------------------------

TOOL_MAP = {
    "web-search": web_search,
    "summarizer": summarize_text,
}


def build_agent_from_skills(skills: list[Skill]) -> Agent:
    """Build an agent whose instructions and tools come from loaded skills."""
    tools = []
    instructions_parts = ["You are a skill-powered assistant with the following capabilities:\n"]

    for skill in skills:
        instructions_parts.append(f"- **{skill.name}**: {skill.description}")
        if skill.usage:
            instructions_parts.append(f"  Instructions: {skill.usage.splitlines()[0]}")
        if skill.name in TOOL_MAP:
            tools.append(TOOL_MAP[skill.name])

    return Agent(
        name="skill-agent",
        model="openai:gpt-4o-mini",
        instructions="\n".join(instructions_parts),
        tools=tools,
    )


if __name__ == "__main__":
    # Create sample skill files
    _ensure_skills_dir()

    # Load skills from registry
    registry = SkillRegistry()
    registry.register_source(str(SKILLS_DIR))
    all_skills = registry.load_all()
    print(f"Loaded {len(all_skills)} skills: {registry.list_names()}")

    # Filter to active agent-type skills
    agent_skills = registry.search(skill_type="agent", active_only=True)
    print(f"Active agent skills: {[s.name for s in agent_skills]}")

    # Build and run the agent
    agent = build_agent_from_skills(agent_skills)
    result = run.sync(agent, "Search for the latest news about AI agents.")
    print(result.output)
