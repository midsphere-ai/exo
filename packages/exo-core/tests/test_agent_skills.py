"""Tests for native skill support in the Agent class."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from exo.agent import Agent
from exo.models.types import ModelResponse  # pyright: ignore[reportMissingImports]
from exo.skills import DictToolResolver, Skill, SkillRegistry
from exo.tool import tool
from exo.types import Usage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@tool
def web_search(query: str) -> str:
    """Search the web."""
    return f"results for {query}"


@tool
def screenshot(url: str) -> str:
    """Take a screenshot."""
    return f"screenshot of {url}"


def _make_registry(*skills: Skill) -> SkillRegistry:
    """Build a SkillRegistry pre-loaded with the given skills."""
    reg = SkillRegistry()
    for sk in skills:
        reg._skills[sk.name] = sk
    return reg


def _mock_provider(content: str = "Hello!") -> AsyncMock:
    resp = ModelResponse(
        id="resp-1",
        model="test-model",
        content=content,
        tool_calls=[],
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )
    provider = AsyncMock()
    provider.complete = AsyncMock(return_value=resp)
    return provider


WEBDEV_SKILL = Skill(
    name="webdev",
    description="Next.js web application development",
    usage="# Webdev\nUse Next.js for all web projects.",
    tool_list={"web_search": ["search"]},
    active=True,
)

MOBILEDEV_SKILL = Skill(
    name="mobiledev",
    description="Expo mobile development",
    usage="# Mobiledev\nUse Expo for mobile.",
    tool_list={"screenshot": ["capture"]},
    active=True,
)

INACTIVE_SKILL = Skill(
    name="legacy",
    description="Legacy tooling",
    usage="# Legacy\nDo not use.",
    active=False,
)

NO_TOOLS_SKILL = Skill(
    name="writing",
    description="Technical writing guidelines",
    usage="# Writing\nUse active voice. Be concise.",
    tool_list={},
    active=True,
)


# ---------------------------------------------------------------------------
# Tests: constructor behavior
# ---------------------------------------------------------------------------


class TestSkillConstructor:
    def test_skills_registers_activate_skill_tool(self) -> None:
        reg = _make_registry(WEBDEV_SKILL)
        agent = Agent(name="a", skills=reg)
        assert "activate_skill" in agent.tools

    def test_no_skills_no_activate_skill(self) -> None:
        agent = Agent(name="a")
        assert "activate_skill" not in agent.tools

    def test_tool_resolver_dict_auto_wrapped(self) -> None:
        reg = _make_registry(WEBDEV_SKILL)
        agent = Agent(
            name="a",
            skills=reg,
            tool_resolver={"webdev": [web_search]},
        )
        assert isinstance(agent._tool_resolver, DictToolResolver)

    def test_tool_resolver_protocol_accepted(self) -> None:
        reg = _make_registry(WEBDEV_SKILL)
        resolver = DictToolResolver({"webdev": [web_search]})
        agent = Agent(name="a", skills=reg, tool_resolver=resolver)
        assert agent._tool_resolver is resolver


# ---------------------------------------------------------------------------
# Tests: activate_skill tool behavior
# ---------------------------------------------------------------------------


class TestActivateSkill:
    async def test_returns_usage(self) -> None:
        reg = _make_registry(WEBDEV_SKILL)
        agent = Agent(name="a", skills=reg)
        result = await agent.tools["activate_skill"].execute(name="webdev")
        assert result == "# Webdev\nUse Next.js for all web projects."

    async def test_adds_tools(self) -> None:
        reg = _make_registry(WEBDEV_SKILL)
        resolver = DictToolResolver({"webdev": [web_search]})
        agent = Agent(name="a", skills=reg, tool_resolver=resolver)

        assert "web_search" not in agent.tools
        await agent.tools["activate_skill"].execute(name="webdev")
        assert "web_search" in agent.tools

    async def test_skips_duplicates(self) -> None:
        reg = _make_registry(WEBDEV_SKILL)
        resolver = DictToolResolver({"webdev": [web_search]})
        agent = Agent(name="a", skills=reg, tool_resolver=resolver, tools=[web_search])

        original_tool = agent.tools["web_search"]
        await agent.tools["activate_skill"].execute(name="webdev")
        # Original tool preserved, no error raised
        assert agent.tools["web_search"] is original_tool

    async def test_not_found_returns_error(self) -> None:
        reg = _make_registry(WEBDEV_SKILL)
        agent = Agent(name="a", skills=reg)
        result = await agent.tools["activate_skill"].execute(name="nonexistent")
        assert "[Error: Skill 'nonexistent' not found." in result
        assert "webdev" in result

    async def test_no_tool_list(self) -> None:
        reg = _make_registry(NO_TOOLS_SKILL)
        resolver = DictToolResolver({})
        agent = Agent(name="a", skills=reg, tool_resolver=resolver)

        result = await agent.tools["activate_skill"].execute(name="writing")
        assert result == "# Writing\nUse active voice. Be concise."

    async def test_no_usage_returns_fallback(self) -> None:
        skill = Skill(name="empty", usage="", active=True)
        reg = _make_registry(skill)
        agent = Agent(name="a", skills=reg)
        result = await agent.tools["activate_skill"].execute(name="empty")
        assert result == "[Skill 'empty' activated (no usage instructions)]"

    async def test_multiple_tools_resolved(self) -> None:
        skill = Skill(
            name="fullstack",
            description="Full stack development",
            usage="# Fullstack",
            tool_list={"web_search": ["search"], "screenshot": ["capture"]},
            active=True,
        )
        reg = _make_registry(skill)
        resolver = DictToolResolver({"fullstack": [web_search, screenshot]})
        agent = Agent(name="a", skills=reg, tool_resolver=resolver)

        await agent.tools["activate_skill"].execute(name="fullstack")
        assert "web_search" in agent.tools
        assert "screenshot" in agent.tools


# ---------------------------------------------------------------------------
# Tests: skill catalog in system prompt
# ---------------------------------------------------------------------------


class TestSkillCatalog:
    async def test_active_skills_in_system_prompt(self) -> None:
        reg = _make_registry(WEBDEV_SKILL, MOBILEDEV_SKILL)
        provider = _mock_provider()
        agent = Agent(name="a", skills=reg)

        await agent.run("hi", provider=provider)

        call_args = provider.complete.call_args
        messages = call_args[0][0]
        system_msg = messages[0]
        assert system_msg.role == "system"
        assert "webdev" in system_msg.content
        assert "mobiledev" in system_msg.content
        assert "activate_skill" in system_msg.content

    async def test_inactive_skills_excluded(self) -> None:
        reg = _make_registry(WEBDEV_SKILL, INACTIVE_SKILL)
        provider = _mock_provider()
        agent = Agent(name="a", skills=reg)

        await agent.run("hi", provider=provider)

        call_args = provider.complete.call_args
        messages = call_args[0][0]
        system_msg = messages[0]
        assert "webdev" in system_msg.content
        assert "legacy" not in system_msg.content

    async def test_skill_descriptions_in_catalog(self) -> None:
        reg = _make_registry(WEBDEV_SKILL)
        provider = _mock_provider()
        agent = Agent(name="a", skills=reg)

        await agent.run("hi", provider=provider)

        call_args = provider.complete.call_args
        messages = call_args[0][0]
        system_msg = messages[0]
        assert "Next.js web application development" in system_msg.content


# ---------------------------------------------------------------------------
# Tests: serialization
# ---------------------------------------------------------------------------


class TestSkillSerialization:
    def test_to_dict_raises_with_skills(self) -> None:
        reg = _make_registry(WEBDEV_SKILL)
        agent = Agent(name="a", skills=reg)
        with pytest.raises(ValueError, match="skill registry"):
            agent.to_dict()

    def test_describe_includes_skills(self) -> None:
        reg = _make_registry(WEBDEV_SKILL, MOBILEDEV_SKILL)
        agent = Agent(name="a", skills=reg)
        info = agent.describe()
        assert "skills" in info
        assert set(info["skills"]) == {"webdev", "mobiledev"}

    def test_describe_no_skills_key_without_registry(self) -> None:
        agent = Agent(name="a")
        info = agent.describe()
        assert "skills" not in info
