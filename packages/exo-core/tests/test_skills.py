"""Tests for exo.skills — multi-source skill registry."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from exo.skills import (
    ConflictStrategy,
    Skill,
    SkillError,
    SkillRegistry,
    _collect_skills,
    extract_front_matter,
    parse_github_url,
)
from exo.types import ExoError

# ---------------------------------------------------------------------------
# SkillError
# ---------------------------------------------------------------------------


class TestSkillError:
    def test_inherits_exo_error(self) -> None:
        assert issubclass(SkillError, ExoError)

    def test_raise_and_catch(self) -> None:
        with pytest.raises(SkillError, match="test error"):
            raise SkillError("test error")


# ---------------------------------------------------------------------------
# ConflictStrategy
# ---------------------------------------------------------------------------


class TestConflictStrategy:
    def test_values(self) -> None:
        assert ConflictStrategy.KEEP_FIRST == "keep_first"
        assert ConflictStrategy.KEEP_LAST == "keep_last"
        assert ConflictStrategy.RAISE == "raise"

    def test_from_string(self) -> None:
        assert ConflictStrategy("keep_first") == ConflictStrategy.KEEP_FIRST


# ---------------------------------------------------------------------------
# Skill dataclass
# ---------------------------------------------------------------------------


class TestSkill:
    def test_minimal(self) -> None:
        s = Skill(name="test")
        assert s.name == "test"
        assert s.description == ""
        assert s.usage == ""
        assert s.tool_list == {}
        assert s.skill_type == ""
        assert s.active is True
        assert s.path == ""

    def test_full(self) -> None:
        s = Skill(
            name="search",
            description="Search the web",
            usage="Use this to search.",
            tool_list={"browser": ["navigate", "click"]},
            skill_type="agent",
            active=False,
            path="/skills/search/skill.md",
        )
        assert s.name == "search"
        assert s.description == "Search the web"
        assert s.tool_list == {"browser": ["navigate", "click"]}
        assert s.skill_type == "agent"
        assert s.active is False

    def test_repr(self) -> None:
        s = Skill(name="test", skill_type="agent", active=False)
        r = repr(s)
        assert "test" in r
        assert "agent" in r
        assert "False" in r


# ---------------------------------------------------------------------------
# parse_github_url
# ---------------------------------------------------------------------------


class TestParseGithubUrl:
    def test_simple_repo(self) -> None:
        result = parse_github_url("https://github.com/user/repo")
        assert result is not None
        assert result["owner"] == "user"
        assert result["repo"] == "repo"
        assert result["branch"] == "main"
        assert result["subdir"] == ""

    def test_with_branch(self) -> None:
        result = parse_github_url("https://github.com/user/repo/tree/develop")
        assert result is not None
        assert result["branch"] == "develop"
        assert result["subdir"] == ""

    def test_with_branch_and_subdir(self) -> None:
        result = parse_github_url("https://github.com/user/repo/tree/main/skills/dir")
        assert result is not None
        assert result["branch"] == "main"
        assert result["subdir"] == "skills/dir"

    def test_not_github(self) -> None:
        assert parse_github_url("/local/path") is None
        assert parse_github_url("https://gitlab.com/user/repo") is None

    def test_http(self) -> None:
        result = parse_github_url("http://github.com/user/repo")
        assert result is not None
        assert result["owner"] == "user"


# ---------------------------------------------------------------------------
# extract_front_matter
# ---------------------------------------------------------------------------


class TestExtractFrontMatter:
    def test_full_front_matter(self) -> None:
        text = """---
name: my_skill
description: A test skill
tool_list: {"browser": ["click", "nav"]}
active: True
type: agent
---
# Usage
Use this skill to do things."""
        meta, body = extract_front_matter(text)
        assert meta["name"] == "my_skill"
        assert meta["description"] == "A test skill"
        assert meta["tool_list"] == {"browser": ["click", "nav"]}
        assert meta["active"] is True
        assert meta["type"] == "agent"
        assert "Use this skill" in body

    def test_no_front_matter(self) -> None:
        text = "# Just a markdown file\nNo front-matter."
        meta, body = extract_front_matter(text)
        assert meta == {}
        assert body == text

    def test_empty_text(self) -> None:
        meta, body = extract_front_matter("")
        assert meta == {}
        assert body == ""

    def test_active_false(self) -> None:
        text = "---\nactive: False\n---\nBody."
        meta, _body = extract_front_matter(text)
        assert meta["active"] is False

    def test_invalid_tool_list_json(self) -> None:
        text = "---\ntool_list: not-json\n---\nBody."
        meta, _ = extract_front_matter(text)
        assert meta["tool_list"] == {}

    def test_unclosed_front_matter(self) -> None:
        text = "---\nname: test\nNo closing delimiter."
        meta, body = extract_front_matter(text)
        assert meta == {}
        assert body == text

    def test_desc_alias(self) -> None:
        text = "---\ndesc: short desc\n---\nBody."
        meta, _ = extract_front_matter(text)
        assert meta["desc"] == "short desc"


# ---------------------------------------------------------------------------
# _collect_skills
# ---------------------------------------------------------------------------


class TestCollectSkills:
    def test_find_skill_md(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text(
            "---\nname: my_skill\ndescription: Test\n---\nUsage here."
        )
        skills = _collect_skills(tmp_path)
        assert "my_skill" in skills
        assert skills["my_skill"].description == "Test"
        assert "Usage here" in skills["my_skill"].usage

    def test_find_skill_md_uppercase(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "upper"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: upper_skill\n---\nContent.")
        skills = _collect_skills(tmp_path)
        assert "upper_skill" in skills

    def test_name_from_directory(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "dir_name"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text("---\ndescription: No name field\n---\nBody.")
        skills = _collect_skills(tmp_path)
        assert "dir_name" in skills

    def test_empty_directory(self, tmp_path: Path) -> None:
        skills = _collect_skills(tmp_path)
        assert skills == {}

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        skills = _collect_skills(tmp_path / "nonexistent")
        assert skills == {}

    def test_nested_skills(self, tmp_path: Path) -> None:
        (tmp_path / "a" / "b").mkdir(parents=True)
        (tmp_path / "a" / "b" / "skill.md").write_text("---\nname: nested\n---\nNested skill.")
        skills = _collect_skills(tmp_path)
        assert "nested" in skills


# ---------------------------------------------------------------------------
# SkillRegistry — local loading
# ---------------------------------------------------------------------------


class TestSkillRegistryLocal:
    def test_load_local_skills(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text("---\nname: my_skill\ndescription: Hello\n---\nUsage.")
        reg = SkillRegistry()
        reg.register_source(str(tmp_path))
        skills = reg.load_all()
        assert "my_skill" in skills
        assert skills["my_skill"].description == "Hello"

    def test_get(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "s1"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text("---\nname: s1\n---\nBody.")
        reg = SkillRegistry()
        reg.register_source(str(tmp_path))
        reg.load_all()
        s = reg.get("s1")
        assert s.name == "s1"

    def test_get_missing_raises(self) -> None:
        reg = SkillRegistry()
        with pytest.raises(SkillError, match="not found"):
            reg.get("missing")

    def test_list_names(self, tmp_path: Path) -> None:
        for name in ["alpha", "beta"]:
            d = tmp_path / name
            d.mkdir()
            (d / "skill.md").write_text(f"---\nname: {name}\n---\nBody.")
        reg = SkillRegistry()
        reg.register_source(str(tmp_path))
        reg.load_all()
        names = reg.list_names()
        assert set(names) == {"alpha", "beta"}

    def test_skills_property_is_copy(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "s"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text("---\nname: s\n---\nBody.")
        reg = SkillRegistry()
        reg.register_source(str(tmp_path))
        reg.load_all()
        copy = reg.skills
        copy.clear()
        assert reg.skills != {}

    def test_missing_source_dir_raises(self) -> None:
        reg = SkillRegistry()
        reg.register_source("/nonexistent/path")
        with pytest.raises(SkillError, match="not found"):
            reg.load_all()


# ---------------------------------------------------------------------------
# SkillRegistry — conflict resolution
# ---------------------------------------------------------------------------


class TestSkillRegistryConflict:
    def _make_sources(self, tmp_path: Path) -> tuple[Path, Path]:
        src1 = tmp_path / "src1" / "dup"
        src1.mkdir(parents=True)
        (src1 / "skill.md").write_text("---\nname: dup\ndescription: first\n---\nBody 1.")

        src2 = tmp_path / "src2" / "dup"
        src2.mkdir(parents=True)
        (src2 / "skill.md").write_text("---\nname: dup\ndescription: second\n---\nBody 2.")
        return tmp_path / "src1", tmp_path / "src2"

    def test_keep_first(self, tmp_path: Path) -> None:
        src1, src2 = self._make_sources(tmp_path)
        reg = SkillRegistry(conflict=ConflictStrategy.KEEP_FIRST)
        reg.register_source(str(src1))
        reg.register_source(str(src2))
        skills = reg.load_all()
        assert skills["dup"].description == "first"

    def test_keep_last(self, tmp_path: Path) -> None:
        src1, src2 = self._make_sources(tmp_path)
        reg = SkillRegistry(conflict=ConflictStrategy.KEEP_LAST)
        reg.register_source(str(src1))
        reg.register_source(str(src2))
        skills = reg.load_all()
        assert skills["dup"].description == "second"

    def test_raise_on_conflict(self, tmp_path: Path) -> None:
        src1, src2 = self._make_sources(tmp_path)
        reg = SkillRegistry(conflict=ConflictStrategy.RAISE)
        reg.register_source(str(src1))
        reg.register_source(str(src2))
        with pytest.raises(SkillError, match="Duplicate skill"):
            reg.load_all()

    def test_string_conflict_strategy(self, tmp_path: Path) -> None:
        src1, src2 = self._make_sources(tmp_path)
        reg = SkillRegistry(conflict="keep_last")
        reg.register_source(str(src1))
        reg.register_source(str(src2))
        skills = reg.load_all()
        assert skills["dup"].description == "second"


# ---------------------------------------------------------------------------
# SkillRegistry — search
# ---------------------------------------------------------------------------


class TestSkillRegistrySearch:
    def _setup_reg(self, tmp_path: Path) -> SkillRegistry:
        skills = [
            ("search", "Search the web", "agent", True),
            ("code", "Code assistant", "", True),
            ("debug", "Debug helper", "", False),
        ]
        for name, desc, stype, active in skills:
            d = tmp_path / name
            d.mkdir()
            active_str = "True" if active else "False"
            (d / "skill.md").write_text(
                f"---\nname: {name}\ndescription: {desc}\ntype: {stype}\nactive: {active_str}\n---\nUsage."
            )
        reg = SkillRegistry()
        reg.register_source(str(tmp_path))
        reg.load_all()
        return reg

    def test_search_by_query(self, tmp_path: Path) -> None:
        reg = self._setup_reg(tmp_path)
        results = reg.search(query="web")
        assert len(results) == 1
        assert results[0].name == "search"

    def test_search_by_type(self, tmp_path: Path) -> None:
        reg = self._setup_reg(tmp_path)
        results = reg.search(skill_type="agent")
        assert len(results) == 1
        assert results[0].name == "search"

    def test_search_active_only(self, tmp_path: Path) -> None:
        reg = self._setup_reg(tmp_path)
        results = reg.search(active_only=True)
        names = {s.name for s in results}
        assert "debug" not in names
        assert "search" in names
        assert "code" in names

    def test_search_no_results(self, tmp_path: Path) -> None:
        reg = self._setup_reg(tmp_path)
        results = reg.search(query="nonexistent")
        assert results == []

    def test_search_combined_filters(self, tmp_path: Path) -> None:
        reg = self._setup_reg(tmp_path)
        results = reg.search(query="assist", active_only=True, skill_type="")
        assert len(results) == 1
        assert results[0].name == "code"

    def test_search_all(self, tmp_path: Path) -> None:
        reg = self._setup_reg(tmp_path)
        results = reg.search()
        assert len(results) == 3

    def test_search_case_insensitive(self, tmp_path: Path) -> None:
        reg = self._setup_reg(tmp_path)
        results = reg.search(query="SEARCH")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# SkillRegistry — GitHub source (mocked)
# ---------------------------------------------------------------------------


class TestSkillRegistryGitHub:
    def test_github_source_clones(self, tmp_path: Path) -> None:
        clone_dir = tmp_path / "cache" / "user" / "repo" / "main"
        clone_dir.mkdir(parents=True)
        skill_dir = clone_dir / "my_skill"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text("---\nname: remote_skill\n---\nRemote body.")

        reg = SkillRegistry(cache_dir=tmp_path / "cache")
        reg.register_source("https://github.com/user/repo")

        with patch("exo.skills._clone_github", return_value=clone_dir):
            skills = reg.load_all()

        assert "remote_skill" in skills

    def test_github_with_subdir(self, tmp_path: Path) -> None:
        sub = tmp_path / "cache" / "user" / "repo" / "main" / "skills"
        sub.mkdir(parents=True)
        sd = sub / "my_skill"
        sd.mkdir()
        (sd / "skill.md").write_text("---\nname: sub_skill\n---\nBody.")

        reg = SkillRegistry(cache_dir=tmp_path / "cache")
        reg.register_source("https://github.com/user/repo/tree/main/skills")

        with patch("exo.skills._clone_github", return_value=sub):
            skills = reg.load_all()

        assert "sub_skill" in skills
