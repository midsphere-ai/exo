"""Multi-source skill registry for loading skills from local paths and GitHub.

Skills are markdown files with YAML front-matter containing metadata
(name, description, tool_list, type, active). The registry loads skills
from local directories and GitHub repositories, caching remote repos
at ``~/.exo/skills/``.

Usage::

    reg = SkillRegistry()
    reg.register_source("/path/to/skills")
    reg.register_source("https://github.com/user/repo/tree/main/skills")
    skills = reg.load_all()
"""

from __future__ import annotations

import json
import re
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Any

from exo.types import ExoError

_GITHUB_RE = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)"
    r"(?:/tree/(?P<branch>[^/]+)(?:/(?P<subdir>.+))?)?"
)

DEFAULT_CACHE_DIR = Path.home() / ".exo" / "skills"
SKILL_FILENAMES = {"skill.md", "SKILL.md"}


class SkillError(ExoError):
    """Raised for skill loading or registry errors."""


class ConflictStrategy(StrEnum):
    """How to handle duplicate skill names across sources."""

    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"
    RAISE = "raise"


class Skill:
    """A loaded skill with metadata and usage content.

    Args:
        name: Unique skill name.
        description: Human-readable description.
        usage: Markdown body content (usage instructions).
        tool_list: Mapping of tool names to action lists.
        skill_type: Skill type — empty string for tool skills, ``"agent"`` for agent skills.
        active: Whether the skill starts active.
        path: Filesystem path to the source skill.md file.
    """

    __slots__ = ("active", "description", "name", "path", "skill_type", "tool_list", "usage")

    def __init__(
        self,
        *,
        name: str,
        description: str = "",
        usage: str = "",
        tool_list: dict[str, list[str]] | None = None,
        skill_type: str = "",
        active: bool = True,
        path: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self.usage = usage
        self.tool_list: dict[str, list[str]] = tool_list or {}
        self.skill_type = skill_type
        self.active = active
        self.path = path

    def __repr__(self) -> str:
        return f"Skill(name={self.name!r}, type={self.skill_type!r}, active={self.active})"


def parse_github_url(url: str) -> dict[str, str] | None:
    """Parse a GitHub URL into owner, repo, branch, and subdirectory.

    Returns:
        Dict with keys ``owner``, ``repo``, ``branch``, ``subdir`` or
        ``None`` if the URL is not a valid GitHub URL.
    """
    m = _GITHUB_RE.match(url)
    if not m:
        return None
    return {
        "owner": m.group("owner"),
        "repo": m.group("repo"),
        "branch": m.group("branch") or "main",
        "subdir": m.group("subdir") or "",
    }


def extract_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Extract YAML front-matter and body from a markdown skill file.

    Args:
        text: Full text content of a skill.md file.

    Returns:
        Tuple of (front-matter dict, body string). Front-matter keys
        are lowercased. The ``tool_list`` value is JSON-parsed if present.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    end = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end < 0:
        return {}, text

    meta: dict[str, Any] = {}
    for line in lines[1:end]:
        colon = line.find(":")
        if colon < 0:
            continue
        key = line[:colon].strip().lower()
        val = line[colon + 1 :].strip()
        if key == "tool_list":
            try:
                meta[key] = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                meta[key] = {}
        elif key == "active":
            meta[key] = val.lower() == "true"
        else:
            meta[key] = val

    body = "\n".join(lines[end + 1 :]).strip()
    return meta, body


def _clone_github(parsed: dict[str, str], cache_dir: Path) -> Path:
    """Shallow-clone a GitHub repo into the cache directory.

    Returns:
        Path to the clone (or subdirectory within it).
    """
    owner = parsed["owner"]
    repo = parsed["repo"]
    branch = parsed["branch"]
    subdir = parsed["subdir"]

    clone_dir = cache_dir / owner / repo / branch
    if not clone_dir.exists():
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/{owner}/{repo}.git"
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch, url, str(clone_dir)],
            check=True,
            capture_output=True,
        )

    if subdir:
        target = clone_dir / subdir
        if not target.is_dir():
            raise SkillError(f"Subdirectory '{subdir}' not found in {owner}/{repo}@{branch}")
        return target
    return clone_dir


def _collect_skills(root: Path) -> dict[str, Skill]:
    """Walk a directory tree and collect skills from skill.md files."""
    skills: dict[str, Skill] = {}
    if not root.is_dir():
        return skills

    for skill_file in root.rglob("*"):
        if skill_file.name not in SKILL_FILENAMES or not skill_file.is_file():
            continue
        text = skill_file.read_text(encoding="utf-8")
        meta, body = extract_front_matter(text)
        name = meta.get("name") or skill_file.parent.name
        skill = Skill(
            name=name,
            description=meta.get("description", meta.get("desc", "")),
            usage=body,
            tool_list=meta.get("tool_list") if isinstance(meta.get("tool_list"), dict) else {},
            skill_type=meta.get("type", ""),
            active=meta.get("active", True) if isinstance(meta.get("active"), bool) else True,
            path=str(skill_file),
        )
        skills[name] = skill
    return skills


class SkillRegistry:
    """Multi-source skill registry.

    Loads skills from local filesystem paths and GitHub repository URLs.
    Remote repositories are shallow-cloned and cached at *cache_dir*.

    Args:
        conflict: Strategy for handling duplicate skill names.
        cache_dir: Directory for caching GitHub clones.
    """

    def __init__(
        self,
        *,
        conflict: ConflictStrategy | str = ConflictStrategy.KEEP_FIRST,
        cache_dir: Path | None = None,
    ) -> None:
        self._conflict = ConflictStrategy(conflict)
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._sources: list[str] = []
        self._skills: dict[str, Skill] = {}

    @property
    def skills(self) -> dict[str, Skill]:
        """All loaded skills keyed by name."""
        return dict(self._skills)

    def register_source(self, source: str) -> None:
        """Add a skill source (local path or GitHub URL)."""
        self._sources.append(source)

    def load_all(self) -> dict[str, Skill]:
        """Load skills from all registered sources.

        Returns:
            Dict mapping skill name to Skill.

        Raises:
            SkillError: On conflict when strategy is ``raise``.
        """
        self._skills.clear()
        for source in self._sources:
            parsed = parse_github_url(source)
            if parsed:
                root = _clone_github(parsed, self._cache_dir)
            else:
                root = Path(source).expanduser().resolve()
                if not root.is_dir():
                    raise SkillError(f"Skill source directory not found: {root}")
            new_skills = _collect_skills(root)
            for name, skill in new_skills.items():
                self._merge(name, skill)
        return dict(self._skills)

    def _merge(self, name: str, skill: Skill) -> None:
        """Merge a skill into the registry, applying conflict strategy."""
        if name not in self._skills:
            self._skills[name] = skill
            return
        if self._conflict == ConflictStrategy.KEEP_FIRST:
            return
        if self._conflict == ConflictStrategy.KEEP_LAST:
            self._skills[name] = skill
            return
        raise SkillError(f"Duplicate skill '{name}' (conflict_strategy=raise)")

    def get(self, name: str) -> Skill:
        """Retrieve a skill by name.

        Raises:
            SkillError: If the skill is not found.
        """
        if name not in self._skills:
            raise SkillError(f"Skill '{name}' not found")
        return self._skills[name]

    def search(
        self,
        *,
        query: str = "",
        skill_type: str | None = None,
        active_only: bool = False,
    ) -> list[Skill]:
        """Search skills by text query, type, and active status.

        Args:
            query: Case-insensitive substring to match against name or description.
            skill_type: Filter by skill type (e.g. ``"agent"``).
            active_only: If ``True``, only return active skills.

        Returns:
            List of matching skills.
        """
        results: list[Skill] = []
        q = query.lower()
        for skill in self._skills.values():
            if active_only and not skill.active:
                continue
            if skill_type is not None and skill.skill_type != skill_type:
                continue
            if q and q not in skill.name.lower() and q not in skill.description.lower():
                continue
            results.append(skill)
        return results

    def list_names(self) -> list[str]:
        """Return all loaded skill names."""
        return list(self._skills.keys())
