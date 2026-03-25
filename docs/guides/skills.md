# Skills

Skills are reusable capability packages defined as markdown files with YAML front-matter. The skill registry loads skills from local directories and GitHub repositories, making it easy to share and discover agent capabilities across projects.

## Basic Usage

```python
from exo.skills import SkillRegistry

registry = SkillRegistry()
registry.register_source("/path/to/my/skills")
skills = registry.load_all()

for name, skill in skills.items():
    print(f"{name}: {skill.description}")
```

## Skill File Format

Skills are defined in markdown files named `skill.md` or `SKILL.md`. The file has YAML front-matter for metadata and a markdown body for usage instructions:

```markdown
---
name: web_search
description: Search the web for information
type: tool
active: true
tool_list: {"search": ["web_search", "image_search"]}
---

# Web Search Skill

Use this skill to search the web for current information.

## Usage

1. Call `web_search(query)` to search for web pages
2. Call `image_search(query)` to search for images

## Examples

- "Search for the latest AI news"
- "Find images of solar panels"
```

### Front-Matter Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | Parent directory name | Unique skill name |
| `description` | `str` | `""` | Human-readable description (also accepts `desc`) |
| `type` | `str` | `""` | Skill type: empty for tool skills, `"agent"` for agent skills |
| `active` | `bool` | `true` | Whether the skill starts active |
| `tool_list` | `dict[str, list[str]]` | `{}` | JSON mapping of tool names to action lists |

The `tool_list` value is JSON-parsed from the front-matter. If it is not a valid JSON object, it defaults to `{}`.

If `name` is not specified in the front-matter, the skill is named after its parent directory.

## Skill Class

A loaded skill is represented by the `Skill` class:

```python
class Skill:
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
    ) -> None: ...
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Unique skill name |
| `description` | `str` | Human-readable description |
| `usage` | `str` | Markdown body content |
| `tool_list` | `dict[str, list[str]]` | Tool name to action list mapping |
| `skill_type` | `str` | Skill type (`""` or `"agent"`) |
| `active` | `bool` | Whether the skill is active |
| `path` | `str` | Filesystem path to the source file |

## SkillRegistry

The `SkillRegistry` manages loading and searching skills from multiple sources.

### Constructor

```python
class SkillRegistry:
    def __init__(
        self,
        *,
        conflict: ConflictStrategy | str = ConflictStrategy.KEEP_FIRST,
        cache_dir: Path | None = None,
    ) -> None: ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conflict` | `ConflictStrategy \| str` | `"keep_first"` | How to handle duplicate skill names |
| `cache_dir` | `Path \| None` | `~/.exo/skills/` | Directory for caching GitHub clones |

### Registering Sources

```python
registry = SkillRegistry()

# Local directory
registry.register_source("/path/to/skills")

# GitHub repository
registry.register_source("https://github.com/user/repo")

# GitHub subdirectory
registry.register_source("https://github.com/user/repo/tree/main/skills")

# Specific branch
registry.register_source("https://github.com/user/repo/tree/develop/skills")
```

### Loading Skills

```python
skills = registry.load_all()
# Returns dict[str, Skill] mapping skill name -> Skill
```

`load_all()` processes all registered sources in order, collecting `skill.md` / `SKILL.md` files recursively. Duplicate names are handled according to the conflict strategy.

### Retrieving a Skill

```python
skill = registry.get("web_search")
# Raises SkillError if not found
```

### Searching Skills

```python
results = registry.search(
    query="search",         # case-insensitive substring match on name/description
    skill_type="tool",      # filter by skill type
    active_only=True,       # only return active skills
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | `""` | Case-insensitive substring match against name and description |
| `skill_type` | `str \| None` | `None` | Filter by skill type |
| `active_only` | `bool` | `False` | Only return active skills |

### Listing Skill Names

```python
names = registry.list_names()
# Returns list[str] of all loaded skill names
```

### Accessing Skills Property

```python
all_skills = registry.skills
# Returns dict[str, Skill] (a copy of the internal dict)
```

## Conflict Strategies

When the same skill name appears in multiple sources, the conflict strategy determines what happens:

```python
from exo.skills import ConflictStrategy, SkillRegistry

# Keep the first occurrence (default)
registry = SkillRegistry(conflict=ConflictStrategy.KEEP_FIRST)

# Keep the last occurrence (later sources override earlier ones)
registry = SkillRegistry(conflict=ConflictStrategy.KEEP_LAST)

# Raise an error on duplicates
registry = SkillRegistry(conflict=ConflictStrategy.RAISE)
```

| Strategy | Value | Behavior |
|----------|-------|----------|
| `KEEP_FIRST` | `"keep_first"` | First source wins (default) |
| `KEEP_LAST` | `"keep_last"` | Last source overrides earlier |
| `RAISE` | `"raise"` | Raises `SkillError` on duplicate |

## GitHub Sources

Skills can be loaded from public GitHub repositories. The registry shallow-clones the repository and caches it locally.

```python
registry = SkillRegistry()

# Full repository
registry.register_source("https://github.com/myorg/skills-repo")

# Subdirectory within a repo
registry.register_source("https://github.com/myorg/skills-repo/tree/main/agents")

# Specific branch
registry.register_source("https://github.com/myorg/skills-repo/tree/v2/skills")
```

### URL Format

GitHub URLs are parsed using this pattern:

```
https://github.com/{owner}/{repo}/tree/{branch}/{subdir}
```

| Component | Required | Default | Description |
|-----------|----------|---------|-------------|
| `owner` | Yes | -- | Repository owner |
| `repo` | Yes | -- | Repository name |
| `branch` | No | `"main"` | Branch name |
| `subdir` | No | `""` | Subdirectory within the repo |

### Cache Directory

Cloned repositories are cached at `~/.exo/skills/{owner}/{repo}/{branch}/`. You can customize the cache location:

```python
from pathlib import Path

registry = SkillRegistry(cache_dir=Path("/tmp/skill-cache"))
```

## Front-Matter Parsing

The `extract_front_matter()` function parses YAML front-matter from skill files:

```python
from exo.skills import extract_front_matter

text = """---
name: my_skill
description: A useful skill
active: true
tool_list: {"tool_a": ["action1", "action2"]}
---

# Skill Usage

Instructions here.
"""

meta, body = extract_front_matter(text)
# meta = {"name": "my_skill", "description": "A useful skill", "active": True, "tool_list": {"tool_a": ["action1", "action2"]}}
# body = "# Skill Usage\n\nInstructions here."
```

## Directory Structure

A typical skills directory:

```
skills/
  web_search/
    skill.md
  code_review/
    SKILL.md
  data_analysis/
    skill.md
```

The registry recursively walks the directory tree looking for files named `skill.md` or `SKILL.md`.

## Error Handling

```python
from exo.skills import SkillError, SkillRegistry

registry = SkillRegistry(conflict="raise")

# Source directory not found
try:
    registry.register_source("/nonexistent/path")
    registry.load_all()
except SkillError as e:
    print(e)  # "Skill source directory not found: ..."

# Skill not found
try:
    registry.get("nonexistent")
except SkillError as e:
    print(e)  # "Skill 'nonexistent' not found"

# Duplicate skill with raise strategy
try:
    registry.load_all()  # two sources with same skill name
except SkillError as e:
    print(e)  # "Duplicate skill 'name' (conflict_strategy=raise)"
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `Skill` | `exo.skills` | A loaded skill with metadata and usage content |
| `SkillRegistry` | `exo.skills` | Multi-source skill registry |
| `ConflictStrategy` | `exo.skills` | Enum for duplicate name handling |
| `SkillError` | `exo.skills` | Skill loading/registry error |
| `extract_front_matter()` | `exo.skills` | Parse YAML front-matter from markdown |
| `parse_github_url()` | `exo.skills` | Parse a GitHub URL into components |
