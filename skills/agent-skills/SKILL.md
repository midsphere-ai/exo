---
name: exo:agent-skills
description: "Use when implementing agent skills in Exo — SkillRegistry, Skill class, skill markdown files, front-matter format, tool_list mapping, skill sources (local and GitHub), conflict strategies, skill search, building agents from skills. Triggers on: exo skill, SkillRegistry, skill.md, agent skill, skill registry, skill loading, register_source, skill search, skill file."
---

> **Branch:** These skills are written for the `rename/orbiter-to-exo` branch. The Exo APIs referenced here may differ on other branches.

# Exo Agent Skills — Implementing Skills for Agents

## When To Use This Skill

Use this skill when the developer needs to:
- Create skill files that Exo agents consume at runtime
- Load skills from local directories or GitHub repositories
- Build a skill-powered agent using `SkillRegistry`
- Search, filter, and compose skills
- Handle skill conflicts across multiple sources
- Map skill metadata to agent instructions and tools

## Decision Guide

1. **Need to define a new skill?** → Create a `skill.md` or `SKILL.md` file with YAML front-matter
2. **Need to load skills from disk?** → `SkillRegistry().register_source("/path/to/skills")`
3. **Need to load from GitHub?** → `registry.register_source("https://github.com/org/repo/tree/main/skills")`
4. **Need to handle duplicate skill names?** → Set `ConflictStrategy` on registry (`KEEP_FIRST`, `KEEP_LAST`, `RAISE`)
5. **Need to wire skills into an agent?** → Use skill metadata to compose instructions + map `tool_list` to actual tools
6. **Need a "tool" skill vs "agent" skill?** → `type: ""` (empty) for tool skills, `type: agent` for agent skills

## Reference

### Skill File Format

Skill files must be named `skill.md` or `SKILL.md`. They use YAML front-matter + markdown body:

```markdown
---
name: web_search
description: Search the web for real-time information
type: agent
active: true
tool_list: {"search": ["web_search", "image_search"], "fetch": ["get_page"]}
---

# Web Search Skill

This skill enables the agent to search the web for current information.

## Usage

1. Call `web_search(query)` to find web pages
2. Call `image_search(query)` to find images
3. Call `get_page(url)` to fetch page content

## When to Use

- User asks about current events or recent data
- User needs information not in the agent's training data
- User asks for specific URLs or web content
```

### Front-Matter Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | Parent directory name | Unique skill identifier |
| `description` | `str` | `""` | Human-readable description (also accepts `desc` alias) |
| `type` | `str` | `""` | `""` for tool skills, `"agent"` for agent skills |
| `active` | `bool` | `true` | Whether the skill is enabled |
| `tool_list` | JSON object | `{}` | Maps tool categories to action name lists |

**Parsing rules:**
- `name` falls back to parent directory name if omitted
- `description` falls back to `desc` field if present
- `tool_list` must be valid JSON — invalid JSON silently defaults to `{}`
- `active` must be `true` or `false` (case-insensitive) — non-boolean defaults to `True`
- All front-matter keys are lowercased during parsing

### Directory Structure

Skills are discovered recursively by walking for `skill.md` / `SKILL.md`:

```
skills/
├── web_search/
│   └── skill.md              ← name: web_search
├── code_review/
│   └── SKILL.md              ← Both filenames recognized
├── data_analysis/
│   └── skill.md
└── nested/
    └── research/
        └── skill.md          ← Recursive discovery finds this
```

### Skill Class

```python
from exo.skills import Skill

skill = Skill(
    name="web_search",                              # Required
    description="Search the web for information",   # Human-readable
    usage="Call web_search(query) to search...",     # Markdown body from file
    tool_list={"search": ["web_search", "image_search"]},  # Tool mapping
    skill_type="agent",                             # "" or "agent"
    active=True,                                     # Enabled flag
    path="/path/to/skill.md",                       # Source file path
)

# Properties
skill.name          # "web_search"
skill.description   # "Search the web for information"
skill.usage         # Markdown body content
skill.tool_list     # {"search": ["web_search", "image_search"]}
skill.skill_type    # "agent"
skill.active        # True
skill.path          # "/path/to/skill.md"
```

### SkillRegistry

```python
from exo.skills import SkillRegistry, ConflictStrategy

registry = SkillRegistry(
    conflict=ConflictStrategy.KEEP_FIRST,  # How to handle duplicate names
    cache_dir=None,                         # GitHub clone cache (default: ~/.exo/skills/)
)
```

**ConflictStrategy options:**
| Strategy | Behavior |
|----------|----------|
| `KEEP_FIRST` | First source wins (default) |
| `KEEP_LAST` | Last source overrides |
| `RAISE` | Raises `SkillError` on duplicates |

### Registering Sources

```python
# Local directory
registry.register_source("/home/user/my-skills")

# GitHub repository (root)
registry.register_source("https://github.com/org/repo")

# GitHub repository (specific branch)
registry.register_source("https://github.com/org/repo/tree/develop")

# GitHub repository (specific subdirectory)
registry.register_source("https://github.com/org/repo/tree/main/skills")
```

**GitHub URL parsing:**
- Format: `https://github.com/{owner}/{repo}(/tree/{branch}(/subdir)?)?`
- Default branch: `main`
- Repos are shallow-cloned (`--depth 1`) to `~/.exo/skills/{owner}/{repo}/{branch}/`

### Loading Skills

```python
# Load from all registered sources
all_skills = registry.load_all()
# Returns: dict[str, Skill] — keyed by skill name

# Clears previously loaded skills before loading
# Applies conflict strategy for duplicates across sources
```

### Retrieving Skills

```python
# Get by exact name (raises SkillError if not found)
skill = registry.get("web_search")

# List all loaded skill names
names = registry.list_names()  # ["web_search", "code_review", ...]

# Access all skills
all_skills = registry.skills  # Returns copy of internal dict
```

### Searching Skills

```python
# Search by text query (case-insensitive, matches name and description)
results = registry.search(query="search")

# Filter by type
agent_skills = registry.search(skill_type="agent")
tool_skills = registry.search(skill_type="")

# Only active skills
active = registry.search(active_only=True)

# Combine filters
results = registry.search(
    query="web",
    skill_type="agent",
    active_only=True,
)
```

### Front-Matter Parsing (Standalone)

```python
from exo.skills import extract_front_matter

text = """---
name: my_skill
description: Does something useful
type: agent
active: true
tool_list: {"search": ["web_search"]}
---

# My Skill

Usage instructions here.
"""

meta, body = extract_front_matter(text)
# meta = {"name": "my_skill", "description": "Does something useful",
#         "type": "agent", "active": True, "tool_list": {"search": ["web_search"]}}
# body = "# My Skill\n\nUsage instructions here."
```

### SkillError

```python
from exo.skills import SkillError

# Raised for:
# - Source directory not found
# - Skill name not found in registry.get()
# - Duplicate skill names with ConflictStrategy.RAISE
```

## Patterns

### Basic Skill-Powered Agent

```python
from exo import Agent, run, tool
from exo.skills import SkillRegistry

# 1. Define actual tool implementations
@tool
async def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Results for '{query}': ..."

@tool
async def summarize(text: str) -> str:
    """Summarize a piece of text."""
    return f"Summary: {text[:100]}..."

# 2. Map skill names to tool implementations
TOOL_MAP = {
    "web_search": web_search,
    "summarize": summarize,
}

# 3. Load skills
registry = SkillRegistry()
registry.register_source("./skills")
skills = registry.load_all()

# 4. Build agent from skills
active_skills = registry.search(active_only=True)
tools = [TOOL_MAP[s.name] for s in active_skills if s.name in TOOL_MAP]

instructions_parts = ["You have the following skills:\n"]
for skill in active_skills:
    instructions_parts.append(f"### {skill.name}")
    instructions_parts.append(f"{skill.description}")
    if skill.usage:
        instructions_parts.append(f"\n{skill.usage}\n")

agent = Agent(
    name="skill-agent",
    model="openai:gpt-4o",
    instructions="\n".join(instructions_parts),
    tools=tools,
)

result = await run(agent, "Search for AI news and summarize it", provider=provider)
```

### Multi-Source Registry

```python
registry = SkillRegistry(conflict=ConflictStrategy.KEEP_LAST)

# Local skills take precedence (loaded last)
registry.register_source("https://github.com/org/shared-skills/tree/main/skills")
registry.register_source("./local-skills")  # Overrides shared skills with same name

skills = registry.load_all()
```

### Skill Type Separation

```python
registry = SkillRegistry()
registry.register_source("./skills")
registry.load_all()

# Tool skills: lightweight, single-action capabilities
tool_skills = registry.search(skill_type="", active_only=True)

# Agent skills: complex, multi-step capabilities
agent_skills = registry.search(skill_type="agent", active_only=True)

# Use tool skills for simple agents
simple_agent = Agent(
    name="helper",
    instructions=_build_instructions(tool_skills),
    tools=[TOOL_MAP[s.name] for s in tool_skills if s.name in TOOL_MAP],
)

# Use agent skills for complex orchestration agents
complex_agent = Agent(
    name="orchestrator",
    instructions=_build_instructions(agent_skills),
    tools=[TOOL_MAP[s.name] for s in agent_skills if s.name in TOOL_MAP],
)
```

### Dynamic Skill Discovery

```python
import os

registry = SkillRegistry()

# Register all skill directories found in a parent directory
skills_root = "./skill-packs"
for entry in os.listdir(skills_root):
    path = os.path.join(skills_root, entry)
    if os.path.isdir(path):
        registry.register_source(path)

skills = registry.load_all()
print(f"Loaded {len(skills)} skills from {len(registry._sources)} sources")
```

### Skill-Driven Instructions Builder

```python
def build_instructions(skills: list[Skill]) -> str:
    """Build structured agent instructions from loaded skills."""
    parts = [
        "You are a versatile assistant with the following capabilities:\n",
    ]

    for skill in skills:
        parts.append(f"## {skill.name}")
        parts.append(f"**Description:** {skill.description}")

        if skill.tool_list:
            tools_str = ", ".join(
                f"`{action}`"
                for actions in skill.tool_list.values()
                for action in actions
            )
            parts.append(f"**Available tools:** {tools_str}")

        if skill.usage:
            # Include first 3 lines of usage as quick reference
            usage_lines = skill.usage.strip().splitlines()[:3]
            parts.append("\n".join(usage_lines))

        parts.append("")  # Blank line separator

    parts.append(
        "\nUse the appropriate skill based on what the user asks. "
        "If a skill has associated tools, prefer using them."
    )

    return "\n".join(parts)
```

### Creating a Skill File Programmatically

```python
from pathlib import Path

def create_skill(
    directory: str,
    name: str,
    description: str,
    skill_type: str = "",
    tool_list: dict | None = None,
    usage: str = "",
) -> Path:
    """Create a skill.md file in the given directory."""
    import json

    skill_dir = Path(directory) / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "skill.md"

    lines = ["---"]
    lines.append(f"name: {name}")
    lines.append(f"description: {description}")
    if skill_type:
        lines.append(f"type: {skill_type}")
    lines.append("active: true")
    if tool_list:
        lines.append(f"tool_list: {json.dumps(tool_list)}")
    lines.append("---")
    lines.append("")
    lines.append(usage)

    skill_file.write_text("\n".join(lines), encoding="utf-8")
    return skill_file
```

## Gotchas

- **File must be named `skill.md` or `SKILL.md`** — any other filename is ignored by `_collect_skills()`
- **`name` fallback** — if `name` is omitted from front-matter, the parent directory name is used. This means directory naming matters.
- **`tool_list` must be valid JSON** — YAML-style mappings don't work. Use `{"key": ["val1", "val2"]}` not `key: [val1, val2]`.
- **`tool_list` is metadata only** — it doesn't auto-register tools. You must map skill names to actual `Tool` implementations yourself.
- **`usage` is the markdown body** — everything after the closing `---` delimiter becomes `skill.usage`. This is the skill's documentation/instructions.
- **GitHub caching** — repos are shallow-cloned once to `~/.exo/skills/{owner}/{repo}/{branch}/`. Delete the cache directory to force re-clone.
- **`load_all()` clears first** — calling `load_all()` clears all previously loaded skills before reloading. It's not additive.
- **Search is substring-based** — `registry.search(query="search")` matches "web_search", "search_engine", "research", etc. It's case-insensitive.
- **Conflict strategy applies per-name** — if two sources define a skill with the same name, the strategy determines which wins. Different names never conflict.
- **Skills are not agents** — a `Skill` is a metadata container. It becomes useful only when you wire it into an `Agent`'s instructions and tools.
- **`skill_type=""` is the default** — empty string means "tool skill". Only set `type: agent` for skills that represent complex multi-step agent capabilities.
