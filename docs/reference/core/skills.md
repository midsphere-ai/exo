# exo.skills

Multi-source skill registry for loading skills from local paths and GitHub repositories.

**Module:** `exo.skills`

```python
from exo.skills import (
    Skill,
    SkillRegistry,
    SkillError,
    ConflictStrategy,
    parse_github_url,
    extract_front_matter,
    DEFAULT_CACHE_DIR,
    SKILL_FILENAMES,
)
```

---

## SkillError

```python
class SkillError(ExoError)
```

Raised for skill loading or registry errors (missing directories, duplicate skills with raise strategy, etc.). Inherits from `ExoError`.

---

## ConflictStrategy

```python
class ConflictStrategy(StrEnum)
```

How to handle duplicate skill names across sources.

### Values

| Member | Value | Description |
|--------|-------|-------------|
| `KEEP_FIRST` | `"keep_first"` | Keep the first skill loaded with a given name (default). |
| `KEEP_LAST` | `"keep_last"` | Replace with the latest skill loaded with a given name. |
| `RAISE` | `"raise"` | Raise a `SkillError` on duplicate names. |

---

## Constants

### DEFAULT_CACHE_DIR

```python
DEFAULT_CACHE_DIR = Path.home() / ".exo" / "skills"
```

Default directory for caching GitHub repository clones.

### SKILL_FILENAMES

```python
SKILL_FILENAMES = {"skill.md", "SKILL.md"}
```

Recognized filenames for skill definitions.

---

## Skill

```python
class Skill
```

A loaded skill with metadata and usage content. Uses `__slots__` for memory efficiency.

### Constructor

```python
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
) -> None
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | *(required)* | Unique skill name. |
| `description` | `str` | `""` | Human-readable description. |
| `usage` | `str` | `""` | Markdown body content (usage instructions). |
| `tool_list` | `dict[str, list[str]] \| None` | `None` | Mapping of tool names to action lists. |
| `skill_type` | `str` | `""` | Skill type -- empty string for tool skills, `"agent"` for agent skills. |
| `active` | `bool` | `True` | Whether the skill starts active. |
| `path` | `str` | `""` | Filesystem path to the source skill.md file. |

### Attributes

| Name | Type | Description |
|------|------|-------------|
| `name` | `str` | Skill name. |
| `description` | `str` | Skill description. |
| `usage` | `str` | Markdown usage content. |
| `tool_list` | `dict[str, list[str]]` | Tool name to action list mapping. |
| `skill_type` | `str` | Skill type identifier. |
| `active` | `bool` | Whether the skill is active. |
| `path` | `str` | Source file path. |

### Example

```python
from exo.skills import Skill

skill = Skill(
    name="web_search",
    description="Search the web for information.",
    usage="Use this skill to find up-to-date information.",
    tool_list={"search_api": ["search", "fetch_page"]},
    active=True,
)
print(skill)  # Skill(name='web_search', type='', active=True)
```

---

## SkillRegistry

```python
class SkillRegistry
```

Multi-source skill registry. Loads skills from local filesystem paths and GitHub repository URLs. Remote repositories are shallow-cloned and cached.

### Constructor

```python
def __init__(
    self,
    *,
    conflict: ConflictStrategy | str = ConflictStrategy.KEEP_FIRST,
    cache_dir: Path | None = None,
) -> None
```

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `conflict` | `ConflictStrategy \| str` | `ConflictStrategy.KEEP_FIRST` | Strategy for handling duplicate skill names. |
| `cache_dir` | `Path \| None` | `None` | Directory for caching GitHub clones. Defaults to `~/.exo/skills/`. |

### Properties

#### skills

```python
@property
def skills(self) -> dict[str, Skill]
```

All loaded skills keyed by name. Returns a copy of the internal dict.

### Methods

#### register_source()

```python
def register_source(self, source: str) -> None
```

Add a skill source (local path or GitHub URL).

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `source` | `str` | *(required)* | Local directory path or GitHub repository URL. |

#### load_all()

```python
def load_all(self) -> dict[str, Skill]
```

Load skills from all registered sources.

**Returns:** Dict mapping skill name to `Skill`.

**Raises:** `SkillError` on conflict when strategy is `RAISE`.

#### get()

```python
def get(self, name: str) -> Skill
```

Retrieve a skill by name.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `name` | `str` | *(required)* | The skill name. |

**Returns:** The `Skill` instance.

**Raises:** `SkillError` if the skill is not found.

#### search()

```python
def search(
    self,
    *,
    query: str = "",
    skill_type: str | None = None,
    active_only: bool = False,
) -> list[Skill]
```

Search skills by text query, type, and active status.

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `query` | `str` | `""` | Case-insensitive substring to match against name or description. |
| `skill_type` | `str \| None` | `None` | Filter by skill type (e.g. `"agent"`). |
| `active_only` | `bool` | `False` | If `True`, only return active skills. |

**Returns:** List of matching skills.

#### list_names()

```python
def list_names(self) -> list[str]
```

Return all loaded skill names.

**Returns:** List of name strings.

### Example

```python
from exo.skills import SkillRegistry, ConflictStrategy

reg = SkillRegistry(conflict=ConflictStrategy.KEEP_LAST)

# Register local skills
reg.register_source("/path/to/my/skills")

# Register GitHub skills
reg.register_source("https://github.com/user/repo/tree/main/skills")

# Load all skills
skills = reg.load_all()
for name, skill in skills.items():
    print(f"{name}: {skill.description}")

# Search
results = reg.search(query="search", active_only=True)

# Get specific skill
web_skill = reg.get("web_search")
print(web_skill.usage)
```

---

## parse_github_url()

```python
def parse_github_url(url: str) -> dict[str, str] | None
```

Parse a GitHub URL into owner, repo, branch, and subdirectory.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `url` | `str` | *(required)* | A GitHub URL. |

### Returns

Dict with keys `owner`, `repo`, `branch`, `subdir`, or `None` if the URL is not a valid GitHub URL.

### Example

```python
from exo.skills import parse_github_url

result = parse_github_url("https://github.com/user/repo/tree/main/skills")
# {"owner": "user", "repo": "repo", "branch": "main", "subdir": "skills"}

result = parse_github_url("https://github.com/user/repo")
# {"owner": "user", "repo": "repo", "branch": "main", "subdir": ""}

result = parse_github_url("not a github url")
# None
```

---

## extract_front_matter()

```python
def extract_front_matter(text: str) -> tuple[dict[str, Any], str]
```

Extract YAML front-matter and body from a markdown skill file.

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `str` | *(required)* | Full text content of a skill.md file. |

### Returns

Tuple of `(front_matter_dict, body_string)`. Front-matter keys are lowercased. The `tool_list` value is JSON-parsed if present. The `active` value is converted to a boolean.

### Example

```python
from exo.skills import extract_front_matter

text = """---
name: my_skill
description: A useful skill
active: true
tool_list: {"search": ["web_search"]}
---
# Usage
Use this skill to search the web.
"""

meta, body = extract_front_matter(text)
# meta = {"name": "my_skill", "description": "A useful skill", "active": True, "tool_list": {"search": ["web_search"]}}
# body = "# Usage\nUse this skill to search the web."
```

---

## Skill File Format

Skills are defined in `skill.md` or `SKILL.md` files with YAML front-matter:

```markdown
---
name: web_search
description: Search the web for current information
type: agent
active: true
tool_list: {"search_api": ["search", "fetch"]}
---
# Web Search Skill

Use this skill when you need to find up-to-date information.

## Usage
1. Call the search_api tool with your query
2. Process the results
```

### Front-Matter Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Skill name (falls back to parent directory name if absent). |
| `description` | `string` | Human-readable description (also accepts `desc`). |
| `type` | `string` | Skill type: empty for tool skills, `"agent"` for agent skills. |
| `active` | `boolean` | Whether the skill starts active (`true`/`false`). |
| `tool_list` | `JSON object` | Mapping of tool names to action lists. |
