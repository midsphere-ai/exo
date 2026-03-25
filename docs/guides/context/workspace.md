# Workspace

The `Workspace` provides versioned artifact storage with filesystem persistence, observer notifications, and optional knowledge store integration. Agents use it to write code, data, and documents that persist across conversation turns and can be referenced by neurons in prompt building.

## Basic Usage

```python
from exo.context import Workspace, ArtifactType

workspace = Workspace(base_dir="/tmp/my_workspace")

# Write an artifact
workspace.write("analysis.md", "# Results\nThe data shows...", artifact_type=ArtifactType.MARKDOWN)

# Read it back
content = workspace.read("analysis.md")
print(content)  # "# Results\nThe data shows..."

# List all artifacts
for name in workspace.list():
    print(name)  # "analysis.md"
```

## Artifact Types

The `ArtifactType` enum categorizes stored artifacts:

| Type | Value | Description |
|------|-------|-------------|
| `CODE` | `"code"` | Source code files |
| `CSV` | `"csv"` | Comma-separated data |
| `IMAGE` | `"image"` | Image files |
| `JSON` | `"json"` | JSON data |
| `MARKDOWN` | `"markdown"` | Markdown documents |
| `TEXT` | `"text"` | Plain text (default) |

```python
workspace.write("data.json", '{"key": "value"}', artifact_type=ArtifactType.JSON)
workspace.write("main.py", "print('hello')", artifact_type=ArtifactType.CODE)
```

## Versioning

Every `write()` creates a new version. The workspace tracks the full version history of each artifact:

```python
# Version 1
workspace.write("report.md", "Draft 1")

# Version 2
workspace.write("report.md", "Draft 2 - revised")

# Version 3
workspace.write("report.md", "Final version")

# Get version history
history = workspace.version_history("report.md")
for version in history:
    print(f"v{version.version}: {version.created_at}")

# Revert to an earlier version
workspace.revert_to_version("report.md", version=1)
content = workspace.read("report.md")  # "Draft 1"
```

Each `ArtifactVersion` is a frozen dataclass with:

- `version` -- integer version number (1-indexed).
- `content` -- the content at that version.
- `artifact_type` -- the `ArtifactType`.
- `created_at` -- ISO timestamp.

## Observer Pattern

Register callbacks to be notified when artifacts change:

```python
def on_write(event: str, name: str, **kwargs):
    print(f"Artifact '{name}' was written")

def on_delete(event: str, name: str, **kwargs):
    print(f"Artifact '{name}' was deleted")

workspace.on("write", on_write)
workspace.on("delete", on_delete)

workspace.write("data.csv", "a,b,c\n1,2,3")
# prints: Artifact 'data.csv' was written

workspace.delete("data.csv")
# prints: Artifact 'data.csv' was deleted
```

Supported events: `"write"`, `"delete"`, `"revert"`.

## Artifact Management

```python
# Read an artifact
content = workspace.read("file.txt")  # returns str or None

# Get full artifact metadata
artifact = workspace.get("file.txt")   # returns Artifact or None
if artifact:
    print(artifact.name)
    print(artifact.artifact_type)
    print(artifact.current_version)

# List all artifact names
names = workspace.list()  # ["file.txt", "data.json", ...]

# Delete an artifact
workspace.delete("file.txt")
```

## Filesystem Persistence

When `base_dir` is provided, artifacts are persisted to disk. The workspace creates the directory structure automatically:

```python
workspace = Workspace(base_dir="/tmp/project")

workspace.write("src/main.py", "print('hello')", artifact_type=ArtifactType.CODE)
# Creates /tmp/project/src/main.py on disk
```

Without `base_dir`, artifacts are stored only in memory.

## Knowledge Store Integration

The workspace can integrate with a `KnowledgeStore` for searchable artifact content. When enabled, written artifacts are automatically indexed:

```python
from exo.context._internal.knowledge import KnowledgeStore

knowledge = KnowledgeStore()
workspace = Workspace(base_dir="/tmp/project", knowledge_store=knowledge)

workspace.write("docs/guide.md", "This guide explains how to use the API...")

# The content is now searchable
results = knowledge.search("API usage")
```

See the [Knowledge guide](knowledge.md) for more on the knowledge store.

## Advanced Patterns

### Workspace in Neurons

The built-in `WorkspaceNeuron` reads the workspace to list artifacts in the prompt:

```python
from exo.context import PromptBuilder

# WorkspaceNeuron (priority 30) automatically lists artifacts
builder = PromptBuilder(ctx)
builder.add("workspace")
prompt = builder.build()
# Includes something like:
# "Workspace artifacts: analysis.md (markdown), data.json (json)"
```

### Versioned Experiment Tracking

Use versioning to track iterative refinements:

```python
# Agent refines a solution over multiple steps
for iteration in range(5):
    solution = agent_step(ctx, iteration)
    workspace.write("solution.py", solution, artifact_type=ArtifactType.CODE)
    # Each write creates a new version

# Review all iterations
for v in workspace.version_history("solution.py"):
    print(f"Iteration {v.version}: {len(v.content)} chars")
```

### Observer-Driven Pipelines

Use observers to trigger downstream processing when artifacts change:

```python
async def index_on_write(event: str, name: str, content: str = "", **kw):
    """Re-index workspace contents when files change."""
    if name.endswith(".md") or name.endswith(".txt"):
        knowledge_store.add(name, content)

workspace.on("write", index_on_write)
```

## API Summary

| Symbol | Module | Description |
|--------|--------|-------------|
| `Workspace` | `exo.context` | Versioned artifact storage with filesystem persistence |
| `Workspace.write(name, content, artifact_type)` | | Write or update an artifact (creates new version) |
| `Workspace.read(name)` | | Read artifact content (latest version) |
| `Workspace.get(name)` | | Get full `Artifact` with metadata |
| `Workspace.list()` | | List all artifact names |
| `Workspace.delete(name)` | | Delete an artifact |
| `Workspace.version_history(name)` | | List all `ArtifactVersion`s for an artifact |
| `Workspace.revert_to_version(name, version)` | | Revert to a previous version |
| `Workspace.on(event, callback)` | | Register observer callback for write/delete/revert |
| `ArtifactType` | `exo.context` | Enum: CODE, CSV, IMAGE, JSON, MARKDOWN, TEXT |
