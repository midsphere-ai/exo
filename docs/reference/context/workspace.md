# exo.context.workspace

Persistent artifact storage with versioning and observer notifications.

## Module Path

```python
from exo.context.workspace import Workspace, ArtifactType, Artifact, ArtifactVersion, WorkspaceError
```

---

## ArtifactType

Classification of stored artifacts.

```python
class ArtifactType(StrEnum):
    CODE = "code"
    CSV = "csv"
    IMAGE = "image"
    JSON = "json"
    MARKDOWN = "markdown"
    TEXT = "text"
```

---

## ArtifactVersion

Immutable snapshot of an artifact at a point in time.

### Constructor

```python
ArtifactVersion(content: str, timestamp: float | None = None)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `content` | `str` | *(required)* | The artifact content at this version |
| `timestamp` | `float \| None` | `None` | Unix timestamp (auto-set to `time.time()` if `None`) |

### Properties

| Property | Type | Description |
|---|---|---|
| `content` | `str` | The version content |
| `timestamp` | `float` | Unix timestamp of this version |

---

## Artifact

A named artifact with type, content, and version history.

### Constructor

```python
Artifact(name: str, content: str, artifact_type: ArtifactType = ArtifactType.TEXT)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | Artifact name |
| `content` | `str` | *(required)* | Initial content |
| `artifact_type` | `ArtifactType` | `TEXT` | Artifact classification |

### Properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | Artifact name |
| `artifact_type` | `ArtifactType` | Classification |
| `content` | `str` | Current (latest) content |
| `version_count` | `int` | Number of versions |
| `versions` | `list[ArtifactVersion]` | All versions (oldest first, copy) |

---

## WorkspaceError

Exception raised for workspace operation errors.

```python
class WorkspaceError(Exception): ...
```

---

## Workspace

Persistent artifact storage with versioning and observer notifications.

### Constructor

```python
Workspace(
    workspace_id: str,
    *,
    storage_path: str | Path | None = None,
    knowledge_store: Any | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `workspace_id` | `str` | *(required)* | Unique workspace identifier (must be non-empty) |
| `storage_path` | `str \| Path \| None` | `None` | Root directory for filesystem persistence (created on first write) |
| `knowledge_store` | `Any \| None` | `None` | Optional `KnowledgeStore` for auto-indexing artifacts |

**Raises:** `WorkspaceError` if `workspace_id` is empty.

### Properties

| Property | Type | Description |
|---|---|---|
| `workspace_id` | `str` | The workspace identifier |
| `storage_path` | `Path \| None` | Filesystem storage path |
| `knowledge_store` | `Any \| None` | Attached KnowledgeStore for auto-indexing |

### CRUD Methods

#### write()

```python
async def write(
    self,
    name: str,
    content: str,
    *,
    artifact_type: ArtifactType = ArtifactType.TEXT,
) -> Artifact
```

Write or update an artifact. If `name` exists, a new version is appended and `on_update` fires. Otherwise a new artifact is created and `on_create` fires. Persists to filesystem if `storage_path` is set. Auto-indexes in knowledge store if attached.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | *(required)* | Artifact name (must be non-empty) |
| `content` | `str` | *(required)* | Content to write |
| `artifact_type` | `ArtifactType` | `TEXT` | Artifact type classification |

**Returns:** The created or updated `Artifact`.

**Raises:** `WorkspaceError` if name is empty.

#### read()

```python
def read(self, name: str) -> str | None
```

Read current content of an artifact by name. Returns `None` if missing.

#### get()

```python
def get(self, name: str) -> Artifact | None
```

Get the full `Artifact` object. Returns `None` if missing.

#### list()

```python
def list(self, *, artifact_type: ArtifactType | None = None) -> list[Artifact]
```

List all artifacts, optionally filtered by type.

#### delete()

```python
async def delete(self, name: str) -> bool
```

Delete an artifact by name. Returns `True` if deleted, `False` if missing. Fires `on_delete` observer.

### Versioning Methods

#### version_history()

```python
def version_history(self, name: str) -> list[ArtifactVersion]
```

Return the version history of an artifact. Empty list if missing.

#### revert_to_version()

```python
def revert_to_version(self, name: str, version: int) -> Artifact
```

Revert an artifact to a previous version (0-indexed). Creates a new version whose content matches the target version.

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Artifact name |
| `version` | `int` | 0-based version index to revert to |

**Raises:** `WorkspaceError` if name or version is invalid.

### Observer Methods

#### on()

```python
def on(self, event: str, callback: ObserverCallback) -> Workspace
```

Register an observer callback for an event. Returns `self` for chaining.

| Event | Trigger |
|---|---|
| `"on_create"` | New artifact created |
| `"on_update"` | Existing artifact updated |
| `"on_delete"` | Artifact deleted |

**Callback signature:** `async (event_name: str, artifact: Artifact) -> None`

### Dunder Methods

| Method | Description |
|---|---|
| `__len__` | Number of artifacts |
| `__repr__` | `Workspace(id='ws-1', artifacts=3)` |

### Example

```python
import asyncio
from exo.context.workspace import Workspace, ArtifactType

async def main():
    ws = Workspace("my-workspace", storage_path="/tmp/workspace")

    # Register observer
    async def on_change(event, artifact):
        print(f"{event}: {artifact.name}")

    ws.on("on_create", on_change)
    ws.on("on_update", on_change)

    # Write artifacts
    art = await ws.write("main.py", "print('hello')", artifact_type=ArtifactType.CODE)
    # Output: on_create: main.py

    # Update (creates version 2)
    await ws.write("main.py", "print('hello world')")
    # Output: on_update: main.py

    # Read
    content = ws.read("main.py")
    print(content)  # "print('hello world')"

    # Version history
    versions = ws.version_history("main.py")
    print(len(versions))  # 2

    # Revert to version 0
    ws.revert_to_version("main.py", 0)
    print(ws.read("main.py"))  # "print('hello')"

    # List and filter
    all_arts = ws.list()
    code_arts = ws.list(artifact_type=ArtifactType.CODE)

    # Delete
    deleted = await ws.delete("main.py")
    print(deleted)  # True

asyncio.run(main())
```
