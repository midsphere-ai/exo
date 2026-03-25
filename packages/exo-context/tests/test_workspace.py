"""Tests for exo.context.workspace — Workspace + artifact system."""

from __future__ import annotations

import pytest

from exo.context.workspace import (  # pyright: ignore[reportMissingImports]
    Artifact,
    ArtifactType,
    ArtifactVersion,
    Workspace,
    WorkspaceError,
)

# ── ArtifactType ─────────────────────────────────────────────────────


class TestArtifactType:
    def test_all_values(self) -> None:
        assert set(ArtifactType) == {"text", "code", "markdown", "json", "csv", "image"}

    def test_is_str_enum(self) -> None:
        assert isinstance(ArtifactType.TEXT, str)
        assert ArtifactType.CODE == "code"


# ── ArtifactVersion ──────────────────────────────────────────────────


class TestArtifactVersion:
    def test_creation(self) -> None:
        v = ArtifactVersion("hello", timestamp=1000.0)
        assert v.content == "hello"
        assert v.timestamp == 1000.0

    def test_auto_timestamp(self) -> None:
        v = ArtifactVersion("data")
        assert v.timestamp > 0

    def test_repr(self) -> None:
        v = ArtifactVersion("abc", timestamp=100.0)
        assert "len=3" in repr(v)


# ── Artifact ─────────────────────────────────────────────────────────


class TestArtifact:
    def test_creation(self) -> None:
        a = Artifact("readme.md", "# Hello", ArtifactType.MARKDOWN)
        assert a.name == "readme.md"
        assert a.content == "# Hello"
        assert a.artifact_type == ArtifactType.MARKDOWN
        assert a.version_count == 1

    def test_default_type(self) -> None:
        a = Artifact("notes", "some notes")
        assert a.artifact_type == ArtifactType.TEXT

    def test_add_version(self) -> None:
        a = Artifact("file", "v1")
        a._add_version("v2")
        assert a.content == "v2"
        assert a.version_count == 2
        assert a.versions[0].content == "v1"
        assert a.versions[1].content == "v2"

    def test_repr(self) -> None:
        a = Artifact("file.py", "pass", ArtifactType.CODE)
        assert "file.py" in repr(a)
        assert "code" in repr(a)


# ── Workspace init ───────────────────────────────────────────────────


class TestWorkspaceInit:
    def test_creation(self) -> None:
        ws = Workspace("ws-1")
        assert ws.workspace_id == "ws-1"
        assert len(ws) == 0
        assert ws.storage_path is None

    def test_empty_id_raises(self) -> None:
        with pytest.raises(WorkspaceError, match="workspace_id"):
            Workspace("")

    def test_with_storage_path(self, tmp_path: object) -> None:
        ws = Workspace("ws-2", storage_path=tmp_path)
        assert ws.storage_path is not None

    def test_repr(self) -> None:
        ws = Workspace("ws-3")
        assert "ws-3" in repr(ws)
        assert "artifacts=0" in repr(ws)


# ── Write / Read / List / Delete ─────────────────────────────────────


class TestWorkspaceCRUD:
    async def test_write_and_read(self) -> None:
        ws = Workspace("ws")
        await ws.write("readme", "# Hello")
        assert ws.read("readme") == "# Hello"

    async def test_read_missing(self) -> None:
        ws = Workspace("ws")
        assert ws.read("missing") is None

    async def test_get_artifact(self) -> None:
        ws = Workspace("ws")
        await ws.write("file", "data", artifact_type=ArtifactType.JSON)
        art = ws.get("file")
        assert art is not None
        assert art.artifact_type == ArtifactType.JSON

    async def test_get_missing(self) -> None:
        ws = Workspace("ws")
        assert ws.get("missing") is None

    async def test_write_empty_name_raises(self) -> None:
        ws = Workspace("ws")
        with pytest.raises(WorkspaceError, match="name"):
            await ws.write("", "content")

    async def test_list_all(self) -> None:
        ws = Workspace("ws")
        await ws.write("a", "1")
        await ws.write("b", "2")
        assert len(ws.list()) == 2

    async def test_list_by_type(self) -> None:
        ws = Workspace("ws")
        await ws.write("code.py", "pass", artifact_type=ArtifactType.CODE)
        await ws.write("notes.md", "# Notes", artifact_type=ArtifactType.MARKDOWN)
        assert len(ws.list(artifact_type=ArtifactType.CODE)) == 1
        assert ws.list(artifact_type=ArtifactType.CODE)[0].name == "code.py"

    async def test_delete(self) -> None:
        ws = Workspace("ws")
        await ws.write("file", "data")
        assert len(ws) == 1
        result = await ws.delete("file")
        assert result is True
        assert len(ws) == 0
        assert ws.read("file") is None

    async def test_delete_missing(self) -> None:
        ws = Workspace("ws")
        result = await ws.delete("missing")
        assert result is False

    async def test_overwrite_adds_version(self) -> None:
        ws = Workspace("ws")
        await ws.write("file", "v1")
        await ws.write("file", "v2")
        assert ws.read("file") == "v2"
        assert ws.get("file") is not None
        assert ws.get("file").version_count == 2  # type: ignore[union-attr]


# ── Versioning ───────────────────────────────────────────────────────


class TestVersioning:
    async def test_version_history(self) -> None:
        ws = Workspace("ws")
        await ws.write("f", "v1")
        await ws.write("f", "v2")
        await ws.write("f", "v3")
        history = ws.version_history("f")
        assert len(history) == 3
        assert history[0].content == "v1"
        assert history[2].content == "v3"

    async def test_version_history_missing(self) -> None:
        ws = Workspace("ws")
        assert ws.version_history("missing") == []

    async def test_revert_to_version(self) -> None:
        ws = Workspace("ws")
        await ws.write("f", "v1")
        await ws.write("f", "v2")
        await ws.write("f", "v3")
        ws.revert_to_version("f", 0)
        assert ws.read("f") == "v1"
        # Revert adds a new version (doesn't truncate)
        assert ws.get("f") is not None
        assert ws.get("f").version_count == 4  # type: ignore[union-attr]

    async def test_revert_missing_raises(self) -> None:
        ws = Workspace("ws")
        with pytest.raises(WorkspaceError, match="not found"):
            ws.revert_to_version("missing", 0)

    async def test_revert_invalid_version_raises(self) -> None:
        ws = Workspace("ws")
        await ws.write("f", "v1")
        with pytest.raises(WorkspaceError, match="out of range"):
            ws.revert_to_version("f", 5)

    async def test_revert_negative_version_raises(self) -> None:
        ws = Workspace("ws")
        await ws.write("f", "v1")
        with pytest.raises(WorkspaceError, match="out of range"):
            ws.revert_to_version("f", -1)


# ── Observer notifications ───────────────────────────────────────────


class TestObservers:
    async def test_on_create_fires(self) -> None:
        events: list[tuple[str, str]] = []

        async def handler(event: str, artifact: Artifact) -> None:
            events.append((event, artifact.name))

        ws = Workspace("ws")
        ws.on("on_create", handler)
        await ws.write("file", "data")
        assert events == [("on_create", "file")]

    async def test_on_update_fires(self) -> None:
        events: list[tuple[str, str]] = []

        async def handler(event: str, artifact: Artifact) -> None:
            events.append((event, artifact.name))

        ws = Workspace("ws")
        ws.on("on_update", handler)
        await ws.write("file", "v1")  # create — no update event
        await ws.write("file", "v2")  # update
        assert events == [("on_update", "file")]

    async def test_on_delete_fires(self) -> None:
        events: list[tuple[str, str]] = []

        async def handler(event: str, artifact: Artifact) -> None:
            events.append((event, artifact.name))

        ws = Workspace("ws")
        ws.on("on_delete", handler)
        await ws.write("file", "data")
        await ws.delete("file")
        assert events == [("on_delete", "file")]

    async def test_multiple_observers(self) -> None:
        calls: list[int] = []

        async def cb1(event: str, artifact: Artifact) -> None:
            calls.append(1)

        async def cb2(event: str, artifact: Artifact) -> None:
            calls.append(2)

        ws = Workspace("ws")
        ws.on("on_create", cb1).on("on_create", cb2)
        await ws.write("file", "data")
        assert calls == [1, 2]

    async def test_chaining(self) -> None:
        async def noop(event: str, artifact: Artifact) -> None:
            pass

        ws = Workspace("ws")
        result = ws.on("on_create", noop)
        assert result is ws  # chaining returns self

    async def test_no_observer_for_event(self) -> None:
        ws = Workspace("ws")
        # No error when no observers registered
        await ws.write("file", "data")


# ── Filesystem persistence ───────────────────────────────────────────


class TestFilesystemPersistence:
    async def test_write_persists(self, tmp_path: object) -> None:
        from pathlib import Path

        path = Path(str(tmp_path))
        ws = Workspace("ws", storage_path=path)
        await ws.write("report", "hello world")

        content_file = path / "report" / "content"
        assert content_file.exists()
        assert content_file.read_text() == "hello world"

        meta_file = path / "report" / "meta.json"
        assert meta_file.exists()
        import json

        meta = json.loads(meta_file.read_text())
        assert meta["name"] == "report"
        assert meta["artifact_type"] == "text"
        assert meta["version_count"] == 1

    async def test_delete_removes_files(self, tmp_path: object) -> None:
        from pathlib import Path

        path = Path(str(tmp_path))
        ws = Workspace("ws", storage_path=path)
        await ws.write("file", "data")
        assert (path / "file").exists()
        await ws.delete("file")
        assert not (path / "file").exists()

    async def test_overwrite_updates_file(self, tmp_path: object) -> None:
        from pathlib import Path

        path = Path(str(tmp_path))
        ws = Workspace("ws", storage_path=path)
        await ws.write("file", "v1")
        await ws.write("file", "v2")
        content_file = path / "file" / "content"
        assert content_file.read_text() == "v2"

    async def test_no_storage_path_no_files(self) -> None:
        ws = Workspace("ws")  # no storage_path
        await ws.write("file", "data")
        # Should not raise — just skips persistence
        assert ws.read("file") == "data"

    async def test_revert_persists(self, tmp_path: object) -> None:
        from pathlib import Path

        path = Path(str(tmp_path))
        ws = Workspace("ws", storage_path=path)
        await ws.write("file", "v1")
        await ws.write("file", "v2")
        ws.revert_to_version("file", 0)
        content_file = path / "file" / "content"
        assert content_file.read_text() == "v1"
