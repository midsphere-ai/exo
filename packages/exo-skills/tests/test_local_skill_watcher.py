"""Tests for LocalFileWatcher — local filesystem skill hot-reloading."""

from __future__ import annotations

import asyncio
from pathlib import Path

from exo.skills import Skill, SkillChangeEvent
from exo_skills.watchers.local import LocalFileWatcher, _diff_snapshots

# ---------------------------------------------------------------------------
# _diff_snapshots helper
# ---------------------------------------------------------------------------


class TestDiffSnapshots:
    def test_empty_both(self) -> None:
        assert _diff_snapshots({}, {}, "/src") == []

    def test_added(self) -> None:
        new = {"s1": Skill(name="s1", description="hello")}
        events = _diff_snapshots({}, new, "/src")
        assert len(events) == 1
        assert events[0].kind == "added"
        assert events[0].skill_name == "s1"
        assert events[0].skill is not None

    def test_removed(self) -> None:
        old = {"s1": Skill(name="s1")}
        events = _diff_snapshots(old, {}, "/src")
        assert len(events) == 1
        assert events[0].kind == "removed"
        assert events[0].skill is None

    def test_modified(self) -> None:
        old = {"s1": Skill(name="s1", description="v1")}
        new = {"s1": Skill(name="s1", description="v2")}
        events = _diff_snapshots(old, new, "/src")
        assert len(events) == 1
        assert events[0].kind == "modified"

    def test_no_change(self) -> None:
        s = Skill(name="s1", description="same")
        events = _diff_snapshots({"s1": s}, {"s1": s}, "/src")
        assert events == []

    def test_mixed(self) -> None:
        old = {"keep": Skill(name="keep"), "remove": Skill(name="remove")}
        new = {"keep": Skill(name="keep"), "add": Skill(name="add")}
        events = _diff_snapshots(old, new, "/src")
        kinds = {e.kind for e in events}
        assert kinds == {"added", "removed"}


# ---------------------------------------------------------------------------
# LocalFileWatcher
# ---------------------------------------------------------------------------


def _background(coro):
    """Create a background task and return it (prevents GC collection)."""
    return asyncio.create_task(coro)


class TestLocalFileWatcher:
    async def test_detects_added_skill(self, tmp_path: Path) -> None:
        """Write a skill file after starting the watcher — expect an 'added' event."""
        # Pre-create the subdirectory so watchfiles only needs to detect the new file
        skill_dir = tmp_path / "new_skill"
        skill_dir.mkdir()
        watcher = LocalFileWatcher(tmp_path, debounce_ms=100)

        async def write_skill() -> None:
            await asyncio.sleep(0.3)
            (skill_dir / "skill.md").write_text(
                "---\nname: new_skill\ndescription: Added\n---\nBody."
            )

        task = _background(write_skill())

        batch = None
        async for b in watcher.watch():
            batch = b
            await watcher.stop()
            break

        await task
        assert batch is not None
        assert any(e.kind == "added" and e.skill_name == "new_skill" for e in batch)

    async def test_detects_modified_skill(self, tmp_path: Path) -> None:
        """Modify an existing skill file — expect a 'modified' event."""
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "skill.md"
        skill_file.write_text("---\nname: my_skill\ndescription: v1\n---\nBody.")

        watcher = LocalFileWatcher(tmp_path, debounce_ms=100)

        async def modify_skill() -> None:
            await asyncio.sleep(0.3)
            skill_file.write_text("---\nname: my_skill\ndescription: v2\n---\nBody.")

        task = _background(modify_skill())

        batch = None
        async for b in watcher.watch():
            batch = b
            await watcher.stop()
            break

        await task
        assert batch is not None
        assert any(e.kind == "modified" and e.skill_name == "my_skill" for e in batch)

    async def test_detects_removed_skill(self, tmp_path: Path) -> None:
        """Delete a skill file — expect a 'removed' event."""
        skill_dir = tmp_path / "doomed"
        skill_dir.mkdir()
        skill_file = skill_dir / "skill.md"
        skill_file.write_text("---\nname: doomed\n---\nBody.")

        watcher = LocalFileWatcher(tmp_path, debounce_ms=100)

        async def remove_skill() -> None:
            await asyncio.sleep(0.3)
            skill_file.unlink()

        task = _background(remove_skill())

        batch = None
        async for b in watcher.watch():
            batch = b
            await watcher.stop()
            break

        await task
        assert batch is not None
        assert any(e.kind == "removed" and e.skill_name == "doomed" for e in batch)

    async def test_stop_terminates_watch(self, tmp_path: Path) -> None:
        """Calling stop() causes watch() to return without blocking forever."""
        watcher = LocalFileWatcher(tmp_path, debounce_ms=100)

        async def stop_soon() -> None:
            await asyncio.sleep(0.2)
            await watcher.stop()

        task = _background(stop_soon())

        batches: list[list[SkillChangeEvent]] = []
        async for b in watcher.watch():
            batches.append(b)

        await task
        # Should have terminated cleanly with no batches (no changes made)
        assert batches == []

    async def test_ignores_non_skill_files(self, tmp_path: Path) -> None:
        """Changes to non-skill files should not produce events."""
        watcher = LocalFileWatcher(tmp_path, debounce_ms=100)

        async def write_non_skill() -> None:
            await asyncio.sleep(0.2)
            (tmp_path / "readme.md").write_text("# Not a skill")
            await asyncio.sleep(0.5)
            await watcher.stop()

        task = _background(write_non_skill())

        batches: list[list[SkillChangeEvent]] = []
        async for b in watcher.watch():
            batches.append(b)

        await task
        assert batches == []
