"""Local filesystem watcher for skill hot-reloading.

Uses ``watchfiles`` to detect changes to skill files in a local directory,
then diffs against a snapshot to produce :class:`SkillChangeEvent` batches.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from watchfiles import Change, awatch

from exo.skills import (
    SKILL_FILENAMES,
    Skill,
    SkillChangeEvent,
    SkillWatcher,
    _collect_skills,
    _skill_fingerprint,
)

logger = logging.getLogger(__name__)


def _diff_snapshots(
    old: dict[str, Skill],
    new: dict[str, Skill],
    source_path: str,
) -> list[SkillChangeEvent]:
    """Compare two skill snapshots and return a list of change events.

    Args:
        old: Previous snapshot mapping skill name to Skill.
        new: Current snapshot mapping skill name to Skill.
        source_path: The root directory being watched (for event metadata).

    Returns:
        List of :class:`SkillChangeEvent` describing added, removed, and
        modified skills.  Returns an empty list when the snapshots are
        identical.
    """
    events: list[SkillChangeEvent] = []

    old_names = set(old)
    new_names = set(new)

    # Added skills
    for name in sorted(new_names - old_names):
        events.append(
            SkillChangeEvent(
                kind="added",
                skill_name=name,
                skill=new[name],
                source_path=source_path,
            )
        )

    # Removed skills
    for name in sorted(old_names - new_names):
        events.append(
            SkillChangeEvent(
                kind="removed",
                skill_name=name,
                skill=None,
                source_path=source_path,
            )
        )

    # Modified skills
    for name in sorted(old_names & new_names):
        if _skill_fingerprint(old[name]) != _skill_fingerprint(new[name]):
            events.append(
                SkillChangeEvent(
                    kind="modified",
                    skill_name=name,
                    skill=new[name],
                    source_path=source_path,
                )
            )

    return events


def _skill_file_filter(change: Change, path: str) -> bool:
    """Filter for ``watchfiles.awatch`` that only passes skill file changes.

    Args:
        change: The type of filesystem change (added, modified, deleted).
        path: Absolute path to the changed file.

    Returns:
        ``True`` if the changed file is a recognised skill filename.
    """
    return Path(path).name in SKILL_FILENAMES


class LocalFileWatcher(SkillWatcher):
    """Watch a local directory for skill file changes using ``watchfiles``.

    On the first iteration of :meth:`watch`, an initial snapshot is taken via
    :func:`_collect_skills`.  Subsequent filesystem events trigger a re-scan
    and diff against the previous snapshot, yielding batches of
    :class:`SkillChangeEvent` objects.

    Args:
        path: Root directory to watch for skill files.
        debounce_ms: Minimum quiet period (in milliseconds) before a batch
            of filesystem events is processed.  Defaults to 500.

    Example::

        watcher = LocalFileWatcher("./skills")
        async for batch in watcher.watch():
            for event in batch:
                print(event.kind, event.skill_name)
    """

    def __init__(self, path: str | Path, debounce_ms: int = 500) -> None:
        self._path = Path(path).expanduser().resolve()
        self._debounce_ms = debounce_ms
        self._stop_event = asyncio.Event()
        self._snapshot: dict[str, Skill] = {}

    async def watch(self) -> AsyncIterator[list[SkillChangeEvent]]:
        """Yield batches of skill change events as they occur.

        Takes an initial snapshot, then watches the directory for changes.
        The iterator terminates when :meth:`stop` is called.
        """
        self._snapshot = _collect_skills(self._path)
        source = str(self._path)
        logger.debug(
            "LocalFileWatcher started on %s — initial snapshot has %d skill(s)",
            source,
            len(self._snapshot),
        )

        async for _changes in awatch(
            self._path,
            watch_filter=_skill_file_filter,
            stop_event=self._stop_event,
            debounce=self._debounce_ms,
        ):
            new_snapshot = _collect_skills(self._path)
            events = _diff_snapshots(self._snapshot, new_snapshot, source)

            if events:
                logger.debug(
                    "LocalFileWatcher detected %d change(s) in %s",
                    len(events),
                    source,
                )
                self._snapshot = new_snapshot
                yield events
            else:
                # Filesystem event fired but skills are identical after re-scan
                # (e.g. a non-semantic whitespace change).
                self._snapshot = new_snapshot

    async def stop(self) -> None:
        """Signal the watcher to stop.

        Causes the ``awatch`` loop in :meth:`watch` to terminate, which in
        turn ends the async iterator.
        """
        logger.debug("LocalFileWatcher stopping for %s", self._path)
        self._stop_event.set()
