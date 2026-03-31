"""GitHub polling watcher for skill hot-reloading.

Polls a GitHub repository on a configurable interval via ``git pull --ff-only``,
then diffs against a snapshot to produce :class:`SkillChangeEvent` batches.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from exo.skills import (
    DEFAULT_CACHE_DIR,
    Skill,
    SkillChangeEvent,
    SkillError,
    SkillWatcher,
    _clone_github,
    _collect_skills,
    _skill_fingerprint,
    parse_github_url,
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
        source_path: The source URL being watched (for event metadata).

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


class GitHubPollingWatcher(SkillWatcher):
    """Watch a GitHub repository for skill changes via periodic ``git pull``.

    On the first iteration of :meth:`watch`, the repository is shallow-cloned
    (if not already cached) and an initial snapshot is taken via
    :func:`_collect_skills`.  On each subsequent poll cycle the watcher runs
    ``git pull --ff-only`` to fetch upstream changes, re-scans the skill
    directory, and yields any detected differences as a batch of
    :class:`SkillChangeEvent` objects.

    Args:
        source_url: GitHub URL to watch (e.g.
            ``https://github.com/owner/repo/tree/branch/subdir``).
        poll_interval: Seconds between polls.  Defaults to 300 (5 minutes).
        cache_dir: Local directory for cloned repos.  Defaults to
            :data:`DEFAULT_CACHE_DIR` (``~/.exo/skills``).

    Raises:
        SkillError: If *source_url* is not a valid GitHub URL.

    Example::

        watcher = GitHubPollingWatcher(
            "https://github.com/acme/skills/tree/main/agents",
            poll_interval=60.0,
        )
        async for batch in watcher.watch():
            for event in batch:
                print(event.kind, event.skill_name)
    """

    def __init__(
        self,
        source_url: str,
        poll_interval: float = 300.0,
        cache_dir: Path | None = None,
    ) -> None:
        parsed = parse_github_url(source_url)
        if parsed is None:
            raise SkillError(f"Invalid GitHub URL: {source_url}")

        self._source_url = source_url
        self._parsed = parsed
        self._poll_interval = poll_interval
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._stop_event = asyncio.Event()
        self._snapshot: dict[str, Skill] = {}

    async def watch(self) -> AsyncIterator[list[SkillChangeEvent]]:
        """Yield batches of skill change events as they are detected.

        Performs an initial clone, takes a snapshot, then enters a poll loop
        that runs ``git pull --ff-only`` at the configured interval.  The
        iterator terminates when :meth:`stop` is called.
        """
        loop = asyncio.get_event_loop()

        # Initial clone (blocking I/O, run in thread executor).
        skills_dir: Path = await loop.run_in_executor(
            None, _clone_github, self._parsed, self._cache_dir
        )

        # The repo root is always cache_dir/owner/repo/branch, regardless of
        # any subdirectory the parsed URL might point to.
        clone_dir_root = (
            self._cache_dir / self._parsed["owner"] / self._parsed["repo"] / self._parsed["branch"]
        )

        self._snapshot = _collect_skills(skills_dir)
        logger.debug(
            "GitHubPollingWatcher started on %s — initial snapshot has %d skill(s)",
            self._source_url,
            len(self._snapshot),
        )

        while True:
            # Sleep until poll interval elapses or stop is requested.
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._poll_interval)
                # stop_event was set — exit the watcher.
                logger.debug("GitHubPollingWatcher stopping for %s", self._source_url)
                return
            except TimeoutError:
                pass  # Poll interval elapsed, continue with pull.

            # Pull latest changes.
            try:
                proc = await asyncio.create_subprocess_exec(
                    "git",
                    "-C",
                    str(clone_dir_root),
                    "pull",
                    "--ff-only",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()

                if proc.returncode != 0:
                    logger.warning(
                        "git pull failed for %s (exit %d): %s",
                        self._source_url,
                        proc.returncode,
                        stderr.decode(errors="replace").strip(),
                    )
                    continue
                else:
                    logger.debug(
                        "git pull for %s: %s",
                        self._source_url,
                        stdout.decode(errors="replace").strip(),
                    )
            except OSError:
                logger.exception("Failed to execute git pull for %s", self._source_url)
                continue

            # Re-scan and diff.
            new_snapshot = _collect_skills(skills_dir)
            events = _diff_snapshots(self._snapshot, new_snapshot, self._source_url)

            if events:
                logger.debug(
                    "GitHubPollingWatcher detected %d change(s) in %s",
                    len(events),
                    self._source_url,
                )
                self._snapshot = new_snapshot
                yield events
            else:
                self._snapshot = new_snapshot

    async def stop(self) -> None:
        """Signal the watcher to stop.

        Sets the internal stop event, causing the poll loop in :meth:`watch`
        to terminate on its next cycle.
        """
        logger.debug("GitHubPollingWatcher stop requested for %s", self._source_url)
        self._stop_event.set()
