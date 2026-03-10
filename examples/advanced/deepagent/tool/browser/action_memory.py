"""
Action memory utilities for the browser-use agent.

Maintains a per-site summary of actions executed by the agent and provides
helpers to surface that history back to the LLM interaction loop. Optionally
persists the data to disk so future runs can build on prior browsing steps.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlsplit, urlunsplit

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

try:  # Soft import for typing without creating a runtime dependency cycle
    from browser_use.agent.message_manager.service import MessageManager
except Exception:  # pragma: no cover - browser_use is available at runtime
    MessageManager = None  # type: ignore


def _truncate(value: str, max_length: int) -> str:
    """Return a truncated representation limited to `max_length` characters."""
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."


def _serialise_params(params: Dict[str, Any], max_length: int = 160) -> str:
    """Serialise action parameters into a compact JSON string."""
    if not params:
        return ""
    try:
        encoded = json.dumps(params, ensure_ascii=True, separators=(",", ":"))
    except (TypeError, ValueError):
        encoded = str(params)
    return _truncate(encoded, max_length)


def _coerce_text(content: Any) -> str:
    """Convert a langchain response into a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    if hasattr(content, "content"):
        return _coerce_text(content.content)
    return str(content).strip()


@dataclass
class SiteMemoryRecord:
    """Single summary entry per visited site."""

    summary: str
    update_count: int = 1
    last_updated: datetime = field(default_factory=datetime.utcnow)


class BrowserActionMemory:
    """
    Tracks per-site browser actions and keeps a rendered summary message that can be
    inserted into the LLM conversation when a site is revisited.
    """

    def __init__(
        self,
        *,
        generic_domains: Optional[Iterable[str]] = None,
        storage_path: str | Path | None = None,
        load_existing: bool = True,
    ) -> None:
        self._history: Dict[str, SiteMemoryRecord] = {}
        self._generic_domains = {
            domain.lower()
            for domain in generic_domains
            or (
                "google.com",
                "www.google.com",
                "youtube.com",
                "www.youtube.com",
            )
        }
        self._message_manager: MessageManager | None = None
        self._current_prompt_site: str | None = None
        self._last_rendered_summary: str | None = None
        self._storage_path: Path | None = None
        self._existing_site_hits: int = 0
        self._session_observations: dict[str, list[str]] = {}

        if storage_path:
            resolved_path = Path(storage_path)
            if resolved_path.suffix == "":
                resolved_path.mkdir(parents=True, exist_ok=True)
                resolved_path = resolved_path / "site_action_memory.jsonl"
            else:
                resolved_path.parent.mkdir(parents=True, exist_ok=True)
            self._storage_path = resolved_path
            if load_existing:
                self._load_existing()
            else:
                """logger.info(
                    "[site-memory] Skipping load of existing memory store at %s",
                    resolved_path,
                )"""
            self._ensure_store_exists()

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    @property
    def history(self) -> Dict[str, SiteMemoryRecord]:
        """Expose the underlying action summary history."""
        return self._history

    def bind_message_manager(self, manager: MessageManager) -> None:
        """Attach the live message manager so summaries can be injected."""
        self._message_manager = manager

    async def update_site_summary(
        self,
        *,
        url: str | None,
        action_name: str,
        params: Dict[str, Any],
        outcome: str,
        summarizer: BaseChatModel | None = None,
    ) -> None:
        """
        Update the per-site summary for `url`. If the URL resolves to a site we care
        about, merge the new observation into the running summary, refresh the prompt,
        and persist to disk if configured.
        """
        site_key = self._normalize_site_key(url)
        if site_key is None:
            return
        if self._should_skip_prompt(site_key):
            return

        observation = self._format_observation(action_name, params, outcome)
        record = self._history.get(site_key)
        self._session_observations.setdefault(site_key, []).append(observation)

        if record:
            summary = await self._generate_summary(
                summarizer=summarizer,
                existing_summary=record.summary,
                new_observation=observation,
            )
            record.summary = summary
            record.last_updated = datetime.now(timezone.utc)
            record.update_count += 1
        else:
            summary = await self._generate_summary(
                summarizer=summarizer,
                existing_summary="",
                new_observation=observation,
            )
            self._history[site_key] = SiteMemoryRecord(
                summary=summary,
                update_count=1,
                last_updated=datetime.now(timezone.utc),
            )

        self._persist_all()
        if record:
            self._existing_site_hits += 1
            """logger.info(
                "[site-memory] Revisited existing site %s (updates this run: %d)",
                site_key,
                record.update_count,
            )"""
        else:
            #logger.info("[site-memory] Created first summary for %s", site_key)
            pass
        self._refresh_summary(site_key)
        #logger.info("[site-memory] refresh_summary done")

    def clear_prompt(self) -> None:
        """Remove any injected memory prompt from the message history."""
        if not self._message_manager:
            return
        self._remove_memory_messages()
        self._current_prompt_site = None
        self._last_rendered_summary = None

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _normalize_site_key(self, url: str | None) -> Optional[str]:
        """Convert a URL into a canonical key identifying the visited site."""
        if not url:
            return None
        parsed = urlsplit(url)
        if not parsed.netloc:
            return None

        hostname = parsed.netloc.lower()
        path = parsed.path or ""
        path = path.rstrip("/") or "/"

        bare_hostname = hostname[4:] if hostname.startswith("www.") else hostname
        if hostname in self._generic_domains or bare_hostname in self._generic_domains:
            path_lower = path.lower()
            if bare_hostname == "google.com" and (
                path_lower.startswith("/maps") or hostname.startswith("maps.")
            ):
                pass
            else:
                return None
        normalized = urlunsplit((parsed.scheme or "https", hostname, path, "", ""))
        return normalized

    def _refresh_summary(self, site_key: str) -> None:
        """Update the rendered memory summary for the provided site."""
        if self._should_skip_prompt(site_key):
            if self._message_manager and self._current_prompt_site == site_key:
                self._remove_memory_messages()
            self._current_prompt_site = None
            self._last_rendered_summary = None
            return

        summary = self._build_summary(site_key)

        if not self._message_manager:
            self._current_prompt_site = site_key if summary else None
            self._last_rendered_summary = summary
            return

        if (
            self._current_prompt_site == site_key
            and self._last_rendered_summary == summary
        ):
            return

        self._remove_memory_messages()

        if not summary:
            self._current_prompt_site = None
            self._last_rendered_summary = None
            return

        memory_message = HumanMessage(
            content=(
                f"Known findings already collected on {site_key}:\n{summary}\n"
                "Before proposing new steps, review this history and continue from the latest useful result. "
                "Do not repeat any listed actions unless you explicitly need to redo them. "
                "If there is nothing new to do, just proceed to a different website to explore other information "
                "Leverage the prior outcomes to move the task forward efficiently."
            )
        )
        # Message is appended to history and token count updated automatically.
        self._message_manager._add_message_with_tokens(  # type: ignore[attr-defined]
            memory_message,
            message_type="site_memory",
        )
        """logger.info(
            "[site-memory] Injected summary into prompt for %s:\n%s",
            site_key,
            summary,
        )"""
        self._current_prompt_site = site_key
        self._last_rendered_summary = summary

    def _build_summary(self, site_key: str) -> str:
        """Return the stored summary text for the site."""
        record = self._history.get(site_key)
        if not record:
            return ""
        return record.summary.strip()

    def _remove_memory_messages(self) -> None:
        """Delete any existing memory prompts from the message history."""
        if not self._message_manager:
            return

        history = self._message_manager.state.history  # type: ignore[attr-defined]
        retained: list[Any] = []
        for managed_message in history.messages:
            if managed_message.metadata.message_type == "site_memory":
                history.current_tokens -= managed_message.metadata.tokens
                continue
            retained.append(managed_message)
        history.messages = retained

    def _should_skip_prompt(self, site_key: str) -> bool:
        """Return True when we never want to inject memory for the given site."""
        try:
            parsed = urlsplit(site_key)
        except ValueError:
            return False
        hostname = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        if hostname.endswith("google.com"):
            if hostname.startswith("maps.") or path.startswith("/maps"):
                return True
        return False

    def _format_observation(
        self,
        action_name: str,
        params: Dict[str, Any],
        outcome: str,
    ) -> str:
        params_text = _serialise_params(params)
        pieces = [f"Action: {action_name}"]
        if params_text:
            pieces.append(f"Parameters: {params_text}")
        pieces.append(f"Result: {outcome.strip()}")
        return "\n".join(pieces).strip()

    async def _generate_summary(
        self,
        *,
        summarizer: BaseChatModel | None,
        existing_summary: str,
        new_observation: str,
    ) -> str:
        """Combine the existing summary with a new observation."""
        new_observation = new_observation.strip()
        if not existing_summary:
            return new_observation

        if summarizer is None:
            if new_observation in existing_summary:
                return existing_summary
            return f"{existing_summary}\n- {new_observation}"

        try:
            messages = [
                SystemMessage(
                    content=(
                        "You maintain a concise running summary of what the agent has already "
                        "discovered on a website. Keep the summary factual, avoid repetition. "
                        "Do not compress too much information but only compress information of repetitive nature."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Current summary:\n{existing_summary or '[empty]'}\n\n"
                        f"New observation:\n{new_observation}\n\n"
                        "Update the summary to reflect all known information."
                    )
                ),
            ]
            response = await summarizer.ainvoke(messages)
            text = _coerce_text(response)
            if text:
                return text
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to summarise site memory: %s", exc, exc_info=True)

        if new_observation in existing_summary:
            return existing_summary
        return f"{existing_summary}\n- {new_observation}"

    def _persist_all(self) -> None:
        """Persist the entire site summary map to disk as JSONL."""
        if not self._storage_path:
            return

        temp_path = self._storage_path.with_name(self._storage_path.name + ".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                for site, record in self._history.items():
                    payload = {
                        "site": site,
                        "summary": record.summary,
                        "update_count": record.update_count,
                        "timestamp": record.last_updated.isoformat(),
                    }
                    json.dump(payload, handle, ensure_ascii=True, separators=(",", ":"))
                    handle.write("\n")
            temp_path.replace(self._storage_path)
        except OSError as exc:
            logger.debug("Could not persist action memory store: %s", exc, exc_info=True)
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    def _load_existing(self) -> None:
        """Load persisted summary entries from disk into memory."""
        if not self._storage_path or not self._storage_path.exists():
            return

        try:
            with self._storage_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("Skipping malformed action memory line: %s", line)
                        continue

                    site_key = payload.get("site")
                    if not site_key:
                        continue

                    timestamp_val = payload.get("timestamp")
                    try:
                        parsed_timestamp = (
                            datetime.fromisoformat(timestamp_val)
                            if timestamp_val
                            else datetime.utcnow()
                        )
                    except (TypeError, ValueError):
                        parsed_timestamp = datetime.utcnow()

                    summary_text = payload.get("summary")
                    if summary_text is None:
                        # Backwards compatibility with the old per-action format.
                        outcome = payload.get("outcome", "")
                        action = payload.get("action", "")
                        params = payload.get("params") or {}
                        summary_text = (
                            outcome
                            or action
                            or _serialise_params(params)
                            or ""
                        )

                    update_count = payload.get("update_count")
                    try:
                        update_count_int = int(update_count) if update_count else 1
                    except (TypeError, ValueError):
                        update_count_int = 1

                    self._history[site_key] = SiteMemoryRecord(
                        summary=str(summary_text),
                        update_count=max(update_count_int, 1),
                        last_updated=parsed_timestamp,
                    )
        except OSError as exc:
            logger.debug("Could not read action memory store: %s", exc, exc_info=True)

    def _ensure_store_exists(self) -> None:
        """Create an empty JSONL file if we are persisting to disk and it is missing."""
        if not self._storage_path:
            return
        try:
            self._storage_path.touch(exist_ok=True)
        except OSError as exc:
            logger.debug("Could not create action memory store: %s", exc, exc_info=True)

    @property
    def existing_site_hits(self) -> int:
        """Return the number of times an existing site was re-indexed this session."""
        return self._existing_site_hits

    def get_session_observations(self) -> dict[str, list[str]]:
        """Return a shallow copy of per-site observations recorded this run."""
        return {site: entries[:] for site, entries in self._session_observations.items()}


class BrowserSessionContext:
    """
    Context object passed into the browser agent so the controller can update
    memory and the LLM can reference prior actions on revisited sites.
    """

    def __init__(
        self,
        *,
        storage_path: str | Path | None = None,
        session_id: str | None = None,
        load_existing: bool = True,
    ) -> None:
        self.session_id = session_id
        self.memory = BrowserActionMemory(
            storage_path=storage_path,
            load_existing=load_existing,
        )

    def bind_message_manager(self, manager: MessageManager) -> None:
        self.memory.bind_message_manager(manager)


__all__ = ["SiteMemoryRecord", "BrowserActionMemory", "BrowserSessionContext"]
