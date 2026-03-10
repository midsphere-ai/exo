"""Action/step tracking for the research loop."""
from __future__ import annotations
import logging
from typing import Any, Callable

logger = logging.getLogger("deepsearch")


class ActionTracker:
    def __init__(self) -> None:
        self._actions: list[dict] = []
        self._thinks: list[str] = []
        self._listeners: list[Callable] = []

    def on_action(self, callback: Callable) -> None:
        self._listeners.append(callback)

    def track_action(self, **kwargs: Any) -> None:
        self._actions.append(kwargs)
        for listener in self._listeners:
            try:
                listener(kwargs)
            except Exception:
                pass

    def track_think(self, think: str, lang: str | None = None, params: dict | None = None) -> None:
        self._thinks.append(think)
        for listener in self._listeners:
            try:
                listener({"think": think})
            except Exception:
                pass

    @property
    def actions(self) -> list[dict]:
        return list(self._actions)
