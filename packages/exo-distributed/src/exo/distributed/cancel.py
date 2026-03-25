"""Cooperative cancellation token for distributed task execution."""

from __future__ import annotations


class CancellationToken:
    """Token checked cooperatively between agent execution steps.

    Workers set :attr:`cancelled` to ``True`` when a cancel signal is
    received for the active task.  The agent execution loop checks this
    token between steps and stops early if cancellation was requested.
    """

    def __init__(self) -> None:
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._cancelled

    def cancel(self) -> None:
        """Request cancellation."""
        self._cancelled = True
