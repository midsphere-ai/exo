"""Abstract base class for typed context windowing hooks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from exo.context.info import ContextWindowInfo


class ContextWindowHook(ABC):
    """Base class for custom context windowing hooks.

    Subclass and implement :meth:`window`. The framework invokes
    ``__call__`` which delegates to your typed ``window()`` method.

    Example::

        class KeepImportant(ContextWindowHook):
            async def window(self, *, messages, info, **_):
                if info.fill_ratio < 0.5:
                    return  # plenty of room
                # ... custom logic to trim messages ...

        agent = Agent(
            name="bot",
            overflow="hook",
            hooks=[(HookPoint.CONTEXT_WINDOW, KeepImportant())],
        )
    """

    @abstractmethod
    async def window(
        self,
        *,
        agent: Any,
        messages: list[Any],
        info: ContextWindowInfo,
        actions: list[Any],
        provider: Any | None = None,
        **extra: Any,
    ) -> None:
        """Modify *messages* in place to apply context windowing.

        Args:
            agent: The Agent instance.
            messages: Mutable message list — modify in place.
            info: Read-only snapshot of context state.
            actions: Mutable list — append ``_ContextAction`` for streaming events.
            provider: LLM provider (available for custom summarization calls).
            **extra: Future-proof kwargs.
        """
        ...

    async def __call__(self, **kwargs: Any) -> None:
        """Hook protocol adapter — delegates to :meth:`window`."""
        await self.window(**kwargs)
