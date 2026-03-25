"""Agent memory persistence — three strategies for conversation state.

Strategies:
- ConversationMemory: full history (all messages from thread)
- SlidingWindowMemory: last N messages (configurable window_size, default 20)
- SummaryMemory: maintains running summary, updates after each turn
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable

from exo_web.database import get_db

_log = logging.getLogger(__name__)

# Type alias for an optional async function that generates a summary from text.
SummarizeFn = Callable[[str], Awaitable[str]]


class MemoryService:
    """Manages agent memory persistence across threads."""

    async def save_turn(
        self,
        agent_id: str,
        thread_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Persist a new user/assistant turn for the given agent+thread."""
        async with get_db() as db:
            user_id = str(uuid.uuid4())
            asst_id = str(uuid.uuid4())
            await db.execute(
                "INSERT INTO agent_memory (id, agent_id, thread_id, role, content) VALUES (?, ?, ?, 'user', ?)",
                (user_id, agent_id, thread_id, user_message),
            )
            await db.execute(
                "INSERT INTO agent_memory (id, agent_id, thread_id, role, content) VALUES (?, ?, ?, 'assistant', ?)",
                (asst_id, agent_id, thread_id, assistant_message),
            )
            await db.commit()

    async def get_memory(
        self,
        agent_id: str,
        thread_id: str,
        *,
        memory_type: str = "conversation",
        window_size: int = 20,
        summarize_fn: SummarizeFn | None = None,
    ) -> list[dict[str, str]]:
        """Return messages to inject into context based on strategy.

        Args:
            agent_id: The agent owning the memory.
            thread_id: The thread/conversation to load from.
            memory_type: One of 'conversation', 'sliding_window', 'summary'.
            window_size: For sliding_window, how many recent messages to keep.
            summarize_fn: Async function for summary generation (required for
                          summary mode on first call — if missing, falls back
                          to conversation mode).

        Returns:
            List of {role, content} dicts ready for injection into message
            history.
        """
        _log.debug("get_memory: agent=%s thread=%s type=%s", agent_id, thread_id, memory_type)
        if memory_type == "sliding_window":
            return await self._get_sliding_window(agent_id, thread_id, window_size)
        if memory_type == "summary":
            return await self._get_summary(agent_id, thread_id, summarize_fn)
        # Default: full conversation history
        return await self._get_conversation(agent_id, thread_id)

    async def clear_memory(self, agent_id: str, thread_id: str) -> None:
        """Reset memory for a specific agent+thread."""
        async with get_db() as db:
            await db.execute(
                "DELETE FROM agent_memory WHERE agent_id = ? AND thread_id = ?",
                (agent_id, thread_id),
            )
            await db.execute(
                "DELETE FROM agent_memory_summary WHERE agent_id = ? AND thread_id = ?",
                (agent_id, thread_id),
            )
            await db.commit()

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    async def _get_conversation(self, agent_id: str, thread_id: str) -> list[dict[str, str]]:
        """ConversationMemory — return full message history."""
        async with get_db() as db:
            cursor = await db.execute(
                "SELECT role, content FROM agent_memory "
                "WHERE agent_id = ? AND thread_id = ? ORDER BY created_at ASC",
                (agent_id, thread_id),
            )
            rows = await cursor.fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    async def _get_sliding_window(
        self, agent_id: str, thread_id: str, window_size: int
    ) -> list[dict[str, str]]:
        """SlidingWindowMemory — return the last N messages."""
        async with get_db() as db:
            cursor = await db.execute(
                "SELECT role, content FROM ("
                "  SELECT role, content, created_at FROM agent_memory "
                "  WHERE agent_id = ? AND thread_id = ? "
                "  ORDER BY created_at DESC LIMIT ?"
                ") sub ORDER BY created_at ASC",
                (agent_id, thread_id, window_size),
            )
            rows = await cursor.fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    async def _get_summary(
        self,
        agent_id: str,
        thread_id: str,
        summarize_fn: SummarizeFn | None,
    ) -> list[dict[str, str]]:
        """SummaryMemory — return stored summary + recent messages.

        The summary is updated after each save_turn call via
        update_summary(). Here we just load the existing summary and the
        last few messages that have not been summarized yet.
        """
        messages: list[dict[str, str]] = []

        async with get_db() as db:
            # Load existing summary
            cursor = await db.execute(
                "SELECT summary FROM agent_memory_summary WHERE agent_id = ? AND thread_id = ?",
                (agent_id, thread_id),
            )
            row = await cursor.fetchone()
            if row and row["summary"]:
                messages.append(
                    {
                        "role": "system",
                        "content": f"Previous conversation summary:\n{row['summary']}",
                    }
                )

            # Also include the last few messages for recency
            cursor = await db.execute(
                "SELECT role, content FROM ("
                "  SELECT role, content, created_at FROM agent_memory "
                "  WHERE agent_id = ? AND thread_id = ? "
                "  ORDER BY created_at DESC LIMIT 6"
                ") sub ORDER BY created_at ASC",
                (agent_id, thread_id),
            )
            rows = await cursor.fetchall()
            messages.extend({"role": r["role"], "content": r["content"]} for r in rows)

        return messages

    async def update_summary(
        self,
        agent_id: str,
        thread_id: str,
        summarize_fn: SummarizeFn,
    ) -> None:
        """Update the running summary for a thread.

        Called after save_turn() when the memory type is 'summary'. Reads all
        stored messages, asks the LLM to summarize, and persists the result.
        """
        async with get_db() as db:
            # Get existing summary
            cursor = await db.execute(
                "SELECT summary FROM agent_memory_summary WHERE agent_id = ? AND thread_id = ?",
                (agent_id, thread_id),
            )
            row = await cursor.fetchone()
            existing_summary = (row["summary"] if row else "") or ""

            # Get recent messages not yet in the summary (last 10 turns)
            cursor = await db.execute(
                "SELECT role, content FROM ("
                "  SELECT role, content, created_at FROM agent_memory "
                "  WHERE agent_id = ? AND thread_id = ? "
                "  ORDER BY created_at DESC LIMIT 20"
                ") sub ORDER BY created_at ASC",
                (agent_id, thread_id),
            )
            rows = await cursor.fetchall()

        if not rows:
            return

        # Build the text to summarize
        parts = []
        if existing_summary:
            parts.append(f"Previous summary:\n{existing_summary}\n")
        parts.append("Recent conversation:")
        for r in rows:
            parts.append(f"{r['role']}: {r['content']}")
        text_to_summarize = "\n".join(parts)

        prompt = (
            "Summarize the following conversation concisely, preserving key facts, "
            "decisions, and context that would be needed to continue the conversation "
            "naturally. Keep the summary under 500 words.\n\n"
            f"{text_to_summarize}"
        )

        new_summary = await summarize_fn(prompt)

        # Upsert the summary
        summary_id = str(uuid.uuid4())
        async with get_db() as db:
            await db.execute(
                "INSERT INTO agent_memory_summary (id, agent_id, thread_id, summary, updated_at) "
                "VALUES (?, ?, ?, ?, datetime('now')) "
                "ON CONFLICT(agent_id, thread_id) DO UPDATE SET summary = excluded.summary, updated_at = datetime('now')",
                (summary_id, agent_id, thread_id, new_summary),
            )
            await db.commit()


# Module-level singleton for convenience
memory_service = MemoryService()
