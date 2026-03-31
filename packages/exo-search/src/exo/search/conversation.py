"""Multi-turn conversation history manager.

Tracks query-answer pairs and injects context into new queries so
agents can resolve references like "tell me more" or "what about X?".

Usage:
    from examples.advanced.exo-search.conversation import ConversationManager
"""

from __future__ import annotations


def format_chat_history(
    chat_history: list[tuple[str, str]],
    total_budget: int = 4000,
) -> str:
    """Format chat history with budget-based truncation.

    Recent turns get full text; older turns are progressively truncated.
    This preserves important context from the most recent exchanges while
    keeping total character count bounded.

    Args:
        chat_history: List of (query, answer) tuples.
        total_budget: Maximum total characters for all history.

    Returns:
        Formatted string with User/Assistant pairs.
    """
    if not chat_history:
        return ""

    parts: list[str] = []
    remaining = total_budget

    # Process in reverse (most recent first) to give recent turns priority
    for q, a in reversed(chat_history):
        q_text = f"User: {q}"
        if remaining <= 0:
            break
        # Most recent turns get more budget
        turn_budget = max(200, remaining // 2) if remaining < total_budget else remaining
        a_text = f"Assistant: {a[:turn_budget]}..." if len(a) > turn_budget else f"Assistant: {a}"
        turn_text = f"{q_text}\n{a_text}"
        remaining -= len(turn_text)
        parts.append(turn_text)

    # Reverse back to chronological order
    parts.reverse()
    return "\n".join(parts)


class ConversationManager:
    """Manages multi-turn conversation history for contextual search."""

    def __init__(self, max_turns: int = 10) -> None:
        self._turns: list[tuple[str, str]] = []
        self._max_turns = max_turns

    def add_turn(self, query: str, answer: str) -> None:
        """Record a query-answer pair.

        Args:
            query: The user's query.
            answer: The generated answer.
        """
        self._turns.append((query, answer))
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns :]

    def get_context_prompt(self, new_query: str) -> str:
        """Build a context-aware prompt for the new query.

        If there is conversation history, prepends it so agents can
        resolve references and follow-ups.

        Args:
            new_query: The new user query.

        Returns:
            The query with conversation context prepended.
        """
        if not self._turns:
            return new_query

        parts = ["Previous conversation:"]
        for i, (q, a) in enumerate(self._turns, 1):
            # Truncate long answers to keep context manageable.
            short_answer = a[:500] + "..." if len(a) > 500 else a
            parts.append(f"Q{i}: {q}")
            parts.append(f"A{i}: {short_answer}")
        parts.append(f"\nCurrent question: {new_query}")
        return "\n".join(parts)

    @property
    def turns(self) -> list[tuple[str, str]]:
        """Return the conversation history."""
        return list(self._turns)

    def clear(self) -> None:
        """Clear conversation history."""
        self._turns.clear()
