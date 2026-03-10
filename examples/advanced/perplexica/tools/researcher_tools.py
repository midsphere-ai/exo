"""Researcher-specific tools: done signal and reasoning preamble."""

from __future__ import annotations

from orbiter import tool


@tool
async def done() -> str:
    """Signal that research is complete. Call this when you have gathered enough information to answer the user's query.

    YOU MUST CALL THIS ACTION TO SIGNAL COMPLETION; DO NOT OUTPUT FINAL ANSWERS DIRECTLY TO THE USER.
    IT WILL BE AUTOMATICALLY TRIGGERED IF MAXIMUM ITERATIONS ARE REACHED SO IF YOU'RE LOW ON ITERATIONS, DON'T CALL IT AND INSTEAD FOCUS ON GATHERING ESSENTIAL INFO FIRST.
    """
    return "Research complete."


@tool
async def reasoning_preamble(plan: str) -> str:
    """State your plan in natural language before any other action. Keep it short, action-focused, and tailored to the current query.

    Args:
        plan: A concise natural-language plan in one short paragraph.
    """
    return plan
