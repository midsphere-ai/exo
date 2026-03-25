"""LLM-powered intent recognition for routing user input to task actions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Intent:
    """Recognized intent from user input.

    Args:
        action: The task action to perform (e.g. ``"create_task"``, ``"pause_task"``).
        task_id: Relevant task ID, if the intent targets an existing task.
        confidence: Model confidence in the classification (0.0-1.0).
        details: Extra structured details extracted from the input.
    """

    action: str
    task_id: str | None = None
    confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


# Recognized task action types.
TASK_ACTIONS = frozenset(
    {
        "create_task",
        "pause_task",
        "resume_task",
        "cancel_task",
        "list_tasks",
        "get_task_status",
        "update_task",
        "unknown",
    }
)

_SYSTEM_PROMPT = """\
You are a task-intent classifier.  Given a user message, return a JSON object with:
- "action": one of {actions}
- "task_id": the task ID mentioned (string or null)
- "confidence": your confidence 0.0-1.0
- "details": any extra extracted info as a dict

Return ONLY the JSON object, no other text."""

_AVAILABLE_TASKS_ADDENDUM = """
The following tasks currently exist:
{tasks}
Use these IDs when the user refers to a task."""


class IntentRecognizer:
    """Classify user input into a task action using an LLM.

    Args:
        model: Model identifier string (e.g. ``"openai:gpt-4o"``).
    """

    def __init__(self, model: str) -> None:
        self._model = model

    async def recognize(
        self,
        input: str,
        *,
        available_tasks: list[dict[str, Any]] | None = None,
        provider: Any | None = None,
    ) -> Intent:
        """Classify *input* into a task ``Intent``.

        Args:
            input: The user message to classify.
            available_tasks: Optional list of task dicts (with ``id`` and ``name``
                keys) to help the model resolve task references.
            provider: Optional pre-built provider with an ``async complete()``
                method.  When ``None``, one is created via ``get_provider``.

        Returns:
            An ``Intent`` describing the recognised action.
        """
        if provider is None:
            from exo.models.provider import (  # pyright: ignore[reportMissingImports]
                get_provider,
            )

            provider = get_provider(self._model)

        assert provider is not None  # guaranteed after fallback above

        actions_str = ", ".join(sorted(TASK_ACTIONS))
        system_text = _SYSTEM_PROMPT.format(actions=actions_str)

        if available_tasks:
            tasks_text = "\n".join(
                f"- {t.get('id', '?')}: {t.get('name', '?')}" for t in available_tasks
            )
            system_text += _AVAILABLE_TASKS_ADDENDUM.format(tasks=tasks_text)

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": input},
        ]

        response = await provider.complete(messages)
        return _parse_intent(response.content)


def _parse_intent(content: str) -> Intent:
    """Parse LLM response content into an ``Intent``.

    Extracts the first JSON object from *content*, falling back to an
    ``unknown`` intent if parsing fails.
    """
    try:
        data = _extract_json(content)
        action = str(data.get("action", "unknown"))
        if action not in TASK_ACTIONS:
            action = "unknown"

        task_id = data.get("task_id")
        if task_id is not None:
            task_id = str(task_id)

        confidence = float(data.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))

        details = data.get("details") or {}
        if not isinstance(details, dict):
            details = {}

        return Intent(
            action=action,
            task_id=task_id,
            confidence=confidence,
            details=details,
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return Intent(action="unknown", confidence=0.0)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from *text*.

    Handles cases where the model wraps JSON in markdown code fences or
    includes surrounding prose.
    """
    # Try direct parse first.
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try extracting from code fences.
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        parsed = json.loads(fence_match.group(1).strip())
        if isinstance(parsed, dict):
            return parsed

    # Try finding a bare JSON object.
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        parsed = json.loads(brace_match.group(0))
        if isinstance(parsed, dict):
            return parsed

    msg = "No JSON object found in response"
    raise json.JSONDecodeError(msg, text, 0)
