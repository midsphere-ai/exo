"""Tests for IntentRecognizer — LLM-powered intent classification for task actions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from exo.task_controller import (
    TASK_ACTIONS,
    Intent,
    IntentRecognizer,
)

# ---------------------------------------------------------------------------
# MockProvider
# ---------------------------------------------------------------------------


@dataclass
class _MockResponse:
    content: str = ""


class MockProvider:
    """Mock LLM provider that returns a configurable response."""

    def __init__(self, content: str = "{}") -> None:
        self._content = content

    async def complete(self, messages: Any, **kwargs: Any) -> _MockResponse:
        return _MockResponse(content=self._content)


class CapturingProvider:
    """Mock provider that captures messages sent to it."""

    def __init__(self, content: str = "{}") -> None:
        self._content = content
        self.captured_messages: list[Any] = []

    async def complete(self, messages: Any, **kwargs: Any) -> _MockResponse:
        self.captured_messages.extend(messages)
        return _MockResponse(content=self._content)


# ---------------------------------------------------------------------------
# Intent dataclass tests
# ---------------------------------------------------------------------------


class TestIntent:
    def test_defaults(self) -> None:
        intent = Intent(action="create_task")
        assert intent.action == "create_task"
        assert intent.task_id is None
        assert intent.confidence == 0.0
        assert intent.details == {}

    def test_all_fields(self) -> None:
        intent = Intent(
            action="pause_task",
            task_id="abc-123",
            confidence=0.95,
            details={"reason": "user requested"},
        )
        assert intent.action == "pause_task"
        assert intent.task_id == "abc-123"
        assert intent.confidence == 0.95
        assert intent.details == {"reason": "user requested"}


# ---------------------------------------------------------------------------
# TASK_ACTIONS tests
# ---------------------------------------------------------------------------


class TestTaskActions:
    def test_expected_actions_present(self) -> None:
        expected = {
            "create_task",
            "pause_task",
            "resume_task",
            "cancel_task",
            "list_tasks",
            "get_task_status",
            "update_task",
            "unknown",
        }
        assert expected == TASK_ACTIONS

    def test_is_frozenset(self) -> None:
        assert isinstance(TASK_ACTIONS, frozenset)


# ---------------------------------------------------------------------------
# IntentRecognizer tests
# ---------------------------------------------------------------------------


class TestIntentRecognizer:
    @pytest.mark.asyncio()
    async def test_create_task_intent(self) -> None:
        response = json.dumps(
            {
                "action": "create_task",
                "task_id": None,
                "confidence": 0.92,
                "details": {"name": "Write unit tests"},
            }
        )
        provider = MockProvider(content=response)

        recognizer = IntentRecognizer("test-model")
        intent = await recognizer.recognize("Create a task to write unit tests", provider=provider)

        assert intent.action == "create_task"
        assert intent.task_id is None
        assert intent.confidence == pytest.approx(0.92)
        assert intent.details == {"name": "Write unit tests"}

    @pytest.mark.asyncio()
    async def test_pause_task_intent(self) -> None:
        response = json.dumps(
            {
                "action": "pause_task",
                "task_id": "task-42",
                "confidence": 0.88,
                "details": {},
            }
        )
        provider = MockProvider(content=response)

        recognizer = IntentRecognizer("test-model")
        intent = await recognizer.recognize("Pause task-42 for now", provider=provider)

        assert intent.action == "pause_task"
        assert intent.task_id == "task-42"
        assert intent.confidence == pytest.approx(0.88)

    @pytest.mark.asyncio()
    async def test_resume_task_intent(self) -> None:
        response = json.dumps(
            {
                "action": "resume_task",
                "task_id": "task-7",
                "confidence": 0.95,
                "details": {},
            }
        )
        provider = MockProvider(content=response)

        recognizer = IntentRecognizer("test-model")
        intent = await recognizer.recognize("Resume task-7", provider=provider)

        assert intent.action == "resume_task"
        assert intent.task_id == "task-7"

    @pytest.mark.asyncio()
    async def test_cancel_task_intent(self) -> None:
        response = json.dumps(
            {
                "action": "cancel_task",
                "task_id": "task-99",
                "confidence": 0.91,
                "details": {"reason": "no longer needed"},
            }
        )
        provider = MockProvider(content=response)

        recognizer = IntentRecognizer("test-model")
        intent = await recognizer.recognize(
            "Cancel task-99, it's no longer needed", provider=provider
        )

        assert intent.action == "cancel_task"
        assert intent.task_id == "task-99"
        assert intent.details == {"reason": "no longer needed"}

    @pytest.mark.asyncio()
    async def test_list_tasks_intent(self) -> None:
        response = json.dumps(
            {
                "action": "list_tasks",
                "task_id": None,
                "confidence": 0.97,
                "details": {},
            }
        )
        provider = MockProvider(content=response)

        recognizer = IntentRecognizer("test-model")
        intent = await recognizer.recognize("Show me all current tasks", provider=provider)

        assert intent.action == "list_tasks"
        assert intent.task_id is None

    @pytest.mark.asyncio()
    async def test_get_task_status_intent(self) -> None:
        response = json.dumps(
            {
                "action": "get_task_status",
                "task_id": "task-5",
                "confidence": 0.89,
                "details": {},
            }
        )
        provider = MockProvider(content=response)

        recognizer = IntentRecognizer("test-model")
        intent = await recognizer.recognize("What's the status of task-5?", provider=provider)

        assert intent.action == "get_task_status"
        assert intent.task_id == "task-5"

    @pytest.mark.asyncio()
    async def test_update_task_intent(self) -> None:
        response = json.dumps(
            {
                "action": "update_task",
                "task_id": "task-3",
                "confidence": 0.85,
                "details": {"priority": 10},
            }
        )
        provider = MockProvider(content=response)

        recognizer = IntentRecognizer("test-model")
        intent = await recognizer.recognize("Set task-3 priority to 10", provider=provider)

        assert intent.action == "update_task"
        assert intent.task_id == "task-3"
        assert intent.details == {"priority": 10}

    @pytest.mark.asyncio()
    async def test_unknown_action_fallback(self) -> None:
        response = json.dumps(
            {
                "action": "delete_universe",
                "task_id": None,
                "confidence": 0.5,
                "details": {},
            }
        )
        provider = MockProvider(content=response)

        recognizer = IntentRecognizer("test-model")
        intent = await recognizer.recognize("Delete the universe", provider=provider)

        assert intent.action == "unknown"

    @pytest.mark.asyncio()
    async def test_available_tasks_passed_to_prompt(self) -> None:
        """When available_tasks is provided, they appear in the system prompt."""
        capturing = CapturingProvider(
            content=json.dumps(
                {
                    "action": "pause_task",
                    "task_id": "t-1",
                    "confidence": 0.9,
                    "details": {},
                }
            )
        )

        recognizer = IntentRecognizer("test-model")
        tasks = [
            {"id": "t-1", "name": "Write docs"},
            {"id": "t-2", "name": "Fix bug"},
        ]
        intent = await recognizer.recognize(
            "Pause the docs task", available_tasks=tasks, provider=capturing
        )

        assert intent.action == "pause_task"
        assert intent.task_id == "t-1"

        system_msg = capturing.captured_messages[0]
        content = system_msg["content"] if isinstance(system_msg, dict) else system_msg.content
        assert "t-1: Write docs" in content
        assert "t-2: Fix bug" in content

    @pytest.mark.asyncio()
    async def test_malformed_json_returns_unknown(self) -> None:
        provider = MockProvider(content="I don't understand the question.")

        recognizer = IntentRecognizer("test-model")
        intent = await recognizer.recognize("Something nonsensical", provider=provider)

        assert intent.action == "unknown"
        assert intent.confidence == 0.0

    @pytest.mark.asyncio()
    async def test_json_in_code_fence(self) -> None:
        response = '```json\n{"action": "list_tasks", "task_id": null, "confidence": 0.9, "details": {}}\n```'
        provider = MockProvider(content=response)

        recognizer = IntentRecognizer("test-model")
        intent = await recognizer.recognize("Show tasks", provider=provider)

        assert intent.action == "list_tasks"
        assert intent.confidence == pytest.approx(0.9)

    @pytest.mark.asyncio()
    async def test_json_with_surrounding_text(self) -> None:
        response = 'Here is the result: {"action": "create_task", "task_id": null, "confidence": 0.8, "details": {"name": "test"}} done.'
        provider = MockProvider(content=response)

        recognizer = IntentRecognizer("test-model")
        intent = await recognizer.recognize("Make a test task", provider=provider)

        assert intent.action == "create_task"
        assert intent.confidence == pytest.approx(0.8)
        assert intent.details == {"name": "test"}

    @pytest.mark.asyncio()
    async def test_confidence_clamped_to_range(self) -> None:
        response = json.dumps(
            {
                "action": "list_tasks",
                "task_id": None,
                "confidence": 1.5,
                "details": {},
            }
        )
        provider = MockProvider(content=response)

        recognizer = IntentRecognizer("test-model")
        intent = await recognizer.recognize("List tasks", provider=provider)

        assert intent.confidence == 1.0

    @pytest.mark.asyncio()
    async def test_negative_confidence_clamped(self) -> None:
        response = json.dumps(
            {
                "action": "list_tasks",
                "task_id": None,
                "confidence": -0.5,
                "details": {},
            }
        )
        provider = MockProvider(content=response)

        recognizer = IntentRecognizer("test-model")
        intent = await recognizer.recognize("List tasks", provider=provider)

        assert intent.confidence == 0.0

    @pytest.mark.asyncio()
    async def test_details_non_dict_becomes_empty(self) -> None:
        response = json.dumps(
            {
                "action": "create_task",
                "task_id": None,
                "confidence": 0.8,
                "details": "not a dict",
            }
        )
        provider = MockProvider(content=response)

        recognizer = IntentRecognizer("test-model")
        intent = await recognizer.recognize("Create task", provider=provider)

        assert intent.details == {}

    @pytest.mark.asyncio()
    async def test_no_available_tasks(self) -> None:
        """When available_tasks is None, no task list appears in prompt."""
        capturing = CapturingProvider(
            content=json.dumps(
                {
                    "action": "list_tasks",
                    "task_id": None,
                    "confidence": 0.9,
                    "details": {},
                }
            )
        )

        recognizer = IntentRecognizer("test-model")
        await recognizer.recognize("Show tasks", provider=capturing)

        system_msg = capturing.captured_messages[0]
        content = system_msg["content"] if isinstance(system_msg, dict) else system_msg.content
        assert "currently exist" not in content
