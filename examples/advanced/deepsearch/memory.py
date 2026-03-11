"""Memory system — combined summaries and insights for agent memory management.

1:1 port of SkyworkAI's GeneralMemorySystem from src/memory/general_memory_system.py.
Simplified for Orbiter: standalone class, no registry, no file_lock utility.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from orbiter.types import SystemMessage, UserMessage

from .llm_utils import call_llm

logger = logging.getLogger("deepagent")


# ---------------------------------------------------------------------------
# Types (1:1 from SkyworkAI src/memory/types.py)
# ---------------------------------------------------------------------------


class EventType(Enum):
    TASK_START = "task_start"
    TOOL_STEP = "tool_step"
    TASK_END = "task_end"


class Importance(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ChatEvent(BaseModel):
    id: str = Field(description="Unique identifier for the event.")
    step_number: int = Field(description="Step number of the event.")
    event_type: EventType = Field(description="Type of the event.")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp.")
    data: dict[str, Any] = Field(default_factory=dict, description="Event data.")
    agent_name: str | None = Field(default=None, description="Agent name.")
    session_id: str | None = Field(default=None, description="Session ID.")
    task_id: str | None = Field(default=None, description="Task ID.")

    def __str__(self) -> str:
        return (
            f"<chat_event>\n"
            f"  ID: {self.id}\n"
            f"  Step: {self.step_number}\n"
            f"  Type: {self.event_type.value}\n"
            f"  Timestamp: {self.timestamp}\n"
            f"  Agent: {self.agent_name}\n"
            f"  Data: {json.dumps(self.data)}\n"
            f"</chat_event>"
        )


class Summary(BaseModel):
    id: str = Field(description="Unique identifier for the summary.")
    importance: Importance = Field(description="Importance level.")
    content: str = Field(description="Summary content.")

    def __str__(self) -> str:
        return f"<summary id={self.id} importance={self.importance.value}>{self.content}</summary>"


class Insight(BaseModel):
    id: str = Field(description="Unique identifier for the insight.")
    content: str = Field(description="Insight content.")
    importance: Importance = Field(description="Importance level.")
    source_event_id: str = Field(description="ID of source event.")
    tags: list[str] = Field(description="Categorization tags.")

    def __str__(self) -> str:
        return f"<insight id={self.id} importance={self.importance.value}>{self.content}</insight>"


# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------


class CombinedMemoryOutput(BaseModel):
    """Structured output for combined summary and insight generation."""
    summaries: list[Summary] = Field(description="List of summary points")
    insights: list[Insight] = Field(description="List of insights")


class ProcessDecision(BaseModel):
    should_process: bool = Field(description="Whether to process the memory")
    reasoning: str = Field(description="Reasoning for the decision")


# ---------------------------------------------------------------------------
# CombinedMemory (1:1 from SkyworkAI)
# ---------------------------------------------------------------------------


class CombinedMemory:
    """Combined memory that handles both summaries and insights."""

    def __init__(
        self,
        model_name: str = "openai:gpt-4o-mini",
        max_summaries: int = 20,
        max_insights: int = 100,
    ) -> None:
        self.model_name = model_name
        self.max_summaries = max_summaries
        self.max_insights = max_insights
        self.events: list[ChatEvent] = []
        self.candidate_chat_history: list[dict[str, str]] = []
        self.summaries: list[Summary] = []
        self.insights: list[Insight] = []

    def add_event(self, event: ChatEvent | list[ChatEvent]) -> None:
        events = [event] if isinstance(event, ChatEvent) else event
        for ev in events:
            self.events.append(ev)
            if ev.event_type in (EventType.TOOL_STEP, EventType.TASK_END):
                content = str(ev)
                role = "assistant" if ev.agent_name else "human"
                self.candidate_chat_history.append({"role": role, "content": content})

    async def check_and_process_memory(self) -> None:
        should_process = await self._check_should_process()
        if should_process:
            await self._process_memory()

    async def _check_should_process(self) -> bool:
        if len(self.candidate_chat_history) <= 5:
            return False

        new_lines = self._get_new_lines_text()
        current_memory = self._get_current_memory_text()

        decision_prompt = f"""You are analyzing a conversation to decide whether to process it and generate summaries and insights.
Current conversation has {len(self.events)} events.

Decision criteria:
1. If there are fewer than 5 events, do not process the memory.
2. If the conversation is repetitive or doesn't add new information, do not process.
3. If there are significant new insights, decisions, or learnings, process.
4. If the conversation is getting long (more than 10 events), process.

Current memory:
{current_memory}

New conversation events:
{new_lines}

Decide if you should process the memory."""

        try:
            response = await call_llm(
                model=self.model_name,
                messages=[
                    SystemMessage(content="You are a memory processing decision system."),
                    UserMessage(content=decision_prompt),
                ],
                response_format=ProcessDecision,
            )

            if response.parsed_model is not None:
                decision = response.parsed_model
                logger.info(f"Memory processing decision: {decision.should_process} - {decision.reasoning}")
                return decision.should_process
            return False

        except Exception as e:
            logger.warning(f"Failed to check if should process memory: {e}")
            return False

    async def _process_memory(self) -> None:
        if not self.candidate_chat_history:
            return

        new_lines = self._get_new_lines_text()
        current_memory = self._get_current_memory_text()

        prompt = f"""Analyze the conversation events and extract both summaries and insights.

For summaries, focus on:
1. Key decisions and tools taken
2. Important information exchanged
3. Task progress and outcomes

For insights, look for:
1. Successful strategies and patterns
2. Mistakes or failures to avoid
3. Key learnings and realizations
4. Actionable insights for future performance

Avoid repeating information already in the summaries or insights.

Current memory:
{current_memory}

New conversation events:
{new_lines}

Generate new summaries and insights."""

        try:
            response = await call_llm(
                model=self.model_name,
                messages=[
                    SystemMessage(content="You are a memory processing system."),
                    UserMessage(content=prompt),
                ],
                response_format=CombinedMemoryOutput,
            )

            if not response.success:
                raise ValueError(f"Model call failed: {response.message}")

            if response.parsed_model is not None:
                output = response.parsed_model
                self.summaries.extend(output.summaries)
                self.insights.extend(output.insights)
                self._sort_and_limit_summaries()
                self._sort_and_limit_insights()
                self.candidate_chat_history.clear()

        except Exception as e:
            logger.warning(f"Failed to process memory: {e}")

    def _get_new_lines_text(self) -> str:
        lines = []
        for msg in self.candidate_chat_history:
            role = msg["role"]
            content = msg["content"]
            lines.append(f"<{role}>\n{content}\n</{role}>")
        return "\n".join(lines)

    def _get_current_memory_text(self) -> str:
        summaries_text = "\n".join(str(s) for s in self.summaries)
        insights_text = "\n".join(str(i) for i in self.insights)
        return f"<summaries>\n{summaries_text}\n</summaries>\n<insights>\n{insights_text}\n</insights>"

    def _sort_and_limit_summaries(self) -> None:
        order = {Importance.HIGH: 0, Importance.MEDIUM: 1, Importance.LOW: 2}
        self.summaries.sort(key=lambda x: order.get(x.importance, 1))
        if len(self.summaries) > self.max_summaries:
            self.summaries = self.summaries[:self.max_summaries]

    def _sort_and_limit_insights(self) -> None:
        order = {Importance.HIGH: 0, Importance.MEDIUM: 1, Importance.LOW: 2}
        self.insights.sort(key=lambda x: order.get(x.importance, 1))
        if len(self.insights) > self.max_insights:
            self.insights = self.insights[:self.max_insights]

    def clear(self) -> None:
        self.events.clear()
        self.candidate_chat_history.clear()
        self.summaries.clear()
        self.insights.clear()

    def size(self) -> int:
        return len(self.events)

    def get_events(self, n: int | None = None) -> list[ChatEvent]:
        if n is None:
            return self.events
        return self.events[-n:] if len(self.events) > n else self.events

    def get_summaries(self, n: int | None = None) -> list[Summary]:
        if n is None:
            return self.summaries
        return self.summaries[-n:] if len(self.summaries) > n else self.summaries

    def get_insights(self, n: int | None = None) -> list[Insight]:
        if n is None:
            return self.insights
        return self.insights[-n:] if len(self.insights) > n else self.insights


# ---------------------------------------------------------------------------
# DeepAgentMemory — top-level memory system
# ---------------------------------------------------------------------------


class DeepAgentMemory:
    """Memory system for the DeepAgent.

    1:1 port of SkyworkAI's GeneralMemorySystem, simplified for Orbiter.
    Uses in-process state with per-session locking.
    """

    def __init__(
        self,
        *,
        model_name: str = "openai:gpt-4o-mini",
        max_summaries: int = 20,
        max_insights: int = 100,
        save_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_summaries = max_summaries
        self.max_insights = max_insights
        self.save_path = save_path

        self._memory: CombinedMemory | None = None
        self._lock = asyncio.Lock()
        self._pending_task: asyncio.Task[None] | None = None

    async def start_session(self, session_id: str = "default") -> str:
        """Start or resume a memory session."""
        if self.save_path and os.path.exists(self.save_path):
            await self._load_from_json(self.save_path)

        if self._memory is None:
            self._memory = CombinedMemory(
                model_name=self.model_name,
                max_summaries=self.max_summaries,
                max_insights=self.max_insights,
            )
        return session_id

    async def end_session(self) -> None:
        """End session, wait for pending processing, and save."""
        if self._pending_task is not None:
            try:
                await asyncio.wait_for(self._pending_task, timeout=60.0)
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Error waiting for memory processing: {e}")

        if self.save_path:
            await self._save_to_json(self.save_path)

    async def add_event(
        self,
        step_number: int,
        event_type: EventType | str,
        data: dict[str, Any],
        agent_name: str = "",
    ) -> None:
        """Add an event to memory."""
        if isinstance(event_type, str):
            try:
                event_type = EventType(event_type)
            except ValueError:
                event_type = EventType.TOOL_STEP

        event_id = f"event_{uuid.uuid4().hex[:8]}"
        event = ChatEvent(
            id=event_id,
            step_number=step_number,
            event_type=event_type,
            data=data,
            agent_name=agent_name,
        )

        if self._memory is None:
            self._memory = CombinedMemory(model_name=self.model_name)

        async with self._lock:
            self._memory.add_event(event)

        # Fire-and-forget background processing
        self._pending_task = asyncio.create_task(self._process_background())

        if self.save_path:
            await self._save_to_json(self.save_path)

    async def _process_background(self) -> None:
        """Background task: check and process memory."""
        try:
            if self._memory is None:
                return
            async with self._lock:
                await self._memory.check_and_process_memory()
            if self.save_path:
                await self._save_to_json(self.save_path)
        except Exception as e:
            logger.warning(f"Background memory processing failed: {e}")

    def get_memory(self) -> CombinedMemory | None:
        """Get the current memory instance."""
        return self._memory

    def get_memory_context(self) -> str:
        """Get formatted memory context for injection into prompts."""
        if self._memory is None:
            return ""

        summaries = self._memory.get_summaries()
        insights = self._memory.get_insights()

        if not summaries and not insights:
            return ""

        parts = []
        if summaries:
            summaries_text = "\n".join(str(s) for s in summaries)
            parts.append(f"<summaries>\n{summaries_text}\n</summaries>")
        if insights:
            insights_text = "\n".join(str(i) for i in insights)
            parts.append(f"<insights>\n{insights_text}\n</insights>")

        return "\n".join(parts)

    async def _save_to_json(self, file_path: str) -> None:
        """Save memory state to JSON file."""
        if self._memory is None:
            return

        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

        data = {
            "metadata": {"memory_system_type": "deep_agent_memory"},
            "events": [e.model_dump(mode="json") for e in self._memory.events],
            "summaries": [s.model_dump(mode="json") for s in self._memory.summaries],
            "insights": [i.model_dump(mode="json") for i in self._memory.insights],
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    async def _load_from_json(self, file_path: str) -> bool:
        """Load memory state from JSON file."""
        if not os.path.exists(file_path):
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if self._memory is None:
                self._memory = CombinedMemory(
                    model_name=self.model_name,
                    max_summaries=self.max_summaries,
                    max_insights=self.max_insights,
                )

            # Restore events
            if "events" in data:
                for event_data in data["events"]:
                    if "timestamp" in event_data and isinstance(event_data["timestamp"], str):
                        event_data["timestamp"] = datetime.fromisoformat(event_data["timestamp"])
                    if "event_type" in event_data and isinstance(event_data["event_type"], str):
                        event_data["event_type"] = EventType(event_data["event_type"])
                    self._memory.events.append(ChatEvent(**event_data))

            # Restore summaries
            if "summaries" in data:
                for s_data in data["summaries"]:
                    if "importance" in s_data and isinstance(s_data["importance"], str):
                        s_data["importance"] = Importance(s_data["importance"])
                    self._memory.summaries.append(Summary(**s_data))

            # Restore insights
            if "insights" in data:
                for i_data in data["insights"]:
                    if "importance" in i_data and isinstance(i_data["importance"], str):
                        i_data["importance"] = Importance(i_data["importance"])
                    self._memory.insights.append(Insight(**i_data))

            logger.info(f"Memory loaded from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load memory from {file_path}: {e}")
            return False
