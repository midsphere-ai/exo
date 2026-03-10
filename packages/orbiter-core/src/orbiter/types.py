"""Core message types for the Orbiter framework."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class OrbiterError(Exception):
    """Base exception for all Orbiter errors."""


class UserMessage(BaseModel):
    """A message from the user."""

    model_config = {"frozen": True}

    role: Literal["user"] = "user"
    content: str


class SystemMessage(BaseModel):
    """A system instruction message."""

    model_config = {"frozen": True}

    role: Literal["system"] = "system"
    content: str


class ToolCall(BaseModel):
    """A request from the LLM to invoke a tool.

    Args:
        id: Unique identifier for this tool call.
        name: Name of the tool to invoke.
        arguments: JSON-encoded string of the tool arguments.
        thought_signature: Opaque signature for round-tripping thought parts (Gemini).
    """

    model_config = {"frozen": True}

    id: str
    name: str
    arguments: str = ""
    thought_signature: bytes | None = None


class AssistantMessage(BaseModel):
    """A response from the LLM assistant.

    May contain text content, tool calls, or both.

    Args:
        content: Text content of the response.
        tool_calls: Tool invocations requested by the assistant.
    """

    model_config = {"frozen": True}

    role: Literal["assistant"] = "assistant"
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning_content: str = ""
    thought_signatures: list[bytes] = Field(default_factory=list)


class ToolResult(BaseModel):
    """The result of executing a tool call.

    Args:
        tool_call_id: The id of the ToolCall this responds to.
        tool_name: Name of the tool that was executed.
        content: The string result from the tool.
        error: Error message if the tool failed.
    """

    model_config = {"frozen": True}

    role: Literal["tool"] = "tool"
    tool_call_id: str
    tool_name: str
    content: str = ""
    error: str | None = None


Message = UserMessage | AssistantMessage | SystemMessage | ToolResult


class Usage(BaseModel):
    """Token usage statistics from an LLM call.

    Args:
        input_tokens: Number of tokens in the prompt.
        output_tokens: Number of tokens in the completion.
        total_tokens: Total tokens consumed.
    """

    model_config = {"frozen": True}

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AgentInput(BaseModel):
    """Normalized input for an agent run.

    Args:
        query: The user's query string.
        messages: Prior conversation messages for context.
    """

    model_config = {"frozen": True}

    query: str
    messages: list[Message] = Field(default_factory=list)


class AgentOutput(BaseModel):
    """Output from a single LLM call within a run.

    Args:
        text: Text content of the LLM response.
        tool_calls: Tool invocations requested by the LLM.
        usage: Token usage for this call.
    """

    model_config = {"frozen": True}

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    reasoning_content: str = ""
    thought_signatures: list[bytes] = Field(default_factory=list)


class ActionModel(BaseModel):
    """A parsed tool action ready for execution.

    Unlike ``ToolCall`` where arguments is a JSON string, here arguments
    is already parsed into a dict.

    Args:
        tool_call_id: Identifier linking back to the originating ToolCall.
        tool_name: Name of the tool to execute.
        arguments: Parsed keyword arguments for the tool.
    """

    model_config = {"frozen": True}

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    """Return type of ``run()`` — the final result of an agent execution.

    Args:
        output: Final text output from the agent.
        messages: Full message history of the run.
        usage: Aggregated token usage across all steps.
        steps: Number of LLM call steps taken.
    """

    model_config = {"frozen": True}

    output: str = ""
    messages: list[Message] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    steps: int = Field(default=0, ge=0)


class TextEvent(BaseModel):
    """Streaming event for a text delta.

    Args:
        type: Discriminator literal, always ``"text"``.
        text: The text chunk.
        agent_name: Name of the agent producing this event.
    """

    model_config = {"frozen": True}

    type: Literal["text"] = "text"
    text: str
    agent_name: str = ""


class ToolCallEvent(BaseModel):
    """Streaming event for a tool call notification.

    Args:
        type: Discriminator literal, always ``"tool_call"``.
        tool_name: Name of the tool being called.
        tool_call_id: Identifier for this tool call.
        agent_name: Name of the agent producing this event.
    """

    model_config = {"frozen": True}

    type: Literal["tool_call"] = "tool_call"
    tool_name: str
    tool_call_id: str
    agent_name: str = ""


class StepEvent(BaseModel):
    """Streaming event for step start/completion.

    Emitted at the start and end of each agent step so consumers
    can show step-by-step progress.

    Args:
        type: Discriminator literal, always ``"step"``.
        step_number: The step index (1-based).
        agent_name: Name of the agent executing this step.
        status: Whether the step is starting or has completed.
        started_at: Timestamp when the step started.
        completed_at: Timestamp when the step completed (None if still running).
        usage: Token usage for this step (None if not yet available).
    """

    model_config = {"frozen": True}

    type: Literal["step"] = "step"
    step_number: int
    agent_name: str
    status: Literal["started", "completed"]
    started_at: float
    completed_at: float | None = None
    usage: Usage | None = None


class ToolResultEvent(BaseModel):
    """Streaming event emitted after each tool execution.

    Args:
        type: Discriminator literal, always ``"tool_result"``.
        tool_name: Name of the tool that was executed.
        tool_call_id: Identifier linking back to the originating tool call.
        arguments: The arguments passed to the tool.
        result: The string result from the tool.
        error: Error message if the tool failed (None on success).
        success: Whether the tool execution succeeded.
        duration_ms: How long the tool execution took in milliseconds.
        agent_name: Name of the agent that invoked this tool.
    """

    model_config = {"frozen": True}

    type: Literal["tool_result"] = "tool_result"
    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    error: str | None = None
    success: bool = True
    duration_ms: float = 0.0
    agent_name: str = ""


class ReasoningEvent(BaseModel):
    """Streaming event for reasoning content from the model.

    Args:
        type: Discriminator literal, always ``"reasoning"``.
        text: The reasoning text content.
        agent_name: Name of the agent producing this event.
    """

    model_config = {"frozen": True}

    type: Literal["reasoning"] = "reasoning"
    text: str
    agent_name: str = ""


class ErrorEvent(BaseModel):
    """Streaming event for errors during agent execution.

    Args:
        type: Discriminator literal, always ``"error"``.
        error: The error message.
        error_type: The type/class of the error.
        agent_name: Name of the agent that encountered the error.
        step_number: The step during which the error occurred (None if not step-specific).
        recoverable: Whether the error is recoverable.
    """

    model_config = {"frozen": True}

    type: Literal["error"] = "error"
    error: str
    error_type: str
    agent_name: str = ""
    step_number: int | None = None
    recoverable: bool = False


class StatusEvent(BaseModel):
    """Streaming event for agent status changes.

    Args:
        type: Discriminator literal, always ``"status"``.
        status: The current status of the agent.
        agent_name: Name of the agent whose status changed.
        message: Human-readable description of the status change.
    """

    model_config = {"frozen": True}

    type: Literal["status"] = "status"
    status: Literal[
        "starting", "running", "waiting_for_tool", "completed", "cancelled", "error"
    ]
    agent_name: str = ""
    message: str = ""


class UsageEvent(BaseModel):
    """Streaming event for per-step token usage.

    Args:
        type: Discriminator literal, always ``"usage"``.
        usage: Token usage statistics for this step.
        agent_name: Name of the agent that consumed the tokens.
        step_number: The step this usage is associated with.
        model: The model identifier used for this step.
    """

    model_config = {"frozen": True}

    type: Literal["usage"] = "usage"
    usage: Usage
    agent_name: str = ""
    step_number: int = 0
    model: str = ""


StreamEvent = (
    TextEvent
    | ToolCallEvent
    | StepEvent
    | ToolResultEvent
    | ReasoningEvent
    | ErrorEvent
    | StatusEvent
    | UsageEvent
)
