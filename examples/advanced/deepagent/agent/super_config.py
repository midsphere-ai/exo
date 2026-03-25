#!/usr/bin/env python
# coding: utf-8
"""Super Agent Configuration.

Standalone Pydantic configuration for the SuperReActAgent.
Uses plain BaseModel classes following the deepsearch/config.py pattern.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Low-level model connection details."""

    model_name: str = Field(default="gpt-4o", description="LLM model identifier.")
    api_key: str | None = Field(default=None, description="Provider API key.")
    api_base: str | None = Field(default=None, description="Provider base URL.")
    timeout: int = Field(default=120, description="Request timeout in seconds.")


class SuperModelConfig(BaseModel):
    """Model configuration wrapper (preserves ``config.model.model_info`` access path)."""

    model_info: ModelInfo = Field(default_factory=ModelInfo)


class ConstraintConfig(BaseModel):
    """Execution constraints for the agent loop.

    Plain Pydantic model for execution constraints.
    """

    max_iteration: int = Field(default=10, description="Maximum iterations for the ReAct loop.")
    reserved_max_chat_rounds: int = Field(
        default=40, description="Reserved max chat rounds for context window management."
    )


class AgentConstraints(BaseModel):
    """High-level agent constraints with a helper to build a ``ConstraintConfig``."""

    max_iteration: int = Field(default=10, description="Maximum iterations for the ReAct loop.")
    max_tool_calls_per_turn: int = Field(default=5, description="Maximum tool calls per turn.")
    reserved_max_chat_rounds: int = Field(
        default=40, description="Reserved max chat rounds for context."
    )

    def to_constraint_config(self) -> ConstraintConfig:
        """Convert to a ``ConstraintConfig`` for use in ``SuperAgentConfig``."""
        return ConstraintConfig(
            max_iteration=self.max_iteration,
            reserved_max_chat_rounds=self.reserved_max_chat_rounds,
        )


class SuperAgentConfig(BaseModel):
    """Enhanced ReAct-style agent configuration.

    Standalone Pydantic model for agent configuration.
    All fields (id, version, description, model, prompt_template, tools,
    constrain, workflows, plugins) are declared explicitly.

    Custom fields:
    - Reasoning model integration (question hints, final-answer extraction)
    - Context-limit retry
    - Tool-call constraints
    - Plan / todo tracking
    - Sub-agent configuration
    """

    # --- Fields formerly inherited from ReActAgentConfig ---
    id: str = Field(default="", description="Agent identifier.")
    version: str = Field(default="0.1", description="Agent version string.")
    description: str = Field(default="", description="Human-readable agent description.")
    model: SuperModelConfig = Field(default_factory=SuperModelConfig, description="LLM model config.")
    prompt_template: list[dict[str, Any]] = Field(default_factory=list, description="System prompt templates.")
    tools: list[str] = Field(default_factory=list, description="Tool name whitelist.")
    constrain: ConstraintConfig = Field(default_factory=ConstraintConfig, description="Execution constraints.")
    workflows: list[dict[str, Any]] = Field(default_factory=list, description="Workflow descriptors (plain dicts).")
    plugins: list[dict[str, Any]] = Field(default_factory=list, description="Plugin descriptors (plain dicts).")

    # --- Custom fields ---
    agent_type: str = Field(default="main", description="Agent type: 'main' or sub-agent name.")

    # Reasoning model integration
    enable_question_hints: bool = Field(default=False, description="Enable question-hints extraction.")
    enable_extract_final_answer: bool = Field(default=False, description="Enable final-answer extraction.")
    open_api_key: str | None = Field(default=None, description="API key for reasoning model.")
    reasoning_model: str = Field(default="o3", description="Reasoning model to use.")
    reasoning_base_url: str | None = Field(default=None, description="Base URL for reasoning model API (for non-OpenAI providers).")

    # Context management
    enable_context_limit_retry: bool = Field(
        default=True, description="Enable context-limit retry with message removal."
    )

    # Tool result keeping
    keep_tool_result: int = Field(
        default=-1, description="Number of tool results to keep in history (-1 = keep all)."
    )

    # Tool call constraints
    max_tool_calls_per_turn: int = Field(default=5, description="Maximum tool calls per turn.")

    # Plan / todo tracking
    enable_todo_plan: bool = Field(default=True, description="Enable todo.md plan tracking.")

    # Sub-agent configuration (main agent only)
    sub_agent_configs: dict[str, SuperAgentConfig] = Field(
        default_factory=dict, description="Sub-agent configurations keyed by agent name."
    )

    # Task guidance
    task_guidance: str = Field(default="", description="Additional guidance for task execution.")


class SuperAgentFactory:
    """Factory helpers for creating ``SuperAgentConfig`` instances.

    Provides convenience methods that wire up ``AgentConstraints`` and
    sensible defaults for main and sub-agent configurations.
    """

    @staticmethod
    def create_main_agent_config(
        agent_id: str,
        agent_version: str,
        description: str,
        model: SuperModelConfig,
        prompt_template: list[dict[str, Any]],
        workflows: list[dict[str, Any]] | None = None,
        plugins: list[dict[str, Any]] | None = None,
        tools: list[str] | None = None,
        max_iteration: int = 20,
        max_tool_calls_per_turn: int = 5,
        enable_question_hints: bool = False,
        enable_extract_final_answer: bool = False,
        open_api_key: str | None = None,
        reasoning_model: str = "o3",
        reasoning_base_url: str | None = None,
        task_guidance: str = "",
        enable_todo_plan: bool = True,
        agent_type: str = "main",
    ) -> SuperAgentConfig:
        """Create a main-agent configuration with the given parameters."""
        constraints = AgentConstraints(
            max_iteration=max_iteration,
            max_tool_calls_per_turn=max_tool_calls_per_turn,
        )
        return SuperAgentConfig(
            id=agent_id,
            version=agent_version,
            description=description,
            model=model,
            prompt_template=prompt_template,
            workflows=workflows or [],
            plugins=plugins or [],
            tools=tools or [],
            constrain=constraints.to_constraint_config(),
            max_tool_calls_per_turn=max_tool_calls_per_turn,
            agent_type=agent_type,
            enable_question_hints=enable_question_hints,
            enable_extract_final_answer=enable_extract_final_answer,
            open_api_key=open_api_key,
            reasoning_model=reasoning_model,
            reasoning_base_url=reasoning_base_url,
            task_guidance=task_guidance,
            enable_todo_plan=enable_todo_plan,
        )

    @staticmethod
    def create_sub_agent_config(
        agent_id: str,
        agent_version: str,
        description: str,
        model: SuperModelConfig,
        prompt_template: list[dict[str, Any]],
        workflows: list[dict[str, Any]] | None = None,
        plugins: list[dict[str, Any]] | None = None,
        tools: list[str] | None = None,
        max_iteration: int = 10,
        max_tool_calls_per_turn: int = 3,
        enable_todo_plan: bool = True,
    ) -> SuperAgentConfig:
        """Create a sub-agent configuration with conservative defaults."""
        constraints = AgentConstraints(
            max_iteration=max_iteration,
            max_tool_calls_per_turn=max_tool_calls_per_turn,
        )
        return SuperAgentConfig(
            id=agent_id,
            version=agent_version,
            description=description,
            model=model,
            prompt_template=prompt_template,
            workflows=workflows or [],
            plugins=plugins or [],
            tools=tools or [],
            constrain=constraints.to_constraint_config(),
            max_tool_calls_per_turn=max_tool_calls_per_turn,
            agent_type=agent_id,
            enable_question_hints=False,
            enable_extract_final_answer=False,
            enable_todo_plan=enable_todo_plan,
        )
