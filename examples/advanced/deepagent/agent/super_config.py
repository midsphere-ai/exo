#!/usr/bin/env python
# coding: utf-8
"""
Super Agent Configuration
Enhanced ReAct Agent Config with advanced features
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field

from openjiuwen.agent.config.react_config import ReActAgentConfig, ConstrainConfig
from openjiuwen.agent.common.schema import PluginSchema, WorkflowSchema
from openjiuwen.core.component.common.configs.model_config import ModelConfig


class AgentConstraints(BaseModel):
    """Agent execution constraints"""
    max_iteration: int = Field(default=10, description="Maximum iterations for ReAct loop")
    max_tool_calls_per_turn: int = Field(default=5, description="Maximum tool calls per turn")
    reserved_max_chat_rounds: int = Field(default=40, description="Reserved max chat rounds for context")

    def to_constrain_config(self) -> ConstrainConfig:
        """Convert to ConstrainConfig for compatibility with ReActAgentConfig"""
        return ConstrainConfig(
            max_iteration=self.max_iteration,
            reserved_max_chat_rounds=self.reserved_max_chat_rounds
        )


class SuperAgentConfig(ReActAgentConfig):
    """
    Enhanced ReAct Agent Config with advanced features:
    - question hints and final answer extraction
    - Context limit handling
    - Sub-agent support
    """

    # Agent type (main or sub-agent name)
    agent_type: str = Field(default="main", description="Agent type: main or sub-agent name")

    # Reasoning model integration
    enable_question_hints: bool = Field(default=False, description="Enable question hints extraction")
    enable_extract_final_answer: bool = Field(default=False, description="Enable final answer extraction")
    open_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    reasoning_model: str = Field(default="o3", description="Reasoning model to use")

    # Context management
    enable_context_limit_retry: bool = Field(default=True, description="Enable context limit retry with message removal")

    # Tool result keeping
    keep_tool_result: int = Field(default=-1, description="Number of tool results to keep in history (-1 = keep all)")

    # Tool call constraints
    max_tool_calls_per_turn: int = Field(default=5, description="Maximum tool calls per turn")

    # Plan/todo tracking
    enable_todo_plan: bool = Field(
        default=True,
        description="Enable todo.md plan tracking and context injection"
    )

    # Sub-agent configuration (for main agent only)
    sub_agent_configs: Dict[str, "SuperAgentConfig"] = Field(
        default_factory=dict,
        description="Sub-agent configurations keyed by agent name"
    )

    # Guidance text for summary generation
    task_guidance: str = Field(
        default="",
        description="Additional guidance for task execution and summary generation"
    )


class SuperAgentFactory:
    """Factory for creating Super agents"""

    @staticmethod
    def create_main_agent_config(
        agent_id: str,
        agent_version: str,
        description: str,
        model: ModelConfig,
        prompt_template: List[Dict],
        workflows: List[WorkflowSchema] = None,
        plugins: List[PluginSchema] = None,
        tools: List[str] = None,
        max_iteration: int = 20,
        max_tool_calls_per_turn: int = 5,
        enable_question_hints: bool = False,
        enable_extract_final_answer: bool = False,
        open_api_key: Optional[str] = None,
        reasoning_model: str = "o3",
        task_guidance: str = "",
        enable_todo_plan: bool = True,
        agent_type: str = "main"
    ) -> SuperAgentConfig:
        """Create main agent configuration"""

        constraints = AgentConstraints(
            max_iteration=max_iteration,
            max_tool_calls_per_turn=max_tool_calls_per_turn
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
            constrain=constraints.to_constrain_config(),  # Convert to ConstrainConfig
            max_tool_calls_per_turn=max_tool_calls_per_turn,
            agent_type=agent_type,
            enable_question_hints=enable_question_hints,
            enable_extract_final_answer=enable_extract_final_answer,
            open_api_key=open_api_key,
            reasoning_model=reasoning_model,
            task_guidance=task_guidance,
            enable_todo_plan=enable_todo_plan
        )

    @staticmethod
    def create_sub_agent_config(
        agent_id: str,
        agent_version: str,
        description: str,
        model: ModelConfig,
        prompt_template: List[Dict],
        workflows: List[WorkflowSchema] = None,
        plugins: List[PluginSchema] = None,
        tools: List[str] = None,
        max_iteration: int = 10,
        max_tool_calls_per_turn: int = 3,
        enable_todo_plan: bool = True
    ) -> SuperAgentConfig:
        """Create sub-agent configuration"""

        constraints = AgentConstraints(
            max_iteration=max_iteration,
            max_tool_calls_per_turn=max_tool_calls_per_turn
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
            constrain=constraints.to_constrain_config(),  # Convert to ConstrainConfig
            max_tool_calls_per_turn=max_tool_calls_per_turn,
            agent_type=agent_id,  # Use agent_id as agent_type for sub-agents
            enable_question_hints=False,  # Sub-agents don't use reasoning model
            enable_extract_final_answer=False,
            enable_todo_plan=enable_todo_plan
        )
