"""Factory functions for creating Super agents.

Constructs ``exo.agent.Agent`` instances.
Follows the ``deepsearch/agents.py`` pattern — tools are passed as instances,
multi-agent orchestration uses ``exo.swarm.Swarm(mode="team")``.
"""

from __future__ import annotations

from typing import Any

from exo.agent import Agent
from exo.swarm import Swarm
from exo.tool import Tool

from .super_config import (
    ModelInfo,
    SuperAgentConfig,
    SuperAgentFactory,
    SuperModelConfig,
)


def create_super_main_agent(
    agent_id: str = "super_main",
    agent_version: str = "1.0",
    description: str = "Super Main Agent",
    model_name: str = "anthropic/claude-sonnet-4.5",
    api_key: str = "",
    api_base: str = "https://openrouter.ai/api/v1",
    system_prompt: str = "",
    max_iteration: int = 20,
    max_tool_calls_per_turn: int = 5,
    tools: list[Tool] | None = None,
    sub_agent_configs: dict[str, SuperAgentConfig] | None = None,
    enable_o3_hints: bool = False,
    enable_o3_final_answer: bool = False,
    o3_api_key: str | None = None,
    task_guidance: str = "",
    enable_todo_plan: bool = True,
) -> Agent:
    """Create a Super main agent backed by Exo's Agent.

    Args:
        agent_id: Agent identifier.
        agent_version: Agent version string.
        description: Human-readable agent description.
        model_name: LLM model name (e.g. ``"anthropic/claude-sonnet-4.5"``).
        api_key: Provider API key.
        api_base: Provider base URL.
        system_prompt: System prompt text.
        max_iteration: Maximum iterations for the agent loop.
        max_tool_calls_per_turn: Maximum tool calls per turn.
        tools: Tool instances available to the agent.
        sub_agent_configs: Sub-agent configurations (stored on config).
        enable_o3_hints: Enable reasoning-model hint extraction.
        enable_o3_final_answer: Enable reasoning-model final answer extraction.
        o3_api_key: OpenAI API key for reasoning model.
        task_guidance: Additional task guidance text.
        enable_todo_plan: Toggle todo.md plan tracking.

    Returns:
        Configured ``Agent`` instance.
    """
    model_info = ModelInfo(
        api_key=api_key,
        api_base=api_base,
        model_name=model_name,
        timeout=600,
    )
    model_config = SuperModelConfig(model_info=model_info)
    prompt_template = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    )
    tool_names = [t.name for t in (tools or [])]

    agent_config = SuperAgentFactory.create_main_agent_config(
        agent_id=agent_id,
        agent_version=agent_version,
        description=description,
        model=model_config,
        prompt_template=prompt_template,
        tools=tool_names,
        max_iteration=max_iteration,
        max_tool_calls_per_turn=max_tool_calls_per_turn,
        enable_question_hints=enable_o3_hints,
        enable_extract_final_answer=enable_o3_final_answer,
        open_api_key=o3_api_key,
        task_guidance=task_guidance,
        enable_todo_plan=enable_todo_plan,
    )
    if sub_agent_configs:
        agent_config.sub_agent_configs = sub_agent_configs

    return _build_agent(agent_config, tools)


def create_super_sub_agent(
    agent_id: str,
    agent_version: str = "1.0",
    description: str = "",
    model_name: str = "anthropic/claude-sonnet-4.5",
    api_key: str = "",
    api_base: str = "https://openrouter.ai/api/v1",
    system_prompt: str = "",
    max_iteration: int = 10,
    max_tool_calls_per_turn: int = 3,
    tools: list[Tool] | None = None,
    enable_todo_plan: bool = True,
) -> Agent:
    """Create a Super sub-agent backed by Exo's Agent.

    Args:
        agent_id: Agent identifier (also used as agent type).
        agent_version: Agent version string.
        description: Human-readable agent description.
        model_name: LLM model name.
        api_key: Provider API key.
        api_base: Provider base URL.
        system_prompt: System prompt text.
        max_iteration: Maximum iterations for the agent loop.
        max_tool_calls_per_turn: Maximum tool calls per turn.
        tools: Tool instances available to the agent.
        enable_todo_plan: Toggle todo.md plan tracking.

    Returns:
        Configured ``Agent`` instance.
    """
    model_info = ModelInfo(
        api_key=api_key,
        api_base=api_base,
        model_name=model_name,
        timeout=600,
    )
    model_config = SuperModelConfig(model_info=model_info)
    prompt_template = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    )
    tool_names = [t.name for t in (tools or [])]

    agent_config = SuperAgentFactory.create_sub_agent_config(
        agent_id=agent_id,
        agent_version=agent_version,
        description=description,
        model=model_config,
        prompt_template=prompt_template,
        tools=tool_names,
        max_iteration=max_iteration,
        max_tool_calls_per_turn=max_tool_calls_per_turn,
        enable_todo_plan=enable_todo_plan,
    )

    return _build_agent(agent_config, tools)


def create_agent_system_with_sub_agents(
    main_agent_params: dict[str, Any],
    sub_agent_configs: dict[str, dict[str, Any]],
) -> Swarm:
    """Create a multi-agent system using Exo's Swarm.

    Builds sub-agents and a main agent, then returns a ``Swarm(mode="team")``
    where the main agent leads and sub-agents are workers.  The Swarm
    auto-generates ``delegate_to_<worker>`` tools for the lead agent.

    Args:
        main_agent_params: Keyword arguments for ``create_super_main_agent``.
        sub_agent_configs: Mapping of agent name to keyword arguments for
            ``create_super_sub_agent``.

    Returns:
        Configured ``Swarm`` instance.

    Example::

        swarm = create_agent_system_with_sub_agents(
            main_agent_params={
                "agent_id": "main",
                "api_key": "...",
                "system_prompt": "...",
                "tools": main_tools,
            },
            sub_agent_configs={
                "agent-browser": {
                    "agent_id": "agent-browser",
                    "description": "Browser agent",
                    "api_key": "...",
                    "tools": browser_tools,
                },
            },
        )
    """
    workers = [
        create_super_sub_agent(**params)
        for params in sub_agent_configs.values()
    ]

    lead = create_super_main_agent(**main_agent_params)

    return Swarm(agents=[lead, *workers], mode="team")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_agent(config: SuperAgentConfig, tools: list[Tool] | None) -> Agent:
    """Construct an ``Agent`` from a ``SuperAgentConfig`` and tool instances.

    Args:
        config: Agent configuration with model/prompt/constraint settings.
        tools: Tool instances to attach.

    Returns:
        Configured ``Agent`` instance.
    """
    model_str = _resolve_model_string(config)
    system_prompt = _extract_system_prompt(config)

    return Agent(
        name=config.id or "agent",
        model=model_str,
        instructions=system_prompt,
        tools=tools or [],
        max_steps=config.constrain.max_iteration,
    )


def _resolve_model_string(config: SuperAgentConfig) -> str:
    """Build an Exo ``"provider:model"`` string from config.

    Args:
        config: Agent configuration.

    Returns:
        Model string such as ``"openrouter:anthropic/claude-sonnet-4.5"``.
    """
    model_name = config.model.model_info.model_name
    api_base = config.model.model_info.api_base or ""
    if "openrouter" in api_base:
        return f"openrouter:{model_name}"
    return model_name


def _extract_system_prompt(config: SuperAgentConfig) -> str:
    """Extract the system prompt text from the config's prompt_template.

    Args:
        config: Agent configuration.

    Returns:
        System prompt string, or empty string if not set.
    """
    for entry in config.prompt_template:
        if entry.get("role") == "system":
            return entry.get("content", "")
    return ""
