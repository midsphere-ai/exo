#!/usr/bin/env python
"""
Tool Call Handler

Handles tool call execution, type conversion, and formatting.
Also manages sub-agent tool creation and execution using Orbiter's Tool system.
"""

import asyncio
import json
import logging
from typing import Any

from orbiter.tool import Tool

logger = logging.getLogger(__name__)


class SubAgentTool(Tool):
    """Tool wrapper that delegates execution to a sub-agent.

    When the LLM calls this tool, the ``subtask`` argument is forwarded
    to the wrapped sub-agent's ``invoke()`` method.

    Args:
        agent_name: Unique name for the tool (should start with 'agent-').
        sub_agent: Agent instance with an ``invoke()`` coroutine.
        description: Human-readable description of the sub-agent's purpose.
    """

    def __init__(self, agent_name: str, sub_agent: Any, description: str) -> None:
        self.name = agent_name
        self.description = (
            f"{description}. Delegate a subtask to this specialized agent "
            "by providing a clear task description."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "subtask": {
                    "type": "string",
                    "description": (
                        "The task or question to delegate to this sub-agent. "
                        "Be specific and provide all necessary context."
                    ),
                },
            },
            "required": ["subtask"],
        }
        self._sub_agent = sub_agent

    async def execute(self, **kwargs: Any) -> str:
        """Execute the sub-agent with the given subtask.

        Args:
            **kwargs: Must include ``subtask`` string.

        Returns:
            The sub-agent's output text.
        """
        subtask = kwargs.get("subtask", "")
        subtask += (
            "\n\nPlease provide the answer and detailed supporting "
            "information of the subtask given to you."
        )
        result = await self._sub_agent.invoke(
            {"query": subtask},
            runtime=None,  # Sub-agent creates its own runtime
        )
        return result.get("output", "No result from sub-agent")


class ToolCallHandler:
    """Handles all tool call related operations.

    Responsibilities:
    - Sub-agent tool creation (creates SubAgentTool wrappers)
    - Type conversion for tool arguments
    - Tool execution (regular tools and sub-agents)
    - Tool call formatting for message history
    """

    def __init__(self, sub_agents: dict[str, Any] | None = None) -> None:
        """Initialize ToolCallHandler.

        Args:
            sub_agents: Dictionary mapping agent names to agent instances.
        """
        self._sub_agents = sub_agents if sub_agents is not None else {}

    def create_sub_agent_tool(self, agent_name: str, sub_agent: Any) -> SubAgentTool:
        """Create a SubAgentTool wrapper for a sub-agent.

        Args:
            agent_name: Name of the sub-agent (should start with 'agent-').
            sub_agent: Agent instance with ``_agent_config.description`` and ``invoke()``.

        Returns:
            A SubAgentTool that delegates to the sub-agent.
        """
        description = sub_agent._agent_config.description or f"Sub-agent: {agent_name}"
        tool = SubAgentTool(agent_name, sub_agent, description)
        logger.info("Created tool wrapper for sub-agent '%s'", agent_name)
        return tool

    def convert_tool_args(self, tool_args: dict[str, Any], tool: Tool) -> dict[str, Any]:
        """Convert tool arguments to correct types based on JSON Schema parameters.

        Args:
            tool_args: Raw arguments from LLM (may be strings).
            tool: Tool instance with ``parameters`` JSON Schema.

        Returns:
            Converted arguments with correct types.
        """
        properties = tool.parameters.get("properties", {})
        if not properties:
            return tool_args

        converted: dict[str, Any] = {}
        for param_name, schema in properties.items():
            if param_name not in tool_args:
                continue

            value = tool_args[param_name]
            param_type = schema.get("type", "string")

            try:
                if param_type == "integer":
                    converted[param_name] = int(value)
                elif param_type == "number":
                    converted[param_name] = float(value)
                elif param_type == "boolean":
                    if isinstance(value, str):
                        converted[param_name] = value.lower() in ("true", "1", "yes")
                    else:
                        converted[param_name] = bool(value)
                elif param_type == "string":
                    converted[param_name] = str(value)
                else:
                    converted[param_name] = value
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Failed to convert %s to %s: %s, using raw value",
                    param_name, param_type, e,
                )
                converted[param_name] = value

        return converted

    async def execute_tool_call(
        self,
        tool_call: Any,
        tools: dict[str, Tool],
    ) -> Any:
        """Execute a single tool call.

        Args:
            tool_call: Tool call object from LLM with ``name`` and ``arguments``.
            tools: Dictionary mapping tool names to Tool instances.

        Returns:
            Tool execution result as a string.
        """
        tool_name = tool_call.name
        try:
            tool_args = (
                json.loads(tool_call.arguments)
                if isinstance(tool_call.arguments, str)
                else tool_call.arguments
            )
        except (json.JSONDecodeError, AttributeError):
            tool_args = {}

        logger.debug(
            "Tool %s raw args: %s, types: %s",
            tool_name, tool_args,
            [(k, type(v).__name__) for k, v in tool_args.items()],
        )

        # Look up tool
        tool = tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        # Sub-agent tools handle their own execution via SubAgentTool.execute()
        if isinstance(tool, SubAgentTool):
            return await tool.execute(**tool_args)

        return await self._execute_regular_tool(tool_name, tool_args, tool)

    async def _execute_regular_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool: Tool,
    ) -> str:
        """Execute a regular (non-sub-agent) tool call.

        Args:
            tool_name: Tool name.
            tool_args: Tool arguments.
            tool: The Tool instance to execute.

        Returns:
            Tool execution result as a string.
        """
        tool_args = self.convert_tool_args(tool_args, tool)
        logger.debug(
            "Tool %s converted args: %s, types: %s",
            tool_name, tool_args,
            [(k, type(v).__name__) for k, v in tool_args.items()],
        )

        if tool_name == "auto_browser_use":
            timeout_seconds = 30 * 60  # 30 minutes safety limit
            try:
                result = await asyncio.wait_for(
                    tool.execute(**tool_args), timeout=timeout_seconds,
                )
            except TimeoutError:
                logger.warning("Tool %s timed out after %s seconds", tool_name, timeout_seconds)
                return "No results obtained due to timeout from the browser use for taking too long"
        else:
            result = await tool.execute(**tool_args)

        # Ensure result is string for downstream processing
        result_str = result if isinstance(result, str) else str(result)

        max_len = 100_000  # 100k chars ≈ 25k tokens
        if len(result_str) > max_len:
            result_str = result_str[:max_len] + "\n... [Result truncated]"
        elif len(result_str) == 0:
            result_str = f"Tool call to {tool_name} completed, but produced no specific output or result."
        return result_str

    @staticmethod
    def format_tool_calls_for_message(tool_calls: Any) -> list[dict[str, Any]] | None:
        """Format tool calls for message history.

        Args:
            tool_calls: Tool calls from LLM response.

        Returns:
            Formatted tool calls for message history, or None if empty.
        """
        if not tool_calls:
            return None

        return [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments,
                },
            }
            for tc in tool_calls
        ]
