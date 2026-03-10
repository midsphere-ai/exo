#!/usr/bin/env python
# coding: utf-8
"""
Tool Call Handler
Handles tool call execution, type conversion, and formatting
Also manages sub-agent tool creation and execution
"""

import json
from typing import Dict, Any, List, Optional
import asyncio
from openjiuwen.core.runtime.runtime import Runtime
from openjiuwen.core.common.logging import logger
from openjiuwen.core.utils.tool.function.function import LocalFunction
from openjiuwen.core.utils.tool.param import Param


class ToolCallHandler:
    """
    Handles all tool call related operations:
    - Sub-agent tool creation (creates LocalFunction wrappers)
    - Type conversion for tool arguments
    - Tool execution (regular tools and sub-agents)
    - Tool call formatting for message history
    """

    def __init__(self, sub_agents: Dict[str, Any] = None):
        """
        Initialize ToolCallHandler

        Args:
            sub_agents: Dictionary of sub-agent instances (optional)
        """
        # Use 'is not None' instead of 'or' to preserve empty dict references
        self._sub_agents = sub_agents if sub_agents is not None else {}

    def create_sub_agent_tool(self, agent_name: str, sub_agent: Any) -> LocalFunction:
        """
        Create a LocalFunction tool wrapper for a sub-agent

        Args:
            agent_name: Name of the sub-agent (should start with 'agent-')
            sub_agent: SuperReActAgent instance

        Returns:
            LocalFunction that delegates to the sub-agent
        """
        # Get description from sub-agent config
        description = sub_agent._agent_config.description or f"Sub-agent: {agent_name}"

        # Create a placeholder function
        # Note: Actual execution is handled by execute_tool_call() → _execute_sub_agent()
        def sub_agent_placeholder(subtask: str) -> str:
            """
            Placeholder function for sub-agent tool.
            Actual execution is intercepted by ToolCallHandler.execute_tool_call()
            """
            return f"Sub-agent {agent_name} invoked with task: {subtask}"

        # Create the tool with proper parameters
        sub_agent_tool = LocalFunction(
            name=agent_name,
            description=f"{description}. Delegate a subtask to this specialized agent by providing a clear task description.",
            params=[
                Param(
                    name="subtask",
                    description="The task or question to delegate to this sub-agent. Be specific and provide all necessary context.",
                    param_type="string",
                    required=True
                )
            ],
            func=sub_agent_placeholder
        )

        logger.info(f"Created tool wrapper for sub-agent '{agent_name}'")
        return sub_agent_tool

    def convert_tool_args(self, tool_args: dict, tool) -> dict:
        """
        Convert tool arguments to correct types based on parameter definitions

        Args:
            tool_args: Raw arguments from LLM (may be strings)
            tool: Tool instance with params definition

        Returns:
            Converted arguments with correct types
        """
        if not hasattr(tool, 'params') or not tool.params:
            return tool_args

        converted = {}
        for param in tool.params:
            param_name = param.name
            if param_name not in tool_args:
                continue

            value = tool_args[param_name]
            param_type = param.type

            # Convert based on type
            try:
                if param_type == 'integer':
                    converted[param_name] = int(value)
                elif param_type == 'number':
                    converted[param_name] = float(value)
                elif param_type == 'boolean':
                    if isinstance(value, str):
                        converted[param_name] = value.lower() in ('true', '1', 'yes')
                    else:
                        converted[param_name] = bool(value)
                elif param_type == 'string':
                    converted[param_name] = str(value)
                else:
                    # Keep as-is for complex types
                    converted[param_name] = value
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to convert {param_name} to {param_type}: {e}, using raw value")
                converted[param_name] = value

        return converted

    async def execute_tool_call(
        self,
        tool_call,
        runtime: Runtime
    ) -> Any:
        """
        Execute a single tool call

        Args:
            tool_call: Tool call object from LLM
            runtime: Runtime instance

        Returns:
            Tool execution result
        """
        # Parse tool call
        tool_name = tool_call.name
        try:
            tool_args = json.loads(tool_call.arguments) if isinstance(
                tool_call.arguments, str
            ) else tool_call.arguments
        except (json.JSONDecodeError, AttributeError):
            tool_args = {}

        logger.debug(f"Tool {tool_name} raw args: {tool_args}, types: {[(k, type(v).__name__) for k, v in tool_args.items()]}")

        # Check if this is a sub-agent call
        if tool_name.startswith("agent-"):
            return await self._execute_sub_agent(tool_name, tool_args)
        else:
            return await self._execute_regular_tool(tool_name, tool_args, runtime)

    async def _execute_sub_agent(self, tool_name: str, tool_args: dict) -> Any:
        """
        Execute a sub-agent call

        Args:
            tool_name: Sub-agent tool name
            tool_args: Tool arguments

        Returns:
            Sub-agent result
        """
        if tool_name not in self._sub_agents:
            raise ValueError(f"Sub-agent not found: {tool_name}")

        sub_agent = self._sub_agents[tool_name]
        subtask = tool_args.get("subtask", "")
        subtask += "\n\nPlease provide the answer and detailed supporting information of the subtask given to you."

        # Execute sub-agent
        result = await sub_agent.invoke(
            {"query": subtask},
            runtime=None  # Sub-agent creates its own runtime
        )

        # Return the output from sub-agent
        return result.get("output", "No result from sub-agent")

    async def _execute_regular_tool(
        self,
        tool_name: str,
        tool_args: dict,
        runtime: Runtime
    ) -> Any:
        """
        Execute a regular tool call

        Args:
            tool_name: Tool name
            tool_args: Tool arguments
            runtime: Runtime instance

        Returns:
            Tool execution result
        """
        # Get tool from runtime
        tool = runtime.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        # Convert arguments to correct types
        tool_args = self.convert_tool_args(tool_args, tool)
        logger.debug(f"Tool {tool_name} converted args: {tool_args}, types: {[(k, type(v).__name__) for k, v in tool_args.items()]}")

        # Execute tool
        if tool_name == "auto_browser_use":
            timeout_seconds = 30 * 60  # 30 minutes safety limit for browser tool
            try:
                result = await asyncio.wait_for(tool.ainvoke(tool_args), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.warning(f"Tool {tool_name} timed out after {timeout_seconds} seconds")
                return "No results obtained due to timeout from the browser use for taking too long"
        else:
            result = await tool.ainvoke(tool_args)
        # logger.info(f"Tool {tool_name} executed with result: {result}")
        # Ensure result is string for downstream processing/logging
        if not isinstance(result, str):
            result_str = str(result)
        else:
            result_str = result

        max_len = 100_000  # 100k chars = 25k tokens
        if len(result_str) > max_len:
            result_str = result_str[:max_len] + "\n... [Result truncated]"
        elif len(result_str) == 0:
            result_str = f"Tool call to {tool_name} completed, but produced no specific output or result."
        return result_str

    @staticmethod
    def format_tool_calls_for_message(tool_calls) -> Optional[List[Dict]]:
        """
        Format tool calls for message history

        Args:
            tool_calls: Tool calls from LLM response

        Returns:
            Formatted tool calls for message history, or None if no tool calls
        """
        if not tool_calls:
            return None

        formatted = []
        for tc in tool_calls:
            formatted.append({
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments
                }
            })
        return formatted
