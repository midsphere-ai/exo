"""Super ReAct Agent — composition-based orchestrator using Orbiter primitives.

Enhanced ReAct Agent with custom context management, plan tracking,
reasoning-model integration, and sub-agent delegation.  Composes around
``orbiter.agent.Agent`` (via ``super_factory``) rather than extending a
framework base class.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from agent.context_manager import ContextManager
from agent.plan_tracker import PlanTracker
from agent.prompt_templates import get_task_instruction_prompt, process_input
from agent.qa_handler import QAHandler
from agent.super_config import SuperAgentConfig
from agent.tool_call_handler import ToolCallHandler
from llm.openrouter_llm import ContextLimitError, OpenRouterLLM
from mcp import StdioServerParameters

from orbiter.mcp.client import (  # pyright: ignore[reportMissingImports]
    MCPServerConfig,
    MCPServerConnection,
    MCPTransport,
)
from orbiter.models.types import ModelResponse  # pyright: ignore[reportMissingImports]
from orbiter.tool import FunctionTool, Tool

logger = logging.getLogger(__name__)

def _make_mcp_call_coroutine(connection: MCPServerConnection, tool_name: str):
    """Create an async callable that invokes an MCP tool via the server connection.

    Args:
        connection: Live MCP server connection to call tools on.
        tool_name: Name of the tool on the MCP server.

    Returns:
        Async callable that forwards kwargs to the MCP tool and returns the result.
    """
    async def _wrapper(**kwargs):
        result = await connection.call_tool(tool_name, kwargs or None)
        # Extract text content from CallToolResult
        parts = []
        for item in result.content:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(text)
            else:
                parts.append(str(item))
        output = "\n".join(parts)
        if result.isError:
            raise RuntimeError(f"MCP tool '{tool_name}' returned error: {output}")
        return output

    return _wrapper


def _normalize_mcp_server_config(
    server_name: str, client_type: str, params
) -> MCPServerConfig:
    """Normalize MCP server parameters into an MCPServerConfig.

    Supports stdio, SSE, and streamable HTTP transports. Parameters can be
    provided as StdioServerParameters, a dict, or a plain URL/command string.

    Args:
        server_name: Name of the MCP server.
        client_type: Transport type ('stdio', 'sse', or 'streamable_http').
        params: Server parameters — StdioServerParameters, dict, or string.

    Returns:
        MCPServerConfig ready for connection.

    Raises:
        ValueError: If required fields are missing for the transport type.
    """
    if client_type == "stdio":
        if isinstance(params, StdioServerParameters):
            return MCPServerConfig(
                name=server_name,
                transport=MCPTransport.STDIO,
                command=params.command,
                args=list(params.args) if params.args else [],
                env=params.env,
                cwd=params.cwd,
            )
        if isinstance(params, dict):
            return MCPServerConfig(
                name=server_name,
                transport=MCPTransport.STDIO,
                command=params.get("command", ""),
                args=params.get("args", []),
                env=params.get("env"),
                cwd=params.get("cwd"),
            )
        # String: treat as command
        return MCPServerConfig(
            name=server_name,
            transport=MCPTransport.STDIO,
            command=str(params),
        )

    # SSE or streamable_http
    transport = (
        MCPTransport.SSE if client_type == "sse" else MCPTransport.STREAMABLE_HTTP
    )
    if isinstance(params, dict):
        url = params.get("server_path") or params.get("url")
        if not url:
            raise ValueError(
                f"MCP server URL is required for client_type '{client_type}'"
            )
        return MCPServerConfig(name=server_name, transport=transport, url=url)
    if isinstance(params, str):
        return MCPServerConfig(name=server_name, transport=transport, url=params)

    raise ValueError(f"MCP server_path is required for client_type '{client_type}'")

class SuperReActAgent:
    """Composition-based ReAct orchestrator using Orbiter primitives.

    Wraps an OpenRouterLLM provider with custom context management,
    plan tracking, QA hint extraction, MCP tool registration, and
    sub-agent delegation.  Uses ``orbiter.tool.Tool`` for the tool
    interface and ``orbiter.models.types.ModelResponse`` for LLM output.

    The companion ``super_factory.py`` constructs ``orbiter.agent.Agent``
    instances from the same ``SuperAgentConfig``; this class provides
    the full orchestration loop (``invoke``) with GAIA-specific input
    processing, context-limit retry, and reasoning-model final-answer
    extraction.

    Args:
        agent_config: Configuration object specifying model, constraints,
            prompt template, and feature toggles.
        tools: Tool instances available to the agent.
    """

    def __init__(
        self,
        agent_config: SuperAgentConfig,
        tools: list[Tool] | None = None,
    ) -> None:
        self._agent_config: SuperAgentConfig = agent_config

        # Tool registry — O(1) lookup by name
        self._tools: dict[str, Tool] = {}
        if tools:
            for t in tools:
                self._tools[t.name] = t

        # LLM instance (OpenRouter) — created eagerly for context manager
        model_config = agent_config.model
        self._llm = OpenRouterLLM(
            api_key=model_config.model_info.api_key or "",
            api_base=model_config.model_info.api_base or "https://openrouter.ai/api/v1",
            model_name=model_config.model_info.model_name,
            timeout=model_config.model_info.timeout,
        )

        # Custom context manager for history + summary generation
        self._context_manager = ContextManager(
            llm=self._llm,
            max_history_length=agent_config.constrain.reserved_max_chat_rounds * 2,
        )

        # QA handler (for hints and final answer extraction)
        self._qa_handler: QAHandler | None = None
        if (
            (agent_config.enable_question_hints or agent_config.enable_extract_final_answer)
            and agent_config.open_api_key
        ):
            self._qa_handler = QAHandler(
                api_key=agent_config.open_api_key,
                enable_message_ids=True,
                reasoning_model=agent_config.reasoning_model,
            )

        # MCP server connections (for lifecycle management)
        self._mcp_connections: list[MCPServerConnection] = []

        # Sub-agent instances (for main agent only)
        self._sub_agents: dict[str, SuperReActAgent] = {}

        # Tool call handler
        self._tool_call_handler = ToolCallHandler(
            sub_agents=self._sub_agents,
        )

    def _get_llm(self) -> OpenRouterLLM:
        """Return the LLM provider instance."""
        return self._llm

    def add_tools(self, tools: list[Tool] | list[FunctionTool]) -> None:
        """Register additional tools on this agent.

        Args:
            tools: Tool instances to add.
        """
        for t in tools:
            self._tools[t.name] = t

    async def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a registered tool by name.

        Supports both sync and async tool functions.

        Args:
            tool_name: Name of the registered tool.
            arguments: Keyword arguments for the tool.

        Returns:
            The tool's execution result.

        Raises:
            ValueError: If the tool is not registered.
            RuntimeError: If the tool has no callable or execution fails.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' is not registered in SuperReActAgent")

        func = getattr(tool, "func", None)
        if func is None:
            raise RuntimeError(f"Tool '{tool_name}' has no 'func' defined")

        try:
            result = func(**(arguments or {}))
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception as e:
            raise RuntimeError(f"Error while executing tool '{tool_name}': {e}") from e

    async def _register_mcp_server_as_local_tools(
        self,
        server_name: str,
        client_type: str,
        params,
    ) -> list[FunctionTool]:
        """Register an MCP server and create FunctionTool wrappers for its tools.

        Connects to the MCP server, discovers available tools, and wraps each
        as an orbiter FunctionTool with the correct JSON Schema parameters.

        Args:
            server_name: Name of the MCP server.
            client_type: Transport type ('stdio', 'sse', or 'streamable_http').
            params: Server parameters (StdioServerParameters, dict, or string).

        Returns:
            List of FunctionTool instances wrapping the MCP server's tools.

        Raises:
            RuntimeError: If the server connection or tool discovery fails.
        """
        config = _normalize_mcp_server_config(server_name, client_type, params)
        connection = MCPServerConnection(config)
        await connection.connect()
        # Store connection for lifecycle management
        self._mcp_connections.append(connection)

        mcp_tools = await connection.list_tools()

        local_tools: list[FunctionTool] = []
        for mcp_tool in mcp_tools:
            tool_name = mcp_tool.name
            schema = mcp_tool.inputSchema
            parameters = (
                dict(schema) if isinstance(schema, dict)
                else {"type": "object", "properties": {}}
            )

            call_fn = _make_mcp_call_coroutine(connection, tool_name)
            ft = FunctionTool(
                call_fn,
                name=tool_name,
                description=mcp_tool.description or f"MCP tool: {tool_name}",
            )
            # Override auto-generated schema with the actual MCP tool schema
            ft.parameters = parameters
            local_tools.append(ft)

        return local_tools

    async def create_mcp_tools(
        self, server_name: str, client_type: str, params
    ) -> list[FunctionTool]:
        """Create FunctionTool wrappers for all tools on an MCP server.

        Args:
            server_name: Name of the MCP server.
            client_type: Transport type ('stdio', 'sse', or 'streamable_http').
            params: Server parameters.

        Returns:
            List of FunctionTool instances ready for agent use.
        """
        return await self._register_mcp_server_as_local_tools(
            server_name=server_name,
            client_type=client_type,
            params=params,
        )

    def register_sub_agent(self, agent_name: str, sub_agent: SuperReActAgent) -> None:
        """Register a sub-agent and expose it as a callable tool.

        Args:
            agent_name: Unique name (should start with ``'agent-'``).
            sub_agent: SuperReActAgent instance to delegate to.
        """
        self._sub_agents[agent_name] = sub_agent
        sub_agent_tool = self._tool_call_handler.create_sub_agent_tool(agent_name, sub_agent)
        self.add_tools([sub_agent_tool])
        logger.info("Registered sub-agent '%s' as tool", agent_name)

    @staticmethod
    def _format_tool_calls_for_message(tool_calls: Any) -> list[dict[str, Any]] | None:
        """Format tool calls for message history."""
        return ToolCallHandler.format_tool_calls_for_message(tool_calls)

    async def call_model(
        self,
        user_input: str,
        is_first_call: bool = False,
        step_id: int = 0,
        plan_tracker: PlanTracker | None = None,
    ) -> ModelResponse:
        """Call the LLM for one reasoning step.

        Builds the message list from the prompt template and context manager
        history, generates tool schemas from ``self._tools``, and calls the
        LLM via ``OpenRouterLLM.complete()``.

        Args:
            user_input: User input or tool result text.
            is_first_call: If ``True``, adds ``user_input`` to context history.
            step_id: Iteration number for logging.
            plan_tracker: Optional plan tracker for step nudges.

        Returns:
            The LLM's ``ModelResponse`` containing content and tool calls.
        """
        if is_first_call:
            self._context_manager.add_user_message(user_input)

        chat_history = self._context_manager.get_history()

        # Build message list: system prompts + optional plan nudge + history
        messages: list[dict[str, Any]] = list(self._agent_config.prompt_template)

        if plan_tracker and plan_tracker.has_plan():
            active_step = plan_tracker.get_active_or_next_step()
            if active_step:
                messages.append({
                    "role": "system",
                    "content": (
                        f"Active plan step: Step {active_step.index}. "
                        f"Begin your response with 'Step {active_step.index}:' "
                        "and include this step number when choosing actions."
                    ),
                })

        messages.extend(chat_history)

        # Build tool schemas from self._tools (replaces runtime.get_tool_info)
        tool_schemas = [t.to_schema() for t in self._tools.values()]

        # Call LLM via the Orbiter ModelProvider.complete() interface
        llm = self._get_llm()
        llm_output = await llm.complete(messages, tools=tool_schemas)  # type: ignore[arg-type]

        # Save assistant message to context
        tool_calls_formatted = self._format_tool_calls_for_message(llm_output.tool_calls) or []
        self._context_manager.add_assistant_message(
            llm_output.content or "",
            tool_calls=tool_calls_formatted,
        )

        return llm_output

    async def _execute_tool_call(self, tool_call: Any) -> Any:
        """Execute a single tool call via the ToolCallHandler.

        Args:
            tool_call: Tool call object from the LLM response.

        Returns:
            Tool execution result as a string.
        """
        return await self._tool_call_handler.execute_tool_call(tool_call, self._tools)

    async def invoke(self, inputs: dict[str, Any], runtime: Any = None) -> dict[str, Any]:
        """Run the full ReAct loop: input processing, tool execution, summary.

        This is the main orchestration entry point.  It processes GAIA-format
        inputs, runs the iterative LLM + tool-call loop, handles context-limit
        retry, generates a summary, and optionally extracts a final answer via
        a reasoning model.

        Args:
            inputs: Input dict ``{"query": ..., "file_path": ...}``.
            runtime: Unused — kept for backward compatibility.

        Returns:
            Dict with ``output`` (summary text), ``result_type``
            (``"answer"`` or ``"error"``), and optional ``extracted_metadata``.
        """
        user_input = inputs.get("query", "")
        if not user_input:
            return {"output": "No query provided", "result_type": "error"}

        file_path = inputs.get("file_path")

        # Process GAIA-format inputs (main agent only)
        if self._agent_config.agent_type == "main":
            user_input = process_input(task_description=user_input, task_file_name=file_path)

        # Extract question hints if enabled (main agent only)
        qs_notes = ""
        if (
            self._agent_config.enable_question_hints
            and self._agent_config.agent_type == "main"
            and self._qa_handler
        ):
            try:
                qa_hints = await self._qa_handler.extract_hints(user_input)
                if qa_hints:
                    qs_notes = (
                        "\n\nBefore you begin, please review the following preliminary "
                        "notes highlighting subtle or easily misunderstood points in "
                        "the question, which might help you avoid common pitfalls "
                        "during your analysis (for reference only; these may not be "
                        f"exhaustive):\n\n{qa_hints}"
                    )
            except Exception as e:
                logger.warning("question hints extraction failed: %s", e)
                qs_notes = ""

        if self._agent_config.agent_type == "main":
            user_input = get_task_instruction_prompt(
                task_description=user_input, qs_notes=qs_notes, use_skill=True,
            )

        logger.info("Complete_user_inputs: %s", user_input)

        # ReAct loop setup
        iteration = 0
        max_iteration = self._agent_config.constrain.max_iteration
        max_tool_calls_per_turn = self._agent_config.max_tool_calls_per_turn
        is_first_call = True
        task_failed = False
        is_main_agent = (
            getattr(self._agent_config, "agent_type", "") == "main"
            or getattr(self._agent_config, "id", "") == "super_react_main_mcp"
        )

        def _llm_call_for_plan(messages: Any, tools: Any) -> Any:
            return self._llm.complete(messages, tools=tools)  # type: ignore[arg-type]

        plan_tracker = PlanTracker(
            base_dir=Path(__file__).resolve().parent,
            on_context_update=self._context_manager.upsert_system_message,
            llm_call=_llm_call_for_plan if is_main_agent else None,
        ) if (is_main_agent and self._agent_config.enable_todo_plan) else None

        extracted_metadata = None

        while iteration < max_iteration:
            iteration += 1
            label = "Main" if is_main_agent else "Sub-agent"
            logger.info("====%s iteration %d==== (%s)", label, iteration, self._agent_config.id)

            try:
                llm_output = await self.call_model(
                    user_input,
                    is_first_call=is_first_call,
                    step_id=iteration,
                    plan_tracker=plan_tracker,
                )
                logger.info("llm's output: %s", llm_output.content)

                if plan_tracker:
                    try:
                        await plan_tracker.process_llm_output(llm_output)
                    except Exception as plan_error:
                        logger.warning("Plan tracking failed: %s", plan_error)

                is_first_call = False

                if not llm_output.tool_calls:
                    logger.info("No tool calls, task completed")
                    break

                num_calls = len(llm_output.tool_calls)
                if num_calls > max_tool_calls_per_turn:
                    logger.warning(
                        "Too many tool calls (%d), processing first %d",
                        num_calls, max_tool_calls_per_turn,
                    )

                for tool_call in llm_output.tool_calls[:max_tool_calls_per_turn]:
                    tool_name = tool_call.name
                    logger.info("Executing tool: %s", tool_name)

                    try:
                        result = await self._execute_tool_call(tool_call)
                        logger.info("Tool %s completed", tool_name)
                        self._context_manager.add_tool_message(tool_call.id, str(result))
                    except Exception as tool_error:
                        logger.error("Tool %s failed: %s", tool_name, tool_error)
                        self._context_manager.add_tool_message(
                            tool_call.id,
                            f"Error executing tool: {tool_error}",
                        )
                        raise

                # Check context limits
                if self._agent_config.enable_context_limit_retry:
                    llm = self._get_llm()
                    temp_summary = f"Summarize the task: {inputs.get('query', '')}"
                    chat_history = self._context_manager.get_history()
                    if not llm.ensure_summary_context(chat_history, temp_summary):
                        logger.warning("Context limit reached, triggering summary")
                        task_failed = True
                        break

            except ContextLimitError:
                logger.warning("Context limit exceeded during execution")
                task_failed = True
                break
            except Exception as e:
                logger.error("Error during iteration %d: %s", iteration, e)
                task_failed = True
                break

        if iteration >= max_iteration:
            logger.warning("Max iterations (%d) reached", max_iteration)
            task_failed = True

        # Generate summary
        summary = await self._context_manager.generate_summary(
            task_description=inputs.get("query", ""),
            task_failed=task_failed,
            system_prompts=self._agent_config.prompt_template,
            agent_type=self._agent_config.agent_type,
        )
        logger.info("Generated summary [final turn of %s]: %s", self._agent_config.agent_type, summary)

        # Final answer extraction (main agent + reasoning model)
        if (
            self._agent_config.enable_extract_final_answer
            and self._agent_config.agent_type == "main"
            and self._qa_handler
        ):
            try:
                answer_type = await self._qa_handler.get_answer_type(inputs.get("query", ""))
                logger.info("answer type detected: %s", answer_type)

                extracted_answer, confidence = await self._qa_handler.extract_final_answer(
                    answer_type=answer_type,
                    task_description=inputs.get("query", ""),
                    summary=summary,
                )
                boxed_answer = self._qa_handler.extract_boxed_answer(extracted_answer)

                self._context_manager.add_assistant_message(
                    f"reasoning model extracted final answer:\n{extracted_answer}",
                    tool_calls=[],
                )

                summary = (
                    "------------------------------------------Original Summary:"
                    "------------------------------------------\n"
                    f"{summary}\n\n"
                    "------------------------------------------Reasoning Model "
                    "Extracted Answer:------------------------------------------\n"
                    f"{extracted_answer}"
                )

                logger.info(
                    "reasoning model final answer extraction completed - "
                    "Answer type: %s, Confidence: %s/100, Boxed answer: %s",
                    answer_type, confidence, boxed_answer,
                )

                extracted_metadata = {
                    "answer_type": answer_type,
                    "confidence": confidence,
                    "boxed_answer": boxed_answer,
                    "full_response": extracted_answer,
                }
            except Exception as e:
                logger.warning("reasoning model final answer extraction failed: %s", e)

        if plan_tracker and plan_tracker.has_plan():
            plan_tracker.finalize(summary, mark_remaining_complete=not task_failed)

        result: dict[str, Any] = {
            "output": summary,
            "result_type": "error" if task_failed else "answer",
        }
        if extracted_metadata:
            result["extracted_metadata"] = extracted_metadata
        return result

    async def stream(
        self, inputs: dict[str, Any], runtime: Any = None,
    ) -> AsyncIterator[Any]:
        """Streaming invoke — delegates to ``invoke()`` for now.

        Args:
            inputs: Same as ``invoke``.
            runtime: Unused — kept for backward compatibility.

        Yields:
            The complete result dict.
        """
        result = await self.invoke(inputs)
        yield result
