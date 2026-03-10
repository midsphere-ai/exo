"""
Super ReAct Agent
Enhanced ReAct Agent with custom context management
Supports both main agent and sub-agent execution with the same class
"""

import json
import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, AsyncIterator, List, Optional, Tuple

import inspect

from openjiuwen.core.agent.agent import BaseAgent
from openjiuwen.core.runtime.runtime import Runtime, Workflow
from openjiuwen.core.utils.tool.base import Tool
from openjiuwen.core.common.logging import logger
from openjiuwen.core.utils.llm.messages import AIMessage
from openjiuwen.core.utils.tool.param import Param
from openjiuwen.core.utils.tool.function.function import LocalFunction


from agent.super_config import SuperAgentConfig
from agent.context_manager import ContextManager
from agent.tool_call_handler import ToolCallHandler
from agent.qa_handler import QAHandler
from agent.plan_tracker import PlanTracker
from llm.openrouter_llm import OpenRouterLLM, ContextLimitError

from openjiuwen.core.utils.tool.mcp.base import ToolServerConfig
from openjiuwen.core.runner.runner import Runner, resource_mgr
from mcp import StdioServerParameters

from agent.prompt_templates import process_input, get_task_instruction_prompt

def _make_mcp_call_coroutine(server_name: str, tool_name: str):
    """
    为某个 MCP 工具生成一个 coroutine 函数：
    - 入参是工具的参数（**kwargs）
    - 内部通过 Runner.run_tool 调用真正的 MCP 工具
    """
    async def _wrapper(**kwargs):
        tool_id = f"{server_name}.{tool_name}"  # 例如：browser-use-server.browser_navigate
        result = await Runner.run_tool(tool_id, kwargs)

        # Test 里约定：如果返回 dict 且有 "result" 字段，就用它
        if isinstance(result, dict) and "result" in result:
            return result["result"]
        return result

    return _wrapper


def _normalize_mcp_server_config(client_type: str, params):
    server_path = None
    params_dict: Dict[str, Any] = {}

    if client_type == "stdio":
        if isinstance(params, StdioServerParameters):
            params_dict = {
                "command": params.command,
                "args": list(params.args) if params.args is not None else None,
                "env": params.env,
                "cwd": params.cwd,
                "encoding_error_handler": getattr(params, "encoding_error_handler", None),
            }
        elif isinstance(params, dict):
            params_dict = params
        elif isinstance(params, str):
            server_path = params

        if not server_path:
            server_path = params_dict.get("command") or "stdio"
        return server_path, params_dict

    if isinstance(params, dict):
        if "server_path" in params:
            server_path = params["server_path"]
            params_dict = {k: v for k, v in params.items() if k != "server_path"}
        elif "url" in params:
            server_path = params["url"]
            params_dict = {k: v for k, v in params.items() if k != "url"}
        else:
            params_dict = params
    else:
        server_path = params

    if server_path is None:
        raise ValueError(f"MCP server_path is required for client_type '{client_type}'")
    return server_path, params_dict

class SuperReActAgent(BaseAgent):
    """
    Enhanced ReAct Agent with custom context management:
    - Custom context management (no ContextEngine dependency)
    - Task logging
    - Reasoning model integration
    - Context limit handling
    - Sub-agent support
    - Main agent and sub-agent use the same class with different instances
    """

    def __init__(
        self,
        agent_config: SuperAgentConfig,
        workflows: List[Workflow] = None,
        tools: List[Tool] = None
    ):
        """
        Initialize Super ReAct Agent

        Args:
            agent_config: Super agent configuration
            workflows: List of workflows
            tools: List of tools
        """
        # Call parent init
        super().__init__(agent_config)

        # Store agent-specific config
        self._agent_config: SuperAgentConfig = agent_config

        # LLM instance (OpenRouter) - create eagerly for context manager
        model_config = agent_config.model
        self._llm = OpenRouterLLM(
            api_key=model_config.model_info.api_key,
            api_base=model_config.model_info.api_base,
            model_name=model_config.model_info.model_name,
            timeout=model_config.model_info.timeout
        )

        # Custom context manager (replaces ContextEngine)
        # Pass LLM for summary generation with retry logic
        self._context_manager = ContextManager(
            llm=self._llm,
            max_history_length=agent_config.constrain.reserved_max_chat_rounds * 2
        )

        # QA handler (for hints and final answer extraction)
        self._qa_handler: Optional[QAHandler] = None
        if agent_config.enable_question_hints or agent_config.enable_extract_final_answer:
            if agent_config.open_api_key: #TODO: make this more general
                self._qa_handler = QAHandler(
                    api_key=agent_config.open_api_key,
                    enable_message_ids=True,
                    reasoning_model=agent_config.reasoning_model
                )

        # Add tools and workflows through BaseAgent interface
        if tools:
            self.add_tools(tools)
        if workflows:
            self.add_workflows(workflows)

        # Sub-agent instances (for main agent only)
        self._sub_agents: Dict[str, "SuperReActAgent"] = {}

        # Tool call handler
        self._tool_call_handler = ToolCallHandler(
            sub_agents=self._sub_agents
        )

    def _get_llm(self) -> OpenRouterLLM:
        """Get LLM instance (always available after __init__)"""
        return self._llm

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        调用一个工具：
        - 从 self._tools 里找到 LocalFunction
        - 支持 func 是同步函数或 async 函数
        - 如果 func 返回的是 coroutine（awaitable），自动 await
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
        params
    ):
        """
        注册一个 MCP server（SSE / stdio ），并把该 server 上所有 tools
        映射成 LocalFunction，返回 List[LocalFunction]，可以直接传给 SuperReActAgent.
        """
        tool_mgr = resource_mgr.tool()

        # 注册 MCP server
        server_path, params_dict = _normalize_mcp_server_config(client_type, params)
        server_cfg = ToolServerConfig(
            server_name=server_name,
            server_path=server_path,
            params=params_dict,
            client_type=client_type,
        )
        
        ok_list = await tool_mgr.add_tool_servers([server_cfg])
        if not ok_list or not ok_list[0]:
            raise RuntimeError(f"Failed to add MCP server: {server_name}")

        # 用 Runner.list_tools 拿到工具列表（McpToolInfo）
        tool_infos = await Runner.list_tools(server_name)

        local_tools = []
        for info in tool_infos:
            info.name = info.name.split(".")[-1]  # 去掉 server_name 前缀
            schema = getattr(info, "input_schema", {}) or {}
            properties = schema.get("properties", {}) or {}
            required = set(schema.get("required", []) or [])

            params_def = []
            for pname, pinfo in properties.items():
                params_def.append(
                    Param(
                        name=pname,
                        description=pinfo.get("description", ""),
                        param_type=pinfo.get("type", "string"),
                        required=pname in required,
                    )
                )

            async_func = _make_mcp_call_coroutine(server_name, info.name)

            mcp_local_tool = LocalFunction(
                name=info.name,
                description=info.description,
                params=params_def,
                func=async_func,   
            )

            local_tools.append(mcp_local_tool)

        return local_tools

    async def create_mcp_tools(self, server_name: str, client_type: str, params) -> List[LocalFunction]:
        """Utility method to create MCP tools based on server type and params"""
        return await self._register_mcp_server_as_local_tools(
            server_name=server_name,
            client_type=client_type,
            params=params,
        )

    def register_sub_agent(self, agent_name: str, sub_agent: "SuperReActAgent"):
        """
        Register a sub-agent instance and add it as a tool

        Args:
            agent_name: Name of the sub-agent (should start with 'agent-' for automatic routing)
            sub_agent: SuperReActAgent instance to register
        """
        # Register sub-agent in the handler's registry
        self._sub_agents[agent_name] = sub_agent

        # Delegate tool creation to ToolCallHandler
        sub_agent_tool = self._tool_call_handler.create_sub_agent_tool(agent_name, sub_agent)

        # Add the tool to this agent's tools
        self.add_tools([sub_agent_tool])

        logger.info(f"Registered sub-agent '{agent_name}' as tool")

    def _format_tool_calls_for_message(self, tool_calls) -> List[Dict]:
        """Format tool calls for message history"""
        return ToolCallHandler.format_tool_calls_for_message(tool_calls)

    async def call_model(
        self,
        user_input: str,
        runtime: Runtime,
        is_first_call: bool = False,
        step_id: int = 0,
        plan_tracker: Optional[PlanTracker] = None
    ) -> AIMessage:
        """
        Call LLM for reasoning

        Args:
            user_input: User input or tool result
            runtime: Runtime instance
            is_first_call: Whether this is the first call (adds user message)
            step_id: Step ID for logging

        Returns:
            AIMessage: LLM output
        """
        # If first call, add user message to context
        if is_first_call:
            self._context_manager.add_user_message(user_input)

        # Get chat history from context manager
        chat_history = self._context_manager.get_history()

        # Format messages with prompt template
        messages = []
        for prompt in self._agent_config.prompt_template:
            messages.append(prompt)

        # Add a system nudge with the active/next plan step so the model prefixes responses
        if plan_tracker and plan_tracker.has_plan():
            active_step = plan_tracker.get_active_or_next_step()
            if active_step:
                messages.append({
                    "role": "system",
                    "content": (
                        f"Active plan step: Step {active_step.index}. "
                        f"Begin your response with 'Step {active_step.index}:' and include this step number when choosing actions."
                    )
                })

        # Add chat history
        messages.extend(chat_history)

        # Get tool definitions from runtime
        # tools = runtime.get_tool_info()
        
        # === 从 runtime 拿到所有工具 ===
        all_tools = runtime.get_tool_info()
        tools = all_tools

        # === 计算当前 agent 允许使用的工具名集合 ===
        allowed_tool_names: set[str] = set()

        try:
            agent_cfg = runtime.get_agent_config()
        except Exception as e:
            agent_cfg = None
            logger.warning(f"Failed to get agent config from runtime: {e}")

        if agent_cfg is not None:
            cfg_tools = getattr(agent_cfg, "tools", None)
            if cfg_tools:
                # cfg_tools 例如 ["tool-vqa", "tool-reading", "tool-code", ...]
                allowed_tool_names.update(cfg_tools)

        # 兜底：用自身 _agent_config.tools（BaseAgent.add_tools 已经维护）
        cfg_tools_self = getattr(self._agent_config, "tools", None)
        if cfg_tools_self:
            allowed_tool_names.update(cfg_tools_self)

        # === 根据 allowed_tool_names 从 all_tools 里筛 ===
        if allowed_tool_names:
            filtered_tools = []
            for t in all_tools:
                # ToolInfo.name 是真正暴露给 LLM 的 function 名
                tool_name = getattr(t, "name", None)

                if tool_name in allowed_tool_names:
                    filtered_tools.append(t)

            tools = filtered_tools
            logger.info(
                f"[SuperReActAgent] Filtered tools for agent {self._agent_config.id}: "
                f"{[getattr(t, 'name', None) for t in tools]}"
            )
        else:
            # 如果没有任何限制配置，就退回到“全量工具”行为，保证兼容性
            logger.warning(
                f"[SuperReActAgent] No tool whitelist found for agent {self._agent_config.id}, "
                f"exposing all {len(all_tools)} tools to LLM"
            )
            tools = all_tools
            
        
        # Call LLM
        llm = self._get_llm()
        llm_output = await llm.ainvoke(
            model_name=self._agent_config.model.model_info.model_name,
            messages=messages,
            tools=tools
        )

        # Save AI message to context
        tool_calls_formatted = self._format_tool_calls_for_message(llm_output.tool_calls)
        self._context_manager.add_assistant_message(
            llm_output.content or "",
            tool_calls=tool_calls_formatted
        )

        return llm_output

    async def _execute_tool_call(
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
        return await self._tool_call_handler.execute_tool_call(tool_call, runtime)

    async def invoke(self, inputs: Dict, runtime: Runtime = None) -> Dict:
        """
        Synchronous invoke - complete ReAct loop

        Args:
            inputs: Input dict {"query": usr_question, "file_path": usr_file} of GAIA
            runtime: Optional runtime (creates one if not provided)

        Returns:
            Result dict with 'output' and 'result_type'
        """
        # Prepare runtime
        runtime_created = False

        if runtime is None:
            runtime = await self._runtime.pre_run(session_id="default", inputs=inputs)
            runtime_created = True

        try:
            user_input = inputs.get("query", "")
            if not user_input:
                return {"output": "No query provided", "result_type": "error"}

            file_path  = inputs.get("file_path", None)
            #1 11.27: Process inputs of GAIA  
            
            if self._agent_config.agent_type == "main":
                user_input = process_input(task_description=user_input, task_file_name=file_path)
            
            
            # Extract question hints if enabled (main agent only)
            qs_notes = ""
            if self._agent_config.enable_question_hints and self._agent_config.agent_type == "main":
                if self._qa_handler:
                    try:
                        qa_hints = await self._qa_handler.extract_hints(user_input)
                        if qa_hints:
                            qs_notes = f"\n\nBefore you begin, please review the following preliminary notes highlighting subtle or easily misunderstood points in the question, which might help you avoid common pitfalls during your analysis (for reference only; these may not be exhaustive):\n\n{qa_hints}"
                    except Exception as e:
                        logger.warning(f"question hints extraction failed: {e}")
                        qs_notes = ""
            
            #2 11.27: add input prompt 
            if self._agent_config.agent_type == "main":
                user_input = get_task_instruction_prompt(task_description=user_input, qs_notes = qs_notes, use_skill=True)
                
            logger.info(f"Complete_user_inputs: {user_input}")
            
            
            # ReAct loop
            iteration = 0
            max_iteration = self._agent_config.constrain.max_iteration
            max_tool_calls_per_turn = self._agent_config.max_tool_calls_per_turn
            is_first_call = True
            task_failed = False
            is_main_agent = (
                getattr(self._agent_config, "agent_type", "") == "main"
                or getattr(self._agent_config, "id", "") == "super_react_main_mcp"
            )
            def _llm_call_for_plan(messages, tools):
                return self._llm._ainvoke(
                    model_name=self._llm.config.model_name,
                    messages=messages,
                    tools=tools,
                )
            plan_tracker = PlanTracker(
                base_dir=Path(__file__).resolve().parent,
                on_context_update=self._context_manager.upsert_system_message,
                llm_call=_llm_call_for_plan if is_main_agent else None,
            ) if (is_main_agent and self._agent_config.enable_todo_plan) else None

            # extracted metadata (populated if final answer extraction succeeds)
            extracted_metadata = None

            while iteration < max_iteration:
                iteration += 1
                if is_main_agent:
                    logger.info(f"====Main iteration {iteration}==== ({self._agent_config.id})")
                else:
                    logger.info(f"====Sub-agent iteration {iteration}==== ({self._agent_config.id})")

                try:
                    # Call model
                    llm_output = await self.call_model(
                        user_input,
                        runtime,
                        is_first_call=is_first_call,
                        step_id=iteration,
                        plan_tracker=plan_tracker
                    )
                    logger.info(f"llm's output: {llm_output.content}")

                    if plan_tracker:
                        try:
                            await plan_tracker.process_llm_output(llm_output)
                        except Exception as plan_error:
                            logger.warning(f"Plan tracking failed to process LLM output: {plan_error}")

                    is_first_call = False

                    # Check for tool calls
                    if not llm_output.tool_calls:
                        logger.info("No tool calls, task completed")
                        break

                    # Execute tool calls
                    num_calls = len(llm_output.tool_calls)
                    if num_calls > max_tool_calls_per_turn:
                        logger.warning(
                            f"Too many tool calls ({num_calls}), processing only first {max_tool_calls_per_turn}"
                        )

                    # Execute all tool calls and collect results (don't add to context yet)
                    for tool_call in llm_output.tool_calls[:max_tool_calls_per_turn]:
                        tool_name = tool_call.name
                        logger.info(f"Executing tool: {tool_name}")

                        try:
                            result = await self._execute_tool_call(tool_call, runtime)
                            # Log out the result to understand inner workings 
                            logger.info(f"Tool {tool_name}'s results: {result} | completed")
                            # logger.info(f"Tool {tool_name} completed")

                            # Add tool result to context immediately after execution
                            self._context_manager.add_tool_message(
                                tool_call.id,
                                str(result)
                            )
                        except Exception as tool_error:
                            logger.error(f"Tool {tool_name} failed: {tool_error}")
                            # Add error as tool result so conversation can continue
                            self._context_manager.add_tool_message(
                                tool_call.id,
                                f"Error executing tool: {str(tool_error)}"
                            )
                            raise  # Re-raise to trigger task_failed

                    # Check context limits (if enabled)
                    if self._agent_config.enable_context_limit_retry:
                        llm = self._get_llm()
                        # Simple prompt for context space estimation
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
                    logger.error(f"Error during iteration {iteration}: {e}")
                    task_failed = True
                    break

            # Check if max iterations reached
            if iteration >= max_iteration:
                logger.warning(f"Max iterations ({max_iteration}) reached")
                task_failed = True

            # Generate summary using context manager
            summary = await self._context_manager.generate_summary(
                task_description=inputs.get("query", ""),
                task_failed=task_failed,
                system_prompts=self._agent_config.prompt_template,
                agent_type=self._agent_config.agent_type
            )
            logger.info(f"Generated summary [final turn of {self._agent_config.agent_type}]: {summary}")

            # final answer extraction (main agent only) based on the pointed reasoning model
            if (self._agent_config.enable_extract_final_answer and
                self._agent_config.agent_type == "main" and self._qa_handler):
                # self._qa_handler and
                # not task_failed):  
                try:
                    # Get answer type
                    answer_type = await self._qa_handler.get_answer_type(inputs.get("query", ""))
                    logger.info(f"answer type detected: {answer_type}")

                    # Extract final answer with type-specific formatting
                    extracted_answer, confidence = await self._qa_handler.extract_final_answer(
                        answer_type=answer_type,
                        task_description=inputs.get("query", ""),
                        summary=summary
                    )

                    # Extract boxed answer for logging
                    boxed_answer = self._qa_handler.extract_boxed_answer(extracted_answer)

                    # Add response to message history
                    # This preserves the reasoning model's analysis in the conversation context
                    self._context_manager.add_assistant_message(
                        f"reasoning model extracted final answer:\n{extracted_answer}",
                        tool_calls=[]
                    )

                    # Concatenate original summary and reasoning answer as final result
                    summary = (
                        f"------------------------------------------Original Summary:------------------------------------------\n"
                        f"{summary}\n\n"
                        f"------------------------------------------Reasoning Model Extracted Answer:------------------------------------------\n"
                        f"{extracted_answer}"
                    )

                    logger.info(
                        f"reasoning model final answer extraction completed - "
                        f"Answer type: {answer_type}, "
                        f"Confidence: {confidence}/100, "
                        f"Boxed answer: {boxed_answer}"
                    )

                    # Store extracted metadata for return
                    extracted_metadata = {
                        "answer_type": answer_type,
                        "confidence": confidence,
                        "boxed_answer": boxed_answer,
                        "full_response": extracted_answer
                    }

                except Exception as e:
                    logger.warning(f"reasoning model final answer extraction failed after retries: {str(e)}")
                    # Continue using original summary

            if plan_tracker and plan_tracker.has_plan():
                plan_tracker.finalize(
                    summary,
                    mark_remaining_complete=not task_failed
                )

            # Build result dict
            result = {
                "output": summary,
                "result_type": "error" if task_failed else "answer"
            }

            # Add extracted metadata if available
            if extracted_metadata:
                result["extracted_metadata"] = extracted_metadata
                
            return result

        finally:
            if runtime_created:
                await runtime.post_run()

    async def stream(self, inputs: Dict, runtime: Runtime = None) -> AsyncIterator[Any]:
        """Streaming invoke - delegates to invoke for now"""
        result = await self.invoke(inputs, runtime)
        yield result
