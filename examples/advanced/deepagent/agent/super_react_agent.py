"""
Super ReAct Agent
Enhanced ReAct Agent with custom context management
Supports both main agent and sub-agent execution with the same class
"""

import json
import asyncio
import re
from dataclasses import dataclass, field
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

@dataclass
class PlanStepState:
    """Represents a single plan step and its progress."""
    index: int
    description: str
    status: str = "pending"
    updates: List[str] = field(default_factory=list)
    next_hint: Optional[str] = None

    def add_update(self, note: str):
        if note:
            self.updates.append(note)

    def mark_completed(self):
        self.status = "completed"


class PlanTracker:
    """
    Extracts step-by-step plans from LLM output, writes them to todo.md,
    and appends concise progress snapshots into the agent context.
    """

    _plan_header_pattern = re.compile(r"#PLAN#(?P<plan>.*)", re.IGNORECASE | re.DOTALL)
    # Accept either "#1." or plain "1." step formats
    _plan_step_pattern = re.compile(r"#?\s*(\d+)\.\s*(.+)")
    _step_reference_pattern = re.compile(r"\bstep[\s#:\-]*?(\d+)", re.IGNORECASE)
    _todo_status_block = re.compile(r"<TODO_STATUS>(.*?)</TODO_STATUS>", re.IGNORECASE | re.DOTALL)
    _todo_plan_block = re.compile(r"<TODO_PLAN>(.*?)</TODO_PLAN>", re.IGNORECASE | re.DOTALL)
    _completion_pattern_template = r"step\s*{step}\b.*?(?:completed|complete|done|finished|resolved|answered)"
    _advance_pattern = re.compile(r"(?:move|moving|proceed|proceeding|next|now)\s+(?:to|onto)\s+step\s*(\d+)", re.IGNORECASE)

    def __init__(self, base_dir: Path, context_manager: ContextManager, llm: Optional[OpenRouterLLM] = None):
        self._todo_path = base_dir / "todo.md"
        self._context_manager = context_manager
        self._llm = llm
        self._steps: List[PlanStepState] = []
        self._last_rendered: str = ""
        self._active_step_index: Optional[int] = None

    def has_plan(self) -> bool:
        return bool(self._steps)

    async def process_llm_output(self, llm_output: AIMessage):
        """Handle an LLM message to extract or update plan progress."""
        content = (llm_output.content or "").strip()
        if not content:
            return

        # Apply explicit TODO plan/status blocks first so the persisted todo stays in sync
        plan_changed = self._apply_todo_plan_block(content)
        status_changed = self._apply_todo_status_block(content)
        if plan_changed or status_changed:
            self._write_and_append_context(force_write=True, force_context=True)

        # Only allow plan extraction once, unless the LLM explicitly emits #PLAN#
        if not self._steps or "#PLAN#" in content.upper():
            try:
                if self._extract_plan(content):
                    logger.info("Captured plan from LLM output and initialized todo.md")
                    return
            except Exception as exc:
                logger.warning(f"Plan extraction failed: {exc}")
                return

        if not self._steps:
            return

        # Try LLM-based interpretation to handle varied formats; fall back to heuristic logic
        llm_updated = False
        if self._llm:
            try:
                llm_updated = await self._llm_update_plan(content)
            except Exception as exc:
                logger.warning(f"LLM plan updater failed: {exc}")
        if llm_updated:
            self._write_and_append_context(force_write=True, force_context=True)
            return

        try:
            self._update_progress(content, bool(llm_output.tool_calls))
            # Ensure todo.md is refreshed every turn for the main agent, even if no updates occurred
            self._write_and_append_context(force_write=True)
        except Exception as exc:  # Guard plan tracking from breaking the agent loop
            logger.warning(f"Plan tracking update failed: {exc}")

    def finalize(self, final_note: str, mark_remaining_complete: bool):
        """
        Persist final plan state once the task is complete or aborted.

        Args:
            final_note: Final agent summary or message to attach.
            mark_remaining_complete: If True, mark any pending steps as completed.
        """
        if not self._steps:
            return

        if mark_remaining_complete:
            for step in self._steps:
                if step.status != "completed":
                    step.mark_completed()
                    self._capture_completion_hint(step)

        # Keep final plan updates concise; avoid dumping the full summary into todo.md
        target = None
        if self._active_step_index is not None:
            target = next((s for s in self._steps if s.index == self._active_step_index), None)
        if target is None and self._steps:
            target = self._steps[-1]
        if target:
            target.add_update(f"Step {target.index} completed. Final summary provided separately.")

        self._write_and_append_context()

    def _extract_plan(self, content: str) -> bool:
        """Parse a #PLAN# block and reset plan state."""
        match = self._plan_header_pattern.search(content)
        if match:
            plan_body = match.group("plan") or ""
        else:
            # Only extract plans without #PLAN# if none exist yet (guarded by caller)
            plan_body = content

        steps: List[PlanStepState] = []

        for line in plan_body.splitlines():
            parsed = self._plan_step_pattern.match(line.strip())
            if parsed:
                index = int(parsed.group(1))
                desc = parsed.group(2).strip()
                if desc:
                    steps.append(PlanStepState(index=index, description=desc))

        if not steps:
            return False

        steps = self._dedupe_and_sort_steps(steps)
        self._steps = steps
        self._active_step_index = self._steps[0].index if self._steps else None
        self._write_and_append_context()
        return True

    def _update_progress(self, message: str, has_tool_calls: bool):
        """
        Update the current plan using explicit step references from the LLM response.
        If the model forgets to reference a step, default to the active or next pending step
        so todo.md still captures progress each turn.
        """
        step_refs = sorted(self._extract_step_references(message))
        if not step_refs:
            fallback_step = self._get_step_by_index(self._active_step_index)
            if not fallback_step:
                fallback_step = next((s for s in self._steps if s.status != "completed"), None)

            if fallback_step:
                note = self._clean_note(message)
                if note and (not fallback_step.updates or fallback_step.updates[-1] != note):
                    fallback_step.add_update(note)
                if fallback_step.status == "pending":
                    fallback_step.status = "in_progress"
                self._active_step_index = fallback_step.index
                logger.info(f"No explicit step reference found; recorded progress under Step {fallback_step.index}.")
            else:
                logger.info("No explicit step reference found and no plan steps available to update; persisting todo.md.")

            self._write_and_append_context(force_write=True)
            return

        note = self._clean_note(message)
        step_specific_notes = self._extract_step_notes(message)
        last_seen_index: Optional[int] = None

        for idx in step_refs:
            step = self._get_step_by_index(idx)
            if not step:
                logger.warning(f"Assistant referenced Step {idx}, which is not in the current plan.")
                continue

            last_seen_index = idx

            note_for_step = step_specific_notes.get(idx, note)
            if note_for_step and (not step.updates or step.updates[-1] != note_for_step):
                step.add_update(note_for_step)

            if self._should_mark_completed(message, idx, has_tool_calls):
                step.mark_completed()
                self._capture_completion_hint(step)
                self._advance_active_step(idx)
            else:
                self._active_step_index = idx

        if last_seen_index is not None and self._active_step_index is None:
            self._active_step_index = last_seen_index

        self._write_and_append_context()

    async def _llm_update_plan(self, message: str) -> bool:
        """
        Use an LLM to interpret the latest message and propose plan updates.
        Returns True if any update was applied.
        """
        if not self._llm or not message:
            return False

        if not self._steps:
            return False

        # Build a compact representation of the current plan
        plan_lines = []
        for step in self._steps:
            latest = step.updates[-1] if step.updates else ""
            plan_lines.append(
                f"{step.index}. {step.description} | status={step.status or 'pending'} | latest_note={latest}"
            )
        plan_text = "\n".join(plan_lines)

        system_prompt = (
            "You update a numbered plan. Given the current plan and the latest assistant message, "
            "return JSON only with keys:\n"
            "- plan_updates: list of {\"step\": int, \"status\": one of [pending,in_progress,completed,failed], \"note\": string (optional)}\n"
            "- active_step: int (optional)\n"
            "Do not renumber steps or invent new ones. Keep responses terse JSON only."
        )
        user_prompt = (
            f"Current plan:\n{plan_text}\n\n"
            f"Latest assistant/tool message:\n{message}\n\n"
            "Return the JSON object now."
        )

        resp = await self._llm._ainvoke(
            model_name=self._llm.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=[],
        )
        raw = (resp.content or "").strip()
        if not raw:
            return False
        if not raw.startswith("{"):
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group(0)
        try:
            data = json.loads(raw)
        except Exception as exc:
            logger.debug(f"LLM plan updater returned non-JSON content: {raw} ({exc})")
            return False

        updates = data.get("plan_updates") or []
        active_step = data.get("active_step")
        status_map = {
            "pending": "pending",
            "in_progress": "in_progress",
            "completed": "completed",
            "failed": "failed",
        }

        changed = False
        for upd in updates:
            try:
                idx = int(upd.get("step"))
            except (TypeError, ValueError):
                continue
            step = self._get_step_by_index(idx)
            if not step:
                continue
            status_val = status_map.get(str(upd.get("status", "")).lower())
            note = upd.get("note")
            if status_val and step.status != status_val:
                step.status = status_val
                changed = True
                if status_val == "completed":
                    self._capture_completion_hint(step)
                    self._advance_active_step(idx)
            if note:
                cleaned = self._clean_note(str(note))
                if cleaned and (not step.updates or step.updates[-1] != cleaned):
                    step.add_update(cleaned)
                    changed = True

        if active_step is not None:
            try:
                active_idx = int(active_step)
                if self._get_step_by_index(active_idx):
                    self._active_step_index = active_idx
                    changed = True
            except (TypeError, ValueError):
                pass

        return changed

    def _apply_todo_plan_block(self, message: str) -> bool:
        """Replace the current plan with steps defined inside <TODO_PLAN>...</TODO_PLAN>."""
        match = self._todo_plan_block.search(message or "")
        if not match:
            return False

        body = match.group(1) or ""
        steps: List[PlanStepState] = []

        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            num_match = re.match(r"^\s*(\d+)[\.\)]\s+(.*)", line)
            dash_match = re.match(r"^\s*-\s+(.*)", line)
            desc = None
            if num_match:
                desc = num_match.group(2).strip()
            elif dash_match:
                desc = dash_match.group(1).strip()
            if desc:
                steps.append(PlanStepState(index=len(steps) + 1, description=desc))

        if not steps:
            return False

        steps = self._dedupe_and_sort_steps(steps)
        self._steps = steps
        self._active_step_index = self._steps[0].index if self._steps else None
        return True

    def _apply_todo_status_block(self, message: str) -> bool:
        """Apply status updates defined in a <TODO_STATUS>...</TODO_STATUS> block."""
        match = self._todo_status_block.search(message or "")
        if not match:
            return False

        block = match.group(1) or ""
        status_pattern = re.compile(r"^\s*(\d+)[\.\)]\s*([A-Z_ ]+)\s*(?:[-|:]\s*(.*))?$", re.IGNORECASE)
        status_map = {
            "DONE": "completed",
            "COMPLETED": "completed",
            "SUCCESS": "completed",
            "FAILED": "failed",
            "FAIL": "failed",
            "BLOCKED": "failed",
            "IN_PROGRESS": "in_progress",
            "WORKING": "in_progress",
            "PENDING": "pending",
            "TODO": "pending",
        }

        changed = False
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            m = status_pattern.match(line)
            if not m:
                continue
            try:
                idx = int(m.group(1))
            except (TypeError, ValueError):
                continue
            status_key = (m.group(2) or "").upper().replace(" ", "_")
            status_val = status_map.get(status_key)
            if not status_val:
                continue
            note = (m.group(3) or "").strip()
            step = self._get_step_by_index(idx)
            if not step:
                continue
            if step.status != status_val:
                step.status = status_val
                changed = True
            if note and (not step.updates or step.updates[-1] != note):
                step.add_update(note)
                changed = True

            if status_val == "completed":
                self._capture_completion_hint(step)
                self._advance_active_step(idx)
            elif status_val == "in_progress":
                self._active_step_index = idx

        return changed

    def _extract_step_references(self, message: str) -> set[int]:
        """Return all step indices explicitly mentioned in the message."""
        refs = set()
        for match in self._step_reference_pattern.finditer(message):
            try:
                refs.add(int(match.group(1)))
            except (ValueError, TypeError):
                continue
        return refs

    def _extract_step_notes(self, message: str) -> Dict[int, str]:
        """
        Return a mapping of step index to the note text bounded by that step mention
        and the next step mention. This helps avoid duplicating the same note across steps.
        """
        notes: Dict[int, str] = {}
        matches = list(re.finditer(self._step_reference_pattern, message))
        if not matches:
            return notes

        for i, match in enumerate(matches):
            try:
                idx = int(match.group(1))
            except (ValueError, TypeError):
                continue

            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(message)
            segment = message[start:end].strip()
            cleaned = self._clean_note(segment)
            if cleaned:
                notes[idx] = cleaned
        return notes

    def _get_step_by_index(self, index: Optional[int]) -> Optional[PlanStepState]:
        if index is None:
            return None
        for step in self._steps:
            if step.index == index:
                return step
        return None

    def _dedupe_and_sort_steps(self, steps: List[PlanStepState]) -> List[PlanStepState]:
        """
        Ensure only one entry per step index (keep the last occurrence) and sort by index.
        This heals cases where the LLM emits duplicated numbered lines (e.g., two '1.' rows).
        """
        deduped: Dict[int, PlanStepState] = {}
        for step in steps:
            deduped[step.index] = step  # last occurrence wins
        return [deduped[idx] for idx in sorted(deduped)]

    def get_active_or_next_step(self) -> Optional[PlanStepState]:
        """Return the active step if set, else the next pending step."""
        if not self._steps:
            return None
        if self._active_step_index is not None:
            step = self._get_step_by_index(self._active_step_index)
            if step:
                return step
        for step in self._steps:
            if step.status != "completed":
                return step
        return self._steps[-1]

    def _advance_active_step(self, completed_index: int):
        """Move the active pointer to the next pending step after a completion."""
        for step in self._steps:
            if step.index > completed_index and step.status != "completed":
                self._active_step_index = step.index
                return
        self._active_step_index = completed_index

    def _should_mark_completed(self, message: str, step_index: int, has_tool_calls: bool) -> bool:
        """Determine if a step should be marked completed based on the latest message."""
        completion_pattern = re.compile(
            self._completion_pattern_template.format(step=step_index),
            re.IGNORECASE
        )
        if completion_pattern.search(message):
            return True

        # If the assistant explicitly moves to the next step, consider the prior one done
        advance = self._advance_pattern.search(message)
        if advance:
            try:
                next_step = int(advance.group(1))
                if next_step - 1 == step_index:
                    return True
            except ValueError:
                pass

        # Generic transition to the next step without numbering
        lower_msg = message.lower()
        if any(phrase in lower_msg for phrase in ["next step", "move on to the next step"]):
            if any(step.index > step_index and step.status != "completed" for step in self._steps):
                return True

        # Success keywords combined with step description keywords
        success_terms = {"created", "written", "saved", "downloaded", "completed", "done", "success", "successful", "successfully", "finished", "verified", "exists", "ready"}
        step_tokens = set(re.findall(r"[a-zA-Z]{3,}", self._get_step_by_index(step_index).description.lower() if self._get_step_by_index(step_index) else ""))
        msg_tokens = set(re.findall(r"[a-zA-Z]{3,}", lower_msg))
        if success_terms & msg_tokens and step_tokens and (step_tokens & msg_tokens):
            return True

        return False

    def _capture_completion_hint(self, step: PlanStepState):
        """Persist the 'Next' hint at the moment a step is completed so it won't be overwritten later."""
        if step.next_hint:
            return
        hint = self._next_action_hint(step)
        if hint:
            step.next_hint = self._clean_note(hint, 200)

    def _clean_note(self, note: str, limit: int = 500) -> str:
        if not note:
            return ""
        compact = " ".join(note.split())
        if len(compact) > limit:
            return compact[:limit] + "..."
        return compact

    def _write_and_append_context(self, force_write: bool = False, force_context: bool = False):
        """Persist todo.md and append a concise snapshot to the context."""
        rendered = self._render_markdown()
        changed = rendered != self._last_rendered

        if changed or force_write:
            self._todo_path.parent.mkdir(parents=True, exist_ok=True)
            self._todo_path.write_text(rendered, encoding="utf-8")
            self._last_rendered = rendered

        if changed or force_context:
            summary = self._render_context_summary()
            try:
                self._context_manager.upsert_system_message(
                    summary,
                    "This is the plan list and what has been accomplished so far"
                )
            except Exception as exc:
                logger.warning(f"Failed to append plan summary to context: {exc}")

    def _render_markdown(self) -> str:
        lines = ["# Agent Plan", ""]
        for step in self._steps:
            status_box = self._status_box(step.status)
            lines.append(f"- {status_box} Step {step.index}: {step.description}")
            latest_update = step.updates[-1] if step.updates else "Pending"
            lines.append(f"  - Update: {self._clean_note(latest_update, 500)}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _render_context_summary(self) -> str:
        lines = [
            "This is the plan list and what has been accomplished so far",
            "Plan progress snapshot (todo.md):"
        ]
        for step in self._steps:
            status = step.status
            latest = step.updates[-1] if step.updates else "No updates yet."
            lines.append(f"[{status}] Step {step.index}: {step.description} | Last update: {self._clean_note(latest, 160)}")
        return "\n".join(lines)

    def _status_box(self, status: str) -> str:
        status_lower = (status or "").lower()
        if status_lower == "completed":
            return "✅"
        if status_lower == "failed":
            return "❌"
        if status_lower == "in_progress":
            return "🔄"
        return "✋"

    def _next_action_hint(self, step: PlanStepState) -> str:
        if (step.status or "").lower() == "completed":
            pending = next((s for s in self._steps if s.index > step.index and s.status != "completed"), None)
            return pending.description if pending else "All steps completed."
        if (step.status or "").lower() == "in_progress":
            return "Continue this step."
        return "Work on this step."

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
            plan_tracker = PlanTracker(
                base_dir=Path(__file__).resolve().parent,
                context_manager=self._context_manager,
                llm=self._llm if is_main_agent else None
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
