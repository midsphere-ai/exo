"""Todo tool — manage task decomposition and step tracking.

1:1 port of SkyworkAI's TodoTool from src/tool/workflow_tools/todo.py.
Simplified for Orbiter: no registry, no SessionContext, uses in-process state.
"""

from __future__ import annotations

import json
import uuid
import os
import asyncio
import re
import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from orbiter.tool import Tool

logger = logging.getLogger("deepagent")


class Step(BaseModel):
    """Step model for todo management."""
    id: str = Field(description="Unique step ID")
    name: str = Field(description="Step name/description")
    parameters: dict[str, Any] | None = Field(default=None, description="Step parameters")
    status: str = Field(default="pending", description="Step status: pending, success, failed")
    result: str | None = Field(default=None, description="Step result (1-3 sentences)")
    priority: str = Field(default="medium", description="Step priority: high, medium, low")
    category: str | None = Field(default=None, description="Step category")
    created_at: str = Field(description="Creation timestamp")
    updated_at: str | None = Field(default=None, description="Last update timestamp")


class Todo:
    """State container for managing todo steps."""

    def __init__(self, todo_file: str, steps_file: str) -> None:
        self.todo_file = todo_file
        self.steps_file = steps_file
        self.steps: list[Step] = []

        # Load existing steps if file exists
        if os.path.exists(steps_file):
            try:
                with open(steps_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.steps = [Step(**step_data) for step_data in data]
            except Exception:
                self.steps = []

    def _generate_step_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{timestamp}_{unique_id}"

    def _save_steps(self) -> None:
        with open(self.steps_file, "w", encoding="utf-8") as f:
            json.dump([step.model_dump() for step in self.steps], f, indent=2, ensure_ascii=False)

    def _sync_to_markdown(self) -> None:
        content = "# Todo List\n\n"

        for step in self.steps:
            priority_emoji = {"high": "H", "medium": "M", "low": "L"}.get(step.priority, "M")
            status_emoji = {"pending": "PENDING", "success": "DONE", "failed": "FAILED"}.get(step.status, "PENDING")
            category_text = f" [{step.category}]" if step.category else ""
            checkbox = "[ ]" if step.status == "pending" else "[x]"

            step_line = f"- {checkbox} **{step.id}** [{priority_emoji}] [{status_emoji}] {step.name}{category_text}"

            if step.parameters:
                step_line += f" *(params: {json.dumps(step.parameters)})*"
            step_line += f" *(created: {step.created_at}*"
            if step.updated_at:
                step_line += f", updated: {step.updated_at}"
            if step.result:
                step_line += f", result: {step.result}"
            step_line += ")"
            content += step_line + "\n"

        with open(self.todo_file, "w", encoding="utf-8") as f:
            f.write(content)

    def add_step(
        self,
        task: str,
        priority: str = "medium",
        category: str | None = None,
        parameters: dict[str, Any] | None = None,
        after_step_id: str | None = None,
        step_id: str | None = None,
    ) -> Step:
        if not task:
            raise ValueError("Step description is required")

        if step_id is None:
            step_id = self._generate_step_id()
        else:
            for step in self.steps:
                if step.id == step_id:
                    raise ValueError(f"Step ID {step_id} already exists")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        new_step = Step(
            id=step_id,
            name=task,
            parameters=parameters,
            status="pending",
            priority=priority,
            category=category,
            created_at=timestamp,
        )

        if after_step_id:
            insert_index = -1
            for i, step in enumerate(self.steps):
                if step.id == after_step_id:
                    insert_index = i + 1
                    break
            if insert_index == -1:
                try:
                    index = int(after_step_id)
                    if 0 <= index < len(self.steps):
                        insert_index = index + 1
                except ValueError:
                    pass
            if insert_index == -1:
                self.steps.append(new_step)
            else:
                self.steps.insert(insert_index, new_step)
        else:
            self.steps.append(new_step)

        self._save_steps()
        self._sync_to_markdown()
        return new_step

    def complete_step(self, step_id: str, status: str, result: str | None = None) -> Step:
        if not step_id:
            raise ValueError("Step ID is required")
        if status not in ["success", "failed"]:
            raise ValueError("Status must be 'success' or 'failed'")

        step = None
        for s in self.steps:
            if s.id == step_id:
                step = s
                break
        if not step:
            raise ValueError(f"Step {step_id} not found")
        if step.status != "pending":
            raise ValueError(f"Step {step_id} is already completed with status: {step.status}")

        step.status = status
        step.result = result
        step.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

        self._save_steps()
        self._sync_to_markdown()
        return step

    def update_step(self, step_id: str, task: str | None = None, parameters: dict[str, Any] | None = None) -> Step:
        if not step_id:
            raise ValueError("Step ID is required")

        step = None
        for s in self.steps:
            if s.id == step_id:
                step = s
                break
        if not step:
            raise ValueError(f"Step {step_id} not found")

        if task:
            step.name = task
        if parameters is not None:
            step.parameters = parameters
        step.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

        self._save_steps()
        self._sync_to_markdown()
        return step

    def clear_completed(self) -> list[Step]:
        completed_steps = [step for step in self.steps if step.status in ["success", "failed"]]
        self.steps = [step for step in self.steps if step.status == "pending"]
        self._save_steps()
        self._sync_to_markdown()
        return completed_steps

    def get_content(self) -> str:
        if not os.path.exists(self.todo_file):
            return "[Current todo.md is empty, fill it with your plan when applicable]"
        try:
            with open(self.todo_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return "[Current todo.md is empty, fill it with your plan when applicable]"


_TODO_TOOL_DESCRIPTION = """Todo tool for managing a todo.md file with task decomposition and step tracking.
When using this tool, only provide parameters that are relevant to the specific operation you are performing.

Available `action` parameters:
1. add: Add a new step to the todo list.
    - task: The description of the step.
    - priority: The priority of the step (high, medium, low).
    - category: The category of the step.
    - parameters: Optional parameters for the step.
    - after_step_id: Optional step ID to insert after.
2. complete: Mark step as completed (success or failed).
    - step_id: The ID of the step to complete.
    - status: Completion status: "success" or "failed".
    - result: Result description (1-3 sentences).
3. update: Update step information.
    - step_id: The ID of the step to update.
    - task: New step description.
    - parameters: New step parameters.
4. list: List all steps with their status.
5. clear: Clear completed steps.
6. show: Show the complete todo.md file content.
7. export: Export todo.md to a specified path.
    - export_path: The target path to export the todo.md file.

Example: {"name": "todo", "args": {"action": "add", "task": "Task description", "priority": "high", "category": "work"}}
Example: {"name": "todo", "args": {"action": "complete", "step_id": "step_1", "status": "success", "result": "Completed successfully"}}
"""


class TodoTool(Tool):
    """Manage task decomposition and step tracking via todo.md.

    1:1 port of SkyworkAI's TodoTool, simplified for Orbiter.
    """

    def __init__(self, *, base_dir: str = "deepagent_output/todo") -> None:
        self.name = "todo"
        self.description = _TODO_TOOL_DESCRIPTION
        self.parameters = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "One of: add, complete, update, list, clear, show, export.",
                    "enum": ["add", "complete", "update", "list", "clear", "show", "export"],
                },
                "task": {"type": "string", "description": "Step description for add/update."},
                "step_id": {"type": "string", "description": "Step identifier for complete/update."},
                "status": {"type": "string", "description": "Completion status for complete action."},
                "result": {"type": "string", "description": "Result summary for complete action."},
                "priority": {"type": "string", "description": "Priority: high, medium, low."},
                "category": {"type": "string", "description": "Category label."},
                "parameters": {"type": "object", "description": "Arbitrary metadata for steps."},
                "after_step_id": {"type": "string", "description": "Insert after this step ID."},
                "export_path": {"type": "string", "description": "Target path for export action."},
            },
            "required": ["action"],
        }
        self._base_dir = base_dir
        os.makedirs(self._base_dir, exist_ok=True)
        self._todo: Todo | None = None
        self._lock = asyncio.Lock()

    def _get_todo(self) -> Todo:
        if self._todo is None:
            todo_file = os.path.join(self._base_dir, "todo.md")
            steps_file = os.path.join(self._base_dir, "steps.json")
            self._todo = Todo(todo_file=todo_file, steps_file=steps_file)
        return self._todo

    async def execute(self, **kwargs: Any) -> str:
        """Execute a todo action.

        Args:
            action: One of add, complete, update, list, clear, show, export.
            task: Step description for add/update.
            step_id: Step identifier for complete/update.
            status: Completion status for complete action.
            result: Result summary for complete action.
            priority: Priority for add action.
            category: Category label for add action.
            parameters: Arbitrary metadata for steps.
            after_step_id: Insert after this step ID.
            export_path: Target path for export action.

        Returns:
            Action result message.
        """
        action: str = kwargs.get("action", "")

        try:
            async with self._lock:
                todo = self._get_todo()

                if action == "add":
                    return self._handle_add(todo, kwargs)
                elif action == "complete":
                    return self._handle_complete(todo, kwargs)
                elif action == "update":
                    return self._handle_update(todo, kwargs)
                elif action == "list":
                    return self._handle_list(todo)
                elif action == "clear":
                    return self._handle_clear(todo)
                elif action == "show":
                    return self._handle_show(todo)
                elif action == "export":
                    return self._handle_export(todo, kwargs)
                else:
                    return f"Unknown action: {action}. Available: add, complete, update, list, clear, show, export"

        except Exception as e:
            logger.error(f"Error in TodoTool: {e}")
            return f"Error executing todo action '{action}': {e}"

    def _handle_add(self, todo: Todo, kwargs: dict[str, Any]) -> str:
        task = kwargs.get("task", "")
        priority = kwargs.get("priority", "medium")
        category = kwargs.get("category")
        parameters = kwargs.get("parameters")
        after_step_id = kwargs.get("after_step_id")
        step_id = kwargs.get("step_id")

        if not task:
            return "Error: Step description is required for add action"

        try:
            new_step = todo.add_step(
                task=task, priority=priority, category=category,
                parameters=parameters, after_step_id=after_step_id, step_id=step_id,
            )
            msg = f"Added step {new_step.id}: {task} (priority: {priority})"
            if after_step_id:
                msg = f"Added step {new_step.id} after {after_step_id}: {task} (priority: {priority})"
            return msg
        except ValueError as e:
            return f"Error: {e}"

    def _handle_complete(self, todo: Todo, kwargs: dict[str, Any]) -> str:
        step_id = kwargs.get("step_id", "")
        status = kwargs.get("status", "")
        result = kwargs.get("result")

        if not step_id:
            return "Error: Step ID is required for complete action"
        if not status:
            return "Error: Status is required for complete action"

        try:
            todo.complete_step(step_id=step_id, status=status, result=result)
            return f"Completed step {step_id} with status: {status}"
        except ValueError as e:
            return f"Error: {e}"

    def _handle_update(self, todo: Todo, kwargs: dict[str, Any]) -> str:
        step_id = kwargs.get("step_id", "")
        task = kwargs.get("task")
        parameters = kwargs.get("parameters")

        if not step_id:
            return "Error: Step ID is required for update action"

        try:
            todo.update_step(step_id=step_id, task=task, parameters=parameters)
            return f"Updated step {step_id}"
        except ValueError as e:
            return f"Error: {e}"

    def _handle_list(self, todo: Todo) -> str:
        if not todo.steps:
            return "No steps found. Use 'add' action to create your first step."

        result = "Todo Steps:\n\n"
        for step in todo.steps:
            status_text = {"pending": "PENDING", "success": "DONE", "failed": "FAILED"}.get(step.status, "PENDING")
            priority_text = {"high": "HIGH", "medium": "MED", "low": "LOW"}.get(step.priority, "MED")
            category_text = f" [{step.category}]" if step.category else ""
            result += f"**{step.id}** [{priority_text}] [{status_text}] {step.name}{category_text}\n"
            if step.parameters:
                result += f"  Parameters: {json.dumps(step.parameters)}\n"
            if step.result:
                result += f"  Result: {step.result}\n"
            result += f"  Created: {step.created_at}"
            if step.updated_at:
                result += f", Updated: {step.updated_at}"
            result += "\n\n"
        return result

    def _handle_clear(self, todo: Todo) -> str:
        completed = todo.clear_completed()
        if not completed:
            return "No completed steps to remove"
        return f"Removed {len(completed)} completed step(s)"

    def _handle_show(self, todo: Todo) -> str:
        content = todo.get_content()
        if content.startswith("[Current todo.md is empty"):
            return "No todo file found. Use 'add' action to create your first step."
        return f"Todo.md content:\n\n{content}"

    def _handle_export(self, todo: Todo, kwargs: dict[str, Any]) -> str:
        export_path = kwargs.get("export_path", "")
        if not export_path:
            return "Error: Export path is required for export action"

        try:
            todo._sync_to_markdown()
            content = todo.get_content()
            if content.startswith("[Current todo.md is empty"):
                return "No todo file found. Use 'add' action to create your first step."

            export_dir = os.path.dirname(export_path)
            if export_dir:
                os.makedirs(export_dir, exist_ok=True)

            with open(export_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully exported todo.md to: {export_path}"
        except Exception as e:
            return f"Error exporting todo.md: {e}"
