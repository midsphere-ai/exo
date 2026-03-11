"""Done tool — signal task completion.

1:1 port of SkyworkAI's DoneTool from src/tool/default_tools/done.py.
"""

from __future__ import annotations

from typing import Any

from orbiter.tool import Tool

_DONE_TOOL_DESCRIPTION = """Done tool for indicating that the task has been completed.
Use this tool to signal that a task or subtask has been finished.
Provide the `result` and `reasoning` of the task in the result and reasoning parameters.

Args:
- result (str): The result of the task completion.
- reasoning (str): The analysis or explanation of the task completion.

Example: {"name": "done", "args": {"reasoning": "The task has been completed successfully.","result": "The task has been completed."}}.
"""


class DoneTool(Tool):
    """Signal task completion with reasoning and result.

    1:1 port of SkyworkAI's DoneTool.
    """

    def __init__(self) -> None:
        self.name = "done"
        self.description = _DONE_TOOL_DESCRIPTION
        self.parameters = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "The reasoning of the task completion.",
                },
                "result": {
                    "type": "string",
                    "description": "The result of the task completion.",
                },
            },
            "required": ["reasoning", "result"],
        }

    async def execute(self, **kwargs: Any) -> str:
        """Indicate that the task has been completed.

        Args:
            reasoning: The reasoning of the task completion.
            result: The result of the task completion.

        Returns:
            Formatted completion message.
        """
        reasoning = kwargs.get("reasoning")
        result = kwargs.get("result")

        if reasoning is None or reasoning == "":
            reasoning = "No reasoning provided"
        else:
            reasoning = str(reasoning)
        if result is None or result == "":
            result = "No result provided"
        else:
            result = str(result)

        return f"Task completed.\n\nReasoning: {reasoning}\n\nResult: {result}"
