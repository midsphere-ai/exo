"""Reformulator tool — extract clean final answers from agent conversations.

1:1 port of SkyworkAI's ReformulatorTool from src/tool/other_tools/reformulator.py.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from orbiter.tool import Tool
from orbiter.types import SystemMessage, UserMessage

from ..llm_utils import call_llm

logger = logging.getLogger("deepagent")


class ReformulatedAnswer(BaseModel):
    """Response format for reformulated final answer."""
    final_answer: str = Field(
        description=(
            "The final answer extracted from the conversation. "
            "Should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. "
            "Must adhere to any formatting instructions specified in the original question."
        )
    )


_REFORMULATOR_TOOL_DESCRIPTION = """Reformulator tool for reformulating final answers from agent conversations.
This tool takes the original task and the conversation history, then uses an LLM to extract and format the final answer.
Use this tool when you need to produce a clean, formatted final answer from a conversation transcript.

Args:
- task (str): The original task/question that was asked
- data (List[str]): Conversation history in the form of a list of message texts.

Example: {"name": "reformulator", "args": {"task": "What is the capital of France?", "data": ["The capital of France is Paris."]}}.
"""


class ReformulatorTool(Tool):
    """Extract and format clean final answers from conversations.

    1:1 port of SkyworkAI's ReformulatorTool.
    """

    def __init__(self, *, model: str = "openai:gpt-4o-mini") -> None:
        self.name = "reformulator"
        self.description = _REFORMULATOR_TOOL_DESCRIPTION
        self.parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The original task/question that was asked.",
                },
                "data": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Conversation history as a list of message texts.",
                },
            },
            "required": ["task", "data"],
        }
        self._model = model

    async def execute(self, **kwargs: Any) -> str:
        """Reformulate the final answer from a conversation transcript.

        Args:
            task: The original task/question.
            data: Conversation history as a list of message texts.

        Returns:
            The reformulated final answer.
        """
        task: str = kwargs.get("task", "")
        data: list[str] = kwargs.get("data", [])

        if not task:
            return "Error: No task provided."
        if not data:
            return "Error: No conversation data provided."

        try:
            system_prompt = "You are a helpful assistant that reformulates the final answer from a conversation transcript."

            data_string = "\n".join(data)

            agent_message_prompt = f"""Original task:
{task}

Conversation history:
{data_string}

Extract and format the final answer from the conversation history above.

Instructions for formatting the final answer:
- Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
- ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
- You MUST pay attention to the required units of the calculation result. For example, if the question asks "how many thousand hours...", then the answer `1000 hours` should be `1`, not `1000`.
- You MUST pay attention to extracting key stage names, personal names, and location names when the task required.
- If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise.
- If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
- If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
- If you are unable to determine the final answer, output 'Unable to determine'

Reformulated answer is:"""

            messages = [
                SystemMessage(content=system_prompt),
                UserMessage(content=agent_message_prompt),
            ]

            response = await call_llm(
                model=self._model,
                messages=messages,
                response_format=ReformulatedAnswer,
            )

            if not response.success:
                return f"Failed to reformulate answer: {response.message}"

            if response.parsed_model is not None:
                reformulated = response.parsed_model
                return reformulated.final_answer
            else:
                # Fallback: parse from text response
                response_text = response.message.strip()
                if "FINAL ANSWER: " in response_text:
                    return response_text.split("FINAL ANSWER: ")[-1].strip()
                return response_text

        except Exception as e:
            logger.error(f"Error in reformulator tool: {e}")
            return f"Error reformulating answer: {e}"
