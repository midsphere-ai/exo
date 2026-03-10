"""
Handles model calls for hints extraction and final answer extraction with type-specific formatting
"""

import re
from typing import Optional, Dict, Tuple
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from openjiuwen.core.common.logging import logger
from agent.prompt_templates import (
    get_question_hints_prompt,
    get_answer_type_prompt,
    get_final_answer_prompt
)


class QAHandler:
    """
    Handler for high-level model integration

    Features:
    - Extract task hints to identify potential challenges
    - Determine answer type (number, date, time, string)
    - Extract final answer with type-specific formatting
    - Confidence scoring
    - Message ID support
    """

    def __init__(self, api_key: str, enable_message_ids: bool = True, reasoning_model: str = "o3"):
        """
        Initialize QA Handler

        Args:
            api_key: e.g., OpenAI API key for O3 model
            enable_message_ids: Whether to add message IDs to requests
        """
        self.client = AsyncOpenAI(api_key=api_key, timeout=600) # TODO: support more reasoning models
        self.enable_message_ids = enable_message_ids
        self.r_model = reasoning_model

    def _generate_message_id(self) -> str:
        """Generate random message ID using common LLM format"""
        import uuid
        return f"msg_{uuid.uuid4().hex[:8]}"

    @retry(wait=wait_exponential(multiplier=15), stop=stop_after_attempt(1))
    async def extract_hints(self, question: str) -> str:
        """
        Use reasoning model to extract task hints

        Args:
            question: Task description/question

        Returns:
            Extracted hints highlighting potential challenges

        Raises:
            ValueError: If model returns empty result
        """
        content = get_question_hints_prompt(question)
        if self.enable_message_ids:
            message_id = self._generate_message_id()
            content = f"[{message_id}] {content}"

        response = await self.client.chat.completions.create(       # TODO: support more reasoning models
            model=self.r_model,
            messages=[{"role": "user", "content": content}],
            reasoning_effort="high"
        )

        logger.debug(f"current reasoning model {self.r_model} hints extraction response: {response}")

        result = response.choices[0].message.content

        # Check if result is empty, raise exception to trigger retry if empty
        if not result or not result.strip():
            raise ValueError(f"{self.r_model} hints extraction returned empty result")

        return result

    @retry(wait=wait_exponential(multiplier=15), stop=stop_after_attempt(5))
    async def get_answer_type(self, task_description: str) -> str:
        """
        Determine the expected answer type for the task

        Args:
            task_description: Task description/question

        Returns:
            Answer type: "number", "date", "time", or "string"

        Raises:
            ValueError: If answer type detection returns empty result
        """
        content = get_answer_type_prompt(task_description)
        logger.debug(f"Answer type instruction: {content}")
        if self.enable_message_ids:
            message_id = self._generate_message_id()
            content = f"[{message_id}] {content}"

        response = await self.client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": content}],
            reasoning_effort="medium"
        )

        answer_type = response.choices[0].message.content

        # Check if result is empty, raise exception to trigger retry if empty
        if not answer_type or not answer_type.strip():
            raise ValueError("Answer type detection returned empty result")

        logger.debug(f"Answer type: {answer_type}")

        return answer_type.strip()

    @retry(wait=wait_exponential(multiplier=15), stop=stop_after_attempt(5))
    async def extract_final_answer(
        self,
        answer_type: str,
        task_description: str,
        summary: str
    ) -> Tuple[str, Optional[int]]:
        """
        Use reasoning model to extract final answer from summary with type-specific formatting

        Args:
            answer_type: Expected answer type (date, number, time, string)
            task_description: Original task description
            summary: Agent's summary of the task execution

        Returns:
            Tuple of (full_response, confidence_score)
            - full_response: Complete reasoning model response with analysis and boxed answer
            - confidence_score: Confidence score (0-100), None if not found

        Raises:
            ValueError: If the reasoning model returns empty result or no boxed answer
        """
        # Get type-specific prompt from template
        content = get_final_answer_prompt(answer_type, task_description, summary)
        logger.debug(f"reasoning model {self.r_model} Extract Final Answer Prompt:")
        logger.debug(content)
        if self.enable_message_ids:
            message_id = self._generate_message_id()
            content = f"[{message_id}] {content}"

        response = await self.client.chat.completions.create(
            model=self.r_model,
            messages=[{"role": "user", "content": content}],
            reasoning_effort="medium",
        )

        result = response.choices[0].message.content

        # Check if result is empty, raise exception to trigger retry if empty
        if not result or not result.strip():
            raise ValueError(f"reasoning model {self.r_model} final answer extraction returned empty result")

        # Verify boxed answer exists
        match = re.search(r"\\boxed{([^}]*)}", result)
        if not match:
            raise ValueError(f"reasoning model {self.r_model} final answer extraction returned empty answer")

        # Extract confidence score if present
        confidence = None
        conf_match = re.search(r"\*\*Confidence:\*\*\s*(\d+)", result)
        if conf_match:
            confidence = int(conf_match.group(1))

        return result, confidence

    def extract_boxed_answer(self, reasoning_response: str) -> Optional[str]:
        """
        Extract the boxed answer from reasoning model response

        Args:
            reasoning_response: Full reasoning model response text

        Returns:
            Boxed answer content, or None if not found
        """
        match = re.search(r"\\boxed{([^}]*)}", reasoning_response)
        if match:
            return match.group(1)
        return None
