"""Human-in-the-loop tool for agent execution with human oversight."""

from __future__ import annotations

import asyncio
import sys
from abc import ABC, abstractmethod
from typing import Any

from exo.tool import Tool, ToolError


class HumanInputHandler(ABC):
    """Protocol for receiving input from a human during agent execution.

    Implement this to provide custom input mechanisms (console, web UI,
    Slack bot, etc.). The handler is called when the agent invokes the
    HumanInputTool to request user confirmation or free-form input.
    """

    @abstractmethod
    async def get_input(self, prompt: str, choices: list[str] | None = None) -> str:
        """Request input from a human.

        Args:
            prompt: The question or instruction shown to the human.
            choices: Optional list of valid choices. If provided, the
                handler should constrain input to these values.

        Returns:
            The human's response string.
        """


class ConsoleHandler(HumanInputHandler):
    """Interactive console handler that reads from stdin.

    Displays the prompt to stderr and reads a line from stdin.
    When choices are provided, validates the input against them.
    """

    async def get_input(self, prompt: str, choices: list[str] | None = None) -> str:
        """Read input from the console.

        Args:
            prompt: The question to display.
            choices: Optional valid choices to display and validate.

        Returns:
            The user's input string.
        """
        display = prompt
        if choices:
            display += "\nChoices: " + ", ".join(choices)
        display += "\n> "

        # Run blocking stdin read in a thread
        line = await asyncio.to_thread(self._read_line, display)
        stripped = line.strip()

        if choices and stripped not in choices:
            return choices[0]  # Default to first choice on invalid input

        return stripped

    @staticmethod
    def _read_line(prompt: str) -> str:
        """Blocking read from stdin with prompt to stderr."""
        sys.stderr.write(prompt)
        sys.stderr.flush()
        return sys.stdin.readline()


class HumanInputTool(Tool):
    """A tool that pauses agent execution to request human input.

    When the LLM calls this tool, execution blocks until the human
    responds via the configured ``HumanInputHandler``.

    Args:
        handler: The input handler to use. Defaults to ``ConsoleHandler``.
        timeout: Maximum seconds to wait for input. ``None`` means no timeout.
    """

    def __init__(
        self,
        *,
        handler: HumanInputHandler | None = None,
        timeout: float | None = None,
    ) -> None:
        self.name = "human_input"
        self.description = "Ask a human for input, confirmation, or clarification."
        self.parameters: dict[str, Any] = {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The question or instruction to show the human.",
                },
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of valid choices for the human to pick from.",
                },
            },
            "required": ["prompt"],
        }
        self._handler = handler or ConsoleHandler()
        self._timeout = timeout

    async def execute(self, **kwargs: Any) -> str:
        """Execute the human input request.

        Args:
            **kwargs: Must include ``prompt`` (str). May include ``choices`` (list[str]).

        Returns:
            The human's response string.

        Raises:
            ToolError: If the input request times out.
        """
        prompt: str = kwargs.get("prompt", "")
        choices: list[str] | None = kwargs.get("choices")

        try:
            if self._timeout is not None:
                return await asyncio.wait_for(
                    self._handler.get_input(prompt, choices),
                    timeout=self._timeout,
                )
            return await self._handler.get_input(prompt, choices)
        except TimeoutError:
            raise ToolError("Human input request timed out") from None
