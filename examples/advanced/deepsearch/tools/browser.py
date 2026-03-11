"""Browser tool — interact with the web via browser automation.

1:1 port of SkyworkAI's BrowserTool from src/tool/workflow_tools/browser.py.
Uses browser_use library for automation.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from orbiter.tool import Tool

logger = logging.getLogger("deepagent")

_BROWSER_TOOL_DESCRIPTION = """Use the browser to interact with the internet to complete the task.
- If you want to navigate to a search website, bing (https://www.bing.com/) is the best option.
"""


class BrowserTool(Tool):
    """Browser automation tool for web interaction.

    1:1 port of SkyworkAI's BrowserTool.
    Uses browser_use library with configurable model via OpenRouter.
    """

    def __init__(
        self,
        *,
        model_name: str = "openrouter/gpt-4.1",
        base_dir: str = "deepagent_output/browser",
        max_steps: int = 50,
    ) -> None:
        self.name = "browser"
        self.description = _BROWSER_TOOL_DESCRIPTION
        self.parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to complete using the browser.",
                },
            },
            "required": ["task"],
        }
        self._model_name = model_name
        self._base_dir = base_dir
        self._max_steps = max_steps

        os.makedirs(self._base_dir, exist_ok=True)

    async def execute(self, **kwargs: Any) -> str:
        """Use the browser to complete a task.

        Args:
            task: The task to complete.

        Returns:
            Browser task result or error message.
        """
        task: str = kwargs.get("task", "")

        if not task.strip():
            return "Error: No task provided."

        try:
            from browser_use import Agent as BrowserAgent
            from browser_use.llm import ChatOpenAI
        except ImportError:
            return "Error: browser_use is required. Install with: pip install browser-use"

        task_id = f"browser_{uuid.uuid4().hex[:8]}"
        save_dir = os.path.join(self._base_dir, task_id)
        os.makedirs(save_dir, exist_ok=True)

        gif_path = os.path.join(save_dir, "browser.gif")
        logs_path = os.path.join(save_dir, "browser.log")

        agent = None
        try:
            agent = BrowserAgent(
                task=task,
                llm=ChatOpenAI(
                    model=self._model_name.split("/")[-1],
                    base_url=os.getenv("OPENROUTER_API_BASE"),
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                ),
                page_extraction_llm=ChatOpenAI(
                    model=self._model_name.split("/")[-1],
                    base_url=os.getenv("OPENROUTER_API_BASE"),
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                ),
                file_system_path=save_dir,
                generate_gif=gif_path,
                save_conversation_path=logs_path,
                max_steps=self._max_steps,
                verbose=True,
            )

            history = await agent.run()

            # Extract result
            try:
                if hasattr(history, "extracted_content"):
                    contents = history.extracted_content()
                    res = "\n".join(contents) if contents else "No extracted content found"
                elif hasattr(history, "final_result"):
                    res = history.final_result() or "No final result available"
                elif hasattr(history, "history") and history.history:
                    last_step = history.history[-1]
                    res = str(getattr(last_step, "action_results", last_step))
                else:
                    res = "Task completed but no specific results available"
            except Exception as e:
                res = f"Task completed but error extracting results: {e}"

            if hasattr(agent, "close"):
                agent.close()

            return f"Browser task completed.\n\nTask: {task}\n\nResult: {res}\n\nSaved to: {save_dir}"

        except Exception as e:
            if agent and hasattr(agent, "close"):
                try:
                    agent.close()
                except Exception:
                    pass
            logger.error(f"Error in browser tool: {e}")
            return f"Error in browser tool: {e}"
