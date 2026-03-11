"""Bash tool — execute shell commands asynchronously.

1:1 port of SkyworkAI's BashTool from src/tool/default_tools/bash.py.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from orbiter.tool import Tool

logger = logging.getLogger("deepagent")

_BASH_TOOL_DESCRIPTION = """Execute bash commands in the shell.

IMPORTANT:
- Use this tool to run system commands, scripts, or any bash operations.
- Be careful with commands that modify the system or require elevated privileges.
- For file operations, ALWAYS use ABSOLUTE paths to avoid path-related issues.
- Input should be a VALID bash command string.

Args:
- command (str): The command to execute. If file path is necessary, it should be an absolute path.

Example: {"name": "bash", "args": {"command": "ls -l /path/to/file.txt"}}.
"""


class BashTool(Tool):
    """Execute bash commands asynchronously with timeout protection.

    1:1 port of SkyworkAI's BashTool.
    """

    def __init__(self, *, timeout: int = 30) -> None:
        self.name = "bash"
        self.description = _BASH_TOOL_DESCRIPTION
        self.parameters = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to execute. If file path is necessary, it should be an absolute path.",
                },
            },
            "required": ["command"],
        }
        self._timeout = timeout

    async def execute(self, **kwargs: Any) -> str:
        """Execute a bash command asynchronously.

        Args:
            command: The bash command to execute.

        Returns:
            Command output (stdout + stderr) or error message.
        """
        command: str = kwargs.get("command", "")

        if not command.strip():
            return "Error: Empty command provided"

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return f"Error: Command timed out after {self._timeout} seconds"

            stdout_str = stdout.decode("utf-8", errors="replace").strip()
            stderr_str = stderr.decode("utf-8", errors="replace").strip()

            result: list[str] = []
            if stdout_str:
                result.append(f"STDOUT:\n{stdout_str}")
            if stderr_str:
                result.append(f"STDERR:\n{stderr_str}")

            exit_code = process.returncode
            if exit_code != 0:
                result.append(f"Exit code: {exit_code}")

            return "\n\n".join(result) if result else f"Command completed with exit code: {exit_code}"

        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return f"Error executing command: {e}"
