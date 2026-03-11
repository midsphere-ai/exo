"""Python interpreter tool — execute Python code in a sandboxed environment.

1:1 port of SkyworkAI's PythonInterpreterTool from src/tool/default_tools/python_interpreter.py.
Uses smolagents.LocalPythonExecutor for safe execution.
"""

from __future__ import annotations

import logging
from typing import Any

from orbiter.tool import Tool

logger = logging.getLogger("deepagent")

_PYTHON_INTERPRETER_TOOL_DESCRIPTION = """Execute Python code and return the output.
Use this tool to run Python scripts, perform calculations, or execute any Python code.
The tool provides a safe execution environment with access to standard Python libraries.

Args:
- code (str): The Python code to execute.

Example: {"name": "python_interpreter", "args": {"code": "print('Hello, World!')"}}.
"""


class PythonInterpreterTool(Tool):
    """Execute Python code in a sandboxed environment.

    1:1 port of SkyworkAI's PythonInterpreterTool.
    Uses smolagents.LocalPythonExecutor with an authorized imports whitelist.
    """

    def __init__(self, *, authorized_imports: list[str] | None = None) -> None:
        self.name = "python_interpreter"
        self.description = _PYTHON_INTERPRETER_TOOL_DESCRIPTION
        self.parameters = {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute.",
                },
            },
            "required": ["code"],
        }

        # Lazy-init the executor
        self._executor = None
        self._authorized_imports = authorized_imports

    def _get_executor(self) -> Any:
        """Lazily initialize the Python executor."""
        if self._executor is not None:
            return self._executor

        try:
            from smolagents.local_python_executor import (
                LocalPythonExecutor,
                BASE_BUILTIN_MODULES,
                BASE_PYTHON_TOOLS,
            )
        except ImportError:
            from smolagents import (
                LocalPythonExecutor,
            )
            BASE_BUILTIN_MODULES = []
            BASE_PYTHON_TOOLS = {}

        additional_imports = [
            "subprocess", "pandas", "numpy", "matplotlib", "seaborn",
            "scipy", "sklearn", "json", "csv", "os",
            "matplotlib", "matplotlib.pyplot",
        ]

        base = list(BASE_BUILTIN_MODULES) if BASE_BUILTIN_MODULES else []
        if self._authorized_imports:
            base = list(set(base) | set(self._authorized_imports))
        final_authorized = list(set(base) | set(additional_imports))

        self._executor = LocalPythonExecutor(
            additional_authorized_imports=final_authorized,
        )
        if hasattr(self._executor, "send_tools") and BASE_PYTHON_TOOLS:
            self._executor.send_tools(dict(BASE_PYTHON_TOOLS))

        return self._executor

    async def execute(self, **kwargs: Any) -> str:
        """Execute the provided Python code.

        Args:
            code: Python code to execute.

        Returns:
            Stdout + output, or error message.
        """
        code: str = kwargs.get("code", "")

        if not code.strip():
            return "Error: Empty code provided"

        try:
            executor = self._get_executor()
            executor.state = {}
            code_output = executor(code)
            output = f"Stdout:\n{code_output.logs}\nOutput: {code_output.output!s}"
            return output
        except Exception as e:
            return f"Error: {e}"
