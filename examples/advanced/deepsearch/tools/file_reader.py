"""File reader tool — read file contents with optional line range.

1:1 port of SkyworkAI's FileReaderTool from src/tool/default_tools/file_reader.py.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from orbiter.tool import Tool

logger = logging.getLogger("deepagent")

_FILE_READER_DESCRIPTION = """File reader tool for reading file contents.

BEST FOR: Reading text files with optional line range:
- Read entire file content
- Read specific line range (start_line to end_line)
- Useful for reviewing reports, logs, code files, etc.

Parameters:
- file_path: Path to the file to read (required)
- start_line: Starting line number (optional, 1-indexed)
- end_line: Ending line number (optional, inclusive)

Examples:
- Read entire file: {"name": "read", "args": {"file_path": "/path/to/report.md"}}
- Read lines 1-400: {"name": "read", "args": {"file_path": "/path/to/report.md", "start_line": 1, "end_line": 400}}
- Read from line 400 to end: {"name": "read", "args": {"file_path": "/path/to/report.md", "start_line": 400}}
"""


class FileReaderTool(Tool):
    """Read file contents with optional line range support.

    1:1 port of SkyworkAI's FileReaderTool. Returns content with line numbers.
    """

    def __init__(self) -> None:
        self.name = "read"
        self.description = _FILE_READER_DESCRIPTION
        self.parameters = {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "ABSOLUTE path to the file to read.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed, optional).",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Ending line number (inclusive, optional).",
                },
            },
            "required": ["file_path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        """Read file contents with optional line range.

        Args:
            file_path: Absolute path to the file.
            start_line: Starting line number (1-indexed).
            end_line: Ending line number (inclusive).

        Returns:
            File contents with line numbers, or error message.
        """
        file_path: str = kwargs.get("file_path", "")
        start_line: int | None = kwargs.get("start_line")
        end_line: int | None = kwargs.get("end_line")

        if not file_path or not file_path.strip():
            return "Error: file_path is required."

        file_path = file_path.strip()

        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"

        if not os.path.isfile(file_path):
            return f"Error: Path is not a file: {file_path}"

        try:
            with open(file_path, encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            return f"Error: Cannot read file as text (binary file?): {file_path}"
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return f"Error reading file: {e}"

        total_lines = len(lines)

        if start_line is not None or end_line is not None:
            # Convert to 0-indexed with automatic boundary handling
            start_idx = (start_line - 1) if start_line else 0
            end_idx = end_line if end_line else total_lines

            if start_idx < 0:
                start_idx = 0
            if start_idx > total_lines:
                start_idx = total_lines
            if end_idx > total_lines:
                end_idx = total_lines
            if end_idx < 0:
                end_idx = 0

            if start_idx >= end_idx:
                if start_line and start_line > total_lines:
                    return (
                        f"File: {file_path}\n"
                        f"Note: start_line ({start_line}) exceeds file length ({total_lines} lines). "
                        f"No content to display."
                    )
                return (
                    f"File: {file_path}\n"
                    f"Note: Line range {start_line}-{end_line} is empty or invalid. "
                    f"File has {total_lines} lines."
                )

            selected_lines = lines[start_idx:end_idx]

            numbered_content = ""
            for i, line in enumerate(selected_lines, start=start_idx + 1):
                numbered_content += f"{i:6}|{line}"

            adjusted_note = ""
            if end_line and end_line > total_lines:
                adjusted_note = f" (requested end_line {end_line} adjusted to {total_lines})"

            logger.info(f"Read file {file_path} lines {start_idx + 1}-{end_idx}")
            return (
                f"File: {file_path}\n"
                f"Lines: {start_idx + 1}-{end_idx} (of {total_lines} total){adjusted_note}\n\n"
                f"{numbered_content}"
            )
        else:
            # Return full content with line numbers
            numbered_content = ""
            for i, line in enumerate(lines, start=1):
                numbered_content += f"{i:6}|{line}"

            logger.info(f"Read file {file_path} ({total_lines} lines)")
            return f"File: {file_path}\nTotal lines: {total_lines}\n\n{numbered_content}"
