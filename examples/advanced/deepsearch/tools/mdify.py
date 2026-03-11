"""Mdify tool — convert various file formats to markdown text.

1:1 port of SkyworkAI's MdifyTool from src/tool/default_tools/mdify.py.
Uses markitdown library for conversion.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from orbiter.tool import Tool

logger = logging.getLogger("deepagent")

_MDIFY_TOOL_DESCRIPTION = """Convert various file formats to markdown text using markitdown and save to base_dir folder.
This tool converts files to markdown format and saves the converted markdown text to the base_dir folder for easy text processing and analysis.
The input should be a file path (absolute path recommended) to the file you want to convert.

Supported file formats:
- Documents: PDF, DOCX, PPTX, XLSX, XLS, CSV, TXT, HTML, EPUB
- Images: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP (with OCR text extraction)
- Audio: MP3, WAV, MP4, M4A (with speech-to-text transcription)
- Archives: ZIP (extracts and converts contents)
- Data: IPYNB (Jupyter notebooks), RSS feeds
- Plain text files

Args:
- file_path (str): The absolute path to the file to convert.
- output_format (str): The output format (default: "markdown").

Example: {"name": "mdify", "args": {"file_path": "/path/to/file.pdf", "output_format": "markdown"}}.
"""


class MdifyTool(Tool):
    """Convert various file formats to markdown text.

    1:1 port of SkyworkAI's MdifyTool.
    """

    def __init__(
        self,
        *,
        base_dir: str | None = None,
        timeout: int = 60,
    ) -> None:
        self.name = "mdify"
        self.description = _MDIFY_TOOL_DESCRIPTION
        self.parameters = {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to convert.",
                },
                "output_format": {
                    "type": "string",
                    "description": "Output format (default: markdown).",
                },
            },
            "required": ["file_path"],
        }
        self._base_dir = base_dir
        self._timeout = timeout
        self._converter = None

        if self._base_dir:
            os.makedirs(self._base_dir, exist_ok=True)

    def _get_converter(self) -> Any:
        """Lazily initialize the markitdown converter."""
        if self._converter is not None:
            return self._converter

        try:
            from markitdown import MarkItDown
            self._converter = MarkItDown()
        except ImportError:
            raise ImportError(
                "markitdown is required. Install with: pip install markitdown"
            )
        return self._converter

    async def execute(self, **kwargs: Any) -> str:
        """Convert a file to markdown.

        Args:
            file_path: Absolute path to the file to convert.
            output_format: Output format.

        Returns:
            Converted markdown content with metadata, or error message.
        """
        file_path: str = kwargs.get("file_path", "")
        output_format: str = kwargs.get("output_format", "markdown")

        try:
            if not file_path.strip():
                return "Error: Empty file path provided"

            if not os.path.exists(file_path):
                return f"Error: File not found: {file_path}"

            if not os.path.isfile(file_path):
                return f"Error: Path is not a file: {file_path}"

            file_size = os.path.getsize(file_path)
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()

            # Check file size (limit to 100MB)
            max_size = 100 * 1024 * 1024
            if file_size > max_size:
                return (
                    f"Error: File too large ({file_size / (1024*1024):.1f}MB). "
                    f"Maximum allowed size is {max_size / (1024*1024)}MB"
                )

            # Run conversion in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._convert_file, file_path, output_format
            )

            if result is None:
                return "Error: Conversion failed - unable to process the file"

            # Save to base_dir if specified
            saved_path = None
            if self._base_dir:
                os.makedirs(self._base_dir, exist_ok=True)
                base_name = os.path.splitext(file_name)[0]
                output_filename = f"{base_name}.md"
                saved_path = os.path.join(self._base_dir, output_filename)
                with open(saved_path, "w", encoding="utf-8") as f:
                    f.write(result)

            response_content = f"Successfully converted file: {file_name}\n"
            response_content += f"File size: {file_size / 1024:.1f} KB\n"
            response_content += f"File extension: {file_ext}\n"
            response_content += f"Output format: {output_format}\n"
            if saved_path:
                response_content += f"Saved to: {saved_path}\n"
            response_content += result

            return response_content

        except asyncio.TimeoutError:
            return f"Error: Conversion timed out after {self._timeout} seconds"
        except Exception as e:
            return f"Error during conversion: {e}"

    def _convert_file(self, file_path: str, output_format: str) -> str | None:
        """Convert file to markdown (synchronous helper)."""
        try:
            converter = self._get_converter()
            result = converter.convert(file_path)
            if result and hasattr(result, "text_content"):
                return result.text_content
            elif result and hasattr(result, "markdown"):
                return result.markdown
            elif isinstance(result, str):
                return result
            else:
                return None
        except Exception as e:
            logger.error(f"Conversion error: {e}")
            return None
