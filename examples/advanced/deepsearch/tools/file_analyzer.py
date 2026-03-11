"""File analyzer tool — reads and analyzes local files and URLs.

Simplified port of SkyworkAI's DeepAnalyzerTool. Supports text file reading
and LLM-based analysis with chunking for large files.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from orbiter.tool import Tool
from orbiter.types import UserMessage

from ..llm_utils import call_llm

logger = logging.getLogger("deepagent")

# File type extensions (from SkyworkAI)
TEXT_EXTENSIONS = {
    ".txt", ".md", ".json", ".csv", ".xml", ".yaml", ".yml",
    ".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".h",
    ".ts", ".tsx", ".jsx", ".go", ".rs", ".rb", ".php", ".sh",
    ".toml", ".ini", ".cfg", ".conf", ".log",
}


class FileAnalyzerTool(Tool):
    """Read and analyze files or URL content for research tasks.

    Simplified port of SkyworkAI's DeepAnalyzerTool. Supports:
    - Local text files: reads content directly
    - URLs: fetches content via httpx
    - LLM-based analysis focused on the given task
    - Chunking for large files
    """

    def __init__(
        self,
        *,
        tool_model: str = "openai:gpt-4o-mini",
        chunk_size: int = 400,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        content_max_length: int = 4096,
    ) -> None:
        self.name = "analyze_file"
        self.description = (
            "Read and analyze a local file or URL content. "
            "Extracts information relevant to the given task from the file."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The analysis task — what information to extract.",
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to a local file or URL to analyze.",
                },
            },
            "required": ["task", "file_path"],
        }
        self._tool_model = tool_model
        self._chunk_size = chunk_size
        self._max_file_size = max_file_size
        self._content_max_length = content_max_length

    async def execute(self, **kwargs: Any) -> str:
        """Analyze a file or URL for the given task.

        Args:
            task: What information to look for.
            file_path: Local file path or URL.

        Returns:
            Analysis results as a string.
        """
        task: str = kwargs.get("task", "")
        file_path: str = kwargs.get("file_path", "")

        if not task:
            return "Error: No analysis task provided."
        if not file_path:
            return "Error: No file path provided."

        try:
            # Determine if URL or local file
            if file_path.startswith(("http://", "https://")):
                content = await self._fetch_url(file_path)
            else:
                content = self._read_local_file(file_path)

            if content.startswith("Error:"):
                return content

            # Analyze content (with chunking for large files)
            return await self._analyze_content(task, content, file_path)

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return f"Error analyzing file: {e}"

    def _read_local_file(self, file_path: str) -> str:
        """Read a local file's content."""
        if not os.path.exists(file_path):
            return f"Error: File not found: {file_path}"

        file_size = os.path.getsize(file_path)
        if file_size > self._max_file_size:
            return f"Error: File too large ({file_size} bytes, max {self._max_file_size})"

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in TEXT_EXTENSIONS and ext != "":
            return f"Error: Unsupported file type: {ext}. Supported: {', '.join(sorted(TEXT_EXTENSIONS))}"

        try:
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, encoding="latin-1") as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {e}"

    async def _fetch_url(self, url: str) -> str:
        """Fetch content from a URL."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            pass

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=httpx.Timeout(20.0),
            ) as client:
                resp = await client.get(
                    url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        ),
                    },
                )
                resp.raise_for_status()

                content_type = resp.headers.get("content-type", "")
                if "text/html" in content_type:
                    try:
                        from bs4 import BeautifulSoup

                        soup = BeautifulSoup(resp.text, "html.parser")
                        for tag in soup(["script", "style", "nav", "footer", "header"]):
                            tag.decompose()
                        text = soup.get_text(separator="\n", strip=True)
                        lines = [line.strip() for line in text.splitlines() if line.strip()]
                        return "\n".join(lines)
                    except ImportError:
                        return resp.text
                return resp.text
        except Exception as e:
            return f"Error fetching URL: {e}"

    async def _analyze_content(
        self, task: str, content: str, file_path: str
    ) -> str:
        """Analyze file content, chunking if necessary.

        Port of SkyworkAI's text file analysis pattern:
        chunk -> analyze each -> combine summaries.
        """
        lines = content.splitlines()
        total_lines = len(lines)

        # If content fits in one chunk, analyze directly
        if total_lines <= self._chunk_size:
            return await self._analyze_single(task, content, file_path)

        # Chunk analysis (from SkyworkAI _analyze_text_file)
        total_chunks = (total_lines + self._chunk_size - 1) // self._chunk_size
        logger.info(f"Analyzing {total_lines} lines in {total_chunks} chunks")

        summaries: list[str] = []
        for chunk_num in range(1, total_chunks + 1):
            start_line = (chunk_num - 1) * self._chunk_size
            end_line = min(start_line + self._chunk_size, total_lines)
            chunk_text = "\n".join(lines[start_line:end_line])

            chunk_summary = await self._analyze_chunk(
                task, chunk_text, chunk_num, start_line + 1, end_line
            )
            summaries.append(chunk_summary)

            # Check if answer was found in this chunk
            if "answer found" in chunk_summary.lower() or "found the answer" in chunk_summary.lower():
                logger.info(f"Answer found in chunk {chunk_num}, stopping early")
                break

        # Combine chunk summaries
        if len(summaries) == 1:
            return summaries[0]

        combined = "\n\n".join(
            f"### Chunk {i + 1}\n{s}" for i, s in enumerate(summaries)
        )

        # Final synthesis
        return await self._synthesize_chunks(task, combined, file_path)

    async def _analyze_single(
        self, task: str, content: str, file_path: str
    ) -> str:
        """Analyze a single piece of content."""
        if len(content) > self._content_max_length * 4:
            content = content[: self._content_max_length * 4] + "\n...(truncated)"

        prompt = (
            f"Analyze the following file content and extract information relevant to the task.\n\n"
            f"Task: {task}\n\n"
            f"File: {file_path}\n\n"
            f"Content:\n{content}\n\n"
            f"Provide a comprehensive analysis that:\n"
            f"1. Directly addresses the task\n"
            f"2. Includes specific details, quotes, and data points\n"
            f"3. Summarizes key findings clearly\n"
            f"4. Notes if the answer to the task was found\n\n"
            f"Return your analysis."
        )

        response = await call_llm(
            model=self._tool_model,
            messages=[UserMessage(content=prompt)],
        )
        return response.message.strip()

    async def _analyze_chunk(
        self,
        task: str,
        chunk_text: str,
        chunk_num: int,
        start_line: int,
        end_line: int,
    ) -> str:
        """Analyze a single chunk (from SkyworkAI _analyze_markdown_chunk)."""
        prompt = (
            f"Analyze this chunk of the document and extract information relevant to the task.\n\n"
            f"Task: {task}\n\n"
            f"Current chunk (lines {start_line}-{end_line}):\n{chunk_text}\n\n"
            f"Extract key information that helps answer the task. "
            f"Provide a concise summary (2-3 sentences) of findings from this chunk.\n"
            f"If this chunk contains the answer to the task, clearly state 'Answer found' "
            f"and provide the answer."
        )

        response = await call_llm(
            model=self._tool_model,
            messages=[UserMessage(content=prompt)],
        )
        return response.message.strip()

    async def _synthesize_chunks(
        self, task: str, combined: str, file_path: str
    ) -> str:
        """Synthesize findings from multiple chunks."""
        prompt = (
            f"Synthesize the following chunk analyses into a comprehensive summary.\n\n"
            f"Task: {task}\n"
            f"File: {file_path}\n\n"
            f"Chunk Analyses:\n{combined}\n\n"
            f"Provide a unified analysis that:\n"
            f"1. Combines all relevant findings\n"
            f"2. Directly addresses the task\n"
            f"3. Resolves any contradictions between chunks\n"
            f"4. States clearly whether the answer was found"
        )

        response = await call_llm(
            model=self._tool_model,
            messages=[UserMessage(content=prompt)],
        )
        return response.message.strip()
