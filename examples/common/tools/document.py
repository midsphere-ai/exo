"""Document tools — read and summarise local files.

Supports plain text, JSON, CSV, and PDF files.  Heavy dependencies
(PyPDF2, pandas) are imported lazily so the tools work without them
for basic text files.

Usage:
    from examples.common.tools.document import read_document, summarize_document
"""

from __future__ import annotations

import json
from pathlib import Path

from exo import tool


@tool
async def read_document(file_path: str) -> str:
    """Read a document and return its text content.

    Supports: .txt, .md, .json, .csv, .pdf

    Args:
        file_path: Path to the file to read.
    """
    path = Path(file_path)
    if not path.exists():
        return f"Error: file not found — {file_path}"
    if not path.is_file():
        return f"Error: not a regular file — {file_path}"

    suffix = path.suffix.lower()

    try:
        if suffix in (".txt", ".md", ".log", ".py", ".yaml", ".yml", ".toml"):
            return _read_text(path)
        if suffix == ".json":
            return _read_json(path)
        if suffix == ".csv":
            return _read_csv(path)
        if suffix == ".pdf":
            return _read_pdf(path)
        return f"Error: unsupported file type '{suffix}'."
    except Exception as exc:
        return f"Error reading {file_path}: {exc}"


@tool
async def summarize_document(file_path: str) -> str:
    """Return a brief summary of a document's structure and size.

    Args:
        file_path: Path to the file to summarise.
    """
    path = Path(file_path)
    if not path.exists():
        return f"Error: file not found — {file_path}"

    size = path.stat().st_size
    suffix = path.suffix.lower()

    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return f"JSON array with {len(data)} items ({size:,} bytes)."
            if isinstance(data, dict):
                keys = list(data.keys())[:10]
                return f"JSON object with keys {keys} ({size:,} bytes)."
        except json.JSONDecodeError:
            pass

    if suffix == ".csv":
        lines = path.read_text(encoding="utf-8").splitlines()
        return f"CSV file: {len(lines)} lines, header: {lines[0][:120] if lines else 'empty'} ({size:,} bytes)."

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return f"{suffix or 'text'} file: {len(lines)} lines ({size:,} bytes)."


# --- helpers ----------------------------------------------------------------


def _read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > 12000:
        text = text[:12000] + "\n… [truncated]"
    return text


def _read_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    formatted = json.dumps(data, indent=2, ensure_ascii=False)
    if len(formatted) > 12000:
        formatted = formatted[:12000] + "\n… [truncated]"
    return formatted


def _read_csv(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > 12000:
        text = text[:12000] + "\n… [truncated]"
    return text


def _read_pdf(path: Path) -> str:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        return "Error: PyPDF2 is required for PDF reading. Install with: pip install PyPDF2"

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text())
    text = "\n\n".join(pages)
    if len(text) > 12000:
        text = text[:12000] + "\n… [truncated]"
    return text
