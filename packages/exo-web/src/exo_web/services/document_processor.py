"""Document text extraction and chunking pipeline.

Supported formats: PDF, DOCX, TXT, MD, CSV, HTML.
"""

from __future__ import annotations

import csv
import io
import re
from html.parser import HTMLParser

# ---------------------------------------------------------------------------
# Text extraction by file type
# ---------------------------------------------------------------------------


def extract_text(content: bytes, file_type: str) -> str:
    """Extract plain text from file content based on file type."""
    ft = file_type.lower().lstrip(".")
    if ft in ("txt", "md"):
        return content.decode("utf-8", errors="replace")
    if ft == "csv":
        return _extract_csv(content)
    if ft == "html":
        return _extract_html(content)
    if ft == "pdf":
        return _extract_pdf(content)
    if ft == "docx":
        return _extract_docx(content)
    msg = f"Unsupported file type: {file_type}"
    raise ValueError(msg)


def _extract_csv(content: bytes) -> str:
    text = content.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    lines: list[str] = []
    for row in reader:
        lines.append(" | ".join(row))
    return "\n".join(lines)


class _HTMLTextExtractor(HTMLParser):
    """Simple HTML-to-text converter that strips tags."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._skip = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)


def _extract_html(content: bytes) -> str:
    text = content.decode("utf-8", errors="replace")
    parser = _HTMLTextExtractor()
    parser.feed(text)
    raw = " ".join(parser._parts)
    # Collapse whitespace
    return re.sub(r"\s+", " ", raw).strip()


def _extract_pdf(content: bytes) -> str:
    """Extract text from PDF using a minimal pure-Python approach.

    Uses pypdf if available, otherwise falls back to a basic binary text extraction.
    """
    try:
        from pypdf import PdfReader  # type: ignore[import-untyped]

        reader = PdfReader(io.BytesIO(content))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        # Fallback: extract readable text fragments from PDF binary
        return _fallback_pdf_extract(content)


def _fallback_pdf_extract(content: bytes) -> str:
    """Best-effort text extraction from PDF bytes without external libraries."""
    # Decode with replacement, then grab printable text runs
    raw = content.decode("latin-1", errors="replace")
    # Find text between BT/ET markers (PDF text objects) or just grab long printable runs
    parts: list[str] = []
    for match in re.finditer(r"[\x20-\x7e]{10,}", raw):
        parts.append(match.group())
    return "\n".join(parts)


def _extract_docx(content: bytes) -> str:
    """Extract text from DOCX (Office Open XML).

    DOCX files are ZIP archives containing XML. We parse the main
    document.xml to extract paragraph text.
    """
    import zipfile
    from xml.etree import ElementTree

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        if "word/document.xml" not in zf.namelist():
            return ""
        xml_content = zf.read("word/document.xml")

    tree = ElementTree.fromstring(xml_content)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

    paragraphs: list[str] = []
    for para in tree.iter(f"{{{ns['w']}}}p"):
        texts: list[str] = []
        for run in para.iter(f"{{{ns['w']}}}t"):
            if run.text:
                texts.append(run.text)
        if texts:
            paragraphs.append("".join(texts))

    return "\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by character count.

    Returns a list of text chunks. Empty input returns an empty list.
    """
    if not text or not text.strip():
        return []

    # Ensure sane values
    chunk_size = max(chunk_size, 64)
    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at a sentence or word boundary
        if end < text_len:
            # Look for last sentence boundary in the chunk
            for sep in ("\n\n", "\n", ". ", "? ", "! ", " "):
                last_sep = chunk.rfind(sep)
                if last_sep > chunk_size // 2:
                    chunk = chunk[: last_sep + len(sep)]
                    break

        stripped = chunk.strip()
        if stripped:
            chunks.append(stripped)

        # Move forward by chunk length minus overlap
        step = len(chunk) - chunk_overlap
        if step <= 0:
            step = max(1, chunk_size - chunk_overlap)
        start += step

    return chunks
