"""Text processing utilities."""
from __future__ import annotations
import re
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import AnswerAction, KnowledgeItem


def remove_extra_line_breaks(text: str) -> str:
    """Remove excessive line breaks."""
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", "", text)


def get_knowledge_str(knowledge: list["KnowledgeItem"]) -> list[str]:
    """Format knowledge items for prompt injection."""
    result = []
    for i, k in enumerate(knowledge):
        refs = ""
        if k.references:
            refs = f"\nReferences: {', '.join(str(r) for r in k.references[:3])}"
        result.append(f"<knowledge-{i + 1}>\nQ: {k.question}\nA: {k.answer}{refs}\n</knowledge-{i + 1}>")
    return result


def build_md_from_answer(answer: "AnswerAction") -> str:
    """Build markdown answer with footnote references."""
    md = answer.answer
    if answer.references:
        md += "\n\n---\n\n"
        for i, ref in enumerate(answer.references):
            md += f"[^{i + 1}]: [{ref.title or ref.url}]({ref.url})"
            if ref.date_time:
                md += f" ({ref.date_time})"
            md += "\n"
    return md


def repair_markdown_final(md: str) -> str:
    """Repair common markdown issues."""
    # Fix code block indentation
    md = re.sub(r"```(\w+)\n\s+", lambda m: f"```{m.group(1)}\n", md)
    # Convert HTML tables to markdown-friendly format
    md = convert_html_tables_to_md(md)
    return md


def convert_html_tables_to_md(text: str) -> str:
    """Pass-through HTML tables (they render fine in most markdown)."""
    return text


def smart_merge_strings(a: str, b: str) -> str:
    """Merge two strings, handling overlapping content."""
    if not a:
        return b
    if not b:
        return a
    # Simple concat with dedup of identical segments
    if b.startswith(a[-100:]):
        overlap = a[-100:]
        idx = b.find(overlap)
        if idx >= 0:
            return a + b[idx + len(overlap):]
    return a + "\n" + b


def choose_k(items: list, k: int) -> list:
    """Choose up to k items from list, preserving order."""
    if len(items) <= k:
        return items
    return random.sample(items, k)


def chunk_text(text: str, max_chunk_size: int = 500) -> dict:
    """Simple text chunking by paragraphs/sentences."""
    chunks = []
    positions = []

    # Split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\n+', text)
    pos = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            pos += 2
            continue

        start = text.find(para, pos)
        if start == -1:
            start = pos

        if len(para) > max_chunk_size:
            # Split long paragraphs by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = ""
            chunk_start = start

            for sent in sentences:
                if len(current_chunk) + len(sent) > max_chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    positions.append([chunk_start, chunk_start + len(current_chunk)])
                    chunk_start = chunk_start + len(current_chunk)
                    current_chunk = sent
                else:
                    current_chunk = (current_chunk + " " + sent).strip()

            if current_chunk:
                chunks.append(current_chunk.strip())
                positions.append([chunk_start, chunk_start + len(current_chunk)])
        else:
            chunks.append(para)
            positions.append([start, start + len(para)])

        pos = start + len(para)

    return {"chunks": chunks, "chunk_positions": positions}
