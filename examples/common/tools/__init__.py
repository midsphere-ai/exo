"""Reusable tool implementations for Exo examples.

Provides ready-made tools for common tasks: web browsing, search APIs,
and document processing. Import individual tools or the ``ALL_TOOLS``
list for quick agent setup.

Usage:
    from examples.common.tools import web_search, browse_url, read_document
    agent = Agent(name="assistant", tools=[web_search, browse_url, read_document])
"""

from examples.common.tools.browser import browse_url, screenshot
from examples.common.tools.document import read_document, summarize_document
from examples.common.tools.search import web_search

ALL_TOOLS = [web_search, browse_url, screenshot, read_document, summarize_document]

__all__ = [
    "ALL_TOOLS",
    "browse_url",
    "read_document",
    "screenshot",
    "summarize_document",
    "web_search",
]
