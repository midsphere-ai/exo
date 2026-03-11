"""Configuration for the Perplexica search engine."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class PerplexicaConfig:
    """Perplexica configuration with environment variable overrides."""

    searxng_url: str = ""
    model: str = ""
    fast_model: str = ""
    embedding_model: str = "gemini-embedding-2-preview"
    max_results: int = 10
    searxng_timeout: int = 15
    research_mode: str = ""  # speed, balanced, quality
    system_instructions: str = ""  # user custom instructions for writer
    sources: list[str] = field(default_factory=lambda: ["web"])

    def __post_init__(self) -> None:
        if not self.searxng_url:
            self.searxng_url = os.environ.get(
                "SEARXNG_URL", "http://localhost:8888"
            )
        if not self.model:
            self.model = os.environ.get("PERPLEXICA_MODEL", "openai:gpt-4o")
        if not self.fast_model:
            self.fast_model = os.environ.get(
                "PERPLEXICA_FAST_MODEL", "openai:gpt-4o-mini"
            )
        if not self.embedding_model:
            self.embedding_model = os.environ.get(
                "PERPLEXICA_EMBEDDING_MODEL", "text-embedding-3-small"
            )
        if not self.research_mode:
            self.research_mode = os.environ.get(
                "PERPLEXICA_RESEARCH_MODE", "balanced"
            )
