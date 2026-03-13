"""Configuration for the Perplexica search engine."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from orbiter.observability.logging import get_logger  # pyright: ignore[reportMissingImports]

_log = get_logger(__name__)


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

    # Performance tuning
    max_iterations: int | None = None  # override per-mode default (speed=2, balanced=6, quality=25)
    use_reasoning_preamble: bool | None = None  # None=auto-detect thinking models, True/False=force
    max_writer_words: int | None = None  # override quality mode 2000-word target
    max_writer_sources: int = 30  # cap sources passed to writer for citation accuracy
    jina_reader_url: str = ""  # Jina Reader for full-page content extraction
    jina_api_key: str = ""  # Jina Cloud API key (enables r.jina.ai / s.jina.ai)
    serper_api_key: str = ""  # Serper API key for Google Search

    def __post_init__(self) -> None:
        if not self.searxng_url:
            self.searxng_url = os.environ.get("SEARXNG_URL", "http://localhost:8888")
        if not self.model:
            self.model = os.environ.get("PERPLEXICA_MODEL", "openai:gpt-4o")
        if not self.fast_model:
            self.fast_model = os.environ.get("PERPLEXICA_FAST_MODEL", "openai:gpt-4o-mini")
        if not self.embedding_model:
            self.embedding_model = os.environ.get(
                "PERPLEXICA_EMBEDDING_MODEL", "text-embedding-3-small"
            )
        if not self.research_mode:
            self.research_mode = os.environ.get("PERPLEXICA_RESEARCH_MODE", "balanced")
        if not self.jina_reader_url:
            self.jina_reader_url = os.environ.get("JINA_READER_URL", "http://127.0.0.1:3000")
        if not self.jina_api_key:
            self.jina_api_key = os.environ.get("JINA_API_KEY", "")
        if not self.serper_api_key:
            self.serper_api_key = os.environ.get("SERPER_API_KEY", "")
        _log.debug(
            "config model=%s fast=%s search=%s",
            self.model,
            self.fast_model,
            self.searxng_url,
        )
