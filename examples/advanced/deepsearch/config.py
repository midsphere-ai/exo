"""Configuration for the DeepSearch engine."""
from __future__ import annotations
import os
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("deepsearch")


@dataclass
class DeepSearchConfig:
    # LLM
    llm_provider: str = "gemini"
    model_name: str = "gemini-2.0-flash"

    # Search
    search_provider: str = "auto"
    content_reader: str = "auto"
    embedding_provider: str = "auto"

    # Self-hosted services
    searxng_url: str = ""  # e.g. http://localhost:8888
    reader_url: str = ""   # e.g. http://localhost:3000

    # API Keys
    jina_api_key: str = ""
    brave_api_key: str = ""
    serper_api_key: str = ""

    # Budget
    token_budget: int = 1_000_000
    max_bad_attempts: int = 2
    step_sleep: float = 1.0

    # Limits
    max_urls_per_step: int = 5
    max_queries_per_step: int = 5
    max_reflect_per_step: int = 2

    # Output
    max_returned_urls: int = 100
    max_references: int = 10
    min_relevance_score: float = 0.80

    # Features
    with_images: bool = False
    team_size: int = 1
    no_direct_answer: bool = False

    def __post_init__(self) -> None:
        # Load from environment
        if not self.searxng_url:
            self.searxng_url = os.environ.get("SEARXNG_URL", "")
        if not self.reader_url:
            self.reader_url = os.environ.get("READER_URL", "")
        if not self.jina_api_key:
            self.jina_api_key = os.environ.get("JINA_API_KEY", "")
        if not self.brave_api_key:
            self.brave_api_key = os.environ.get("BRAVE_API_KEY", "")
        if not self.serper_api_key:
            self.serper_api_key = os.environ.get("SERPER_API_KEY", "")

        # Auto-detect providers — prefer self-hosted, then API-based, then free
        if self.search_provider == "auto":
            if self.searxng_url:
                self.search_provider = "searxng"
            elif self.jina_api_key:
                self.search_provider = "jina"
            elif self.brave_api_key:
                self.search_provider = "brave"
            elif self.serper_api_key:
                self.search_provider = "serper"
            else:
                self.search_provider = "duck"

        if self.content_reader == "auto":
            if self.reader_url:
                self.content_reader = "selfhosted"
            elif self.jina_api_key:
                self.content_reader = "jina"
            else:
                self.content_reader = "http"

        if self.embedding_provider == "auto":
            if self.jina_api_key:
                self.embedding_provider = "jina"
            else:
                self.embedding_provider = "local"

    @property
    def model_string(self) -> str:
        return f"{self.llm_provider}:{self.model_name}"
