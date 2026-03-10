"""Perplexica prompt library."""

from .instructions import (
    ACADEMIC_SEARCH_PROMPT,
    CLASSIFIER_PROMPT,
    DONE_PROMPT,
    REASONING_PREAMBLE_PROMPT,
    SCRAPE_URL_PROMPT,
    SOCIAL_SEARCH_PROMPT,
    WEB_SEARCH_BALANCED_PROMPT,
    WEB_SEARCH_QUALITY_PROMPT,
    WEB_SEARCH_SPEED_PROMPT,
    get_researcher_prompt,
    get_suggestion_prompt,
    get_web_search_prompt,
    get_writer_prompt,
)

__all__ = [
    "ACADEMIC_SEARCH_PROMPT",
    "CLASSIFIER_PROMPT",
    "DONE_PROMPT",
    "REASONING_PREAMBLE_PROMPT",
    "SCRAPE_URL_PROMPT",
    "SOCIAL_SEARCH_PROMPT",
    "WEB_SEARCH_BALANCED_PROMPT",
    "WEB_SEARCH_QUALITY_PROMPT",
    "WEB_SEARCH_SPEED_PROMPT",
    "get_researcher_prompt",
    "get_suggestion_prompt",
    "get_web_search_prompt",
    "get_writer_prompt",
]
