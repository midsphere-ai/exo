"""Confidence scoring for search answers.

Computes a 0-1 confidence score from four components:
- **Citation rate** (40%): verified / total citations
- **Source authority** (20%): domain-based scoring
- **Content richness** (20%): enriched / total sources ratio
- **Sub-question coverage** (20%): keyword match of sub-questions in answer

No LLM calls — pure heuristic computation.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from pydantic import BaseModel

from exo.observability.logging import get_logger  # pyright: ignore[reportMissingImports]

from ..types import CitationVerification, SearchResult

_log = get_logger(__name__)


class ConfidenceBreakdown(BaseModel):
    """Detailed breakdown of confidence score components."""

    citation_rate: float = 0.0  # 0-1: verified / total citations
    source_authority: float = 0.0  # 0-1: average domain authority
    content_richness: float = 0.0  # 0-1: enriched / total sources
    sub_question_coverage: float = 0.0  # 0-1: covered / total sub-questions
    citation_weight: float = 0.4
    authority_weight: float = 0.2
    richness_weight: float = 0.2
    coverage_weight: float = 0.2


# ---------------------------------------------------------------------------
# Domain authority scoring
# ---------------------------------------------------------------------------

_GOV_EDU_SCORE = 1.0
_AUTHORITATIVE_SCORE = 0.8
_UNKNOWN_SCORE = 0.5
_UGC_SCORE = 0.3

_AUTHORITATIVE_DOMAINS: set[str] = {
    "nature.com",
    "arxiv.org",
    "wikipedia.org",
    "sciencedirect.com",
    "springer.com",
    "ieee.org",
    "acm.org",
    "nih.gov",
    "cdc.gov",
    "who.int",
    "un.org",
    "bbc.com",
    "bbc.co.uk",
    "reuters.com",
    "apnews.com",
    "nytimes.com",
    "washingtonpost.com",
    "theguardian.com",
    "economist.com",
    "ft.com",
    "bloomberg.com",
    "wsj.com",
    "github.com",
    "stackoverflow.com",
    "docs.python.org",
    "developer.mozilla.org",
    "w3.org",
    "pnas.org",
    "sciencemag.org",
    "thelancet.com",
    "bmj.com",
    "nejm.org",
    "pubmed.ncbi.nlm.nih.gov",
    "scholar.google.com",
    "britannica.com",
    "merriam-webster.com",
    "worldbank.org",
    "imf.org",
    "europa.eu",
}

_UGC_DOMAINS: set[str] = {
    "reddit.com",
    "quora.com",
    "answers.yahoo.com",
    "medium.com",
    "tumblr.com",
    "wordpress.com",
    "blogspot.com",
    "livejournal.com",
    "tiktok.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
}


def _score_domain(url: str) -> float:
    """Score a URL's domain on a 0-1 authority scale."""
    try:
        hostname = urlparse(url).hostname or ""
    except Exception:
        return _UNKNOWN_SCORE

    hostname = hostname.lower().removeprefix("www.")

    # .gov / .edu
    if hostname.endswith(".gov") or hostname.endswith(".edu"):
        return _GOV_EDU_SCORE

    # Known authoritative
    for domain in _AUTHORITATIVE_DOMAINS:
        if hostname == domain or hostname.endswith("." + domain):
            return _AUTHORITATIVE_SCORE

    # UGC
    for domain in _UGC_DOMAINS:
        if hostname == domain or hostname.endswith("." + domain):
            return _UGC_SCORE

    return _UNKNOWN_SCORE


# ---------------------------------------------------------------------------
# Sub-question coverage (keyword match, no LLM)
# ---------------------------------------------------------------------------

_STOPWORDS: set[str] = {
    "the",
    "and",
    "was",
    "were",
    "have",
    "been",
    "this",
    "that",
    "with",
    "from",
    "which",
    "their",
    "about",
    "would",
    "could",
    "should",
    "other",
    "what",
    "how",
    "does",
    "will",
    "are",
    "for",
    "has",
    "its",
    "can",
    "who",
    "when",
    "where",
    "why",
    "is",
    "not",
    "but",
    "any",
    "all",
}


def _compute_coverage(sub_questions: list[str], answer: str) -> float:
    """Compute what fraction of sub-questions are covered in the answer."""
    if not sub_questions:
        return 1.0  # no sub-questions = fully covered (single-focus query)

    answer_lower = answer.lower()
    covered = 0

    for sq in sub_questions:
        # Extract non-stopword keywords from sub-question
        words = re.findall(r"[a-z]+", sq.lower())
        keywords = [w for w in words if len(w) > 3 and w not in _STOPWORDS]
        if not keywords:
            covered += 1
            continue
        # Check if enough keywords appear in answer
        matched = sum(1 for kw in keywords if kw in answer_lower)
        if len(keywords) > 0 and matched / len(keywords) >= 0.4:
            covered += 1

    return covered / len(sub_questions)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_confidence(
    verification: CitationVerification | None,
    sources: list[SearchResult],
    sub_questions: list[str],
    answer: str,
) -> tuple[float, ConfidenceBreakdown]:
    """Compute a 0-1 confidence score for a search answer.

    Returns ``(score, breakdown)`` where breakdown has component values.
    """
    breakdown = ConfidenceBreakdown()

    # 1. Citation rate (40%)
    if verification and verification.total_citations > 0:
        breakdown.citation_rate = verification.verified / verification.total_citations
    else:
        # No citations to verify — neutral
        breakdown.citation_rate = 0.5

    # 2. Source authority (20%)
    if sources:
        scores = [_score_domain(s.url) for s in sources]
        breakdown.source_authority = sum(scores) / len(scores)
    else:
        breakdown.source_authority = 0.0

    # 3. Content richness (20%)
    if sources:
        enriched_count = sum(1 for s in sources if s.enriched)
        breakdown.content_richness = enriched_count / len(sources)
    else:
        breakdown.content_richness = 0.0

    # 4. Sub-question coverage (20%)
    breakdown.sub_question_coverage = _compute_coverage(sub_questions, answer)

    # Weighted sum
    score = (
        breakdown.citation_rate * breakdown.citation_weight
        + breakdown.source_authority * breakdown.authority_weight
        + breakdown.content_richness * breakdown.richness_weight
        + breakdown.sub_question_coverage * breakdown.coverage_weight
    )

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    _log.info(
        "confidence=%.2f citation=%.2f authority=%.2f richness=%.2f coverage=%.2f",
        score,
        breakdown.citation_rate,
        breakdown.source_authority,
        breakdown.content_richness,
        breakdown.sub_question_coverage,
    )

    return score, breakdown
