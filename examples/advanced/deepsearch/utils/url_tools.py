"""URL normalization, ranking, filtering, and processing utilities."""
from __future__ import annotations
import logging
import re
from urllib.parse import urlparse, urlunparse, unquote
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import SearchSnippet, BoostedSearchSnippet

logger = logging.getLogger("deepsearch")


def normalize_url(url_string: str) -> str | None:
    """Normalize a URL for deduplication."""
    try:
        url_string = url_string.strip()
        if not url_string:
            return None

        # Skip search engine and example URLs
        if any(x in url_string for x in ["google.com/search", "baidu.com/s?", "example.com"]):
            return None

        parsed = urlparse(url_string)
        if parsed.scheme not in ("http", "https"):
            return None

        # Normalize hostname
        hostname = parsed.hostname or ""
        hostname = hostname.lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]

        # Normalize path
        path = parsed.path.rstrip("/") or "/"
        # Remove double slashes
        path = re.sub(r"/+", "/", path)

        # Filter tracking params
        tracking_params = {
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "fbclid", "gclid", "msclkid", "mc_cid", "mc_eid",
        }
        if parsed.query:
            params = []
            for param in parsed.query.split("&"):
                key = param.split("=")[0].lower()
                if key not in tracking_params:
                    params.append(param)
            query = "&".join(params)
        else:
            query = ""

        # Reconstruct
        normalized = urlunparse((parsed.scheme, hostname, path, "", query, ""))
        return normalized
    except Exception:
        return None


def normalize_hostname(url: str) -> str:
    """Extract and normalize hostname from URL."""
    try:
        parsed = urlparse(url)
        hostname = (parsed.hostname or "").lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        return hostname
    except Exception:
        return ""


def add_to_all_urls(snippet: "SearchSnippet", all_urls: dict[str, "SearchSnippet"], weight_delta: float = 1.0) -> int:
    """Add a search snippet to the URL collection. Returns 1 if new, 0 if existing."""
    url = normalize_url(snippet.url)
    if not url:
        return 0
    if url in all_urls:
        existing = all_urls[url]
        all_urls[url] = existing.model_copy(update={"weight": existing.weight + weight_delta})
        return 0
    all_urls[url] = snippet.model_copy(update={"url": url})
    return 1


def rank_urls(
    snippets: list["SearchSnippet"],
    question: str,
    boost_hostnames: list[str] | None = None,
) -> list["BoostedSearchSnippet"]:
    """Rank URLs by relevance. Simple weight-based ranking."""
    from ..types import BoostedSearchSnippet

    ranked = []
    for s in snippets:
        score = s.weight
        hostname = normalize_hostname(s.url)

        # Boost specific hostnames
        hostname_boost = 0.0
        if boost_hostnames and hostname in boost_hostnames:
            hostname_boost = 2.0
            score += hostname_boost

        ranked.append(BoostedSearchSnippet(
            title=s.title, url=s.url, description=s.description,
            weight=s.weight, date=s.date,
            freq_boost=0.0, hostname_boost=hostname_boost,
            path_boost=0.0, jina_rerank_boost=0.0,
            final_score=score,
        ))

    ranked.sort(key=lambda x: x.final_score, reverse=True)
    return ranked


def filter_urls(
    all_urls: dict[str, "SearchSnippet"],
    visited: list[str],
    bad_hostnames: list[str] | None = None,
    only_hostnames: list[str] | None = None,
) -> list["SearchSnippet"]:
    """Filter URLs removing visited and bad hostnames."""
    result = []
    for url, snippet in all_urls.items():
        if url in visited:
            continue
        hostname = normalize_hostname(url)
        if bad_hostnames and hostname in bad_hostnames:
            continue
        if only_hostnames and hostname not in only_hostnames:
            continue
        result.append(snippet)
    return result


def keep_k_per_hostname(results: list, k: int = 2) -> list:
    """Keep at most k results per hostname for diversity."""
    counts: dict[str, int] = {}
    filtered = []
    for r in results:
        hostname = normalize_hostname(r.url)
        counts[hostname] = counts.get(hostname, 0) + 1
        if counts[hostname] <= k:
            filtered.append(r)
    return filtered


def sort_select_urls(urls: list["BoostedSearchSnippet"], limit: int = 20) -> list[dict]:
    """Sort and select top URLs for display in prompt."""
    sorted_urls = sorted(urls, key=lambda x: x.final_score, reverse=True)[:limit]
    return [
        {"url": u.url, "score": u.final_score, "merged": f"{u.title}: {u.description}"[:50]}
        for u in sorted_urls
    ]


def extract_urls_with_description(text: str) -> list["SearchSnippet"]:
    """Extract URLs from text content."""
    from ..types import SearchSnippet

    url_pattern = re.compile(r'https?://[^\s\)\]\}>"\']+')
    results = []
    for match in url_pattern.finditer(text):
        url = normalize_url(match.group())
        if url:
            results.append(SearchSnippet(title="", url=url, description=""))
    return results
