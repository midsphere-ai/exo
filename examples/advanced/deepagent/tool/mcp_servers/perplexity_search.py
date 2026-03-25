"""
JiuWenAgent/src/tools/perplexity_search.py

Perplexity search utility (function-call style, non-MCP).

Exports:
- class PerplexitySearch(model: str = "sonar")
  - .academic_search(..., return_json=False) -> str|dict
  - .web_search(..., return_json=False) -> str|dict
  - .search(query, **kwargs) -> str|dict
"""
from __future__ import annotations

import json
import os
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

PROXIES = None
VERIFY_SSL = False if os.environ.get("PPLX_VERIFY_SSL", "0").lower() in {"", "0", "false", "no"} else True
DEFAULT_TIMEOUT = float(os.environ.get("PPLX_HTTP_TIMEOUT", 30))
RETRY_MAX_TRIES = int(os.environ.get("PPLX_HTTP_RETRIES", 3))

PPLX_API_URL = os.environ.get("PPLX_API_URL", "https://api.perplexity.ai/chat/completions")
DEFAULT_MODEL = os.environ.get("PPLX_MODEL", "sonar-pro")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
class _HTTP:
    @staticmethod
    def post(url: str, json_payload: Dict[str, Any], headers: Dict[str, str], timeout: float = DEFAULT_TIMEOUT) -> requests.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(1, RETRY_MAX_TRIES + 1):
            try:
                resp = requests.post(
                    url,
                    json=json_payload,
                    headers=headers,
                    timeout=timeout,
                    proxies=PROXIES,
                    verify=VERIFY_SSL,
                )
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                last_exc = e
                if attempt >= RETRY_MAX_TRIES:
                    break
                time.sleep(min(0.5 * (2 ** (attempt - 1)), 4.0))
        assert last_exc is not None
        raise last_exc


def _fmt_citations(cites: List[Dict[str, Any]]) -> str:
    if not cites:
        return ""
    lines: List[str] = []
    for i, c in enumerate(cites, 1):
        title = c.get("title") or c.get("source", "(untitled)")
        url = c.get("url") or c.get("source_url") or c.get("link") or ""
        host = c.get("hostname") or ""
        meta = f" ({host})" if host and (host not in title) else ""
        lines.append(f"[{i}] {title}{meta} {url}")
    return "\n".join(lines)


def _extract_citations(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Perplexity has used different keys, try several
    keys = [
        "search_results",
        "web_search_results",
        "citations",
        "sources",
    ]
    for k in keys:
        lst = resp.get(k)
        if isinstance(lst, list) and lst:
            # normalize records
            norm: List[Dict[str, Any]] = []
            for it in lst:
                if isinstance(it, dict):
                    norm.append(it)
                elif isinstance(it, str):
                    norm.append({"title": it, "url": it})
            return norm
    return []


def _choose_model(user_model: Optional[str]) -> str:
    return (user_model or DEFAULT_MODEL).strip()


def _validate_recency(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    allowed = {"day", "week", "month", "year"}
    v2 = v.strip().lower()
    return v2 if v2 in allowed else None


# -----------------------------------------------------------------------------
# Core client
# -----------------------------------------------------------------------------
class PerplexitySearch:

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable is not set")
        self.model = _choose_model(model)
        self.url = PPLX_API_URL

    def _make_request(
        self,
        query: str,
        *,
        model: Optional[str] = None,
        search_mode: Optional[str] = None,  # can change to "academic"
        search_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_recency_filter: Optional[str] = None,
        search_after_date: Optional[str] = None,
        search_before_date: Optional[str] = None,
        search_context_size: str = "high",
        system_prompt: str = "Be precise and concise. Provide citations for all facts.",
        max_tokens: int = 4096,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 5,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 1.0,
        stream: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": _choose_model(model) or self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            "max_tokens": max_tokens,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "return_images": False,
            "return_related_questions": False,
            "top_k": int(top_k),
            "stream": bool(stream),
            "presence_penalty": float(presence_penalty),
            "frequency_penalty": float(frequency_penalty),
            "web_search_options": {"search_context_size": search_context_size},
        }

        if search_mode:
            payload["search_mode"] = search_mode

        # Domain filters (API limit of 10)
        domains: List[str] = []
        if search_domains:
            domains.extend(search_domains)
        if exclude_domains:
            domains.extend([f"-{d}" for d in exclude_domains])
        if domains:
            payload["search_domain_filter"] = domains[:10]

        # Recency filter
        rec = _validate_recency(search_recency_filter)
        if rec:
            payload["search_recency_filter"] = rec

        if search_after_date:
            payload["search_after_date_filter"] = search_after_date
        if search_before_date:
            payload["search_before_date_filter"] = search_before_date

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = _HTTP.post(self.url, json_payload=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
        try:
            data = resp.json()
        except Exception as e:
            raise ValueError(f"Failed to parse Perplexity response as JSON: {e}")
        # Basic sanity check
        if not isinstance(data, dict) or "choices" not in data:
            raise ValueError("Invalid response format from Perplexity API")
        return data

    @staticmethod
    def _format_response_with_citations(response_data: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        if "choices" not in response_data or not response_data["choices"]:
            raise ValueError("Invalid response format from Perplexity API (no choices)")
        answer = response_data["choices"][0]["message"].get("content", "").strip()
        citations = _extract_citations(response_data)
        return answer, citations

    @staticmethod
    def _to_text(answer: str, citations: List[Dict[str, Any]]) -> str:
        cite_block = _fmt_citations(citations)
        return f"{answer}\n\n## Citations\n{cite_block}" if cite_block else answer

    def academic_search(
        self,
        query: str,
        *,
        search_after_date: Optional[str] = None,
        search_before_date: Optional[str] = None,
        max_tokens: int = 4096,
        model: Optional[str] = None,
        return_json: bool = False,
    ) -> Any:
        system_prompt = (
            "You are an academic research assistant. Provide comprehensive answers based on "
            "peer-reviewed papers, academic journals, and scholarly sources. "
            "Always cite your sources with proper academic citations."
        )
        data = self._make_request(
            query=query,
            model=model,
            search_mode="academic",
            search_after_date=search_after_date,
            search_before_date=search_before_date,
            search_context_size="high",
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        answer, citations = self._format_response_with_citations(data)
        if return_json:
            return {"answer": answer, "citations": citations, "raw": data}
        return self._to_text(answer, citations)

    def web_search(
        self,
        query: str,
        *,
        search_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        search_recency_filter: Optional[str] = None,
        search_context_size: str = "high",
        max_tokens: int = 4096,
        model: Optional[str] = None,
        return_json: bool = False,
    ) -> Any:
        system_prompt = (
            "Provide accurate and comprehensive answers based on current web information. "
            "Always cite your sources with numbered references."
        )
        data = self._make_request(
            query=query,
            model=model,
            search_domains=search_domains,
            exclude_domains=exclude_domains,
            search_recency_filter=search_recency_filter,
            search_context_size=search_context_size,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )
        answer, citations = self._format_response_with_citations(data)
        if return_json:
            return {"answer": answer, "citations": citations, "raw": data}
        return self._to_text(answer, citations)

    def search(self, query: str, *, return_json: bool = False, **kwargs) -> Any:
        data = self._make_request(query, **kwargs)
        answer, citations = self._format_response_with_citations(data)
        if return_json:
            return {"answer": answer, "citations": citations, "raw": data}
        return self._to_text(answer, citations)


# -----------------------------------------------------------------------------
# Convenience wrappers
# -----------------------------------------------------------------------------

def perplexity_academic_search(
    query: str,
    *,
    model: str = DEFAULT_MODEL,
    search_after_date: Optional[str] = None,
    search_before_date: Optional[str] = None,
    max_tokens: int = 4096,
    return_json: bool = False,
) -> Any:
    client = PerplexitySearch(model=model)
    return client.academic_search(
        query=query,
        search_after_date=search_after_date,
        search_before_date=search_before_date,
        max_tokens=max_tokens,
        return_json=return_json,
    )


def perplexity_web_search(
    query: str,
    *,
    model: str = DEFAULT_MODEL,
    search_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    search_recency_filter: Optional[str] = None,
    search_context_size: str = "high",
    max_tokens: int = 4096,
    return_json: bool = False,
) -> Any:
    client = PerplexitySearch(model=model)
    return client.web_search(
        query=query,
        search_domains=search_domains,
        exclude_domains=exclude_domains,
        search_recency_filter=search_recency_filter,
        search_context_size=search_context_size,
        max_tokens=max_tokens,
        return_json=return_json,
    )


def perplexity_search(
    query: str,
    *,
    model: str = DEFAULT_MODEL,
    return_json: bool = False,
    **kwargs,
) -> Any:
    client = PerplexitySearch(model=model)
    return client.search(query=query, return_json=return_json, **kwargs)


# -----------------------------------------------------------------------------
# Tool-call specs for easy registration (OpenAI-style function calling)
# -----------------------------------------------------------------------------
FUNCTION_SPECS = [
    {
        "schema": {
            "name": "perplexity_academic_search",
            "description": "Academic search focusing on scholarly sources (Perplexity). Returns formatted text unless return_json is true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Academic search query"},
                    "model": {"type": "string", "description": "Perplexity model (e.g., 'sonar', 'sonar-pro')", "default": DEFAULT_MODEL},
                    "search_after_date": {"type": "string", "description": "Include results after this date (YYYY-MM-DD)"},
                    "search_before_date": {"type": "string", "description": "Include results before this date (YYYY-MM-DD)"},
                    "max_tokens": {"type": "integer", "default": 4096},
                    "return_json": {"type": "boolean", "default": False},
                },
                "required": ["query"],
            },
        },
        "func": perplexity_academic_search,
    },
    {
        "schema": {
            "name": "perplexity_web_search",
            "description": "General web search with optional domain filters and recency (Perplexity). Returns formatted text unless return_json is true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Web search query"},
                    "model": {"type": "string", "description": "Perplexity model (e.g., 'sonar', 'sonar-pro')", "default": DEFAULT_MODEL},
                    "search_domains": {"type": "array", "items": {"type": "string"}},
                    "exclude_domains": {"type": "array", "items": {"type": "string"}},
                    "search_recency_filter": {"type": "string", "enum": ["day", "week", "month", "year"]},
                    "search_context_size": {"type": "string", "enum": ["low", "medium", "high"], "default": "high"},
                    "max_tokens": {"type": "integer", "default": 4096},
                    "return_json": {"type": "boolean", "default": False},
                },
                "required": ["query"],
            },
        },
        "func": perplexity_web_search,
    },
    {
        "schema": {
            "name": "perplexity_search",
            "description": "Raw Perplexity search with custom keyword args. Returns formatted text unless return_json is true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "model": {"type": "string", "default": DEFAULT_MODEL},
                    "return_json": {"type": "boolean", "default": False},
                    # Any additional keys will be forwarded (e.g., search_mode, search_domains, etc.)
                },
                "required": ["query"],
                "additionalProperties": True,
            },
        },
        "func": perplexity_search,
    },
]


if __name__ == "__main__":
    try:
        print(perplexity_web_search("What is the atomic number of the element mentioned in the first line of Nikki Giovanni’s poem “The Laws of Motion”?", return_json=False))
        print("\n" + "-" * 80 + "\n")
        print(perplexity_academic_search("Find the essay that begins on page 32 of Volume 8, Issue 8 of the Russell Sage Foundation Journal of the Social Sciences. What two-word term do the authors use beginning on page 34 that describes how peoples’ actions are influenced by the actions of people they come into contact with?", return_json=False))
    except Exception as e:
        print(f"[Test Error] {e}")
