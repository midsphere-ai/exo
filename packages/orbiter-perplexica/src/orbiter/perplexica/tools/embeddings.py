"""Semantic reranking tools using embedding-based cosine similarity.

Reranks search results by computing cosine similarity between query
and result embeddings.  Supports OpenAI, Gemini, and Vertex AI embedding
providers.  Falls back to keyword overlap scoring when no API key is
configured.

Provider auto-detection order:
    1. Gemini  — if ``GEMINI_API_KEY`` is set
    2. Vertex AI — if ``GOOGLE_CLOUD_PROJECT`` is set (uses ADC)
    3. OpenAI  — if ``OPENAI_API_KEY`` is set
    4. Keyword overlap fallback

Usage:
    from examples.advanced.perplexica.tools.embeddings import rerank_by_embeddings
"""

from __future__ import annotations

import asyncio
import json
import os
import re

from orbiter import tool
from orbiter.observability.logging import get_logger  # pyright: ignore[reportMissingImports]

_log = get_logger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _keyword_score(query_words: list[str], text: str) -> float:
    """Compute keyword overlap score between query words and text.

    Args:
        query_words: List of lowercase query words.
        text: Text to score against.
    """
    text_lower = text.lower()
    return sum(1 for w in query_words if w in text_lower) / max(
        len(query_words), 1
    )


def _parse_results(results_json: str) -> list[dict[str, str]]:
    """Parse search results from JSON array or numbered text format.

    Args:
        results_json: JSON string or numbered-list text of search results.
    """
    # Try JSON first.
    try:
        parsed = json.loads(results_json)
        if isinstance(parsed, list):
            results = []
            for item in parsed:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("snippet", ""),
                    }
                )
            return results
    except (json.JSONDecodeError, TypeError):
        pass

    # Fall back to parsing numbered text format: [N] Title | URL | Snippet
    results = []
    for line in results_json.strip().splitlines():
        match = re.match(r"\[(\d+)\]\s*(.+?)\s*\|\s*(\S+)\s*\|\s*(.+)", line)
        if match:
            results.append(
                {
                    "title": match.group(2).strip(),
                    "url": match.group(3).strip(),
                    "snippet": match.group(4).strip(),
                }
            )
    return results


def _get_openai_embeddings(
    texts: list[str], api_key: str, model: str
) -> list[list[float]]:
    """Fetch embeddings from the OpenAI API.

    Args:
        texts: List of text strings to embed.
        api_key: OpenAI API key.
        model: Embedding model name.
    """
    import urllib.request

    payload = json.dumps({"input": texts, "model": model}).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))

    # Sort by index to ensure correct ordering.
    sorted_data = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_data]


def _get_gemini_embeddings(
    texts: list[str], api_key: str, model: str
) -> list[list[float]]:
    """Fetch embeddings from the Gemini API via REST.

    Args:
        texts: List of text strings to embed.
        api_key: Gemini API key.
        model: Embedding model name (e.g. "text-embedding-004").
    """
    import urllib.request

    url = (
        f"https://generativelanguage.googleapis.com/v1beta"
        f"/models/{model}:batchEmbedContents?key={api_key}"
    )
    requests_list = [
        {
            "model": f"models/{model}",
            "content": {"parts": [{"text": t}]},
            "taskType": "SEMANTIC_SIMILARITY",
            "outputDimensionality": 3072,
        }
        for t in texts
    ]
    payload = json.dumps({"requests": requests_list}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))

    return [emb["values"] for emb in data["embeddings"]]


def _get_vertex_embeddings(
    texts: list[str], model: str
) -> list[list[float]]:
    """Fetch embeddings from Vertex AI using the google-genai library.

    Uses Application Default Credentials (no API key needed).

    Args:
        texts: List of text strings to embed.
        model: Embedding model name (e.g. "text-embedding-004").
    """
    from google import genai

    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    client = genai.Client(
        vertexai=True, project=project, location=location
    )
    response = client.models.embed_content(
        model=model,
        contents=texts,
        config={"task_type": "SEMANTIC_SIMILARITY", "output_dimensionality": 3072},
    )
    return [emb.values for emb in response.embeddings]


def _get_embeddings(texts: list[str]) -> list[list[float]] | None:
    """Auto-detect provider and fetch embeddings.

    Tries OpenAI, Gemini, then Vertex AI based on available credentials.
    Returns None if no provider is available.

    Args:
        texts: List of text strings to embed.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if gemini_key:
        model = os.environ.get(
            "PERPLEXICA_EMBEDDING_MODEL", "gemini-embedding-2-preview"
        )
        _log.debug("embedding provider=gemini model=%s texts=%d", model, len(texts))
        return _get_gemini_embeddings(texts, gemini_key, model)

    if gcp_project:
        model = os.environ.get(
            "PERPLEXICA_EMBEDDING_MODEL", "gemini-embedding-2-preview"
        )
        _log.debug("embedding provider=vertex model=%s texts=%d", model, len(texts))
        return _get_vertex_embeddings(texts, model)

    if openai_key:
        model = os.environ.get(
            "PERPLEXICA_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        _log.debug("embedding provider=openai model=%s texts=%d", model, len(texts))
        return _get_openai_embeddings(texts, openai_key, model)

    return None


async def rerank_search_results(
    query: str,
    results: list,
    top_k: int | None = None,
) -> list:
    """Rerank SearchResult objects by semantic relevance to the query.

    Uses embedding cosine similarity when an API key is available,
    falls back to keyword overlap scoring otherwise.
    """
    if not results:
        return results
    if top_k is None:
        top_k = len(results)

    texts = [query] + [f"{r.title} {r.content[:500]}" for r in results]

    scored: list[tuple[float, object]] = []
    try:
        embeddings = await asyncio.to_thread(_get_embeddings, texts)
        if embeddings is not None:
            query_emb = embeddings[0]
            for i, r in enumerate(results):
                sim = _cosine_similarity(query_emb, embeddings[i + 1])
                scored.append((sim, r))
        else:
            raise ValueError("No embedding provider available")
    except Exception as exc:
        _log.warning("embedding failed, falling back to keyword scoring: %s", exc)
        query_words = query.lower().split()
        for r in results:
            text = f"{r.title} {r.content[:500]}"
            score = _keyword_score(query_words, text)
            scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    _log.debug("reranked %d results", len(scored))
    return [r for _, r in scored[:top_k]]


@tool
async def rerank_by_embeddings(
    query: str, results_json: str, top_k: int = 6
) -> str:
    """Rerank search results using embedding-based cosine similarity.

    Args:
        query: The original user query.
        results_json: JSON string containing search results to rerank.
            Each result should have title, url, and snippet fields.
        top_k: Number of top results to return after reranking.
    """
    results = _parse_results(results_json)
    if not results:
        return "No results to rerank."

    scored: list[tuple[float, dict[str, str]]] = []
    texts = [query] + [
        r.get("snippet", "") or r.get("title", "") for r in results
    ]

    try:
        embeddings = _get_embeddings(texts)
        if embeddings is not None:
            query_emb = embeddings[0]
            for i, result in enumerate(results):
                sim = _cosine_similarity(query_emb, embeddings[i + 1])
                scored.append((sim, result))
        else:
            raise ValueError("No embedding provider available")
    except Exception:
        # Fall back to keyword scoring.
        query_words = query.lower().split()
        for result in results:
            text = f"{result.get('title', '')} {result.get('snippet', '')}"
            score = _keyword_score(query_words, text)
            scored.append((score, result))

    # Sort by score descending, take top_k.
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    lines: list[str] = []
    for i, (score, result) in enumerate(top, start=1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        snippet = result.get("snippet", "")
        lines.append(f"[{i}] ({score:.2f}) {title} | {url} | {snippet}")
    return "\n".join(lines)
