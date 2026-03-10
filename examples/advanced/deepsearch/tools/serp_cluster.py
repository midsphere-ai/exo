"""Search result clustering by insight."""
from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger("deepsearch")


async def cluster_results(
    results: list[dict],
    generate_fn,
    schema_gen,
) -> list[dict]:
    """Cluster search results by insight for knowledge extraction.

    Args:
        results: List of search result dicts with title, url, description
        generate_fn: LLM generate function
        schema_gen: Schema generator

    Returns:
        List of cluster dicts with question, insight, urls
    """
    system = """You are a search engine result analyzer. You look at the SERP API response and group them into meaningful cluster.

Each cluster should contain a summary of the content, key data and insights, the corresponding URLs and search advice. Respond in JSON format."""

    import json
    user = json.dumps(results, default=str)

    try:
        schema = schema_gen.get_serp_cluster_schema()
        result = await generate_fn(schema=schema, system=system, prompt=user)
        clusters = result.clusters if hasattr(result, 'clusters') else []
        return [
            {"question": c.question, "insight": c.insight, "urls": c.urls}
            for c in clusters
        ]
    except Exception as e:
        logger.warning("SERP clustering failed: %s", e)
        return []
