"""Built-in evaluators for the evaluation framework.

Each evaluator takes an expected value and an actual (agent) response and
returns a score between 0.0 and 1.0.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any


def exact_match(expected: str, actual: str) -> float:
    """Return 1.0 if expected == actual (case-sensitive), else 0.0."""
    return 1.0 if expected.strip() == actual.strip() else 0.0


def contains(expected: str, actual: str) -> float:
    """Return 1.0 if expected text appears anywhere in actual, else 0.0."""
    return 1.0 if expected.strip() in actual else 0.0


def regex_match(expected: str, actual: str) -> float:
    """Return 1.0 if the expected regex pattern matches actual, else 0.0.

    The expected value is treated as a regex pattern.
    """
    try:
        return 1.0 if re.search(expected.strip(), actual) else 0.0
    except re.error:
        return 0.0


def semantic_similarity(expected: str, actual: str) -> float:
    """Return a similarity ratio between 0.0 and 1.0 using SequenceMatcher.

    This is a lightweight local approximation; for production use, swap in
    an embedding-based similarity.
    """
    return SequenceMatcher(None, expected.strip(), actual.strip()).ratio()


async def llm_as_judge(
    expected: str,
    actual: str,
    *,
    provider_resolver: Any = None,
    provider_type: str = "",
    model_name: str = "",
    user_id: str = "",
) -> float:
    """Use an LLM to judge whether the actual response matches the criteria.

    The *expected* field is treated as a criteria description (not a literal
    match).  The model returns a score from 0 to 10 which is normalized to
    0.0-1.0.

    If no provider is available, falls back to semantic_similarity.
    """
    if provider_resolver is None:
        return semantic_similarity(expected, actual)

    prompt = (
        "You are an evaluation judge. Score the following response on a scale "
        "of 0 to 10 based on how well it meets the criteria.\n\n"
        f"Criteria: {expected}\n\n"
        f"Response: {actual}\n\n"
        "Return ONLY a single integer between 0 and 10, nothing else."
    )

    try:
        provider = await provider_resolver(provider_type, model_name, user_id)
        resp = await provider.complete(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
        )
        score_text = resp.content.strip()
        score = int(score_text)
        return max(0.0, min(1.0, score / 10.0))
    except Exception:
        return semantic_similarity(expected, actual)


# Registry of evaluator names to callables.  Sync evaluators are wrapped
# so the route code can always ``await`` them uniformly.
EVALUATORS: dict[str, str] = {
    "exact_match": "exact_match",
    "contains": "contains",
    "regex_match": "regex_match",
    "llm_as_judge": "llm_as_judge",
    "semantic_similarity": "semantic_similarity",
}


async def run_evaluator(
    evaluator_type: str,
    expected: str,
    actual: str,
    **kwargs: Any,
) -> float:
    """Dispatch to the appropriate evaluator and return a 0.0-1.0 score."""
    if evaluator_type == "llm_as_judge":
        return await llm_as_judge(expected, actual, **kwargs)
    fn_map = {
        "exact_match": exact_match,
        "contains": contains,
        "regex_match": regex_match,
        "semantic_similarity": semantic_similarity,
    }
    fn = fn_map.get(evaluator_type, semantic_similarity)
    return fn(expected, actual)
