"""7-cognitive-persona query expansion system.

Ported from node-DeepResearch/src/tools/query-rewriter.ts.
Expands user search queries by analyzing potential intents through 7 layers
and generating optimized queries from 7 cognitive personas.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

from pydantic import BaseModel

from ..types import SearchAction

logger = logging.getLogger("deepsearch")


def _get_prompt(query: str, think: str, context: str) -> tuple[str, str]:
    """Build (system, user) prompt pair for query rewriting.

    Args:
        query: The original search query string.
        think: The agent's reasoning/motivation for this search.
        context: Soundbite context from preliminary search results.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    current_time = datetime.now(timezone.utc)
    current_year = current_time.year
    current_month = current_time.month

    system = f"""
You are an expert search query expander with deep psychological understanding.
You optimize user queries by extensively analyzing potential user intents and generating comprehensive query variations.

The current time is {current_time.isoformat()}. Current year: {current_year}, current month: {current_month}.

<intent-mining>
To uncover the deepest user intent behind every query, analyze through these progressive layers:

1. Surface Intent: The literal interpretation of what they're asking for
2. Practical Intent: The tangible goal or problem they're trying to solve
3. Emotional Intent: The feelings driving their search (fear, aspiration, anxiety, curiosity)
4. Social Intent: How this search relates to their relationships or social standing
5. Identity Intent: How this search connects to who they want to be or avoid being
6. Taboo Intent: The uncomfortable or socially unacceptable aspects they won't directly state
7. Shadow Intent: The unconscious motivations they themselves may not recognize

Map each query through ALL these layers, especially focusing on uncovering Shadow Intent.
</intent-mining>

<cognitive-personas>
Generate ONE optimized query from each of these cognitive perspectives:

1. Expert Skeptic: Focus on edge cases, limitations, counter-evidence, and potential failures. Generate a query that challenges mainstream assumptions and looks for exceptions.
2. Detail Analyst: Obsess over precise specifications, technical details, and exact parameters. Generate a query that drills into granular aspects and seeks definitive reference data.
3. Historical Researcher: Examine how the subject has evolved over time, previous iterations, and historical context. Generate a query that tracks changes, development history, and legacy issues.
4. Comparative Thinker: Explore alternatives, competitors, contrasts, and trade-offs. Generate a query that sets up comparisons and evaluates relative advantages/disadvantages.
5. Temporal Context: Add a time-sensitive query that incorporates the current date ({current_year}-{current_month}) to ensure recency and freshness of information.
6. Globalizer: Identify the most authoritative language/region for the subject matter (not just the query's origin language). For example, use German for BMW (German company), English for tech topics, Japanese for anime, Italian for cuisine, etc. Generate a search in that language to access native expertise.
7. Reality-Hater-Skepticalist: Actively seek out contradicting evidence to the original query. Generate a search that attempts to disprove assumptions, find contrary evidence, and explore "Why is X false?" or "Evidence against X" perspectives.

Ensure each persona contributes exactly ONE high-quality query that follows the schema format. These 7 queries will be combined into a final array.
</cognitive-personas>

<rules>
Leverage the soundbites from the context user provides to generate queries that are contextually relevant.

1. Query content rules:
   - Split queries for distinct aspects
   - Add operators only when necessary
   - Ensure each query targets a specific intent
   - Remove fluff words but preserve crucial qualifiers
   - Keep 'q' field short and keyword-based (2-5 words ideal)

2. Schema usage rules:
   - Always include the 'q' field in every query object (should be the last field listed)
   - Use 'tbs' for time-sensitive queries (remove time constraints from 'q' field)
   - Include 'location' only when geographically relevant
   - Never duplicate information in 'q' that is already specified in other fields
   - List fields in this order: tbs, location, q

<query-operators>
For the 'q' field content:
- +term : must include term; for critical terms that must appear
- -term : exclude term; exclude irrelevant or ambiguous terms
- filetype:pdf/doc : specific file type
Note: A query can't only have operators; and operators can't be at the start of a query
</query-operators>
</rules>

<examples>
<example-1>
Input Query: \u5b9d\u9a6c\u4e8c\u624b\u8f66\u4ef7\u683c
<think>
\u5b9d\u9a6c\u4e8c\u624b\u8f66\u4ef7\u683c...\u54ce\uff0c\u8fd9\u4eba\u5e94\u8be5\u662f\u60f3\u4e70\u4e8c\u624b\u5b9d\u9a6c\u5427\u3002\u8868\u9762\u4e0a\u662f\u67e5\u4ef7\u683c\uff0c\u5b9e\u9645\u4e0a\u80af\u5b9a\u662f\u60f3\u4e70\u53c8\u6015\u8e29\u5751\u3002\u8c01\u4e0d\u60f3\u5f00\u4e2a\u5b9d\u9a6c\u554a\uff0c\u9762\u5b50\u5341\u8db3\uff0c\u4f46\u53c8\u62c5\u5fc3\u517b\u4e0d\u8d77\u3002\u8fd9\u5e74\u5934\uff0c\u5f00\u4ec0\u4e48\u8f66\u90fd\u662f\u8eab\u4efd\u7684\u8c61\u5f81\uff0c\u5c24\u5176\u662f\u5b9d\u9a6c\u8fd9\u79cd\u8c6a\u8f66\uff0c\u4e00\u770b\u5c31\u662f\u6709\u70b9\u6210\u7ee9\u7684\u4eba\u3002\u4f46\u5f88\u591a\u4eba\u5176\u5b9e\u56ca\u4e2d\u7f9e\u6da9\uff0c\u786c\u6491\u7740\u4e70\u4e86\u5b9d\u9a6c\uff0c\u7ed3\u679c\u6bcf\u5929\u90fd\u5728\u7ea0\u7ed3\u6cb9\u8d39\u4fdd\u517b\u8d39\u3002\u8bf4\u5230\u5e95\uff0c\u53ef\u80fd\u5c31\u662f\u60f3\u901a\u8fc7\u7269\u8d28\u6765\u83b7\u5f97\u5b89\u5168\u611f\u6216\u586b\u8865\u5185\u5fc3\u7684\u67d0\u79cd\u7a7a\u865a\u5427\u3002

\u8981\u5e2e\u4ed6\u7684\u8bdd\uff0c\u5f97\u591a\u65b9\u4f4d\u601d\u8003\u4e00\u4e0b...\u4e8c\u624b\u5b9d\u9a6c\u80af\u5b9a\u6709\u4e0d\u5c11\u95ee\u9898\uff0c\u5c24\u5176\u662f\u90a3\u4e9b\u8f66\u4e3b\u4e0d\u4f1a\u4e3b\u52a8\u544a\u8bc9\u4f60\u7684\u9690\u60a3\uff0c\u7ef4\u4fee\u8d77\u6765\u53ef\u80fd\u8981\u547d\u3002\u4e0d\u540c\u7cfb\u5217\u7684\u5b9d\u9a6c\u4ef7\u683c\u5dee\u5f02\u4e5f\u633a\u5927\u7684\uff0c\u5f97\u770b\u770b\u8be6\u7ec6\u6570\u636e\u548c\u5b9e\u9645\u516c\u91cc\u6570\u3002\u4ef7\u683c\u8fd9\u4e1c\u897f\u4e5f\u4e00\u76f4\u5728\u53d8\uff0c\u53bb\u5e74\u7684\u884c\u60c5\u548c\u4eca\u5e74\u7684\u53ef\u4e0d\u4e00\u6837\uff0c{current_year}\u5e74\u6700\u65b0\u7684\u8d8b\u52bf\u600e\u4e48\u6837\uff1f\u5b9d\u9a6c\u548c\u5954\u9a70\u8fd8\u6709\u4e00\u4e9b\u66f4\u5e73\u4ef7\u7684\u8f66\u6bd4\u8d77\u6765\uff0c\u5230\u5e95\u503c\u4e0d\u503c\u8fd9\u4e2a\u94b1\uff1f\u5b9d\u9a6c\u662f\u5fb7\u56fd\u8f66\uff0c\u5fb7\u56fd\u4eba\u5bf9\u8fd9\u8f66\u7684\u4e86\u89e3\u80af\u5b9a\u6700\u6df1\uff0c\u5fb7\u56fd\u8f66\u4e3b\u7684\u771f\u5b9e\u8bc4\u4ef7\u4f1a\u66f4\u6709\u53c2\u8003\u4ef7\u503c\u3002\u6700\u540e\uff0c\u73b0\u5b9e\u70b9\u770b\uff0c\u80af\u5b9a\u6709\u4eba\u4e70\u4e86\u5b9d\u9a6c\u540e\u6094\u7684\uff0c\u90a3\u4e9b\u8840\u6cea\u6559\u8bad\u4e0d\u80fd\u4e0d\u542c\u554a\uff0c\u5f97\u627e\u627e\u90a3\u4e9b\u771f\u5b9e\u6848\u4f8b\u3002
</think>
queries: [
  {{
    "q": "\u4e8c\u624b\u5b9d\u9a6c \u7ef4\u4fee\u5669\u68a6 \u9690\u85cf\u7f3a\u9677"
  }},
  {{
    "q": "\u5b9d\u9a6c\u5404\u7cfb\u4ef7\u683c\u533a\u95f4 \u91cc\u7a0b\u5bf9\u6bd4"
  }},
  {{
    "tbs": "qdr:y",
    "q": "\u4e8c\u624b\u5b9d\u9a6c\u4ef7\u683c\u8d8b\u52bf"
  }},
  {{
    "q": "\u4e8c\u624b\u5b9d\u9a6cvs\u5954\u9a70vs\u5965\u8fea \u6027\u4ef7\u6bd4"
  }},
  {{
    "tbs": "qdr:m",
    "q": "\u5b9d\u9a6c\u884c\u60c5"
  }},
  {{
    "q": "BMW Gebrauchtwagen Probleme"
  }},
  {{
    "q": "\u4e8c\u624b\u5b9d\u9a6c\u540e\u6094\u6848\u4f8b \u6700\u5dee\u6295\u8d44"
  }}
]
</example-1>

<example-2>
Input Query: sustainable regenerative agriculture soil health restoration techniques
<think>
Sustainable regenerative agriculture soil health restoration techniques... interesting search. They're probably looking to fix depleted soil on their farm or garden. Behind this search though, there's likely a whole story - someone who's read books like "The Soil Will Save Us" or watched documentaries on Netflix about how conventional farming is killing the planet. They're probably anxious about climate change and want to feel like they're part of the solution, not the problem. Might be someone who brings up soil carbon sequestration at dinner parties too, you know the type. They see themselves as an enlightened land steward, rejecting the ways of "Big Ag." Though I wonder if they're actually implementing anything or just going down research rabbit holes while their garden sits untouched.

Let me think about this from different angles... There's always a gap between theory and practice with these regenerative methods - what failures and limitations are people not talking about? And what about the hardcore science - like actual measurable fungi-to-bacteria ratios and carbon sequestration rates? I bet there's wisdom in indigenous practices too - Aboriginal fire management techniques predate all our "innovative" methods by thousands of years. Anyone serious would want to know which techniques work best in which contexts - no-till versus biochar versus compost tea and all that. {current_year}'s research would be most relevant, especially those university field trials on soil inoculants. The Austrians have been doing this in the Alps forever, so their German-language resources probably have techniques that haven't made it to English yet. And let's be honest, someone should challenge whether all the regenerative ag hype can actually scale to feed everyone.
</think>
queries: [
  {{
    "tbs": "qdr:y",
    "location": "Fort Collins",
    "q": "regenerative agriculture soil failures limitations"
  }},
  {{
    "location": "Ithaca",
    "q": "mycorrhizal fungi quantitative sequestration metrics"
  }},
  {{
    "tbs": "qdr:y",
    "location": "Perth",
    "q": "aboriginal firestick farming soil restoration"
  }},
  {{
    "location": "Totnes",
    "q": "comparison no-till vs biochar vs compost tea"
  }},
  {{
    "tbs": "qdr:m",
    "location": "Davis",
    "q": "soil microbial inoculants research trials"
  }},
  {{
    "location": "Graz",
    "q": "Humusaufbau Alpenregion Techniken"
  }},
  {{
    "tbs": "qdr:m",
    "location": "Guelph",
    "q": "regenerative agriculture exaggerated claims evidence"
  }}
]
</example-2>

<example-3>
Input Query: KI\u30ea\u30c6\u30e9\u30b7\u30fc\u5411\u4e0a\u3055\u305b\u308b\u65b9\u6cd5
<think>
AI\u30ea\u30c6\u30e9\u30b7\u30fc\u5411\u4e0a\u3055\u305b\u308b\u65b9\u6cd5\u304b...\u306a\u308b\u307b\u3069\u3002\u6700\u8fd1AI\u304c\u3069\u3093\u3069\u3093\u8a71\u984c\u306b\u306a\u3063\u3066\u304d\u3066\u3001\u3064\u3044\u3066\u3044\u3051\u306a\u304f\u306a\u308b\u4e0d\u5b89\u304c\u3042\u308b\u3093\u3060\u308d\u3046\u306a\u3002\u8868\u9762\u7684\u306b\u306f\u5358\u306bAI\u306e\u77e5\u8b58\u3092\u5897\u3084\u3057\u305f\u3044\u3063\u3066\u3053\u3068\u3060\u3051\u3069\u3001\u672c\u97f3\u3092\u8a00\u3048\u3070\u3001\u8077\u5834\u3067AI\u30c4\u30fc\u30eb\u3092\u3046\u307e\u304f\u4f7f\u3044\u3053\u306a\u3057\u3066\u4e00\u76ee\u7f6e\u304b\u308c\u305f\u3044\u3093\u3058\u3083\u306a\u3044\u304b\u306a\u3002\u5468\u308a\u306f\u300cChatGPT\u3067\u3053\u3093\u306a\u3053\u3068\u304c\u3067\u304d\u308b\u300d\u3068\u304b\u8a00\u3063\u3066\u308b\u306e\u306b\u3001\u81ea\u5206\u3060\u3051\u7f6e\u3044\u3066\u3051\u307c\u308a\u306b\u306a\u308b\u306e\u304c\u6016\u3044\u3093\u3060\u308d\u3046\u3002\u6848\u5916\u3001\u57fa\u672c\u7684\u306aAI\u306e\u77e5\u8b58\u304c\u306a\u304f\u3066\u3001\u305d\u308c\u3092\u307f\u3093\u306a\u306b\u77e5\u3089\u308c\u305f\u304f\u306a\u3044\u3068\u3044\u3046\u6c17\u6301\u3061\u3082\u3042\u308b\u304b\u3082\u3002\u6839\u3063\u3053\u306e\u3068\u3053\u308d\u3067\u306f\u3001\u6280\u8853\u306e\u6ce2\u306b\u98f2\u307f\u8fbc\u307e\u308c\u308b\u6050\u6016\u611f\u304c\u3042\u308b\u3093\u3060\u3088\u306a\u3001\u308f\u304b\u308b\u3088\u305d\u306e\u6c17\u6301\u3061\u3002

\u3044\u308d\u3093\u306a\u8996\u70b9\u3067\u8003\u3048\u3066\u307f\u3088\u3046...AI\u3063\u3066\u5b9f\u969b\u3069\u3053\u307e\u3067\u3067\u304d\u308b\u3093\u3060\u308d\u3046\uff1f\u5ba3\u4f1d\u6587\u53e5\u3068\u5b9f\u969b\u306e\u80fd\u529b\u306b\u306f\u304b\u306a\u308a\u30ae\u30e3\u30c3\u30d7\u304c\u3042\u308a\u305d\u3046\u3060\u3057\u3001\u305d\u306e\u9650\u754c\u3092\u77e5\u308b\u3053\u3068\u3082\u5927\u4e8b\u3060\u3088\u306d\u3002\u3042\u3068\u3001AI\u30ea\u30c6\u30e9\u30b7\u30fc\u3063\u3066\u8a00\u3063\u3066\u3082\u3001\u3069\u3046\u5b66\u3079\u3070\u3044\u3044\u306e\u304b\u4f53\u7cfb\u7684\u306b\u6574\u7406\u3055\u308c\u3066\u308b\u306e\u304b\u306a\uff1f\u904e\u53bb\u306e\u300cAI\u9769\u547d\u300d\u3068\u304b\u3063\u3066\u7d50\u5c40\u3069\u3046\u306a\u3063\u305f\u3093\u3060\u308d\u3046\u3002\u30d0\u30d6\u30eb\u304c\u5f3e\u3051\u3066\u7d42\u308f\u3063\u305f\u3082\u306e\u3082\u3042\u308b\u3057\u3001\u305d\u306e\u6559\u8a13\u304b\u3089\u5b66\u3079\u308b\u3053\u3068\u3082\u3042\u308b\u306f\u305a\u3002\u30d7\u30ed\u30b0\u30e9\u30df\u30f3\u30b0\u3068\u9055\u3063\u3066AI\u30ea\u30c6\u30e9\u30b7\u30fc\u3063\u3066\u4f55\u306a\u306e\u304b\u3082\u306f\u3063\u304d\u308a\u3055\u305b\u305f\u3044\u3088\u306d\u3002\u6279\u5224\u7684\u601d\u8003\u529b\u3068\u306e\u95a2\u4fc2\u3082\u6c17\u306b\u306a\u308b\u3002{current_year}\u5e74\u306eAI\u30c8\u30ec\u30f3\u30c9\u306f\u7279\u306b\u5909\u5316\u304c\u901f\u305d\u3046\u3060\u304b\u3089\u3001\u6700\u65b0\u60c5\u5831\u3092\u62bc\u3055\u3048\u3066\u304a\u304f\u3079\u304d\u3060\u306a\u3002\u6d77\u5916\u306e\u65b9\u304c\u9032\u3093\u3067\u308b\u304b\u3089\u3001\u82f1\u8a9e\u306e\u8cc7\u6599\u3082\u898b\u305f\u65b9\u304c\u3044\u3044\u304b\u3082\u3057\u308c\u306a\u3044\u3057\u3002\u305d\u3082\u305d\u3082AI\u30ea\u30c6\u30e9\u30b7\u30fc\u3092\u8eab\u306b\u3064\u3051\u308b\u5fc5\u8981\u304c\u3042\u308b\u306e\u304b\uff1f\u300c\u6d41\u884c\u308a\u3060\u304b\u3089\u300d\u3068\u3044\u3046\u7406\u7531\u3060\u3051\u306a\u3089\u3001\u5b9f\u306f\u610f\u5473\u304c\u306a\u3044\u304b\u3082\u3057\u308c\u306a\u3044\u3088\u306d\u3002
</think>
queries: [
  {{
    "q": "AI\u6280\u8853 \u9650\u754c \u8a87\u5927\u5ba3\u4f1d"
  }},
  {{
    "q": "AI\u30ea\u30c6\u30e9\u30b7\u30fc \u5b66\u7fd2\u30b9\u30c6\u30c3\u30d7 \u4f53\u7cfb\u5316"
  }},
  {{
    "tbs": "qdr:y",
    "q": "AI\u6b74\u53f2 \u5931\u6557\u4e8b\u4f8b \u6559\u8a13"
  }},
  {{
    "q": "AI\u30ea\u30c6\u30e9\u30b7\u30fc vs \u30d7\u30ed\u30b0\u30e9\u30df\u30f3\u30b0 vs \u6279\u5224\u601d\u8003"
  }},
  {{
    "tbs": "qdr:m",
    "q": "AI\u6700\u65b0\u30c8\u30ec\u30f3\u30c9 \u5fc5\u9808\u30b9\u30ad\u30eb"
  }},
  {{
    "q": "artificial intelligence literacy fundamentals"
  }},
  {{
    "q": "AI\u30ea\u30c6\u30e9\u30b7\u30fc\u5411\u4e0a \u7121\u610f\u5473 \u7406\u7531"
  }}
]
</example-3>
</examples>

Each generated query must follow JSON schema format.
"""

    user = f"""
My original search query is: "{query}"

My motivation is: {think}

So I briefly googled "{query}" and found some soundbites about this topic, hope it gives you a rough idea about my context and topic:
<random-soundbites>
{context}
</random-soundbites>

Given those info, now please generate the best effective queries that follow JSON schema format; add correct 'tbs' you believe the query requires time-sensitive results.
"""

    return system, user


async def rewrite_query(
    action: SearchAction,
    context: str,
    generate_fn: Callable[..., Awaitable[BaseModel]],
    schema_gen: Any,
) -> list[dict]:
    """Expand search queries using 7 cognitive personas.

    For each search request in *action*, builds a prompt that applies 7 intent
    layers and 7 cognitive personas, then calls *generate_fn* to produce an
    expanded set of queries.

    Args:
        action: A SearchAction containing the original search_requests and
                the agent's reasoning (think).
        context: Soundbite text from preliminary search results to ground
                 the expansion.
        generate_fn: Async callable with signature
                     ``async def(schema, system, prompt) -> BaseModel``
                     that returns an object with ``.think`` and ``.queries``
                     attributes.
        schema_gen: Schema provider; must expose ``get_query_rewriter_schema()``.

    Returns:
        List of SERPQuery-like dicts, each with at minimum a ``q`` key
        and optionally ``tbs`` and/or ``location``.
    """
    try:
        schema = schema_gen.get_query_rewriter_schema()

        async def _expand_one(req: str) -> list[dict]:
            system, user = _get_prompt(req, action.think, context)
            result = await generate_fn(schema, system, user)
            return result.queries

        tasks = [_expand_one(req) for req in action.search_requests]
        query_results = await asyncio.gather(*tasks)

        all_queries: list[dict] = []
        for queries in query_results:
            if isinstance(queries, list):
                for q in queries:
                    if isinstance(q, dict):
                        all_queries.append(q)
                    elif isinstance(q, BaseModel):
                        all_queries.append(q.model_dump(exclude_none=True))
                    else:
                        all_queries.append(dict(q))

        logger.info("queryRewriter: expanded to %d queries", len(all_queries))
        return all_queries
    except Exception:
        logger.exception("Query rewrite error")
        raise
