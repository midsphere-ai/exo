"""Error analysis for failed search/answer attempts.

Ported from node-DeepResearch/src/tools/error-analyzer.ts.
Analyzes a sequence of search steps to identify what went wrong and
suggest improvements for the next attempt.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable

from pydantic import BaseModel

logger = logging.getLogger("deepsearch")


def _get_prompt(diary_context: list[str]) -> tuple[str, str]:
    """Build (system, user) prompt pair for error analysis.

    Args:
        diary_context: List of step descriptions from the research diary.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    system = """You are an expert at analyzing search and reasoning processes. Your task is to analyze the given sequence of steps and identify what went wrong in the search process.

<rules>
1. The sequence of actions taken
2. The effectiveness of each step
3. The logic between consecutive steps
4. Alternative approaches that could have been taken
5. Signs of getting stuck in repetitive patterns
6. Whether the final answer matches the accumulated information

Analyze the steps and provide detailed feedback following these guidelines:
- In the recap: Summarize key actions chronologically, highlight patterns, and identify where the process started to go wrong
- In the blame: Point to specific steps or patterns that led to the inadequate answer
- In the improvement: Provide actionable suggestions that could have led to a better outcome
</rules>

<example>
<input>
<steps>

At step 1, you took the **search** action and look for external information for the question: "how old is jina ai ceo?".
In particular, you tried to search for the following keywords: "jina ai ceo age".
You found quite some information and add them to your URL list and **visit** them later when needed.


At step 2, you took the **visit** action and deep dive into the following URLs:
https://www.linkedin.com/in/hxiao87
https://www.crunchbase.com/person/han-xiao
You found some useful information on the web and add them to your knowledge for future reference.


At step 3, you took the **search** action and look for external information for the question: "how old is jina ai ceo?".
In particular, you tried to search for the following keywords: "Han Xiao birthdate, Jina AI founder birthdate".
You found quite some information and add them to your URL list and **visit** them later when needed.


At step 4, you took the **search** action and look for external information for the question: "how old is jina ai ceo?".
In particular, you tried to search for the following keywords: han xiao birthday.
But then you realized you have already searched for these keywords before.
You decided to think out of the box or cut from a completely different angle.


At step 5, you took the **search** action and look for external information for the question: "how old is jina ai ceo?".
In particular, you tried to search for the following keywords: han xiao birthday.
But then you realized you have already searched for these keywords before.
You decided to think out of the box or cut from a completely different angle.


At step 6, you took the **visit** action and deep dive into the following URLs:
https://kpopwall.com/han-xiao/
https://www.idolbirthdays.net/han-xiao
You found some useful information on the web and add them to your knowledge for future reference.


At step 7, you took **answer** action but evaluator thinks it is not a good answer:

</steps>

Original question:
how old is jina ai ceo?

Your answer:
The age of the Jina AI CEO cannot be definitively determined from the provided information.

The evaluator thinks your answer is bad because:
The answer is not definitive and fails to provide the requested information.  Lack of information is unacceptable, more search and deep reasoning is needed.
</input>


<output>
{
  "recap": "The search process consisted of 7 steps with multiple search and visit actions. The initial searches focused on basic biographical information through LinkedIn and Crunchbase (steps 1-2). When this didn't yield the specific age information, additional searches were conducted for birthdate information (steps 3-5). The process showed signs of repetition in steps 4-5 with identical searches. Final visits to entertainment websites (step 6) suggested a loss of focus on reliable business sources.",

  "blame": "The root cause of failure was getting stuck in a repetitive search pattern without adapting the strategy. Steps 4-5 repeated the same search, and step 6 deviated to less reliable entertainment sources instead of exploring business journals, news articles, or professional databases. Additionally, the process didn't attempt to triangulate age through indirect information like education history or career milestones.",

  "improvement": "1. Avoid repeating identical searches and implement a strategy to track previously searched terms. 2. When direct age/birthdate searches fail, try indirect approaches like: searching for earliest career mentions, finding university graduation years, or identifying first company founding dates. 3. Focus on high-quality business sources and avoid entertainment websites for professional information. 4. Consider using industry event appearances or conference presentations where age-related context might be mentioned. 5. If exact age cannot be determined, provide an estimated range based on career timeline and professional achievements.",
}
</output>
</example>"""

    user = "\n".join(diary_context)

    return system, user


async def analyze_steps(
    diary_context: list[str],
    generate_fn: Callable[..., Awaitable[BaseModel]],
    schema_gen: Any,
) -> dict:
    """Analyze a failed answer attempt and diagnose what went wrong.

    Examines the sequence of search/visit/answer steps recorded in the
    research diary and produces a structured analysis with recap, blame,
    and improvement sections.

    Args:
        diary_context: List of step description strings from the research
                       diary, to be joined and sent as the user prompt.
        generate_fn: Async callable with signature
                     ``async def(schema, system, prompt) -> BaseModel``
                     that returns an object with ``.recap``, ``.blame``,
                     and ``.improvement`` attributes.
        schema_gen: Schema provider; must expose ``get_error_analysis_schema()``.

    Returns:
        Dict with keys ``recap``, ``blame``, and ``improvement``, each a
        string describing the analysis findings.
    """
    try:
        schema = schema_gen.get_error_analysis_schema()
        system, user = _get_prompt(diary_context)

        result = await generate_fn(schema, system, user)

        analysis = {
            "recap": result.recap,
            "blame": result.blame,
            "improvement": result.improvement,
        }

        logger.info("errorAnalyzer: %s", analysis)
        return analysis
    except Exception:
        logger.exception("Error in errorAnalyzer")
        raise
