"""Deep researcher tool — multi-round web research with completeness evaluation.

1:1 port of SkyworkAI's DeepResearcherTool. Performs:
1. Generate search query via LLM
2. Parallel search (web search + optional LLM search models)
3. Merge results
4. Evaluate completeness via structured LLM output
5. If complete -> break, else -> next round
6. Generate final summary
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from pydantic import BaseModel, Field

from orbiter.tool import Tool
from orbiter.types import SystemMessage, UserMessage

from ..llm_utils import call_llm
from ..reporter import Report
from .web_search import WebSearchTool

logger = logging.getLogger("deepagent")


# ---------------------------------------------------------------------------
# Structured output models (1:1 from SkyworkAI)
# ---------------------------------------------------------------------------


class CompletenessEvaluation(BaseModel):
    """Response format for evaluating if research is complete."""

    is_complete: bool = Field(
        description="Whether the summary provides a complete answer to the research task"
    )
    reasoning: str = Field(
        description="Brief explanation of why the answer is or isn't complete"
    )


class ResearchSummary(BaseModel):
    """Summary of the research report."""

    summary: str = Field(description="Comprehensive summary of the research findings")
    answer_found: bool = Field(
        description="Whether a complete answer was found to the research task"
    )
    answer_status: str = Field(
        description="Clear statement about whether the answer was found or not found"
    )


# ---------------------------------------------------------------------------
# DeepResearcherTool (1:1 port of SkyworkAI)
# ---------------------------------------------------------------------------


class DeepResearcherTool(Tool):
    """Multi-round deep research tool with automatic query refinement.

    Faithful port of SkyworkAI's DeepResearcherTool workflow:
    1. Generate search query via LLM
    2. Parallel search (web_search + optional LLM models)
    3. Merge results
    4. Evaluate completeness
    5. Loop or generate final summary
    """

    def __init__(
        self,
        *,
        tool_model: str = "openai:gpt-4o-mini",
        web_search: WebSearchTool | None = None,
        max_rounds: int = 3,
        num_results: int = 5,
        use_llm_search: bool = False,
        search_llm_models: list[str] | None = None,
        base_dir: str = "deepagent_output/deep_researcher",
    ) -> None:
        self.name = "deep_research"
        self.description = (
            "Deep research tool that performs multi-round web search and content analysis. "
            "Automatically generates search queries, evaluates completeness, and refines "
            "searches across multiple rounds. Best for complex research questions."
        )
        self.parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The research task or question to investigate.",
                },
                "filter_year": {
                    "type": "integer",
                    "description": "Optional year filter for search results.",
                },
                "title": {
                    "type": "string",
                    "description": 'Title for the report. Defaults to "Research Report".',
                },
            },
            "required": ["task"],
        }
        self._tool_model = tool_model
        self._web_search = web_search or WebSearchTool()
        self._max_rounds = max_rounds
        self._num_results = num_results
        self._use_llm_search = use_llm_search and bool(search_llm_models)
        self._search_llm_models = search_llm_models or []
        self._base_dir = base_dir

        os.makedirs(self._base_dir, exist_ok=True)

    async def execute(self, **kwargs: Any) -> str:
        """Execute deep research workflow.

        1:1 port of SkyworkAI DeepResearcherTool.__call__().

        Args:
            task: The research task or question.
            filter_year: Optional year filter.
            title: Optional report title.

        Returns:
            Research summary with report file path.
        """
        task: str = kwargs.get("task", "")
        filter_year: int | None = kwargs.get("filter_year")
        title: str | None = kwargs.get("title")

        if not task:
            return "Error: No research task provided."

        try:
            logger.info(f"Starting deep research for task: {task}")

            # Generate unique ID for this research session
            research_id = f"deep_researcher_{uuid.uuid4().hex[:8]}"

            # Per-call local variables (thread-safe)
            research_history: list[dict[str, Any]] = []

            # Create file path and Report instance
            md_filename = f"{research_id}.md"
            file_path = os.path.join(self._base_dir, md_filename)

            report_title = title if title is not None else "Research Report"
            report = Report(
                title=report_title,
                model_name=self._tool_model,
                report_file_path=file_path,
            )

            # Add initial task information
            task_content = f"## Research Task\n\n{task}\n\n"
            await report.add_item(content=task_content)

            final_evaluation = None

            # Multi-round search loop (1:1 from SkyworkAI)
            for round_num in range(1, self._max_rounds + 1):
                logger.info(f"Starting round {round_num}/{self._max_rounds}")

                # Generate search query
                query = await self._generate_search_query(
                    task, round_num, research_history
                )
                logger.info(f"Generated query for round {round_num}: {query}")

                # Execute parallel searches
                search_results = await self._parallel_search(task, query, filter_year)

                if not search_results:
                    logger.warning(f"All searches failed in round {round_num}")
                    empty_content = (
                        f"## Round {round_num}\n\n"
                        f"### Search Query\n\n{query}\n\n"
                        f"### Search Results\n\nNo search results found.\n\n"
                    )
                    await report.add_item(content=empty_content)
                    continue

                # Merge search results
                merged_summary = self._merge_search_results(search_results)
                logger.info(
                    f"Merged {len(search_results)} search results: {merged_summary[:500]}..."
                )

                # Record round information
                round_info = {
                    "round_number": round_num,
                    "query": query,
                    "summary": merged_summary,
                }
                research_history.append(round_info)

                # Evaluate completeness
                evaluation = await self._evaluate_completeness(task, merged_summary)

                # Add round content to report
                round_content = (
                    f"## Round {round_num}\n\n"
                    f"### Search Query\n\n{query}\n\n"
                    f"### Search Results\n\n{merged_summary}\n\n"
                )
                if evaluation:
                    round_content += (
                        f"### Evaluation\n\n"
                        f"- **Answer Found**: {'Yes' if evaluation.is_complete else 'No'}\n"
                        f"- **Reasoning**: {evaluation.reasoning}\n\n"
                    )

                await report.add_item(content=round_content)

                if evaluation.is_complete:
                    logger.info(
                        f"Answer found in round {round_num}: {evaluation.reasoning[:100]}..."
                    )
                    final_evaluation = evaluation
                    break

                logger.info(f"Round {round_num} completed, continuing to next round")

            # Finalize report and generate summary
            answer_found = final_evaluation.is_complete if final_evaluation else False

            if file_path:
                # Generate final report using Report.complete()
                final_report_content = await report.complete()

                # Generate summary from the final report
                summary = await self._generate_summary(
                    task, final_report_content, answer_found, final_evaluation
                )

                return f"Deep research summary: {summary}\n\nReport saved to: {file_path}"
            else:
                # Fallback
                if answer_found and final_evaluation:
                    final_message = (
                        f"{research_history[-1]['summary']}\n\n"
                        f"## Evaluation\n\n{final_evaluation.reasoning}"
                    )
                elif research_history:
                    final_message = (
                        f"Research incomplete after {self._max_rounds} rounds.\n\n"
                        f"{research_history[-1]['summary']}"
                    )
                else:
                    final_message = "No search results found in any round."

                return f"Deep research summary: {final_message}"

        except Exception as e:
            logger.error(f"Error in deep research: {e}")
            return f"Error during deep research: {e}"

    # ------------------------------------------------------------------
    # Query generation (1:1 from SkyworkAI)
    # ------------------------------------------------------------------

    async def _generate_search_query(
        self,
        task: str,
        round_num: int,
        research_history: list[dict[str, Any]],
    ) -> str:
        """Generate search query using LLM based on task and round number."""
        system_prompt = "You are a helpful assistant that can analyze tasks and images to generate optimized search queries."

        # Build context from previous rounds (last 2)
        previous_summaries = []
        if research_history:
            for round_info in research_history[-2:]:
                previous_summaries.append(
                    f"Round {round_info['round_number']} query: {round_info['query']}"
                )
                previous_summaries.append(
                    f"Summary: {round_info['summary'][:200]}..."
                )

        previous_context = (
            "\n".join(previous_summaries)
            if previous_summaries
            else "No previous searches yet."
        )
        round_context = (
            f"Round: {round_num}" if round_num > 1 else "Round: 1 (initial)"
        )

        if round_num > 1:
            instruction = (
                "generate a new search query that might help find missing information. "
                "Focus on different aspects, use different keywords, or explore related topics."
            )
        else:
            instruction = (
                "generate an optimized search query for web research. "
                "Focus on the most important keywords and concepts."
            )

        user_prompt = (
            f'Given this research task: "{task}"\n\n'
            f"No image provided\n"
            f"{round_context}\n"
            f"Previous search results:\n{previous_context}\n\n"
            f"Analyze the image if provided and combine it with the text task to {instruction}\n\n"
            f"IMPORTANT: The search query must be concise and focused. "
            f"Use only the most important keywords (typically 3-8 words). "
            f"Avoid long phrases or complete sentences. Keep it short and search-friendly.\n\n"
            f"Return only the search query, nothing else."
        )

        messages = [
            SystemMessage(content=system_prompt),
            UserMessage(content=user_prompt),
        ]

        response = await call_llm(model=self._tool_model, messages=messages)
        return response.message.strip()

    # ------------------------------------------------------------------
    # Parallel search (1:1 from SkyworkAI)
    # ------------------------------------------------------------------

    async def _parallel_search(
        self,
        task: str,
        query: str,
        filter_year: int | None,
    ) -> list[dict[str, Any]]:
        """Execute parallel searches using web_search or LLM models."""
        search_tasks = []

        if self._use_llm_search:
            if not self._search_llm_models:
                logger.warning(
                    "use_llm_search is True but no search_llm_models configured"
                )
                return []

            for model_name in self._search_llm_models:

                def create_llm_task(model: str) -> Any:
                    async def llm_search_task() -> dict[str, Any]:
                        try:
                            summary = await self._llm_search(model, task, query)
                            return {
                                "source": model,
                                "summary": summary,
                                "success": True,
                            }
                        except Exception as e:
                            logger.warning(f"LLM search with {model} failed: {e}")
                            return {
                                "source": model,
                                "summary": None,
                                "success": False,
                                "error": str(e),
                            }

                    return llm_search_task

                search_tasks.append(create_llm_task(model_name)())
        else:

            async def web_search_task() -> dict[str, Any]:
                try:
                    result = await self._web_search.execute(
                        query=query,
                        num_results=self._num_results,
                        filter_year=filter_year,
                    )
                    return {
                        "source": "web_search",
                        "summary": result,
                        "success": True,
                    }
                except Exception as e:
                    logger.warning(f"Web search failed: {e}")
                    return {
                        "source": "web_search",
                        "summary": None,
                        "success": False,
                        "error": str(e),
                    }

            search_tasks.append(web_search_task())

        # Execute all searches in parallel
        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Filter successful results
        search_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Search task raised exception: {result}")
                continue
            if result.get("success") and result.get("summary"):
                search_results.append(result)
            else:
                logger.warning(
                    f"Search from {result.get('source', 'unknown')} failed: "
                    f"{result.get('error', 'Unknown error')}"
                )

        return search_results

    async def _llm_search(
        self, model_name: str, task: str, query: str
    ) -> str:
        """Use LLM model to search the web and return summary.

        1:1 from SkyworkAI DeepResearcherTool._llm_search().
        """
        prompt = (
            "You are an expert web researcher. Based on the research task and search query, "
            "perform a comprehensive web search and provide a detailed summary.\n\n"
            f"Research Task: {task}\n"
            f"Search Query: {query}\n\n"
            "Please search the web for information related to this task and query, "
            "then provide a comprehensive summary that:\n"
            "1. Directly addresses the research task\n"
            "2. Includes relevant information from multiple sources\n"
            "3. Provides detailed insights and findings\n"
            "4. Includes citations or references when possible\n"
            "5. Is well-structured and easy to read\n\n"
            "Return your research findings as a comprehensive summary."
        )

        logger.info(f"Using LLM {model_name} to search the web.")

        response = await call_llm(
            model=model_name,
            messages=[UserMessage(content=prompt)],
        )

        logger.info(f"LLM {model_name} response: {response.message[:200]}...")

        if response.success and response.message.strip():
            return response.message.strip()
        raise ValueError(f"LLM {model_name} returned empty response")

    # ------------------------------------------------------------------
    # Merge results (1:1 from SkyworkAI)
    # ------------------------------------------------------------------

    def _merge_search_results(
        self, search_results: list[dict[str, Any]]
    ) -> str:
        """Merge multiple search results into a single summary."""
        if not search_results:
            return "No search results available."

        if len(search_results) == 1:
            return search_results[0]["summary"]

        # Combine with source labels
        combined_parts = []
        for i, result in enumerate(search_results, 1):
            source = result.get("source", f"Source {i}")
            summary = result.get("summary", "")
            combined_parts.append(f"## {source}\n\n{summary}\n")

        return "\n".join(combined_parts)

    # ------------------------------------------------------------------
    # Completeness evaluation (1:1 from SkyworkAI)
    # ------------------------------------------------------------------

    async def _evaluate_completeness(
        self, task: str, summary: str
    ) -> CompletenessEvaluation:
        """Evaluate if we have found a complete answer using structured LLM output."""
        prompt = (
            "Evaluate if the following summary provides a complete answer to the research task.\n\n"
            f"Research Task: {task}\n\n"
            f"Summary from web search:\n{summary}\n\n"
            "Determine if this summary provides enough information to answer the research task completely.\n\n"
            "Consider:\n"
            "- Does the information directly address the task?\n"
            "- Is there sufficient detail and depth?\n"
            "- Are there multiple perspectives or sources mentioned?\n"
            "- Is the information comprehensive enough?"
        )

        try:
            response = await call_llm(
                model=self._tool_model,
                messages=[UserMessage(content=prompt)],
                response_format=CompletenessEvaluation,
            )

            if response.parsed_model is not None:
                evaluation = response.parsed_model
                logger.info(
                    f"Evaluation: is_complete={evaluation.is_complete}, "
                    f"reasoning={evaluation.reasoning[:100]}..."
                )
                return evaluation

            # Fallback: text parsing (1:1 from SkyworkAI)
            logger.warning(
                "Failed to parse structured response, falling back to text parsing"
            )
            is_complete = False
            reasoning = "Failed to parse structured response."
            if response.success and response.message.strip():
                answer = response.message.strip().upper()
                is_complete = answer.startswith("YES")
                reasoning = f"Text-based evaluation: {response.message.strip()}"

            return CompletenessEvaluation(
                is_complete=is_complete, reasoning=reasoning
            )

        except Exception as e:
            # Fallback heuristic (1:1 from SkyworkAI)
            logger.warning(f"Failed to evaluate completeness with LLM: {e}")
            task_lower = task.lower()
            summary_lower = summary.lower()
            key_terms = [
                term for term in task_lower.split() if len(term) > 3
            ]

            is_complete = len(summary) > 500 and any(
                term in summary_lower for term in key_terms
            )
            reasoning = (
                f"Fallback heuristic evaluation: summary length={len(summary)}, "
                f"keyword match={'yes' if is_complete else 'no'}"
            )
            return CompletenessEvaluation(
                is_complete=is_complete, reasoning=reasoning
            )

    # ------------------------------------------------------------------
    # Summary generation (1:1 from SkyworkAI)
    # ------------------------------------------------------------------

    async def _generate_summary(
        self,
        task: str,
        report_content: str,
        answer_found: bool,
        evaluation: CompletenessEvaluation | None,
    ) -> str:
        """Generate a summary from the final report content."""
        answer_status = f"Answer Found: {'Yes' if answer_found else 'No'}"
        evaluation_text = (
            f"\n\nEvaluation: {evaluation.reasoning}" if evaluation else ""
        )

        summary_prompt = (
            "Based on the research report below, generate a comprehensive summary.\n\n"
            f"Research Task: {task}\n"
            f"Answer Found Status: {answer_status}{evaluation_text}\n\n"
            f"Research Report:\n{report_content}\n\n"
            "Generate a summary that:\n"
            '1. MUST start with a clear statement: "Answer Found: Yes" or "Answer Found: No"\n'
            "2. Provides a concise overview of the key findings\n"
            "3. Highlights the most important information discovered\n"
            "4. Mentions the number of research rounds conducted\n"
            "5. If answer was found, summarize the key answer points\n"
            "6. If answer was not found, explain what was discovered and what gaps remain\n\n"
            "The summary should be clear and informative, suitable as a final response message.\n"
            "The first line must explicitly state whether the answer was found."
        )

        summary_messages = [
            SystemMessage(
                content=(
                    "You are an expert at summarizing research reports. "
                    "Generate clear, informative summaries that MUST explicitly state "
                    "whether answers were found in the first line using "
                    "'Answer Found: Yes' or 'Answer Found: No'."
                )
            ),
            UserMessage(content=summary_prompt),
        ]

        summary_response = await call_llm(
            model=self._tool_model,
            messages=summary_messages,
            response_format=ResearchSummary,
        )

        # Extract summary text (1:1 from SkyworkAI)
        if summary_response.parsed_model is not None:
            research_summary = summary_response.parsed_model
            summary_text = research_summary.summary
            if research_summary.answer_found != answer_found:
                logger.warning(
                    f"Answer found mismatch: model says {research_summary.answer_found}, "
                    f"actual is {answer_found}"
                )
        else:
            summary_text = summary_response.message.strip()

        # Ensure answer status at beginning
        if not summary_text.startswith("Answer Found:"):
            return f"{answer_status}\n\n{summary_text}"
        return summary_text
