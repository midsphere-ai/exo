"""Core research loop — the heart of DeepSearch.

This is a direct port of ``getResponse()`` from ``node-DeepResearch/src/agent.ts``.
It uses Orbiter's model providers for LLM calls but implements its own custom
Search-Read-Reason loop with explicit state management.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from orbiter.models.provider import get_provider
from orbiter.types import AssistantMessage, SystemMessage, UserMessage

from .config import DeepSearchConfig
from .schemas import Schemas
from .prompts import get_prompt, compose_msgs
from .types import (
    AnswerAction,
    BoostedSearchSnippet,
    CodingAction,
    KnowledgeItem,
    ReflectAction,
    Reference,
    ResearchResult,
    SearchAction,
    SearchSnippet,
    StepAction,
    VisitAction,
    WebContent,
)
from .utils.token_tracker import TokenTracker
from .utils.action_tracker import ActionTracker
from .utils.url_tools import (
    add_to_all_urls,
    extract_urls_with_description,
    filter_urls,
    keep_k_per_hostname,
    normalize_url,
    rank_urls,
)
from .utils.text_tools import (
    build_md_from_answer,
    choose_k,
    chunk_text,
    remove_extra_line_breaks,
    remove_html_tags,
    repair_markdown_final,
)
from .utils.date_tools import format_date_based_on_type, format_date_range
from .tools.search import get_search_provider
from .tools.reader import get_reader
from .tools.embeddings import get_embedding_provider
from .tools.evaluator import evaluate_question, evaluate_answer
from .tools.query_rewriter import rewrite_query
from .tools.dedup import dedup_queries
from .tools.serp_cluster import cluster_results
from .tools.error_analyzer import analyze_steps
from .tools.code_sandbox import CodeSandbox
from .tools.finalizer import build_references, finalize_answer
from .context_manager import DeepSearchContextManager

logger = logging.getLogger("deepsearch")


class DeepSearchEngine:
    """Iterative deep-research engine.

    Manages a Search-Read-Reason loop that converges on a high-quality,
    citation-backed answer.
    """

    def __init__(self, config: DeepSearchConfig) -> None:
        self.config = config
        self.token_tracker = TokenTracker(config.token_budget)
        self.action_tracker = ActionTracker()
        self.schema_gen = Schemas()

        # Orbiter context management for bounded context growth
        self.context_mgr = DeepSearchContextManager(
            token_budget=config.token_budget,
            max_knowledge_items=40,
            max_diary_entries=12,
            keep_recent_diary=6,
            max_knowledge_answer_len=600,
        )

        # Providers
        self.provider = get_provider(config.model_string)
        self.search_provider = get_search_provider(config)
        self.reader = get_reader(config)
        self.embedding_provider = get_embedding_provider(config)

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    async def _generate_object(
        self,
        schema: type[BaseModel],
        system: str,
        prompt: str,
        *,
        messages: list[dict[str, str]] | None = None,
        tool_name: str = "agent",
    ) -> BaseModel:
        """Generate a structured JSON object from the LLM."""
        json_schema = schema.model_json_schema()
        full_system = (
            f"{system}\n\n"
            "You MUST respond with valid JSON matching this schema:\n"
            f"{json.dumps(json_schema, indent=2)}"
        )

        orbiter_msgs: list[Any] = [SystemMessage(content=full_system)]
        for msg in messages or []:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                orbiter_msgs.append(UserMessage(content=content))
            elif role == "assistant":
                orbiter_msgs.append(AssistantMessage(content=content))

        if prompt:
            orbiter_msgs.append(UserMessage(content=prompt))

        response = await self.provider.complete(
            orbiter_msgs, temperature=0.7, max_tokens=8192
        )
        self.token_tracker.track_usage(
            tool_name, response.usage.input_tokens, response.usage.output_tokens
        )

        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        def _try_parse(text: str) -> dict:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Fix invalid escape sequences (common LLM issue)
                fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    raise

        try:
            data = _try_parse(content)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", content)
            if match:
                data = _try_parse(match.group())
            else:
                err = ValueError(f"Could not parse JSON: {content[:200]}")
                err.raw_text = content  # type: ignore[attr-defined]
                raise err

        # Convert camelCase keys to snake_case for Pydantic compatibility
        data = self._camel_to_snake_keys(data)

        # If the LLM echoed the schema wrapper, unwrap it
        if "properties" in data and isinstance(data["properties"], dict):
            # LLM returned {properties: {field: {description:..., ...}, ...}, field_val: val}
            # Try to extract actual values that sit alongside "properties"
            unwrapped = {k: v for k, v in data.items() if k != "properties"}
            if unwrapped:
                data = unwrapped
            else:
                # Try extracting default/value from properties entries
                props = data["properties"]
                data = {}
                for k, v in props.items():
                    if isinstance(v, dict):
                        data[k] = v.get("default", v.get("const", v.get("enum", [None])[0] if "enum" in v else None))
                    else:
                        data[k] = v

        # Fix common LLM mistakes: string where sub-object expected
        # e.g. answer: "some text" instead of answer: {answer: "some text"}
        if isinstance(data, dict) and data.get("action") == "answer" and isinstance(data.get("answer"), str):
            data["answer"] = {"answer": data["answer"]}

        # Coerce data to fit schema constraints (LLMs often exceed max_length/max_items)
        json_schema = schema.model_json_schema()
        data = self._coerce_to_schema(data, json_schema)
        return schema.model_validate(data)

    @staticmethod
    def _camel_to_snake_keys(data: Any) -> Any:
        """Recursively convert camelCase dict keys to snake_case."""
        if isinstance(data, dict):
            converted = {}
            for k, v in data.items():
                # Two-step: handle acronyms (URLTargets->URL_Targets) then normal camel
                snake = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', k)
                snake = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', snake).lower()
                converted[snake] = DeepSearchEngine._camel_to_snake_keys(v)
            return converted
        if isinstance(data, list):
            return [DeepSearchEngine._camel_to_snake_keys(item) for item in data]
        return data

    async def _generate_text(
        self, system: str, prompt: str, tool_name: str = "finalizer"
    ) -> str:
        """Generate plain text (for the finalizer)."""
        msgs: list[Any] = [SystemMessage(content=system), UserMessage(content=prompt)]
        response = await self.provider.complete(msgs, temperature=0.7, max_tokens=8192)
        self.token_tracker.track_usage(
            tool_name, response.usage.input_tokens, response.usage.output_tokens
        )
        return response.content

    @staticmethod
    def _coerce_to_schema(data: Any, schema: dict, defs: dict | None = None) -> Any:
        """Best-effort coerce *data* to fit JSON schema constraints.

        Truncates strings exceeding ``maxLength`` and trims lists exceeding
        ``maxItems`` so that Pydantic validation doesn't reject otherwise
        valid LLM output.
        """
        if defs is None:
            defs = schema.get("$defs", {})

        if isinstance(data, dict):
            props = schema.get("properties", {})
            for key, val in list(data.items()):
                if key in props:
                    data[key] = DeepSearchEngine._coerce_to_schema(val, props[key], defs)
                # Handle $ref
            # Check for anyOf / allOf / $ref at top level
            ref = schema.get("$ref")
            if ref and ref.startswith("#/$defs/"):
                ref_name = ref.split("/")[-1]
                if ref_name in defs:
                    return DeepSearchEngine._coerce_to_schema(data, defs[ref_name], defs)
            for variant in schema.get("anyOf", []):
                ref = variant.get("$ref")
                if ref and ref.startswith("#/$defs/") and isinstance(data, dict):
                    ref_name = ref.split("/")[-1]
                    if ref_name in defs:
                        data = DeepSearchEngine._coerce_to_schema(data, defs[ref_name], defs)
            return data

        if isinstance(data, str):
            max_len = schema.get("maxLength")
            if max_len and len(data) > max_len:
                data = data[:max_len]
            return data

        if isinstance(data, list):
            max_items = schema.get("maxItems")
            if max_items and len(data) > max_items:
                data = data[:max_items]
            items_schema = schema.get("items", {})
            ref = items_schema.get("$ref")
            if ref and ref.startswith("#/$defs/"):
                ref_name = ref.split("/")[-1]
                if ref_name in defs:
                    items_schema = defs[ref_name]
            return [DeepSearchEngine._coerce_to_schema(item, items_schema, defs) for item in data]

        return data

    async def _generate_fn(
        self, schema: type[BaseModel], system: str, prompt: str, **kw: Any
    ) -> BaseModel:
        """Convenience wrapper passed to tool functions."""
        return await self._generate_object(schema, system, prompt, **kw)

    def _schema_gen_fn(self, eval_type: str) -> type[BaseModel]:
        """Adapter: turns an eval-type string into a Pydantic schema.

        The evaluator calls ``schema_gen(eval_type)`` so we dispatch here.
        """
        dispatch: dict[str, Any] = {
            "question_evaluate": self.schema_gen.get_question_evaluate_schema,
            "definitive": lambda: self.schema_gen.get_evaluator_schema("definitive"),
            "freshness": lambda: self.schema_gen.get_evaluator_schema("freshness"),
            "plurality": lambda: self.schema_gen.get_evaluator_schema("plurality"),
            "completeness": lambda: self.schema_gen.get_evaluator_schema("completeness"),
            "attribution": lambda: self.schema_gen.get_evaluator_schema("attribution"),
            "strict": lambda: self.schema_gen.get_evaluator_schema("strict"),
        }
        fn = dispatch.get(eval_type)
        if fn is None:
            raise ValueError(f"Unknown schema type: {eval_type}")
        return fn()

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_action(result: BaseModel) -> StepAction:
        """Convert a raw LLM object into the correct StepAction."""
        action = getattr(result, "action", "answer")
        think = getattr(result, "think", "")

        if action == "search":
            obj = getattr(result, "search", None)
            reqs = (
                getattr(obj, "search_requests", None)
                or getattr(obj, "searchRequests", None)
                or getattr(result, "search_requests", None)
                or getattr(result, "searchRequests", [])
            )
            return SearchAction(think=think, search_requests=list(reqs))

        if action == "visit":
            obj = getattr(result, "visit", None)
            targets = (
                getattr(obj, "url_targets", None)
                or getattr(obj, "URLTargets", None)
                or getattr(result, "url_targets", None)
                or getattr(result, "URLTargets", [])
            )
            return VisitAction(think=think, url_targets=list(targets))

        if action == "reflect":
            obj = getattr(result, "reflect", None)
            qs = (
                getattr(obj, "questions_to_answer", None)
                or getattr(obj, "questionsToAnswer", None)
                or getattr(result, "questions_to_answer", None)
                or getattr(result, "questionsToAnswer", [])
            )
            return ReflectAction(think=think, questions_to_answer=list(qs))

        if action == "coding":
            obj = getattr(result, "coding", None)
            issue = (
                getattr(obj, "coding_issue", None)
                or getattr(obj, "codingIssue", None)
                or getattr(result, "coding_issue", None)
                or getattr(result, "codingIssue", "")
            )
            return CodingAction(think=think, coding_issue=str(issue))

        # Default: answer
        obj = getattr(result, "answer", None)
        if isinstance(obj, str):
            text = obj
        elif obj is not None and hasattr(obj, "answer"):
            text = obj.answer
        else:
            text = str(obj) if obj else ""
        return AnswerAction(think=think, answer=text)

    # ------------------------------------------------------------------
    # Sub-routines
    # ------------------------------------------------------------------

    async def _execute_search(
        self,
        queries: list[dict[str, Any]],
        all_urls: dict[str, SearchSnippet],
        web_contents: dict[str, WebContent],
    ) -> tuple[list[KnowledgeItem], list[str]]:
        """Run search queries. Returns (new_knowledge, searched_queries)."""
        new_knowledge: list[KnowledgeItem] = []
        searched: list[str] = []

        for qdict in queries:
            q = qdict.get("q", "")
            tbs = qdict.get("tbs")
            location = qdict.get("location")
            try:
                results = await self.search_provider.search(
                    q, num_results=20, tbs=tbs, location=location
                )
                if not results:
                    continue

                for r in results:
                    url = normalize_url(r.get("url", ""))
                    if not url:
                        continue
                    snippet = SearchSnippet(
                        title=r.get("title", ""),
                        url=url,
                        description=(r.get("description", "") or "")[:500],
                        date=r.get("date"),
                    )
                    add_to_all_urls(snippet, all_urls)
                    web_contents[url] = WebContent(
                        title=snippet.title,
                        chunks=[snippet.description] if snippet.description else [],
                        chunk_positions=(
                            [[0, len(snippet.description)]] if snippet.description else []
                        ),
                    )

                searched.append(q)

                # Cluster for knowledge extraction
                try:
                    clusters = await cluster_results(
                        results, self._generate_fn, self.schema_gen
                    )
                    for c in clusters:
                        new_knowledge.append(
                            KnowledgeItem(
                                question=c.get("question", ""),
                                answer=c.get("insight", ""),
                                references=c.get("urls", []),
                                type="url",
                            )
                        )
                except Exception:
                    pass

                descriptions = " ".join(
                    remove_html_tags(r.get("description", "")) for r in results
                )
                new_knowledge.append(
                    KnowledgeItem(
                        question=f'What does the Internet say about "{q}"?',
                        answer=descriptions[:2000],
                        type="side-info",
                        updated=format_date_range(qdict) if tbs else None,
                    )
                )
            except Exception as exc:
                logger.warning("Search failed for '%s': %s", q, exc)

            await asyncio.sleep(self.config.step_sleep)

        return new_knowledge, searched

    async def _process_urls(
        self,
        urls: list[str],
        knowledge: list[KnowledgeItem],
        all_urls: dict[str, SearchSnippet],
        visited: list[str],
        bad_urls: list[str],
        web_contents: dict[str, WebContent],
        question: str,
    ) -> bool:
        """Read URLs, add to knowledge. Returns True on any success."""
        any_success = False
        for url in urls:
            try:
                result = await self.reader.read(
                    url, with_links=True, with_images=self.config.with_images
                )
                if not result.success or not result.content:
                    bad_urls.append(url)
                    continue

                visited.append(url)
                any_success = True

                content = result.content[:30_000]
                chunked = chunk_text(content)
                web_contents[url] = WebContent(
                    title=result.title,
                    full=content,
                    chunks=chunked["chunks"],
                    chunk_positions=chunked["chunk_positions"],
                )
                knowledge.append(
                    KnowledgeItem(
                        question=f"What is the content of {url}?",
                        answer=content[:2000],
                        references=[url],
                        type="url",
                        updated=format_date_based_on_type(datetime.now(timezone.utc)),
                    )
                )
            except Exception as exc:
                logger.warning("Failed to read %s: %s", url, exc)
                bad_urls.append(url)
        return any_success

    async def _beast_mode(
        self,
        question: str,
        messages: list[dict[str, str]],
        knowledge: list[KnowledgeItem],
        diary: list[str],
        all_questions: list[str],
        all_keywords: list[str],
        weighted_urls: list[BoostedSearchSnippet],
        final_answer_pip: list[str],
    ) -> AnswerAction:
        """Last resort: force an answer with everything we have."""
        prompt_result = get_prompt(
            context=diary,
            all_questions=all_questions,
            all_keywords=all_keywords,
            allow_reflect=False,
            allow_answer=False,
            allow_read=False,
            allow_search=False,
            allow_coding=False,
            knowledge=knowledge,
            all_urls=weighted_urls,
            beast_mode=True,
        )
        schema = self.schema_gen.get_agent_schema(False, False, True, False, False, question)
        msg_k = compose_msgs(messages, knowledge, question, final_answer_pip)
        try:
            raw = await self._generate_object(
                schema, prompt_result["system"], "",
                messages=msg_k, tool_name="agentBeastMode",
            )
            step = self._parse_action(raw)
            return AnswerAction(
                think=getattr(step, "think", ""),
                answer=getattr(step, "answer", ""),
                is_final=True,
            )
        except Exception as exc:
            logger.error("Beast mode failed: %s", exc)
            return AnswerAction(answer="Unable to find a satisfactory answer.", is_final=True)

    # ------------------------------------------------------------------
    # Main research loop
    # ------------------------------------------------------------------

    async def research(
        self,
        question: str,
        messages: list[dict[str, str]] | None = None,
    ) -> ResearchResult:
        """Run the full research loop (port of getResponse)."""

        question = (question or "").strip()
        if messages:
            messages = [m for m in messages if m.get("role") != "system"]
        if not messages:
            messages = [{"role": "user", "content": question}]

        last = messages[-1]
        if last.get("role") == "user":
            question = last["content"].strip()

        # Language detection
        self.schema_gen.set_generate_fn(self._generate_fn)
        await self.schema_gen.detect_language(question)

        # State
        gaps: list[str] = [question]
        all_urls: dict[str, SearchSnippet] = {}
        all_knowledge: list[KnowledgeItem] = []  # canonical list (full, unwindowed)
        all_web_contents: dict[str, WebContent] = {}
        visited_urls: list[str] = []
        bad_urls: list[str] = []
        all_keywords: list[str] = []
        all_questions: list[str] = [question]
        diary_context: list[str] = []
        evaluation_metrics: dict[str, list[dict[str, Any]]] = {}
        final_answer_pip: list[str] = []

        # Context manager tracks windowed views for LLM calls
        ctx = self.context_mgr

        weighted_urls: list[BoostedSearchSnippet] = []
        allow_answer = True
        allow_search = True
        allow_read = True
        allow_reflect = True
        allow_coding = False

        this_step: StepAction = AnswerAction()
        trivial_question = False
        regular_budget = int(self.config.token_budget * 0.85)
        step = 0
        total_step = 0

        # Seed URLs from messages
        for msg in messages:
            for snippet in extract_urls_with_description(msg.get("content", "")):
                add_to_all_urls(snippet, all_urls)

        # ===== MAIN LOOP =====
        while self.token_tracker.total_tokens < regular_budget:
            step += 1
            total_step += 1
            budget_pct = self.token_tracker.total_tokens / self.config.token_budget * 100
            logger.info("Step %d | Budget %.1f%% | Gaps %d", total_step, budget_pct, len(gaps))

            allow_reflect = allow_reflect and len(gaps) <= self.config.max_reflect_per_step
            current_question = gaps[total_step % len(gaps)]

            # Evaluation metrics (step 1 only for original question)
            if current_question.strip() == question and total_step == 1:
                eval_types = await evaluate_question(
                    current_question, self._generate_fn, self._schema_gen_fn
                )
                evaluation_metrics[current_question] = [
                    {"type": t, "num_evals_required": self.config.max_bad_attempts}
                    for t in eval_types
                ]
                evaluation_metrics[current_question].append(
                    {"type": "strict", "num_evals_required": self.config.max_bad_attempts}
                )
            elif current_question.strip() != question and current_question not in evaluation_metrics:
                evaluation_metrics[current_question] = []

            if total_step == 1 and any(
                e["type"] == "freshness" for e in evaluation_metrics.get(current_question, [])
            ):
                allow_answer = False
                allow_reflect = False

            # URL ranking
            if all_urls:
                filtered = filter_urls(all_urls, visited_urls)
                weighted_urls = rank_urls(filtered, current_question)
                weighted_urls = keep_k_per_hostname(weighted_urls, 2)

            allow_read = allow_read and len(weighted_urls) > 0
            allow_search = allow_search and len(weighted_urls) < 50

            # --- Context management: fire processors before LLM call ---
            ctx.update_token_usage(self.token_tracker.total_tokens)
            await ctx.pre_llm_call()
            windowed_knowledge = ctx.get_knowledge()
            windowed_diary = ctx.get_diary()
            if len(windowed_knowledge) < len(all_knowledge) or len(windowed_diary) < len(diary_context):
                logger.info(
                    "Context managed: knowledge %d→%d, diary %d→%d",
                    len(all_knowledge), len(windowed_knowledge),
                    len(diary_context), len(windowed_diary),
                )

            # Build prompt (using windowed context)
            prompt_result = get_prompt(
                context=windowed_diary, all_questions=all_questions,
                all_keywords=all_keywords, allow_reflect=allow_reflect,
                allow_answer=allow_answer, allow_read=allow_read,
                allow_search=allow_search, allow_coding=allow_coding,
                knowledge=windowed_knowledge, all_urls=weighted_urls, beast_mode=False,
            )
            system_prompt = prompt_result["system"]
            url_list: list[str] = prompt_result.get("url_list", [])

            schema = self.schema_gen.get_agent_schema(
                allow_reflect, allow_read, allow_answer, allow_search,
                allow_coding, current_question,
            )
            msg_with_knowledge = compose_msgs(
                messages, windowed_knowledge, current_question,
                final_answer_pip if current_question == question else None,
            )

            # Generate action
            try:
                raw = await self._generate_object(
                    schema, system_prompt, "", messages=msg_with_knowledge, tool_name="agent",
                )
                this_step = self._parse_action(raw)
            except Exception as exc:
                # Smart models often just answer in plain text instead of JSON.
                # Extract the full text and treat it as an answer action.
                raw_text = getattr(exc, "raw_text", "")
                if raw_text and len(raw_text) > 50:
                    logger.info("LLM returned plain text instead of JSON; treating as answer")
                    this_step = AnswerAction(think="Direct answer from LLM", answer=raw_text)
                else:
                    logger.error("Action generation failed: %s", exc)
                    break
            action = this_step.action
            logger.info("Step %d -> %s", total_step, action)
            self.action_tracker.track_action(total_step=total_step, action=action)

            # Reset per-step flags
            allow_answer = True
            allow_reflect = True
            allow_read = True
            allow_search = True
            allow_coding = True

            # ==================== EXECUTE ACTION ====================

            if action == "answer" and isinstance(this_step, AnswerAction) and this_step.answer:
                if total_step == 1 and not self.config.no_direct_answer:
                    this_step.is_final = True
                    trivial_question = True
                    break

                current_evals = evaluation_metrics.get(current_question, [])
                eval_result: dict[str, Any] = {"pass": True, "think": ""}
                active = [e["type"] for e in current_evals if e["num_evals_required"] > 0]
                if active:
                    eval_result = await evaluate_answer(
                        current_question, this_step, active,
                        self._generate_fn, self._schema_gen_fn, all_knowledge,
                    )

                if current_question.strip() == question:
                    allow_coding = False
                    if eval_result.get("pass"):
                        entry = (
                            f"At step {step}, you answered the original question.\n"
                            f"Question: {current_question}\nAnswer: {this_step.answer}\n"
                            f"Evaluator: {eval_result.get('think', '')}\n"
                        )
                        diary_context.append(entry)
                        ctx.add_diary_entry(entry)
                        this_step.is_final = True
                        break

                    failed_type = eval_result.get("type")
                    evaluation_metrics[current_question] = [
                        {**e, "num_evals_required": e["num_evals_required"] - (1 if e["type"] == failed_type else 0)}
                        for e in evaluation_metrics[current_question]
                    ]
                    evaluation_metrics[current_question] = [
                        e for e in evaluation_metrics[current_question] if e["num_evals_required"] > 0
                    ]
                    if failed_type == "strict" and eval_result.get("improvement_plan"):
                        final_answer_pip.append(eval_result["improvement_plan"])
                    if not evaluation_metrics[current_question]:
                        this_step.is_final = False
                        break

                    entry = (
                        f"At step {step}, answer rejected.\nQuestion: {current_question}\n"
                        f"Answer: {this_step.answer}\nReason: {eval_result.get('think', '')}\n"
                    )
                    diary_context.append(entry)
                    ctx.add_diary_entry(entry)
                    try:
                        ea = await analyze_steps(diary_context, self._generate_fn, self.schema_gen)
                        ki = KnowledgeItem(
                            question=f"Why is this answer bad for: {current_question}?",
                            answer=f"{eval_result.get('think','')}\n{ea.get('recap','')}\n{ea.get('blame','')}\n{ea.get('improvement','')}",
                            type="qa",
                        )
                        all_knowledge.append(ki)
                        ctx.add_knowledge([ki])
                    except Exception as exc:
                        logger.warning("Error analysis failed: %s", exc)

                    allow_answer = False
                    diary_context = []
                    ctx.reset_diary()
                    step = 0

                elif eval_result.get("pass"):
                    entry = f"At step {step}, answered sub-question: {current_question}\n"
                    diary_context.append(entry)
                    ctx.add_diary_entry(entry)
                    ki = KnowledgeItem(
                        question=current_question, answer=this_step.answer, type="qa",
                        updated=format_date_based_on_type(datetime.now(timezone.utc)),
                    )
                    all_knowledge.append(ki)
                    ctx.add_knowledge([ki])
                    if current_question in gaps:
                        gaps.remove(current_question)

            elif action == "reflect" and isinstance(this_step, ReflectAction):
                questions = this_step.questions_to_answer
                try:
                    questions = await dedup_queries(questions, all_questions, self.embedding_provider)
                except Exception:
                    pass
                questions = choose_k(questions, self.config.max_reflect_per_step)
                if questions:
                    entry = f"At step {step}, reflected:\n" + "\n".join(f"- {q}" for q in questions)
                    diary_context.append(entry)
                    ctx.add_diary_entry(entry)
                    gaps.extend(questions)
                    all_questions.extend(questions)
                else:
                    entry = f"At step {step}, reflected but all questions already asked."
                    diary_context.append(entry)
                    ctx.add_diary_entry(entry)
                allow_reflect = False

            elif action == "search" and isinstance(this_step, SearchAction):
                reqs = this_step.search_requests
                try:
                    reqs = await dedup_queries(reqs, [], self.embedding_provider)
                except Exception:
                    pass
                reqs = choose_k(reqs, self.config.max_queries_per_step)

                new_knowledge, searched = await self._execute_search(
                    [{"q": q} for q in reqs], all_urls, all_web_contents
                )
                all_keywords.extend(searched)
                all_knowledge.extend(new_knowledge)
                ctx.add_knowledge(new_knowledge)
                sound_bites = " ".join(k.answer for k in new_knowledge)

                try:
                    rewritten = await rewrite_query(this_step, sound_bites, self._generate_fn, self.schema_gen)
                    rw_qs = [q.get("q", q) if isinstance(q, dict) else q for q in rewritten]
                    try:
                        rw_qs = await dedup_queries(rw_qs, all_keywords, self.embedding_provider)
                    except Exception:
                        pass
                    rw_qs = choose_k(rw_qs, self.config.max_queries_per_step)
                    if rw_qs:
                        more_kn, more_sq = await self._execute_search(
                            [{"q": q} for q in rw_qs], all_urls, all_web_contents
                        )
                        all_keywords.extend(more_sq)
                        all_knowledge.extend(more_kn)
                        ctx.add_knowledge(more_kn)
                except Exception as exc:
                    logger.warning("Query rewrite failed: %s", exc)

                entry = f"At step {step}, searched for: {', '.join(searched)}\n"
                diary_context.append(entry)
                ctx.add_diary_entry(entry)
                allow_search = False
                allow_answer = False

            elif action == "visit" and isinstance(this_step, VisitAction) and url_list:
                resolved: list[str] = []
                for idx in this_step.url_targets:
                    if isinstance(idx, int) and 0 < idx <= len(url_list):
                        u = normalize_url(url_list[idx - 1])
                        if u and u not in visited_urls:
                            resolved.append(u)
                for wu in weighted_urls:
                    if wu.url not in resolved and wu.url not in visited_urls:
                        resolved.append(wu.url)
                    if len(resolved) >= self.config.max_urls_per_step:
                        break
                resolved = list(dict.fromkeys(resolved))[:self.config.max_urls_per_step]

                if resolved:
                    pre_len = len(all_knowledge)
                    ok = await self._process_urls(
                        resolved, all_knowledge, all_urls, visited_urls,
                        bad_urls, all_web_contents, current_question,
                    )
                    # Sync newly added knowledge items to context manager
                    if len(all_knowledge) > pre_len:
                        ctx.add_knowledge(all_knowledge[pre_len:])
                    entry = (
                        f"At step {step}, visited:\n" + "\n".join(resolved)
                        + ("\nFound useful info." if ok else "\nFailed to read.")
                    )
                    diary_context.append(entry)
                    ctx.add_diary_entry(entry)
                allow_read = False

            elif action == "coding" and isinstance(this_step, CodingAction):
                try:
                    sandbox = CodeSandbox(generate_fn=self._generate_fn, schema_gen=self.schema_gen)
                    cr = await sandbox.solve(this_step.coding_issue)
                    ki = KnowledgeItem(
                        question=f"Solution to: {this_step.coding_issue}",
                        answer=str(cr["solution"]["output"]),
                        source_code=cr["solution"]["code"],
                        type="coding",
                        updated=format_date_based_on_type(datetime.now(timezone.utc)),
                    )
                    all_knowledge.append(ki)
                    ctx.add_knowledge([ki])
                    entry = f"At step {step}, solved: {this_step.coding_issue}"
                    diary_context.append(entry)
                    ctx.add_diary_entry(entry)
                except Exception as exc:
                    logger.warning("Coding failed: %s", exc)
                    entry = f"At step {step}, coding failed: {this_step.coding_issue}"
                    diary_context.append(entry)
                    ctx.add_diary_entry(entry)
                allow_coding = False

            await asyncio.sleep(self.config.step_sleep)

        # ===== BEAST MODE =====
        if not getattr(this_step, "is_final", False):
            logger.info("Beast mode! Budget %.1f%%",
                        self.token_tracker.total_tokens / self.config.token_budget * 100)
            this_step = await self._beast_mode(
                question, messages, all_knowledge, diary_context,
                all_questions, all_keywords, weighted_urls, final_answer_pip,
            )

        # ===== POST-PROCESSING =====
        answer_step = this_step if isinstance(this_step, AnswerAction) else AnswerAction(answer="")

        if trivial_question:
            answer_step.md_answer = build_md_from_answer(answer_step)
        else:
            try:
                finalized = await finalize_answer(
                    answer_step.answer, all_knowledge, self._generate_text, self.schema_gen,
                )
                answer_step.answer = repair_markdown_final(finalized)
            except Exception as exc:
                logger.warning("Finalization failed: %s", exc)

            try:
                modified, refs = await build_references(
                    answer_step.answer, all_web_contents, self.embedding_provider,
                    max_ref=self.config.max_references,
                    min_rel_score=self.config.min_relevance_score,
                )
                answer_step.answer = modified
                answer_step.references = [
                    Reference(
                        exact_quote=r.get("exact_quote", ""),
                        url=r.get("url", ""),
                        title=r.get("title", ""),
                        date_time=r.get("date_time"),
                        relevance_score=r.get("relevance_score"),
                    )
                    for r in refs
                ]
            except Exception as exc:
                logger.warning("Reference building failed: %s", exc)

            answer_step.md_answer = build_md_from_answer(answer_step)

        return ResearchResult(
            answer=answer_step.answer,
            md_answer=answer_step.md_answer or answer_step.answer,
            references=answer_step.references,
            visited_urls=[u.url for u in weighted_urls[:self.config.max_returned_urls]] if weighted_urls else [],
            read_urls=[u for u in visited_urls if u not in bad_urls],
            all_urls=[u.url for u in weighted_urls] if weighted_urls else [],
            usage=self.token_tracker.get_breakdown(),
        )
