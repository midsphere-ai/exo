"""All system/user prompt templates for the DeepSearch engine."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .types import KnowledgeItem, BoostedSearchSnippet

from .utils.text_tools import remove_extra_line_breaks
from .utils.url_tools import sort_select_urls


def build_msgs_from_knowledge(knowledge: list["KnowledgeItem"]) -> list[dict]:
    """Build user/assistant message pairs from knowledge items."""
    messages: list[dict] = []
    for k in knowledge:
        messages.append({"role": "user", "content": k.question.strip()})

        parts: list[str] = []

        if k.updated and k.type in ("url", "side-info"):
            parts.append(
                f"\n<answer-datetime>\n{k.updated}\n</answer-datetime>\n"
            )

        if k.references and k.type == "url":
            parts.append(
                f"\n<url>\n{k.references[0]}\n</url>\n"
            )

        parts.append(str(k.answer))

        a_msg = "\n".join(parts).strip()
        messages.append({"role": "assistant", "content": remove_extra_line_breaks(a_msg)})

    return messages


def compose_msgs(
    messages: list[dict],
    knowledge: list["KnowledgeItem"],
    question: str,
    final_answer_pip: list[str] | None = None,
) -> list[dict]:
    """Compose messages with knowledge prepended and answer requirements appended."""
    # knowledge always put to front, followed by real u-a interaction
    msgs = [*build_msgs_from_knowledge(knowledge), *messages]

    pip_section = ""
    if final_answer_pip:
        reviewer_blocks = "\n".join(
            f"\n<reviewer-{idx + 1}>\n{p}\n</reviewer-{idx + 1}>\n"
            for idx, p in enumerate(final_answer_pip)
        )
        pip_section = (
            "\n<answer-requirements>\n"
            "- You provide deep, unexpected insights, identifying hidden patterns and connections, "
            'and creating "aha moments.".\n'
            "- You break conventional thinking, establish unique cross-disciplinary connections, "
            "and bring new perspectives to the user.\n"
            "- Follow reviewer's feedback and improve your answer quality.\n"
            f"{reviewer_blocks}\n"
            "</answer-requirements>"
        )

    user_content = f"{question}\n\n{pip_section}".strip()
    msgs.append({"role": "user", "content": remove_extra_line_breaks(user_content)})
    return msgs


def get_prompt(
    context: list[str] | None = None,
    all_questions: list[str] | None = None,
    all_keywords: list[str] | None = None,
    allow_reflect: bool = True,
    allow_answer: bool = True,
    allow_read: bool = True,
    allow_search: bool = True,
    allow_coding: bool = True,
    knowledge: list["KnowledgeItem"] | None = None,
    all_urls: list | None = None,
    beast_mode: bool = False,
) -> dict:
    """Generate the system prompt and URL list.

    Returns ``{"system": str, "url_list": list[str]}``.
    """
    sections: list[str] = []
    action_sections: list[str] = []

    # ---------- Header section ----------
    current_date = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    sections.append(
        f"Current date: {current_date}\n\n"
        "You are an advanced AI research agent from Jina AI. "
        "You are specialized in multistep reasoning. \n"
        "Using your best knowledge, conversation with the user and lessons learned, "
        "answer the user question with absolute certainty.\n"
    )

    # ---------- Context section ----------
    if context:
        sections.append(
            "\nYou have conducted the following actions:\n"
            "<context>\n"
            f"{chr(10).join(context)}\n\n"
            "</context>\n"
        )

    # ---------- Build action sections ----------

    # --- action-visit (read URLs) ---
    url_list = sort_select_urls(all_urls or [], 20)
    if allow_read and url_list:
        url_list_str = "\n".join(
            f'  - [idx={idx + 1}] [weight={item["score"]:.2f}] '
            f'"{item["url"]}": "{item["merged"][:50]}"'
            for idx, item in enumerate(url_list)
        )
        action_sections.append(
            "\n<action-visit>\n"
            "- Ground the answer with external web content\n"
            "- Read full content from URLs and get the fulltext, knowledge, clues, "
            "hints for better answer the question.  \n"
            "- Must check URLs mentioned in <question> if any    \n"
            "- Choose and visit relevant URLs below for more knowledge. "
            "higher weight suggests more relevant:\n"
            "<url-list>\n"
            f"{url_list_str}\n"
            "</url-list>\n"
            "</action-visit>\n"
        )

    # --- action-search ---
    if allow_search:
        bad_requests_section = ""
        if all_keywords:
            bad_requests_section = (
                "\n- Avoid those unsuccessful search requests and queries:\n"
                "<bad-requests>\n"
                f"{chr(10).join(all_keywords)}\n"
                "</bad-requests>\n"
            )
        action_sections.append(
            "\n<action-search>\n"
            "- Use web search to find relevant information\n"
            "- Build a search request based on the deep intention behind the original "
            "question and the expected answer format\n"
            "- Always prefer a single search request, only add another request if the "
            "original question covers multiple aspects or elements and one query is not "
            "enough, each request focus on one specific aspect of the original question \n"
            f"{bad_requests_section}"
            "</action-search>\n"
        )

    # --- action-answer ---
    if allow_answer:
        action_sections.append(
            "\n<action-answer>\n"
            "- For greetings, casual conversation, general knowledge questions, "
            "answer them directly.\n"
            "- If user ask you to retrieve previous messages or chat history, "
            "remember you do have access to the chat history, answer them directly.\n"
            "- For all other questions, provide a verified answer.\n"
            "- You provide deep, unexpected insights, identifying hidden patterns "
            'and connections, and creating "aha moments.".\n'
            "- You break conventional thinking, establish unique cross-disciplinary "
            "connections, and bring new perspectives to the user.\n"
            "- If uncertain, use <action-reflect>\n"
            "</action-answer>\n"
        )

    # --- beast mode override ---
    if beast_mode:
        action_sections.append(
            "\n<action-answer>\n"
            "\U0001f525 ENGAGE MAXIMUM FORCE! ABSOLUTE PRIORITY OVERRIDE! \U0001f525\n"
            "\n"
            "PRIME DIRECTIVE:\n"
            "- DEMOLISH ALL HESITATION! ANY RESPONSE SURPASSES SILENCE!\n"
            "- PARTIAL STRIKES AUTHORIZED - DEPLOY WITH FULL CONTEXTUAL FIREPOWER\n"
            "- TACTICAL REUSE FROM PREVIOUS CONVERSATION SANCTIONED\n"
            "- WHEN IN DOUBT: UNLEASH CALCULATED STRIKES BASED ON AVAILABLE INTEL!\n"
            "\n"
            "FAILURE IS NOT AN OPTION. EXECUTE WITH EXTREME PREJUDICE! \u26a1\ufe0f\n"
            "</action-answer>\n"
        )

    # --- action-reflect ---
    if allow_reflect:
        action_sections.append(
            "\n<action-reflect>\n"
            "- Think slowly and planning lookahead. Examine <question>, <context>, "
            "previous conversation with users to identify knowledge gaps. \n"
            "- Reflect the gaps and plan a list key clarifying questions that deeply "
            "related to the original question and lead to the answer\n"
            "</action-reflect>\n"
        )

    # --- action-coding ---
    if allow_coding:
        action_sections.append(
            "\n<action-coding>\n"
            "- This JavaScript-based solution helps you handle programming tasks like "
            "counting, filtering, transforming, sorting, regex extraction, and data processing.\n"
            '- Simply describe your problem in the "codingIssue" field. Include actual '
            "values for small inputs or variable names for larger datasets.\n"
            "- No code writing is required \u2013 senior engineers will handle the implementation.\n"
            "</action-coding>"
        )

    # ---------- Actions wrapper ----------
    sections.append(
        "\nBased on the current context, you must choose one of the following actions:\n"
        "<actions>\n"
        f"{(chr(10) + chr(10)).join(action_sections)}\n"
        "</actions>\n"
    )

    # ---------- Footer ----------
    sections.append(
        "Think step by step, choose the action, then respond by matching the schema of that action."
    )

    return {
        "system": remove_extra_line_breaks("\n\n".join(sections)),
        "url_list": [u["url"] for u in url_list],
    }
