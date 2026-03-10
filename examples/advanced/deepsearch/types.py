"""Data models for the DeepSearch research engine."""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class SERPQuery(BaseModel):
    q: str
    location: str | None = None
    tbs: str | None = None


class SearchSnippet(BaseModel):
    title: str
    url: str
    description: str
    weight: float = 1.0
    date: str | None = None


class BoostedSearchSnippet(SearchSnippet):
    freq_boost: float = 0.0
    hostname_boost: float = 0.0
    path_boost: float = 0.0
    jina_rerank_boost: float = 0.0
    final_score: float = 0.0


class Reference(BaseModel):
    exact_quote: str
    url: str
    title: str
    date_time: str | None = None
    relevance_score: float | None = None
    answer_chunk: str | None = None
    answer_chunk_position: list[int] | None = None


class KnowledgeItem(BaseModel):
    question: str
    answer: str
    references: list[str] | list[Reference] | None = None
    type: Literal["qa", "side-info", "chat-history", "url", "coding"]
    updated: str | None = None
    source_code: str | None = None


class WebContent(BaseModel):
    full: str | None = None
    chunks: list[str] = Field(default_factory=list)
    chunk_positions: list[list[int]] = Field(default_factory=list)
    title: str = ""


class SearchAction(BaseModel):
    action: Literal["search"] = "search"
    think: str = ""
    search_requests: list[str] = Field(default_factory=list)


class VisitAction(BaseModel):
    action: Literal["visit"] = "visit"
    think: str = ""
    url_targets: list[int] = Field(default_factory=list)


class ReflectAction(BaseModel):
    action: Literal["reflect"] = "reflect"
    think: str = ""
    questions_to_answer: list[str] = Field(default_factory=list)


class AnswerAction(BaseModel):
    action: Literal["answer"] = "answer"
    think: str = ""
    answer: str = ""
    references: list[Reference] = Field(default_factory=list)
    is_final: bool = False
    md_answer: str | None = None
    is_aggregated: bool = False


class CodingAction(BaseModel):
    action: Literal["coding"] = "coding"
    think: str = ""
    coding_issue: str = ""


StepAction = SearchAction | VisitAction | ReflectAction | AnswerAction | CodingAction

EvaluationType = Literal["definitive", "freshness", "plurality", "attribution", "completeness", "strict"]


class RepeatEvaluationType(BaseModel):
    type: EvaluationType
    num_evals_required: int


class EvaluationResponse(BaseModel):
    passed: bool = Field(alias="pass", default=False)
    think: str = ""
    type: EvaluationType | None = None
    freshness_analysis: dict | None = None
    plurality_analysis: dict | None = None
    completeness_analysis: dict | None = None
    improvement_plan: str | None = None
    exact_quote: str | None = None

    model_config = {"populate_by_name": True}


class CodeGenResponse(BaseModel):
    think: str = ""
    code: str = ""


class ErrorAnalysisResponse(BaseModel):
    recap: str = ""
    blame: str = ""
    improvement: str = ""


class ReadResponse(BaseModel):
    code: int = 200
    status: int = 200
    data: dict | None = None
    name: str | None = None
    message: str | None = None


class ResearchResult(BaseModel):
    answer: str = ""
    md_answer: str = ""
    references: list[Reference] = Field(default_factory=list)
    visited_urls: list[str] = Field(default_factory=list)
    read_urls: list[str] = Field(default_factory=list)
    all_urls: list[str] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)


# OpenAI API compat types
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "deepsearch"
    messages: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False
    budget_tokens: int | None = None
    max_attempts: int | None = None
    no_direct_answer: bool = False
    max_returned_urls: int = 100
    boost_hostnames: list[str] = Field(default_factory=list)
    bad_hostnames: list[str] = Field(default_factory=list)
    only_hostnames: list[str] = Field(default_factory=list)
    max_annotations: int = 10
    min_annotation_relevance: float = 0.80
    with_images: bool = False
    language_code: str | None = None
    search_language_code: str | None = None
    search_provider: str | None = None
    team_size: int = 1
