"""Pydantic data models for the Perplexica search engine."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class ResearchMode(str, Enum):
    """Research quality modes matching Perplexica."""
    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"


class Classification(BaseModel):
    """Perplexica's classifier output labels."""
    skip_search: bool = Field(default=False, alias="skipSearch")
    personal_search: bool = Field(default=False, alias="personalSearch")
    academic_search: bool = Field(default=False, alias="academicSearch")
    discussion_search: bool = Field(default=False, alias="discussionSearch")
    show_weather_widget: bool = Field(default=False, alias="showWeatherWidget")
    show_stock_widget: bool = Field(default=False, alias="showStockWidget")
    show_calculation_widget: bool = Field(default=False, alias="showCalculationWidget")

    model_config = {"populate_by_name": True}


class ClassifierOutput(BaseModel):
    """Full classifier response."""
    classification: Classification
    standalone_follow_up: str = Field(default="", alias="standaloneFollowUp")

    model_config = {"populate_by_name": True}


class SearchResult(BaseModel):
    """A single search result from SearXNG."""
    title: str = ""
    url: str = ""
    content: str = ""


class Source(BaseModel):
    """A cited source in the final response."""
    title: str
    url: str
    content: str = ""


class QueryPlan(BaseModel):
    """Structured output for adaptive query generation."""
    queries: list[str] = Field(default_factory=list)
    sufficient: bool = Field(default=False)


class SuggestionOutput(BaseModel):
    """Structured suggestion output."""
    suggestions: list[str] = Field(default_factory=list)


class PipelineEvent(BaseModel):
    """Pipeline stage transition event."""
    type: str = "pipeline"
    stage: str  # "classifier", "researcher", "writer", "suggestions"
    status: str  # "started", "completed"
    message: str = ""


class PerplexicaResponse(BaseModel):
    """Final structured search response."""
    answer: str
    sources: list[Source] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    query: str = ""
    mode: str = ""
