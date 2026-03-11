"""DeepAgent configuration — mirrors SkyworkAI DeepResearchAgent settings."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field


class DeepAgentConfig(BaseModel):
    """Configuration for the DeepAgent research system.

    Mirrors the configuration parameters from SkyworkAI's DeepResearchAgent,
    adapted for the Orbiter framework.
    """

    # Model configuration
    lead_model: str = Field(
        default="openai:gpt-4o",
        description="Model for the lead planning agent.",
    )
    researcher_model: str = Field(
        default="openai:gpt-4o-mini",
        description="Model for the researcher worker agent.",
    )
    tool_model: str = Field(
        default="openai:gpt-4o-mini",
        description="Model for internal tool LLM calls (query gen, evaluation, summarization).",
    )

    # Search configuration
    search_provider: str = Field(
        default="duckduckgo",
        description="Search provider: duckduckgo | brave | serper | jina.",
    )
    search_num_results: int = Field(
        default=5,
        description="Number of search results per query.",
    )
    search_max_rounds: int = Field(
        default=3,
        description="Maximum search rounds for deep research.",
    )

    # Content reader configuration
    content_reader: str = Field(
        default="httpx",
        description="Content reader backend: httpx | jina.",
    )
    content_max_length: int = Field(
        default=4096,
        description="Maximum content length for fetched pages.",
    )

    # LLM search configuration (parallel LLM-based web search)
    use_llm_search: bool = Field(
        default=False,
        description="Whether to use LLM models for parallel web search.",
    )
    search_llm_models: list[str] = Field(
        default_factory=list,
        description="LLM models for parallel search (e.g. ['openai:o3-mini']).",
    )

    # Agent configuration
    lead_max_steps: int = Field(
        default=15,
        description="Maximum tool-calling steps for the lead agent.",
    )
    researcher_max_steps: int = Field(
        default=10,
        description="Maximum tool-calling steps for each worker agent.",
    )

    # API keys
    brave_api_key: str | None = Field(default=None, description="Brave Search API key.")
    serper_api_key: str | None = Field(default=None, description="Serper API key.")
    jina_api_key: str | None = Field(default=None, description="Jina API key.")

    # Output
    output_dir: str = Field(
        default="deepagent_output",
        description="Directory for saving research reports.",
    )

    # === New fields from SkyworkAI full port ===

    # Working directory for agent operations
    workdir: str = Field(
        default=".",
        description="Working directory for agent file operations.",
    )

    # Memory system
    use_memory: bool = Field(
        default=False,
        description="Enable/disable memory system.",
    )

    # Todo system
    use_todo: bool = Field(
        default=True,
        description="Enable/disable todo tracking.",
    )

    # Review steps
    review_steps: int = Field(
        default=3,
        description="Number of recent steps to show in full.",
    )

    # Browser tool
    browser_model: str = Field(
        default="openrouter/gpt-4.1",
        description="Model for browser tool.",
    )
    browser_max_steps: int = Field(
        default=50,
        description="Max steps for browser automation.",
    )

    # Mdify
    mdify_timeout: int = Field(
        default=60,
        description="Timeout for file conversion (seconds).",
    )

    # Bash
    bash_timeout: int = Field(
        default=30,
        description="Timeout for bash commands (seconds).",
    )

    # Python interpreter
    python_authorized_imports: list[str] = Field(
        default_factory=lambda: [
            "subprocess", "pandas", "numpy", "matplotlib", "seaborn",
            "scipy", "sklearn", "json", "csv", "os",
        ],
        description="Allowed Python imports for the interpreter.",
    )

    @classmethod
    def from_env(cls, **overrides: Any) -> DeepAgentConfig:
        """Build config from environment variables with optional overrides."""
        env_map: dict[str, str] = {
            "lead_model": "DEEPAGENT_LEAD_MODEL",
            "researcher_model": "DEEPAGENT_RESEARCHER_MODEL",
            "tool_model": "DEEPAGENT_TOOL_MODEL",
            "search_provider": "DEEPAGENT_SEARCH_PROVIDER",
            "search_num_results": "DEEPAGENT_SEARCH_NUM_RESULTS",
            "search_max_rounds": "DEEPAGENT_SEARCH_MAX_ROUNDS",
            "content_reader": "DEEPAGENT_CONTENT_READER",
            "content_max_length": "DEEPAGENT_CONTENT_MAX_LENGTH",
            "use_llm_search": "DEEPAGENT_USE_LLM_SEARCH",
            "lead_max_steps": "DEEPAGENT_LEAD_MAX_STEPS",
            "researcher_max_steps": "DEEPAGENT_RESEARCHER_MAX_STEPS",
            "brave_api_key": "BRAVE_API_KEY",
            "serper_api_key": "SERPER_API_KEY",
            "jina_api_key": "JINA_API_KEY",
            "output_dir": "DEEPAGENT_OUTPUT_DIR",
            "workdir": "DEEPAGENT_WORKDIR",
            "use_memory": "DEEPAGENT_USE_MEMORY",
            "use_todo": "DEEPAGENT_USE_TODO",
            "review_steps": "DEEPAGENT_REVIEW_STEPS",
            "browser_model": "DEEPAGENT_BROWSER_MODEL",
            "browser_max_steps": "DEEPAGENT_BROWSER_MAX_STEPS",
            "mdify_timeout": "DEEPAGENT_MDIFY_TIMEOUT",
            "bash_timeout": "DEEPAGENT_BASH_TIMEOUT",
        }

        kwargs: dict[str, Any] = {}
        for field_name, env_var in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                # Type coercion for non-string fields
                field_info = cls.model_fields[field_name]
                annotation = field_info.annotation
                if annotation is int or (hasattr(annotation, "__origin__") and annotation.__origin__ is int):
                    val = int(val)
                elif annotation is bool:
                    val = val.lower() in ("true", "1", "yes")
                kwargs[field_name] = val

        # Parse comma-separated LLM search models
        llm_models_env = os.environ.get("DEEPAGENT_SEARCH_LLM_MODELS", "")
        if llm_models_env:
            kwargs["search_llm_models"] = [m.strip() for m in llm_models_env.split(",") if m.strip()]

        # Parse comma-separated Python authorized imports
        py_imports_env = os.environ.get("DEEPAGENT_PYTHON_AUTHORIZED_IMPORTS", "")
        if py_imports_env:
            kwargs["python_authorized_imports"] = [m.strip() for m in py_imports_env.split(",") if m.strip()]

        kwargs.update(overrides)
        return cls(**kwargs)
