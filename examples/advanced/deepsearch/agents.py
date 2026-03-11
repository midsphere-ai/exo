"""Agent assembly — builds the DeepAgent Swarm from config.

Creates a Swarm(mode="team") with:
- Lead agent (planner): delegates tasks to workers
- Researcher worker: ALL tools (deep_research, web_search, read_webpage, bash,
  python_interpreter, read, edit, done, mdify, todo, browser, reformulator)
- Analyzer worker: ALL tools (analyze_file, read_webpage, bash, python_interpreter,
  read, edit, done, mdify, todo)

The Swarm auto-generates delegate_to_researcher and delegate_to_analyzer
tools for the lead agent via _DelegateTool.

1:1 port of SkyworkAI's agent assembly with full tool set.
"""

from __future__ import annotations

from orbiter.agent import Agent
from orbiter.swarm import Swarm

from .config import DeepAgentConfig
from .prompts import ANALYZER_PROMPT, LEAD_PROMPT, RESEARCHER_PROMPT

# Existing tools
from .tools.content_reader import ContentReaderTool
from .tools.deep_researcher import DeepResearcherTool
from .tools.file_analyzer import FileAnalyzerTool
from .tools.web_search import WebSearchTool

# New tools (1:1 ports from SkyworkAI)
from .tools.bash import BashTool
from .tools.python_interpreter import PythonInterpreterTool
from .tools.file_reader import FileReaderTool
from .tools.file_editor import FileEditorTool
from .tools.done import DoneTool
from .tools.mdify import MdifyTool
from .tools.todo import TodoTool
from .tools.browser import BrowserTool
from .tools.reformulator import ReformulatorTool
from .tools.tool_generator import ToolGeneratorTool
from .tools.skill_generator import SkillGeneratorTool


def build_deep_agent(config: DeepAgentConfig | None = None) -> Swarm:
    """Build the DeepAgent research swarm.

    Creates a team-mode Swarm that mirrors SkyworkAI's architecture:
    PlanningAgent + AgentBus + ToolCallingAgent -> Swarm(team) with workers.

    Each worker gets ALL tools (1:1 from SkyworkAI's tool assignment).

    Args:
        config: Optional configuration. Defaults to DeepAgentConfig.from_env().

    Returns:
        Configured Swarm instance ready for run() or stream().
    """
    cfg = config or DeepAgentConfig.from_env()

    # --- Build tools ---

    # Existing tools
    web_search = WebSearchTool(
        provider=cfg.search_provider,
        num_results=cfg.search_num_results,
        max_length=cfg.content_max_length,
        model=cfg.tool_model,
        brave_api_key=cfg.brave_api_key,
        serper_api_key=cfg.serper_api_key,
        jina_api_key=cfg.jina_api_key,
    )

    content_reader = ContentReaderTool(
        mode=cfg.content_reader,
        max_length=cfg.content_max_length,
        jina_api_key=cfg.jina_api_key,
    )

    deep_researcher = DeepResearcherTool(
        tool_model=cfg.tool_model,
        web_search=web_search,
        max_rounds=cfg.search_max_rounds,
        num_results=cfg.search_num_results,
        use_llm_search=cfg.use_llm_search,
        search_llm_models=cfg.search_llm_models,
        base_dir=f"{cfg.output_dir}/deep_researcher",
    )

    file_analyzer = FileAnalyzerTool(
        tool_model=cfg.tool_model,
        content_max_length=cfg.content_max_length,
    )

    # New tools (1:1 from SkyworkAI)
    bash = BashTool(timeout=cfg.bash_timeout)
    python_interpreter = PythonInterpreterTool(
        authorized_imports=cfg.python_authorized_imports,
    )
    file_reader = FileReaderTool()
    file_editor = FileEditorTool()
    done = DoneTool()
    mdify = MdifyTool(
        base_dir=f"{cfg.output_dir}/mdify",
        timeout=cfg.mdify_timeout,
    )
    todo = TodoTool(base_dir=f"{cfg.output_dir}/todo")
    browser = BrowserTool(
        model_name=cfg.browser_model,
        base_dir=f"{cfg.output_dir}/browser",
        max_steps=cfg.browser_max_steps,
    )
    reformulator = ReformulatorTool(model=cfg.tool_model)
    tool_generator = ToolGeneratorTool(
        model=cfg.tool_model,
        base_dir=f"{cfg.output_dir}/tool_generator",
    )
    skill_generator = SkillGeneratorTool(
        model=cfg.tool_model,
        base_dir=f"{cfg.output_dir}/skill_generator",
    )

    # --- Build worker agents with ALL tools (1:1 from SkyworkAI) ---

    researcher = Agent(
        name="researcher",
        model=cfg.researcher_model,
        instructions=RESEARCHER_PROMPT,
        tools=[
            deep_researcher, web_search, content_reader,  # existing
            bash, python_interpreter, file_reader, file_editor,  # new default tools
            done, mdify, todo, browser,  # new workflow tools
            reformulator, tool_generator, skill_generator,  # new other tools
        ],
        max_steps=cfg.researcher_max_steps,
    )

    analyzer = Agent(
        name="analyzer",
        model=cfg.researcher_model,
        instructions=ANALYZER_PROMPT,
        tools=[
            file_analyzer, content_reader,  # existing
            bash, python_interpreter, file_reader, file_editor,  # new default tools
            done, mdify, todo,  # new workflow tools
        ],
        max_steps=cfg.researcher_max_steps,
    )

    # --- Build lead agent (Swarm auto-adds delegate_to_researcher, delegate_to_analyzer) ---

    lead = Agent(
        name="planner",
        model=cfg.lead_model,
        instructions=LEAD_PROMPT,
        max_steps=cfg.lead_max_steps,
    )

    return Swarm(agents=[lead, researcher, analyzer], mode="team")
