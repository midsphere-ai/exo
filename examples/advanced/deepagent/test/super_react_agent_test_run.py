#!/usr/bin/env python
# coding: utf-8
"""
Super ReAct Agent Example
Demonstrates how to use the SuperReActAgent with custom context management
"""

import os
import sys
import asyncio
import atexit
import socket
import subprocess
import time
from datetime import datetime
import json
from dotenv import load_dotenv
from pathlib import Path
from urllib.parse import urlsplit

load_dotenv()

# Ensure both repo root and `examples/` are importable
CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
# EXAMPLES_DIR = os.path.abspath(os.path.join(REPO_ROOT, "examples"))

for path in [REPO_ROOT]: #b1 
    if path not in sys.path:
        sys.path.insert(0, path)

from agent.super_react_agent import SuperReActAgent
from agent.super_config import SuperAgentFactory
from agent.prompt_templates import get_main_agent_system_prompt, get_browsing_agent_system_prompt, get_coding_agent_system_prompt
from openjiuwen.core.component.common.configs.model_config import ModelConfig
from openjiuwen.core.utils.llm.base import BaseModelInfo
from openjiuwen.core.utils.tool.function.function import LocalFunction
from openjiuwen.core.utils.tool.param import Param

from openjiuwen.core.utils.tool.mcp.base import ToolServerConfig
from openjiuwen.core.runner.runner import Runner, resource_mgr
from mcp import StdioServerParameters

# Environment configuration
API_BASE = os.getenv("API_BASE", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("API_KEY", "your_api_key_here")
MODEL_NAME = os.getenv("MODEL_NAME", "anthropic/claude-3.7-sonnet")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openrouter")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
REASONING_MODEL_NAME = os.getenv("REASONING_MODEL_NAME", "o3")
os.environ.setdefault("LLM_SSL_VERIFY", "false")

# Prefer the tooling venv for MCP servers so required deps are available.
TOOLING_VENV_DIR = os.getenv(
    "MCP_TOOL_VENV_DIR",
    os.path.join(REPO_ROOT, "tool", ".venv-tool"),
)

if os.name == "nt":
    tooling_python = os.path.join(TOOLING_VENV_DIR, "Scripts", "python.exe")
    #TODO fix tooling_bin unused for windows
    tooling_bin = os.path.join(TOOLING_VENV_DIR, "Scripts") # not sure 
else:
    tooling_python = os.path.join(TOOLING_VENV_DIR, "bin", "python")
    tooling_bin = os.path.join(TOOLING_VENV_DIR, "bin")

MCP_TOOL_PYTHON = tooling_python if os.path.exists(tooling_python) else sys.executable
MCP_TOOL_BIN = tooling_bin if os.path.isdir(tooling_bin) else None
MCP_TOOL_CWD = REPO_ROOT

pythonpath_entries = [REPO_ROOT] #b2 EXAMPLES_DIR
existing_pythonpath = os.getenv("PYTHONPATH")
if existing_pythonpath:
    pythonpath_entries.append(existing_pythonpath)
MCP_TOOL_ENV_BASE = {"PYTHONPATH": os.pathsep.join(pythonpath_entries)}


def build_tool_env(extra: dict | None = None) -> dict[str, str]:
    env = dict(MCP_TOOL_ENV_BASE)
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            env[key] = str(value)
    return env

# To decide whether to run auto_browser on SSE and STDIO (SSE works - STDIO doesn't work at the moment)
AUTO_BROWSER_TRANSPORT = (os.getenv("AUTO_BROWSER_TRANSPORT") or "sse").strip().lower()
AUTO_BROWSER_SSE_URL = os.getenv("AUTO_BROWSER_SSE_URL", "http://127.0.0.1:8930/sse")
AUTO_BROWSER_START_SSE = (os.getenv("AUTO_BROWSER_START_SSE") or "true").strip().lower()

_AUTO_BROWSER_SSE_PROCESS: subprocess.Popen | None = None

AUTO_BROWSER_ENV = build_tool_env({
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    "ANTHROPIC_BASE_URL": os.getenv("ANTHROPIC_BASE_URL"),
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
    "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
    "OPENROUTER_BASE_URL": os.getenv("OPENROUTER_BASE_URL"),
    "ENABLE_CLAUDE_VISION": "true",
    "ENABLE_OPENAI_VISION": "true",
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    "MCP_STDIO_SAFE": os.getenv("MCP_STDIO_SAFE"),
    "BROWSER_USE_STDIO_SAFE": os.getenv("BROWSER_USE_STDIO_SAFE"),
})


def _parse_sse_host_port(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8930
    return host, port


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _stop_autobrowser_sse_server() -> None:
    global _AUTO_BROWSER_SSE_PROCESS
    if _AUTO_BROWSER_SSE_PROCESS and _AUTO_BROWSER_SSE_PROCESS.poll() is None:
        try:
            _AUTO_BROWSER_SSE_PROCESS.terminate()
            _AUTO_BROWSER_SSE_PROCESS.wait(timeout=5)
        except Exception:
            _AUTO_BROWSER_SSE_PROCESS.kill()
    _AUTO_BROWSER_SSE_PROCESS = None


# Function to automatically start SSE server for autobrowser
def ensure_autobrowser_sse_server() -> None:
    global _AUTO_BROWSER_SSE_PROCESS
    if AUTO_BROWSER_TRANSPORT != "sse":
        return
    if AUTO_BROWSER_START_SSE in {"0", "false", "no", "off"}:
        return

    host, port = _parse_sse_host_port(AUTO_BROWSER_SSE_URL)
    if _is_port_open(host, port):
        return

    env = dict(os.environ)
    env.update(AUTO_BROWSER_ENV)
    args = [
        "-u",
        "-m",
        "tool.mcp_servers.browser_use_mcp_server",
        "--transport",
        "sse",
        "--host",
        host,
        "--port",
        str(port),
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    _AUTO_BROWSER_SSE_PROCESS = subprocess.Popen(
        [MCP_TOOL_PYTHON, *args],
        cwd=MCP_TOOL_CWD,
        env=env,
        creationflags=creationflags,
    )
    atexit.register(_stop_autobrowser_sse_server)

    for _ in range(50):
        if _is_port_open(host, port):
            return
        time.sleep(0.2)

    print(
        f"[WARN] Failed to detect browser-use SSE server on {host}:{port}. "
        "Check logs and ensure dependencies are installed."
    )

if AUTO_BROWSER_TRANSPORT == "sse":
    autobrowser_client_type = "sse"
    autobrowser_params = AUTO_BROWSER_SSE_URL
else:
    autobrowser_client_type = "stdio"
    autobrowser_params = StdioServerParameters(
        command=MCP_TOOL_PYTHON,
        args=["-u", "-m", "tool.mcp_servers.browser_use_mcp_server", "--transport", "stdio"],
        env=build_tool_env({
            "ANTHROPIC_API_KEY":  os.getenv("ANTHROPIC_API_KEY"),
            "ANTHROPIC_BASE_URL": os.getenv("ANTHROPIC_BASE_URL"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
            "ENABLE_CLAUDE_VISION": "true",
            "ENABLE_OPENAI_VISION": "true",
            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
            "MCP_STDIO_SAFE": os.getenv("MCP_STDIO_SAFE"),
            "BROWSER_USE_STDIO_SAFE": os.getenv("BROWSER_USE_STDIO_SAFE"),
        }),
        cwd=MCP_TOOL_CWD,
    )

# ===== GAIA dataset file path =====
GAIA_DATASET_FILE_PATH = "data/test.jsonl"
with open(GAIA_DATASET_FILE_PATH, 'r', encoding='utf-8') as f:
    GAIA_DATASET = [json.loads(line.strip()) for line in f if line.strip()]
# ===== MCP 工具组与实际 MCP server 的映射 =====

# 每个“tool-*”代表一组 MCP 工具（某个 MCP server 上的所有 tools）
MCP_TOOL_GROUPS = {
    # 主 / 子 agent 共用
    "tool-autobrowser": {
        "server_name": "browser-use-server",
        "client_type": autobrowser_client_type,
        "params": autobrowser_params,
    },
    "tool-transcribe": {
        "server_name": "audio-mcp-server",
        "client_type": "stdio",
        "params": StdioServerParameters(
                    command=MCP_TOOL_PYTHON,
                    args=["-u", "-m", "tool.mcp_servers.audio_mcp_server", "--transport", "stdio"],
                    env=build_tool_env({
                        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                    }),
                    cwd=MCP_TOOL_CWD,
                ),
    },

    "tool-reasoning": {
        "server_name": "reasoning-mcp-server",
        "client_type": "stdio",
        "params": StdioServerParameters(
                    command=MCP_TOOL_PYTHON,
                    args=["-u", "-m", "tool.mcp_servers.reasoning_mcp_server", "--transport", "stdio"],
                    env=build_tool_env({
                        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
                        "ANTHROPIC_BASE_URL": os.getenv("ANTHROPIC_BASE_URL"),
                        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
                        "OPENROUTER_BASE_URL": os.getenv("OPENROUTER_BASE_URL"),
                    }),
                    cwd=MCP_TOOL_CWD,
                ),
    },
    "tool-reading": {
        "server_name": "reading-mcp-server",
        "client_type": "stdio",
        "params": StdioServerParameters(
                    command=MCP_TOOL_PYTHON,
                    args=["-u", "-m", "tool.mcp_servers.reading_mcp_server", "--transport", "stdio"],
                    env=build_tool_env({
                        "PATH": os.pathsep.join([MCP_TOOL_BIN, os.getenv("PATH", "")])}
                        ),
                    cwd=MCP_TOOL_CWD,
                ),
    },
    "tool-searching": {
        "server_name": "searching-mcp-server",
        "client_type": "stdio",
        "params": StdioServerParameters(
                        command=MCP_TOOL_PYTHON,
                        args=["-u", "-m", "tool.mcp_servers.searching_mcp_server", "--transport", "stdio"],
                        env=build_tool_env({
                            "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
                            "JINA_API_KEY": os.getenv("JINA_API_KEY"),
                            "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
                        }),
                        cwd=MCP_TOOL_CWD,
                    ),
    },
    "tool-vqa": {
        "server_name": "vision-mcp-server",
        "client_type": "stdio",
        "params": StdioServerParameters(
                    command=MCP_TOOL_PYTHON,
                    args=["-u", "-m", "tool.mcp_servers.vision_mcp_server", "--transport", "stdio"],
                    env=build_tool_env({
                        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
                        "ANTHROPIC_BASE_URL": os.getenv("ANTHROPIC_BASE_URL"),
                        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
                        "ENABLE_CLAUDE_VISION": "true",
                        "ENABLE_OPENAI_VISION": "true",
                        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
                        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
                        "OPENROUTER_BASE_URL": os.getenv("OPENROUTER_BASE_URL"),
                    }),
                    cwd=MCP_TOOL_CWD,
                ),
    },
    "tool-code": {
        "server_name": "e2b-python-interpreter",
        "client_type": "stdio",
        "params": StdioServerParameters(
                    command=MCP_TOOL_PYTHON,
                    args=["-u", "-m", "tool.mcp_servers.python_server", "--transport", "stdio"],
                    env=build_tool_env({
                        "E2B_API_KEY": os.getenv("E2B_API_KEY"),
                        "E2B_TEMPLATE_ID": os.getenv("E2B_TEMPLATE_ID"),
                    }),
                    cwd=MCP_TOOL_CWD,
                ),
    },
}


async def build_mcp_tool_groups(agent: SuperReActAgent) -> dict[str, list[LocalFunction]]:
    """
    使用某个 SuperReActAgent 实例，把上面 MCP_TOOL_GROUPS 里的所有 server都注册成 LocalFunction 工具，并按 tool-group 名字归类返回。

    返回：
        {
          "tool-autobrowser": [LocalFunction(...), ...],  # browser-use-server.* MCP 工具
          "tool-transcribe": [...],
          ...
        }
    """
    tool_groups: dict[str, list[LocalFunction]] = {}

    for group_name, cfg in MCP_TOOL_GROUPS.items():
        tools = await agent.create_mcp_tools(
            server_name=cfg["server_name"],
            client_type=cfg["client_type"],
            params=cfg["params"],
        )
        tool_groups[group_name] = tools
    return tool_groups


def create_model_config():
    """Create separate coding model configuration for SuperReActAgent"""
    model_info = BaseModelInfo(
        api_key=API_KEY,
        api_base=API_BASE,
        model=MODEL_NAME,
        timeout=60  # Increased timeout for API calls
    )

    return ModelConfig(
        model_provider=MODEL_PROVIDER,
        model_info=model_info
    )

# ========= MCP 工具封装成 LocalFunction 的通用方法 =========

def _make_mcp_call_coroutine(server_name: str, tool_name: str):
    """
    为某个 MCP 工具生成一个 coroutine 函数：
    - 入参是工具的参数（**kwargs）
    - 内部通过 Runner.run_tool 调用真正的 MCP 工具
    """
    async def _wrapper(**kwargs):
        tool_id = f"{server_name}.{tool_name}"  # 例如：browser-use-server.browser_navigate
        result = await Runner.run_tool(tool_id, kwargs)

        # Test 里约定：如果返回 dict 且有 "result" 字段，就用它
        if isinstance(result, dict) and "result" in result:
            return result["result"]
        return result

    return _wrapper


async def _register_mcp_server_as_local_tools(
    server_name: str,
    client_type: str,
    params,
):
    """
    注册一个 MCP server（SSE / stdio / playwright），并把该 server 上所有 tools
    映射成 LocalFunction，返回 List[LocalFunction]，可以直接传给 SuperReActAgent.

    :param server_name: MCP server 名字，如 "browser-use-server"
    :param client_type: "sse" / "stdio" / "playwright"
    :param params: ToolServerConfig.params，对应：
                   - sse: "http://127.0.0.1:8930/sse"
                   - stdio: StdioServerParameters(...)
                   - playwright: url 或 StdioServerParameters
    """
    tool_mgr = resource_mgr.tool()

    server_path = None
    params_dict = {}

    if client_type == "stdio":
        if isinstance(params, StdioServerParameters):
            params_dict = {
                "command": params.command,
                "args": list(params.args) if params.args is not None else None,
                "env": params.env,
                "cwd": params.cwd,
                "encoding_error_handler": getattr(params, "encoding_error_handler", None),
            }
        elif isinstance(params, dict):
            params_dict = params
        elif isinstance(params, str):
            server_path = params

        if not server_path:
            server_path = params_dict.get("command") or "stdio"
    else:
        if isinstance(params, dict):
            if "server_path" in params:
                server_path = params["server_path"]
                params_dict = {k: v for k, v in params.items() if k != "server_path"}
            elif "url" in params:
                server_path = params["url"]
                params_dict = {k: v for k, v in params.items() if k != "url"}
            else:
                params_dict = params
        else:
            server_path = params

    if server_path is None:
        raise ValueError(f"MCP server_path is required for client_type '{client_type}'")

    # 1. 注册 MCP server
    server_cfg = ToolServerConfig(
        server_name=server_name,
        server_path=server_path,
        params=params_dict,
        client_type=client_type,
    )
    ok_list = await tool_mgr.add_tool_servers([server_cfg])
    if not ok_list or not ok_list[0]:
        raise RuntimeError(f"Failed to add MCP server: {server_name}")

    # 2. 用 Runner.list_tools 拿到工具列表（McpToolInfo）
    tool_infos = await Runner.list_tools(server_name)

    local_tools = []

    for info in tool_infos:
        schema = getattr(info, "schema", {}) or {}
        properties = schema.get("properties", {}) or {}
        required = set(schema.get("required", []) or [])

        # 3. 把 JSON-Schema 转成 Param 列表
        params_def = []
        for pname, pinfo in properties.items():
            params_def.append(
                Param(
                    name=pname,
                    description=pinfo.get("description", ""),
                    param_type=pinfo.get("type", "string"),
                    required=pname in required,
                )
            )

        # 4. 为每个 tool 生成自己独立的 coroutine wrapper
        async_func = _make_mcp_call_coroutine(server_name, info.name)

        #  LocalFunction 支持 func 是 async 函数?
        mcp_local_tool = LocalFunction(
            name=info.name,
            description=getattr(info, "description", "") or f"MCP tool {info.name} from {server_name}",
            params=params_def,
            func=async_func,   # 传入的是 async 函数
        )

        local_tools.append(mcp_local_tool)

    return local_tools


# 便捷包装
async def create_sse_mcp_tools(server_name: str, sse_url: str):
    """
    将一个 SSE MCP server 上的所有工具注册为 LocalFunction
    """
    return await _register_mcp_server_as_local_tools(
        server_name=server_name,
        client_type="sse",
        params=sse_url,
    )

async def create_stdio_mcp_tools(server_name: str, command: str, args: list[str]):
    """
    将一个 stdio MCP server 上的所有工具注册为 LocalFunction
    """
    params = StdioServerParameters(command=command, args=args)
    return await _register_mcp_server_as_local_tools(
        server_name=server_name,
        client_type="stdio",
        params=params,
    )

async def example_mcp_main_and_sub_agents(queries: list | None = None):
    """
    Example: 主 Agent + browsing 子 Agent，使用不同 MCP 工具集
    当 queries 传入多个问题时，会在同一次 Runner 生命周期中依次运行，
    每次运行前都会清空上下文，避免串话。
    main_agent:
      tools: [tool-reasoning]
    sub_agents:
      agent-browsing:
        tools: [tool-searching, tool-reading, tool-vqa, tool-autobrowser, tool-transcribe]
      agent-coding:
        tools: [tool-code, tool-vqa, tool-reading]
    """
    assert queries is not None, "query is required"
    assert len(queries) > 0, "query is required"
    assert isinstance(queries, list), "queries must be a list"


    print("\n" + "=" * 60)
    print("Example: Super ReAct Agent Test Run")
    print("=" * 60)

    if AUTO_BROWSER_TRANSPORT == "sse":
        ensure_autobrowser_sse_server()

    # 启动 Runner
    await Runner.start()

    # 构造 main_agent 配置
    main_agent_config = SuperAgentFactory.create_main_agent_config(
        agent_id="super_react_main_mcp",
        agent_type = "main",
        agent_version="1.0",
        description="Main MCP agent with multiple tool groups",
        model=create_model_config(),
        prompt_template=[
            {
                "role": "system",
                "content": get_main_agent_system_prompt(datetime.now()),
            }
        ],
        max_iteration=20,              
        max_tool_calls_per_turn=1,  # ISSUE: 
        enable_question_hints=True,
        open_api_key = OPENAI_API_KEY,
        enable_extract_final_answer=True,
        reasoning_model=REASONING_MODEL_NAME,
        enable_todo_plan=False,
    )

    # 构造子 agent 配置
    browsing_agent_config = SuperAgentFactory.create_main_agent_config(
        agent_id="agent-browsing",
        agent_type = "sub",
        agent_version="1.0",
        description="Browsing specialist sub-agent using searching + browser + vqa + reading + code",
        model=create_model_config(),
        prompt_template=[
            {
                "role": "system",
                "content": get_browsing_agent_system_prompt(datetime.now()),
            }
        ],
        max_iteration=20,             
        max_tool_calls_per_turn=1,
        enable_question_hints=True,
        open_api_key = OPENAI_API_KEY,
        enable_extract_final_answer=True,
        enable_todo_plan=False,
    )

    coding_agent_config = SuperAgentFactory.create_main_agent_config(
        agent_id="agent-coding",
        agent_type = "sub",
        agent_version="1.0",
        description="Coding specialist sub-agent using code + vqa + reading",
        model=create_model_config(),
        prompt_template=[
            {
                "role": "system",
                "content": get_coding_agent_system_prompt(datetime.now()),
            }
        ],
        max_iteration=20,             
        max_tool_calls_per_turn=1,
        enable_question_hints=True,
        open_api_key = OPENAI_API_KEY,
        enable_extract_final_answer=True,
        enable_todo_plan=False,
    )

    # 实例化 main_agent 和 browsing_agent （MCP tools 在后续部分添加）
    main_agent = SuperReActAgent(
        agent_config=main_agent_config,
        tools=None,
        workflows=None,
        
    )

    browsing_agent = SuperReActAgent(
        agent_config=browsing_agent_config,
        tools=None,
        workflows=None,
        
    )

    coding_agent = SuperReActAgent(
        agent_config = coding_agent_config,
        tools=None,
        workflows=None,
    )

    # 用 main_agent 构建出所有 MCP 工具组（只调用一次）
    tool_groups = await build_mcp_tool_groups(main_agent)
    # tool_groups 形如：
    # {
    #   "tool-autobrowser": [LocalFunction(...), ...],
    #   "tool-transcribe": [...],
    #   "tool-reasoning":  [...],
    #   "tool-reading":    [...],
    #   "tool-searching":  [...],
    #   "tool-vqa":        [...],
    #   "tool-code":       [...],
    # }

    # 把工具分配到 main_agent / browsing_agent

    MAIN_AGENT_TOOL_GROUPS = [
        "tool-reasoning"
    ]

    CODING_AGENT_TOOL_GROUPS = [
        "tool-code",
        "tool-vqa",
        "tool-reading",
    ]

    BROWSING_AGENT_TOOL_GROUPS = [
        "tool-searching",
        "tool-vqa",
        "tool-reading",
        "tool-autobrowser",
        "tool-transcribe",
    ]

    # 给 main_agent 添加工具
    for group in MAIN_AGENT_TOOL_GROUPS:
        if group in tool_groups:
            main_agent.add_tools(tool_groups[group])
        else:
            print(f"[WARN] tool group '{group}' not found in MCP_TOOL_GROUPS")

    # 给 browsing_agent 添加工具
    for group in BROWSING_AGENT_TOOL_GROUPS:
        if group in tool_groups:
            browsing_agent.add_tools(tool_groups[group])
        else:
            print(f"[WARN] tool group '{group}' not found in MCP_TOOL_GROUPS")

    # 给 coding_agent 添加工具
    for group in CODING_AGENT_TOOL_GROUPS:
        if group in tool_groups:
            coding_agent.add_tools(tool_groups[group])
        else:
            print(f"[WARN] tool group '{group}' not found in MCP_TOOL_GROUPS")

    # 把子agent 注册到 main_agent 中
    print("Registering sub-agent 'agent-browsing' as a tool on main_agent...")
    main_agent.register_sub_agent("agent-browsing", browsing_agent)
    print("Registering sub-agent 'agent-coding' as a tool on main_agent...")
    main_agent.register_sub_agent("agent-coding", coding_agent)    
    print(f"Sub-agents registered on main_agent: {list(main_agent._sub_agents.keys())}")

    results: list[dict] = []

    # At the beginning of your function, create the output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(f"evaluation_results_{timestamp}.txt")
    json_output_file = Path(f"evaluation_results_{timestamp}.json")

    # Open file in append mode
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

    for idx, query in enumerate(queries, start=1):
        print(f"Query #{idx}: {query}")
        # if not query or not isinstance(query, str):
        #     print(f"[WARN] Skipping invalid query at index {idx - 1}")
        #     continue
        usr_question, usr_file, gt = query
        # 在新问题前清空上下文，确保不会串话
        main_agent._context_manager.clear()
        browsing_agent._context_manager.clear()
        coding_agent._context_manager.clear()

        print(f"\nMain query #{idx}:\n{query}\n")

        result = await main_agent.invoke({"query": usr_question, "file_path": usr_file})
        # results.append({"query": query, "result": result, 'ground_truth': gt})

        # Extract prediction
        prediction = result.get('extracted_metadata', 'No output')
        prediction = prediction.get('boxed_answer', 'No boxed answer') if isinstance(prediction, dict) else 'No boxed answer'
        result_type = result.get('result_type', 'unknown')

        # Context lengths for logging
        main_context_len = len(main_agent._context_manager.get_history())
        browsing_context_len = len(browsing_agent._context_manager.get_history())
        coding_context_len = len(coding_agent._context_manager.get_history())
        
        # Compare with ground truth
        is_correct = str(prediction).strip() == str(gt).strip()
        
        # Store result
        result_dict = {
            "query_id": idx,
            "query": usr_question,
            "file_path": usr_file,
            "result": result,
            "prediction": prediction,
            "ground_truth": gt,
            "result_type": result_type,
            "is_correct": is_correct,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "main_context_length": main_context_len,
            "browsing_context_length": browsing_context_len,
            "coding_context_length": coding_context_len
        }
        results.append(result_dict)
        
        # Write to text file immediately
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"Query #{idx}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Question: {usr_question}\n")
            f.write(f"File: {usr_file}\n")
            f.write(f"Result Type: {result_type}\n")
            f.write(f"\nPrediction:\n{prediction}\n")
            f.write(f"\nGround Truth:\n{gt}\n")
            f.write(f"\nStatus: {'✓ CORRECT' if is_correct else '✗ WRONG'}\n")
            f.write(f"\nMain agent context messages: {main_context_len}\n")
            f.write(f"Browsing sub-agent context messages: {browsing_context_len}\n")
            f.write(f"Coding sub-agent context messages: {coding_context_len}\n")
            f.write("\n" + "=" * 80 + "\n\n")

        print(f"Result type: {result.get('result_type', 'unknown')}")
        print(f"Output:\n{result.get('output', 'No output')}")
        print("-" * 80 + "\n")
        print(f"\nMain agent context messages: {main_context_len}")
        print(f"Browsing sub-agent context messages: {browsing_context_len}")
        print(f"Coding sub-agent context messages: {coding_context_len}")

    # Write summary at the end
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct"))
    accuracy = (correct / total * 100) if total > 0 else 0

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total queries: {total}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Wrong: {total - correct}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")

    # Also save as JSON for easier processing
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "results": results
        }, f, indent=2, ensure_ascii=False)

    # # 资源清理（MCP server & Runner）
    # tool_mgr = resource_mgr.tool()
    # # 如果想显式移除所有 server，可以逐个 remove
    # for server_name in {cfg["server_name"] for cfg in MCP_TOOL_GROUPS.values()}:
    #     try:
    #         await tool_mgr.remove_tool_server(server_name)
    #     except RuntimeError as e:
    #         if "cancel scope" in str(e):
    #             print(f"Ignoring SSE shutdown error for {server_name}: {e}")
    #         else:
    #             raise

    try:
        await Runner.stop()
    except RuntimeError as e:
        if "cancel scope" in str(e):
            print("Ignore MCP SSE shutdown RuntimeError during Runner.stop:", e)
        else:
            raise

    return results

async def main():
    """Main function to run all examples"""
    print("\n" + "=" * 70)
    print("Super ReAct Agent Test Run")
    print("=" * 70)
    print("This test run runs a main agent and a browsing sub-agent with different MCP tool groups.")
    print("The intention is to run whatever questions you want to test on the GAIA dataset.")
    print("\nMake sure your API configuration is correct before running.")
    try:
        gaia_queries = []
        for entry in GAIA_DATASET:
            # Prefer GAIA's `label_answer`; fall back to `ground_truth` if absent
            task = entry.get("task_question")
            file = entry.get("file_path")
            gt = entry.get("label_answer", entry.get("ground_truth"))
            gaia_queries.append((task, file, gt))
    
        if not gaia_queries:
            print("[WARN] No valid questions found in GAIA dataset, falling back to default sample query.")
            
        results = await example_mcp_main_and_sub_agents(gaia_queries)

        for item in results:
            print("\n" + "=" * 70)
            print(f"Query: {item.get('query', 'Unknown query')}")
            print(f"Prediction: {item.get('prediction')}")
            print(f"Ground truth: {item.get('ground_truth')}")
            print(f"Correct: {item.get('is_correct')}")
            # print(f"Result: {item.get('result')}")
        print("\n" + "=" * 70)
        print("All queries completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] Error occurred: {e}")
        import traceback
        traceback.print_exc()


    
if __name__ == "__main__":
    async def _runner():
        try:
            await main()
        finally:
            # 给底层一点时间处理 cancel/shutdown
            await asyncio.sleep(0.1)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(_runner())
    finally:
        # 找出所有还没结束的任务
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            # 把所有任务的异常都“吃掉”，避免 unhandled exception 提示
            loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        loop.close()