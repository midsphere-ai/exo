
# openJiuwen DeepAgent

openJiuwen agentcore is an AI agent designed for all scenarios of ToC and ToB. It aims to provide a flexible, powerful and easy-to-use agent development framework, support the creation of AI agents for various complex tasks, achieve efficient and accurate execution of AI agents, help teams of all sizes and across industries build reliable AI agents that can be deployed in production environments, and help users and enterprises implement Agent AI technology.


## 1. Data Preparation

Please place your data under `/data`.
Two sample entries have already been provided in `test.jsonl` for your reference.


## 2. Environment Setup
Our project requires **two** Python environments: one for running the openJiuwen framework, and another for running the MCP services.

### 2.1 Prepare openJiuwen framework
```

1. Install [uv](https://docs.astral.sh/uv/) if it is not already on your PATH.
2. From the repo root (cd ./deepagent) run `uv sync` to create/update the virtual environment with the pinned dependencies in `uv.lock`.
3. Run `uv pip install tiktoken` separately

```

### 2.2 Prepare MCP tool environment

#### Python Environment

- Python 3.12+ required (as specified in `/tool/pyproject.toml`)
- Use any manager to install the dependencies (uv, conda, venv, etc.)

#### Setup

**Option 1: Managed Environment (Recommended)**

The startup scripts will automatically detect and use a virtual environment in `/tool/`. They prefer, in order:
- `.venv-tool`
- `.venv`
- `venv`

If you need to create the venv manually:

```bash
# Navigate to tooling directory
cd ./tool

# Create venv
uv venv .venv-tool --python 3.12


# Activate venv:
## Windows (PowerShell)
.\.venv-tool\Scripts\Activate.ps1
## macOS/Linux
source .venv-tool/bin/activate


# Install dependencies:
## Windows (PowerShell)
uv pip install --no-deps -r requirements.txt 
## macOS/Linux
uv pip install --no-deps -r requirements_mac.txt 
```

**Option 2: User-Managed Environment**


If you prefer to manage your own Python environment (venv, conda, system Python, etc.):

1. Activate your environment first
2. Install dependencies: `pip install --no-deps -r requirements.txt` (`requirements_mac.txt` for macOS)
3. Run the scripts with the `--no-env` flag (see below)

#### Additonal Notes about Node.js

You can check whether Node.js is installed by running the following commands:
```
node -v
npm -v
npx -v
```
, because the startup of some MCP tools depends on this, for example,
```
    serper_server_params = StdioServerParameters(
        command="npx",
        args=["-y", "serper-search-scrape-mcp-server"],
        env={"SERPER_API_KEY": SERPER_API_KEY},
    )
```

### Environment Variables


**Main Agent Configurations:**
- `API_BASE`
- `API_KEY`
- `MODEL_NAME`
- `MODEL_PROVIDER`
- `REASONING_MODEL_NAME`


**Common API Keys:**
Some mcp servers may require specific API keys. Set these in your `.env` file or environment:
- `OPENAI_API_KEY` - Used by audio, vision, doubter servers
- `GEMINI_API_KEY` - Used by vision, browser_use_mcp, doubter, searching servers
- `OPENROUTER_API_KEY` - Used by vision, reasoning, doubter, searching servers

**Server-Specific Keys:**
- `E2B_API_KEY` - Required for python server (sandbox execution)
- `E2B_TEMPLATE_ID` - template ID, we provide what we use as the default
- `SERPER_API_KEY` - For Google search in searching server
- `JINA_API_KEY` - For Jina deep search in searching server
- `PERPLEXITY_API_KEY` - For Perplexity search in searching server
- `ACR_ACCESS_KEY`, `ACR_ACCESS_SECRET` - For audio metadata recognition (optional)

**Others:**
- `CHROME_PATH` - the location of your browser
- `CHROME_USER_PROFILE_DIR` - the location of your browser user profile, since some websites restrict access for users who are not logged in
- `DATA_DIR` - data path





# 2. Usage
  ```
  deactivate (if you have currently activated the tool environment, e.g., .venv-tool)
  cd ../ (jump to  DeepAgent folder)
  # macOS/Linux
  source ./.venv/bin/activate
  uv run ./test/super_react_agent_test_run.py
  ```
