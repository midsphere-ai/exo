---
title: Getting Started
description: Install Exo, create your first agent, and test it in the playground
section: getting-started
order: 1
---

# Getting Started with Exo

Welcome to Exo — a platform for building, testing, and deploying AI agents. This guide walks you through installation, initial setup, creating your first agent, and testing it.

## Prerequisites

- **Python 3.11+** installed
- **Node.js 20+** installed (for the frontend)
- **uv** package manager ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
- An API key from at least one model provider (OpenAI, Anthropic, Google, etc.)

## Installation

1. Clone the repository and install dependencies:

```bash
git clone <your-repo-url> exo
cd exo
uv sync
```

2. Install frontend dependencies:

```bash
cd packages/exo-web
npm install
```

3. Return to the project root:

```bash
cd ../..
```

## First Run

Start the Exo platform:

```bash
# Start the backend API server
cd packages/exo-web
uv run uvicorn exo_web.app:app --reload --port 8000

# In a separate terminal, start the frontend dev server
cd packages/exo-web
npm run dev
```

The platform will be available at **http://localhost:4321** (frontend) with API calls proxied to the backend at port 8000.

On first launch, the database is created automatically and all migrations run.

## Creating Your First User

Before logging in, create an admin user via the CLI:

```bash
cd packages/exo-web
uv run python -m exo_web.cli create-user \
  --email admin@example.com \
  --password your-secure-password \
  --admin
```

You can also create regular (non-admin) users:

```bash
uv run python -m exo_web.cli create-user \
  --email developer@example.com \
  --password their-password \
  --role developer
```

Available roles: `admin`, `developer`, `viewer`.

Now log in at **http://localhost:4321/login** with your credentials.

## Step 1: Create a Project

1. Navigate to **Projects** in the sidebar
2. Click **New Project**
3. Give it a name (e.g., "My First Project") and an optional description
4. Click **Create**

Projects organize your agents, workflows, and resources into logical groups.

## Step 2: Configure a Model Provider

Before creating agents, you need at least one model provider configured with an API key.

1. Go to **Settings** from the sidebar (gear icon at the bottom)
2. Open the **Providers** tab
3. Click **Add Provider** and choose your provider:
   - **OpenAI** — GPT-4o, GPT-4o-mini, etc.
   - **Anthropic** — Claude Sonnet, Claude Haiku, etc.
   - **Google Gemini** — Gemini 2.0 Flash, etc.
   - **Ollama** — Local models (requires Ollama running)
   - **Custom** — Any OpenAI-compatible endpoint
4. Enter your API key and click **Save**
5. Click **Test Connection** to verify your key works

## Step 3: Create Your First Agent

1. Navigate to **Agents** in the sidebar
2. Click **New Agent**
3. Fill in the basics:
   - **Name**: Give your agent a descriptive name (e.g., "Research Assistant")
   - **Project**: Select the project you just created
   - **Model**: Select a model from your configured providers
   - **Instructions**: Define the agent's behavior, e.g. "You are a helpful research assistant. Answer questions clearly and concisely."
4. Click **Create**

## Step 4: Test in the Playground

1. Go to **Playground** in the sidebar
2. Select your agent from the dropdown
3. Type a message (e.g., "What are the main features of Python 3.12?") and press Enter
4. Watch your agent respond in real-time with streaming tokens

The playground supports:
- **Streaming responses** — See tokens appear in real time
- **Conversation history** — Continue multi-turn conversations
- **Model comparison** — Compare responses from different agents side by side

## Step 5: Add Tools (Optional)

Tools give your agent the ability to take actions — search the web, execute code, query databases, and more.

1. Open your agent's edit page (click the agent name, then **Edit**)
2. Scroll to the **Tools** section
3. Click **Add Tool** and select from the built-in catalog, or create a custom tool
4. Save the agent

Built-in tools include: web search, code execution (sandboxed Python), file operations, and more.

## Troubleshooting

### Common Startup Errors

**`ModuleNotFoundError: No module named 'exo_web'`**
- Make sure you ran `uv sync` from the project root
- Ensure you're running commands from the `packages/exo-web` directory

**`ERROR: Database directory is not writable`**
- The default database is created at `exo.db` in the current directory
- Ensure the directory has write permissions
- Or set `EXO_DATABASE_URL` to a writable path

**`Address already in use (port 8000)`**
- Another process is using port 8000
- Either stop that process or use a different port: `--port 8001`

### Database Issues

**`OperationalError: no such table`**
- Migrations may not have run. Run them manually:
  ```bash
  uv run python -m exo_web.cli migrate
  ```
- Check migration status:
  ```bash
  uv run python -m exo_web.cli migrate --status
  ```

**`OperationalError: database is locked`**
- SQLite WAL mode is enabled by default, which helps with concurrent reads
- If you see this, ensure only one write process is active
- Restart the server if the lock persists

### Port Conflicts

**Frontend (Astro) port 4321 in use:**
- Change it in `astro.config.mjs` or use: `npm run dev -- --port 4322`

**Backend (Uvicorn) port 8000 in use:**
- Start with a different port: `uv run uvicorn exo_web.app:app --port 8001`
- Update the Vite proxy in `astro.config.mjs` to match

### Authentication Issues

**"Invalid credentials" on login:**
- Verify you created a user with `create-user` CLI command
- Check the email and password match exactly

**CSRF errors on POST/PUT/DELETE:**
- The browser automatically includes CSRF tokens via the PageLayout
- If calling the API directly, first get a token: `GET /api/v1/auth/csrf`
- Include it as `X-CSRF-Token` header on mutating requests

## Environment Variables Reference

All configuration is via environment variables prefixed with `EXO_`:

| Variable | Default | Description |
|----------|---------|-------------|
| `EXO_DATABASE_URL` | `sqlite+aiosqlite:///exo.db` | Database connection URL. SQLite path or async-compatible URL |
| `EXO_SECRET_KEY` | `change-me-in-production` | Secret key for session cookies and API key encryption. **Must change in production** |
| `EXO_DEBUG` | `false` | Enable debug mode (`true`/`false`). Shows detailed error traces |
| `EXO_SESSION_EXPIRY_HOURS` | `72` | Session cookie lifetime in hours |
| `EXO_RATE_LIMIT_AUTH` | `5` | Max auth attempts per minute (login, register) |
| `EXO_RATE_LIMIT_GENERAL` | `60` | Max general API requests per minute per user |
| `EXO_RATE_LIMIT_AGENT` | `10` | Max agent execution requests per minute per user |
| `EXO_MAX_UPLOAD_MB` | `50` | Maximum file upload size in megabytes |
| `EXO_UPLOAD_DIR` | `data/uploads/` | Directory for uploaded files |
| `EXO_ARTIFACT_DIR` | `data/artifacts/` | Directory for generated artifacts |
| `EXO_CLEANUP_INTERVAL_HOURS` | `6` | Interval between automatic cleanup runs |
| `EXO_CORS_ORIGINS` | *(empty)* | Comma-separated list of allowed CORS origins. Leave empty for same-origin only |

### Production Checklist

Before deploying to production:

1. Set `EXO_SECRET_KEY` to a strong random value (e.g., `python -c "import secrets; print(secrets.token_hex(32))"`)
2. Set `EXO_DEBUG=false`
3. Configure `EXO_CORS_ORIGINS` if the frontend is on a different domain
4. Use a persistent path for `EXO_DATABASE_URL`
5. Set up regular database backups

## Next Steps

- **[Workflows](/docs/workflows)** — Build multi-step automation with a visual canvas
- **[Knowledge Base](/docs/knowledge-base)** — Give agents access to your documents via RAG
- **[Deployments](/docs/deployments)** — Deploy agents as API endpoints or embeddable widgets
- **[Monitoring](/docs/monitoring)** — Track runs, costs, and performance
- **[API Reference](/docs/api-reference)** — Full REST API documentation
