# exo-search UI — Design Spec

## Context

exo-search is a powerful AI search engine with query classification, parallel research agents, result reranking, and citation generation across 3 quality modes (speed/balanced/quality). It already has a `server.py` with basic API endpoints, but no frontend. This spec defines a minimal, clean web UI that lives inside the existing `packages/exo-search/` package, allowing users to configure API keys and model settings in-browser and run searches with streamed results.

## Architecture

Everything lives inside `packages/exo-search/`:

```
packages/exo-search/
├── src/exo/search/
│   ├── server.py          ← extend (new API endpoints + static file serving)
│   └── ...                ← existing search code (untouched)
├── ui/
│   ├── index.html         ← single-page app
│   ├── styles.css         ← zen theme + responsive layout
│   └── app.js             ← search logic, SSE, settings, chat history
├── Dockerfile
└── docker-compose.yml
```

**Frontend:** Vanilla HTML/CSS/JS, no build step. Bricolage Grotesque font and `marked.js` loaded via CDN. FastAPI serves the `ui/` directory as static files at `/`.

**Backend:** Extend the existing `server.py` with new endpoints. Keep existing endpoints intact.

## Backend API

### `POST /api/search`

Request body:

```json
{
  "query": "How does CRISPR work?",
  "mode": "balanced",
  "session_id": "uuid-string",
  "config": {
    "serper_api_key": "...",
    "jina_api_key": "...",
    "model": "openai:gpt-4o",
    "fast_model": "openai:gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "api_key": "sk-...",
    "base_url": "https://api.openai.com/v1"
  }
}
```

The server:
1. Builds a `SearchConfig` from `config` fields
2. Sets provider API keys (e.g. `OPENAI_API_KEY`) as env vars for the request duration
3. Calls `configure_search_keys()` with serper/jina keys
4. Runs `run_search_pipeline()`
5. Returns `SearchResponse.model_dump()`

Maintains a `ConversationManager` per `session_id` for multi-turn context.

### `GET /api/search/stream`

SSE endpoint. Query params: `q`, `mode`, `session_id`. Before opening the SSE stream, the frontend calls `POST /api/config/{session_id}` to cache the config server-side (in-memory dict keyed by session_id). The stream endpoint reads from this cache. Config cache is cleaned up on `DELETE /api/search/{session_id}` or after 1 hour of inactivity.

SSE event types (matching existing `server.py` patterns):
- `status` — pipeline stage transitions (`{"stage": "researcher", "status": "started", "message": "..."}`)
- `answer` — streamed answer text chunks
- `sources` — source list JSON
- `suggestions` — follow-up suggestions JSON
- `done` — completion signal

### `DELETE /api/search/{session_id}`

Clears conversation history for a session (existing endpoint pattern).

### Static file serving

Mount `ui/` directory at `/` using `StaticFiles(directory="ui", html=True)`.

## Frontend

### Visual Design

**Zen colorscheme** (from exo-web):
- Light: `--zen-paper: #f2f0e3`, `--zen-dark: #2e2e2e`, `--zen-muted: #8a877a`, `--zen-subtle: #e8e6d9`, `--zen-coral: #f76f53`, `--zen-blue: #6287f5`, `--zen-green: #63f78b`
- Dark: `--zen-paper: #1f1f1f`, `--zen-dark: #d1cfc0`, `--zen-subtle: #2e2e2e` (accents unchanged)

**Typography:** Bricolage Grotesque via `@fontsource` CDN, weights 400-700. Body default weight 500. `--font-sans: "Bricolage Grotesque", system-ui, sans-serif`.

**Component patterns:**
- Borders: `1px solid` with subtle opacity (8-12%)
- Border radius: 8px inputs, 12-16px cards/search bar
- Hover: subtle background shifts, no heavy shadows
- Focus: coral-tinted ring
- Transitions: 150-200ms ease

### Layout States

**State 1 — Landing (no active search):**
- Full-width, no sidebar
- Top bar: logo left, theme toggle + settings gear right
- Centered hero: heading ("What do you want to know?"), subtitle, search bar with inline mode selector (speed/balanced/quality toggle), submit button
- Suggestion chips below search bar (hardcoded starters)

**State 2 — Results (after first search):**
- Sidebar slides in from left (CSS transition, ~240px wide):
  - Logo + "New Search" button at top
  - Chat history grouped by date (Today/Yesterday/Older)
  - Theme toggle + settings gear at bottom
- Main panel:
  - Top bar: current query title + mode selector
  - Answer area (Perplexity-style, see below)
  - Input bar pinned at bottom

**"New Search"** returns to the centered landing state (sidebar hidden).

### Answer Rendering (Perplexity-style)

The answer area follows Perplexity's visual structure:

**Sources row (top of answer):**
- Horizontal row of numbered source cards, scrollable if overflow
- Each card: circled number (1, 2, 3...) + favicon (via `https://www.google.com/s2/favicons?domain=...&sz=16`) + domain name + truncated title
- Cards are compact pill/chips, not large boxes
- Clicking a source card opens the URL in a new tab

**Answer body:**
- Clean article-style prose with markdown rendering (headings, bold, lists, code blocks)
- Inline citation numbers `[1]`, `[2]` rendered as small superscript badges with a subtle background (e.g. `background: var(--zen-subtle); border-radius: 4px; padding: 0 4px; font-size: 0.75em`)
- Hovering a citation number highlights the corresponding source card in the row above
- Clicking a citation number opens the source URL in a new tab
- Generous line-height (1.7-1.8) and readable max-width (~720px)
- Headings within the answer use `font-weight: 600`, slightly smaller than page titles

**Follow-up suggestions (bottom of answer):**
- Section labeled "Related"
- Rendered as a vertical list of clickable rows with a right arrow icon, not chips
- Each row has the suggestion text + `→` on the right
- Subtle border between rows
- Clicking a suggestion submits it as a follow-up query

**Multi-turn display:**
- Follow-up answers append below the previous answer in the same scrollable area
- Each turn separated by a subtle divider and the user's query shown as a small header above the new answer
- Sources re-numbered per turn (each turn has its own source row)

### Streaming UX

1. User submits query → stage indicator appears below the source row area ("Classifying...")
2. SSE `status` events update the indicator ("Researching... 12 results found", "Writing...") with a subtle animated dot
3. SSE `sources` event renders the source cards row (appears first, before the answer text)
4. SSE `answer` events stream markdown text, rendered progressively with `marked.js` — text appears smoothly as it arrives
5. SSE `suggestions` event renders the "Related" section below the answer
6. SSE `done` event finalizes — saves the exchange to chat history

### Settings Modal

Gear icon opens a modal overlay with two sections:

**Search Backend** (radio toggle):
- **Serper** (default) — shows: Serper API Key (password input)
- **SearXNG** — shows: SearXNG URL (text input, placeholder: `http://localhost:8888`)

**Content Enrichment** (radio toggle):
- **Jina Cloud** (default) — shows: Jina API Key (password input)
- **Self-hosted Jina Reader** — shows: Jina Reader URL (text input, placeholder: `http://127.0.0.1:3000`)

**LLM Configuration:**
- Model (text input, placeholder: `openai:gpt-4o`)
- Fast Model (text input, placeholder: `openai:gpt-4o-mini`)
- API Key (password input — for the LLM provider)
- Base URL (text input, optional — for custom endpoints)
- Embedding Model (text input, placeholder: `text-embedding-3-small`)

All saved to `localStorage`. On first visit with no saved config, modal auto-opens. "Save" validates non-empty required fields (at least one search backend configured + model + API key) and closes.

Config is sent in the request body of every search call. The `config` object includes whichever backend/enrichment option is selected:

```json
{
  "config": {
    "serper_api_key": "...",       // if Serper selected
    "searxng_url": "...",          // if SearXNG selected
    "jina_api_key": "...",         // if Jina Cloud selected
    "jina_reader_url": "...",      // if self-hosted Jina selected
    "model": "openai:gpt-4o",
    "fast_model": "openai:gpt-4o-mini",
    "embedding_model": "text-embedding-3-small",
    "api_key": "sk-...",
    "base_url": "https://api.openai.com/v1"
  }
}
```

### Theme Switching

- `data-theme` attribute on `<html>` (light/dark)
- Blocking script in `<head>` reads `localStorage` or `prefers-color-scheme`
- Toggle button swaps value, persists to `localStorage`

### Chat History

- Stored in `localStorage` as `Array<{id, title, messages: Array<{role, content, sources?, suggestions?}>, mode, created_at}>`
- Each search creates or continues a session (keyed by `session_id`)
- Sidebar shows sessions grouped by date, clicking loads conversation
- "New Search" generates a fresh `session_id` and returns to landing
- Title auto-set from first query text (truncated)

## Dockerfile

Located at `packages/exo-search/Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY . .
RUN uv sync
EXPOSE 8000
CMD ["uv", "run", "python", "-m", "exo.search", "--serve", "--host", "0.0.0.0", "--port", "8000"]
```

Requires the full workspace context (exo-search depends on exo-core, exo-models). No env vars baked in — all config arrives from the frontend per-request.

`docker-compose.yml`:

```yaml
services:
  exo-search:
    build:
      context: ../..
      dockerfile: packages/exo-search/Dockerfile
    ports:
      - "8000:8000"
```

Build context is the workspace root so `uv sync` can resolve workspace dependencies.

## Verification

1. `uv run python -m exo.search --serve` — server starts, UI loads at `http://localhost:8000`
2. Open settings modal → enter API keys and model config → save
3. Type a query → streaming indicators appear → answer renders with citations and source cards
4. Click a follow-up suggestion → follow-up search works with conversation context
5. Click "New Search" → returns to centered landing
6. Toggle theme → light/dark switches, persists on reload
7. Reload page → settings and chat history survive from `localStorage`
8. `docker compose up` → same UI accessible at `http://localhost:8000`
9. Existing `GET /search` and `POST /chat` endpoints still work (backward compat)
