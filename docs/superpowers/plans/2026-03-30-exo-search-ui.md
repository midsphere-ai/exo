# exo-search UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal, Perplexity-style web UI to the existing exo-search package with streaming results, settings modal, chat history, and Docker support.

**Architecture:** Vanilla HTML/CSS/JS frontend served as static files by the existing FastAPI server. New API endpoints accept per-request config (API keys, model strings) from the browser. No build step, no framework.

**Tech Stack:** FastAPI (existing), vanilla JS, SSE (EventSource), marked.js (CDN), Bricolage Grotesque (CDN), CSS custom properties for zen theme.

**Spec:** `docs/superpowers/specs/2026-03-30-exo-search-ui-design.md`

---

## File Structure

```
packages/exo-search/
├── src/exo/search/
│   └── server.py                ← MODIFY: add /api/* endpoints + static file mount
├── ui/
│   ├── index.html               ← CREATE: single-page app shell
│   ├── styles.css               ← CREATE: zen theme, layout, components
│   └── app.js                   ← CREATE: search logic, SSE, settings, history
├── Dockerfile                   ← CREATE
└── docker-compose.yml           ← CREATE
```

---

### Task 1: CSS Theme & Layout (`ui/styles.css`)

**Files:**
- Create: `packages/exo-search/ui/styles.css`

This is the foundation — zen colorscheme, Bricolage Grotesque, both layout states, all component styles. No JS behavior yet.

- [ ] **Step 1: Create `ui/styles.css` with CSS custom properties, reset, typography**

```css
/* packages/exo-search/ui/styles.css */

/* === Font === */
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@400;500;600;700&display=swap');

/* === Reset === */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* === Theme variables === */
:root {
  --zen-paper: #f2f0e3;
  --zen-dark: #2e2e2e;
  --zen-muted: #8a877a;
  --zen-subtle: #e8e6d9;
  --zen-coral: #f76f53;
  --zen-blue: #6287f5;
  --zen-green: #63f78b;
  --font-sans: 'Bricolage Grotesque', system-ui, sans-serif;
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;
  --transition: 150ms ease;
}

[data-theme='dark'] {
  --zen-paper: #1f1f1f;
  --zen-dark: #d1cfc0;
  --zen-muted: #8a877a;
  --zen-subtle: #2e2e2e;
}

html, body {
  height: 100%;
  font-family: var(--font-sans);
  font-weight: 500;
  background: var(--zen-paper);
  color: var(--zen-dark);
  line-height: 1.6;
  transition: background var(--transition), color var(--transition);
}
```

- [ ] **Step 2: Add layout structure styles — app shell, top bar, sidebar, main panel**

```css
/* === App Shell === */
#app {
  display: flex;
  height: 100vh;
  overflow: hidden;
}

/* === Sidebar === */
.sidebar {
  width: 260px;
  border-right: 1px solid color-mix(in srgb, var(--zen-dark) 8%, transparent);
  padding: 16px;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  transform: translateX(-260px);
  transition: transform 300ms ease;
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  z-index: 10;
  background: var(--zen-paper);
}

.sidebar.open {
  transform: translateX(0);
}

/* === Main Panel === */
.main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  transition: margin-left 300ms ease;
}

.sidebar.open ~ .main {
  margin-left: 260px;
}

/* === Top Bar === */
.top-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 32px;
  border-bottom: 1px solid color-mix(in srgb, var(--zen-dark) 8%, transparent);
  flex-shrink: 0;
}

.logo {
  font-weight: 700;
  font-size: 18px;
}

.logo span {
  color: var(--zen-muted);
  font-weight: 400;
}
```

- [ ] **Step 3: Add landing hero styles — centered search bar, mode selector, suggestion chips**

```css
/* === Landing Hero === */
.landing {
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 1;
  padding: 0 40px;
}

.landing-inner {
  max-width: 680px;
  width: 100%;
  text-align: center;
}

.landing h1 {
  font-size: 28px;
  font-weight: 600;
  margin-bottom: 8px;
}

.landing .subtitle {
  font-size: 14px;
  color: var(--zen-muted);
  margin-bottom: 28px;
}

/* === Search Bar === */
.search-bar {
  background: color-mix(in srgb, var(--zen-dark) 4%, var(--zen-paper));
  border: 1.5px solid color-mix(in srgb, var(--zen-dark) 12%, transparent);
  border-radius: var(--radius-lg);
  padding: 14px 18px;
  display: flex;
  align-items: center;
  gap: 12px;
  transition: border-color var(--transition);
}

.search-bar:focus-within {
  border-color: color-mix(in srgb, var(--zen-coral) 40%, transparent);
}

.search-bar input {
  flex: 1;
  border: none;
  background: none;
  font-family: var(--font-sans);
  font-size: 15px;
  font-weight: 500;
  color: var(--zen-dark);
  outline: none;
}

.search-bar input::placeholder {
  color: var(--zen-muted);
}

/* === Mode Selector === */
.mode-selector {
  display: flex;
  gap: 6px;
}

.mode-btn {
  font-family: var(--font-sans);
  font-size: 12px;
  font-weight: 500;
  padding: 6px 14px;
  border-radius: var(--radius-sm);
  border: 1px solid color-mix(in srgb, var(--zen-dark) 12%, transparent);
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
  transition: all var(--transition);
}

.mode-btn.active {
  color: var(--zen-dark);
  border-color: var(--zen-dark);
}

.mode-btn:hover:not(.active) {
  background: color-mix(in srgb, var(--zen-dark) 4%, transparent);
}

/* === Submit Button === */
.submit-btn {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  background: var(--zen-dark);
  color: var(--zen-paper);
  border: none;
  cursor: pointer;
  font-size: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  transition: opacity var(--transition);
}

.submit-btn:hover {
  opacity: 0.85;
}

/* === Suggestion Chips === */
.suggestions {
  display: flex;
  gap: 8px;
  justify-content: center;
  flex-wrap: wrap;
  margin-top: 20px;
}

.suggestion-chip {
  font-family: var(--font-sans);
  font-size: 12px;
  color: var(--zen-muted);
  border: 1px solid color-mix(in srgb, var(--zen-dark) 8%, transparent);
  border-radius: 20px;
  padding: 6px 16px;
  background: color-mix(in srgb, var(--zen-dark) 2%, var(--zen-paper));
  cursor: pointer;
  transition: all var(--transition);
}

.suggestion-chip:hover {
  border-color: color-mix(in srgb, var(--zen-dark) 20%, transparent);
  background: color-mix(in srgb, var(--zen-dark) 6%, var(--zen-paper));
}
```

- [ ] **Step 4: Add results view styles — source cards, answer, citations, related section, bottom input**

```css
/* === Results View === */
.results-view {
  display: none;
  flex-direction: column;
  flex: 1;
  overflow: hidden;
}

.results-view.active {
  display: flex;
}

.results-header {
  padding: 14px 36px;
  border-bottom: 1px solid color-mix(in srgb, var(--zen-dark) 8%, transparent);
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}

.results-header .query-title {
  flex: 1;
  font-size: 15px;
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* === Answer Area === */
.answer-area {
  flex: 1;
  overflow-y: auto;
  padding: 24px 40px;
}

.answer-content {
  max-width: 720px;
}

/* === Source Cards === */
.source-cards {
  display: flex;
  gap: 8px;
  margin-bottom: 20px;
  overflow-x: auto;
  padding-bottom: 4px;
}

.source-card {
  flex-shrink: 0;
  padding: 8px 12px;
  background: color-mix(in srgb, var(--zen-dark) 3%, var(--zen-paper));
  border: 1px solid color-mix(in srgb, var(--zen-dark) 8%, transparent);
  border-radius: 10px;
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  transition: all var(--transition);
  text-decoration: none;
  color: inherit;
}

.source-card:hover {
  border-color: color-mix(in srgb, var(--zen-dark) 20%, transparent);
}

.source-num {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  font-size: 11px;
  font-weight: 600;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  flex-shrink: 0;
}

.source-card .favicon {
  width: 16px;
  height: 16px;
  border-radius: 2px;
}

.source-card .source-domain {
  font-size: 11px;
  font-weight: 500;
}

.source-card .source-title {
  font-size: 10px;
  color: var(--zen-muted);
  max-width: 140px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* === Answer Prose === */
.answer-prose {
  font-size: 14px;
  line-height: 1.8;
}

.answer-prose h1, .answer-prose h2, .answer-prose h3 {
  font-weight: 600;
  margin: 20px 0 8px;
}

.answer-prose h2 { font-size: 18px; }
.answer-prose h3 { font-size: 16px; }

.answer-prose p {
  margin-bottom: 14px;
}

.answer-prose ul, .answer-prose ol {
  margin-bottom: 14px;
  padding-left: 24px;
}

.answer-prose li {
  margin-bottom: 6px;
}

.answer-prose code {
  background: color-mix(in srgb, var(--zen-dark) 6%, var(--zen-paper));
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 13px;
}

.answer-prose pre {
  background: color-mix(in srgb, var(--zen-dark) 6%, var(--zen-paper));
  padding: 14px 16px;
  border-radius: var(--radius-sm);
  overflow-x: auto;
  margin-bottom: 14px;
}

.answer-prose pre code {
  background: none;
  padding: 0;
}

/* citation badges */
.citation {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: color-mix(in srgb, var(--zen-blue) 15%, var(--zen-paper));
  color: var(--zen-blue);
  font-size: 0.7em;
  font-weight: 600;
  padding: 1px 5px;
  border-radius: 4px;
  cursor: pointer;
  text-decoration: none;
  vertical-align: super;
  transition: all var(--transition);
  margin: 0 1px;
}

.citation:hover {
  background: color-mix(in srgb, var(--zen-blue) 25%, var(--zen-paper));
}

/* === Related Section === */
.related-section {
  margin-top: 28px;
  border-top: 1px solid color-mix(in srgb, var(--zen-dark) 8%, transparent);
  padding-top: 16px;
}

.related-section .related-label {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--zen-muted);
  margin-bottom: 8px;
}

.related-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 0;
  border-bottom: 1px solid color-mix(in srgb, var(--zen-dark) 5%, transparent);
  cursor: pointer;
  color: var(--zen-dark);
  transition: color var(--transition);
  font-size: 14px;
}

.related-item:hover {
  color: var(--zen-coral);
}

.related-item .arrow {
  color: var(--zen-muted);
  font-size: 14px;
}

/* === Pipeline Status === */
.pipeline-status {
  padding: 12px 0;
  font-size: 13px;
  color: var(--zen-muted);
  display: flex;
  align-items: center;
  gap: 8px;
}

.pipeline-status .dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--zen-coral);
  animation: pulse 1.2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 0.4; transform: scale(0.9); }
  50% { opacity: 1; transform: scale(1.1); }
}

/* === Follow-up Turn Divider === */
.turn-divider {
  margin: 24px 0 16px;
  padding: 8px 12px;
  background: color-mix(in srgb, var(--zen-dark) 4%, var(--zen-paper));
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 500;
}

/* === Bottom Input === */
.bottom-input {
  padding: 16px 40px;
  border-top: 1px solid color-mix(in srgb, var(--zen-dark) 8%, transparent);
  flex-shrink: 0;
}

.bottom-input .search-bar {
  max-width: 720px;
}
```

- [ ] **Step 5: Add sidebar content styles, icon buttons, settings modal, scrollbar**

```css
/* === Sidebar Content === */
.sidebar .logo { margin-bottom: 20px; }

.new-search-btn {
  font-family: var(--font-sans);
  width: 100%;
  background: var(--zen-dark);
  color: var(--zen-paper);
  border: none;
  border-radius: var(--radius-sm);
  padding: 10px 14px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  margin-bottom: 20px;
  transition: opacity var(--transition);
}

.new-search-btn:hover { opacity: 0.85; }

.history-group-label {
  font-size: 10px;
  color: var(--zen-muted);
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 6px;
  margin-top: 12px;
}

.history-item {
  padding: 8px 10px;
  border-radius: var(--radius-sm);
  font-size: 12px;
  color: var(--zen-muted);
  cursor: pointer;
  transition: all var(--transition);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.history-item:hover {
  background: color-mix(in srgb, var(--zen-dark) 6%, var(--zen-paper));
}

.history-item.active {
  background: color-mix(in srgb, var(--zen-dark) 4%, var(--zen-paper));
  border: 1px solid color-mix(in srgb, var(--zen-dark) 8%, transparent);
  color: var(--zen-dark);
  font-weight: 500;
}

.sidebar-footer {
  margin-top: auto;
  display: flex;
  gap: 8px;
  padding-top: 12px;
  border-top: 1px solid color-mix(in srgb, var(--zen-dark) 8%, transparent);
}

.history-list {
  flex: 1;
  overflow-y: auto;
}

/* === Icon Buttons === */
.icon-btn {
  width: 32px;
  height: 32px;
  border-radius: var(--radius-sm);
  border: 1px solid color-mix(in srgb, var(--zen-dark) 8%, transparent);
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  transition: all var(--transition);
}

.icon-btn:hover {
  background: color-mix(in srgb, var(--zen-dark) 6%, var(--zen-paper));
  border-color: color-mix(in srgb, var(--zen-dark) 15%, transparent);
}

/* === Settings Modal === */
.modal-overlay {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.4);
  z-index: 100;
  align-items: center;
  justify-content: center;
}

.modal-overlay.open {
  display: flex;
}

.modal {
  background: var(--zen-paper);
  border: 1px solid color-mix(in srgb, var(--zen-dark) 12%, transparent);
  border-radius: var(--radius-md);
  padding: 28px 32px;
  max-width: 520px;
  width: 90%;
  max-height: 85vh;
  overflow-y: auto;
  animation: modalSlideIn 200ms ease-out;
}

@keyframes modalSlideIn {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}

.modal h2 {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 20px;
}

.modal h3 {
  font-size: 14px;
  font-weight: 600;
  margin: 20px 0 12px;
  color: var(--zen-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.form-group {
  margin-bottom: 14px;
}

.form-group label {
  display: block;
  font-size: 13px;
  font-weight: 500;
  margin-bottom: 4px;
}

.form-group input,
.form-group select {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid color-mix(in srgb, var(--zen-dark) 12%, transparent);
  border-radius: var(--radius-sm);
  background: color-mix(in srgb, var(--zen-dark) 3%, var(--zen-paper));
  font-family: var(--font-sans);
  font-size: 13px;
  font-weight: 500;
  color: var(--zen-dark);
  outline: none;
  transition: border-color var(--transition);
}

.form-group input:focus,
.form-group select:focus {
  border-color: color-mix(in srgb, var(--zen-coral) 40%, transparent);
}

.form-group .hint {
  font-size: 11px;
  color: var(--zen-muted);
  margin-top: 3px;
}

/* Radio toggle group */
.radio-group {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}

.radio-btn {
  font-family: var(--font-sans);
  font-size: 12px;
  font-weight: 500;
  padding: 6px 14px;
  border-radius: var(--radius-sm);
  border: 1px solid color-mix(in srgb, var(--zen-dark) 12%, transparent);
  background: none;
  color: var(--zen-muted);
  cursor: pointer;
  transition: all var(--transition);
}

.radio-btn.active {
  color: var(--zen-dark);
  border-color: var(--zen-dark);
  background: color-mix(in srgb, var(--zen-dark) 5%, var(--zen-paper));
}

.modal-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 24px;
}

.btn-primary {
  font-family: var(--font-sans);
  padding: 10px 24px;
  background: var(--zen-dark);
  color: var(--zen-paper);
  border: none;
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: opacity var(--transition);
}

.btn-primary:hover { opacity: 0.85; }

.btn-secondary {
  font-family: var(--font-sans);
  padding: 10px 24px;
  background: none;
  color: var(--zen-muted);
  border: 1px solid color-mix(in srgb, var(--zen-dark) 12%, transparent);
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all var(--transition);
}

.btn-secondary:hover {
  background: color-mix(in srgb, var(--zen-dark) 4%, var(--zen-paper));
}

/* === Custom Scrollbar === */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: color-mix(in srgb, var(--zen-dark) 15%, transparent);
  border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
  background: color-mix(in srgb, var(--zen-dark) 25%, transparent);
}
```

- [ ] **Step 6: Verify CSS loads correctly**

Open `packages/exo-search/ui/styles.css` in a browser (via the server once Task 3 is done) and verify:
- Light/dark theme switching via `data-theme` attribute on `<html>`
- Font loads correctly
- No CSS errors in console

- [ ] **Step 7: Commit**

```bash
git add packages/exo-search/ui/styles.css
git commit -m "feat(exo-search): add zen theme CSS for search UI"
```

---

### Task 2: HTML Shell (`ui/index.html`)

**Files:**
- Create: `packages/exo-search/ui/index.html`

The complete HTML structure for both layout states — landing and results.

- [ ] **Step 1: Create `ui/index.html` with head, theme detection, font/CSS/JS loading**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>exo search</title>
  <link rel="stylesheet" href="/styles.css">
  <script>
    // Blocking theme detection — runs before first paint
    (function() {
      var stored = localStorage.getItem('theme');
      var preferred = stored || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      document.documentElement.setAttribute('data-theme', preferred);
    })();
  </script>
</head>
```

- [ ] **Step 2: Add body with app shell — sidebar, main panel with top bar**

```html
<body>
  <div id="app">
    <!-- Sidebar (hidden until first search) -->
    <aside class="sidebar" id="sidebar">
      <div class="logo">exo<span>search</span></div>
      <button class="new-search-btn" id="newSearchBtn">+ New Search</button>
      <div class="history-list" id="historyList"></div>
      <div class="sidebar-footer">
        <button class="icon-btn" id="themeToggleSidebar" title="Toggle theme">☀</button>
        <button class="icon-btn" id="settingsBtnSidebar" title="Settings">⚙</button>
      </div>
    </aside>

    <!-- Main Panel -->
    <div class="main">
      <!-- Top Bar -->
      <div class="top-bar">
        <div class="logo">exo<span>search</span></div>
        <div style="display:flex;gap:10px;align-items:center;">
          <button class="icon-btn" id="themeToggle" title="Toggle theme">☀</button>
          <button class="icon-btn" id="settingsBtn" title="Settings">⚙</button>
        </div>
      </div>
```

- [ ] **Step 3: Add landing view with hero search bar, mode selector, suggestion chips**

```html
      <!-- Landing View -->
      <div class="landing" id="landingView">
        <div class="landing-inner">
          <h1>What do you want to know?</h1>
          <p class="subtitle">AI-powered deep research with cited sources</p>
          <form class="search-bar" id="landingSearchBar">
            <input type="text" placeholder="Ask anything..." id="landingInput" autocomplete="off">
            <div class="mode-selector" id="landingModeSelector">
              <button type="button" class="mode-btn" data-mode="speed">⚡ speed</button>
              <button type="button" class="mode-btn active" data-mode="balanced">⚖ balanced</button>
              <button type="button" class="mode-btn" data-mode="quality">🔬 quality</button>
            </div>
            <button type="submit" class="submit-btn">→</button>
          </form>
          <div class="suggestions" id="landingSuggestions">
            <button class="suggestion-chip">How does CRISPR gene editing work?</button>
            <button class="suggestion-chip">Latest developments in quantum computing</button>
            <button class="suggestion-chip">Compare React vs Svelte for web apps</button>
            <button class="suggestion-chip">Explain transformer architecture in AI</button>
          </div>
        </div>
      </div>
```

- [ ] **Step 4: Add results view with header, answer area, bottom input**

```html
      <!-- Results View -->
      <div class="results-view" id="resultsView">
        <div class="results-header">
          <div class="query-title" id="queryTitle"></div>
          <div class="mode-selector" id="resultsModeSelector">
            <button type="button" class="mode-btn" data-mode="speed">⚡</button>
            <button type="button" class="mode-btn active" data-mode="balanced">⚖</button>
            <button type="button" class="mode-btn" data-mode="quality">🔬</button>
          </div>
        </div>
        <div class="answer-area" id="answerArea">
          <div class="answer-content" id="answerContent"></div>
        </div>
        <div class="bottom-input">
          <form class="search-bar" id="followUpBar">
            <input type="text" placeholder="Ask a follow-up..." id="followUpInput" autocomplete="off">
            <button type="submit" class="submit-btn">→</button>
          </form>
        </div>
      </div>
    </div>
  </div>
```

- [ ] **Step 5: Add settings modal HTML**

```html
  <!-- Settings Modal -->
  <div class="modal-overlay" id="settingsModal">
    <div class="modal">
      <h2>Settings</h2>

      <h3>Search Backend</h3>
      <div class="radio-group" id="searchBackendToggle">
        <button type="button" class="radio-btn active" data-value="serper">Serper</button>
        <button type="button" class="radio-btn" data-value="searxng">SearXNG</button>
      </div>
      <div id="serperFields">
        <div class="form-group">
          <label>Serper API Key</label>
          <input type="password" id="serperApiKey" placeholder="Enter Serper API key">
        </div>
      </div>
      <div id="searxngFields" style="display:none;">
        <div class="form-group">
          <label>SearXNG URL</label>
          <input type="text" id="searxngUrl" placeholder="http://localhost:8888">
        </div>
      </div>

      <h3>Content Enrichment</h3>
      <div class="radio-group" id="enrichmentToggle">
        <button type="button" class="radio-btn active" data-value="jina-cloud">Jina Cloud</button>
        <button type="button" class="radio-btn" data-value="jina-self">Self-hosted Jina</button>
      </div>
      <div id="jinaCloudFields">
        <div class="form-group">
          <label>Jina API Key</label>
          <input type="password" id="jinaApiKey" placeholder="Enter Jina API key">
        </div>
      </div>
      <div id="jinaSelfFields" style="display:none;">
        <div class="form-group">
          <label>Jina Reader URL</label>
          <input type="text" id="jinaReaderUrl" placeholder="http://127.0.0.1:3000">
        </div>
      </div>

      <h3>LLM Configuration</h3>
      <div class="form-group">
        <label>Model</label>
        <input type="text" id="modelInput" placeholder="openai:gpt-4o">
        <div class="hint">Format: provider:model (e.g. openai:gpt-4o, anthropic:claude-sonnet-4-20250514, gemini:gemini-2.5-pro)</div>
      </div>
      <div class="form-group">
        <label>Fast Model</label>
        <input type="text" id="fastModelInput" placeholder="openai:gpt-4o-mini">
        <div class="hint">Used for speed mode and classifier</div>
      </div>
      <div class="form-group">
        <label>API Key</label>
        <input type="password" id="llmApiKey" placeholder="Provider API key">
      </div>
      <div class="form-group">
        <label>Base URL <span style="color:var(--zen-muted)">(optional)</span></label>
        <input type="text" id="baseUrl" placeholder="https://api.openai.com/v1">
        <div class="hint">For custom or self-hosted endpoints</div>
      </div>
      <div class="form-group">
        <label>Embedding Model</label>
        <input type="text" id="embeddingModel" placeholder="text-embedding-3-small">
      </div>

      <div class="modal-actions">
        <button type="button" class="btn-secondary" id="settingsCancel">Cancel</button>
        <button type="button" class="btn-primary" id="settingsSave">Save</button>
      </div>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 6: Verify HTML structure**

Open `index.html` directly in a browser to confirm structure renders (no JS yet, just visual check).

- [ ] **Step 7: Commit**

```bash
git add packages/exo-search/ui/index.html
git commit -m "feat(exo-search): add HTML shell for search UI"
```

---

### Task 3: Backend API Endpoints (`server.py`)

**Files:**
- Modify: `packages/exo-search/src/exo/search/server.py`

Add new `/api/*` endpoints and static file serving. Keep all existing endpoints intact.

- [ ] **Step 1: Add config cache, new request models, and `POST /api/config/{session_id}`**

Add these imports and code to `server.py` (after the existing `_conversations` dict):

```python
import os
import time
from pathlib import Path

from fastapi.staticfiles import StaticFiles

# In-memory config cache: session_id -> (SearchConfig, timestamp)
_config_cache: dict[str, tuple[SearchConfig, float]] = {}
_CONFIG_TTL = 3600  # 1 hour


class UISearchRequest(BaseModel):
    query: str
    mode: str = "balanced"
    session_id: str = "default"
    config: dict = {}


class UIConfigRequest(BaseModel):
    serper_api_key: str = ""
    jina_api_key: str = ""
    searxng_url: str = ""
    jina_reader_url: str = ""
    model: str = ""
    fast_model: str = ""
    embedding_model: str = ""
    api_key: str = ""
    base_url: str = ""


def _build_config(data: UIConfigRequest | dict) -> SearchConfig:
    """Build SearchConfig from UI-provided fields and set env vars."""
    if isinstance(data, dict):
        data = UIConfigRequest(**data)
    # Set provider API key as env var for the duration
    if data.api_key:
        # Detect provider from model string and set appropriate env var
        model = data.model or "openai:gpt-4o"
        provider = model.split(":")[0] if ":" in model else "openai"
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "vertex": "GOOGLE_CLOUD_PROJECT",
        }
        env_key = env_map.get(provider, "OPENAI_API_KEY")
        os.environ[env_key] = data.api_key
    if data.base_url:
        os.environ["OPENAI_BASE_URL"] = data.base_url
    return SearchConfig(
        serper_api_key=data.serper_api_key,
        jina_api_key=data.jina_api_key,
        searxng_url=data.searxng_url or "",
        jina_reader_url=data.jina_reader_url or "",
        model=data.model or "",
        fast_model=data.fast_model or "",
        embedding_model=data.embedding_model or "",
    )


def _clean_stale_configs() -> None:
    """Remove config entries older than TTL."""
    now = time.time()
    stale = [k for k, (_, ts) in _config_cache.items() if now - ts > _CONFIG_TTL]
    for k in stale:
        del _config_cache[k]


@app.post("/api/config/{session_id}")
async def set_session_config(session_id: str, body: UIConfigRequest):
    """Cache config for a session (used before SSE stream)."""
    _clean_stale_configs()
    cfg = _build_config(body)
    _config_cache[session_id] = (cfg, time.time())
    return {"status": "ok", "session_id": session_id}
```

- [ ] **Step 2: Add `POST /api/search` endpoint**

```python
@app.post("/api/search")
async def ui_search_endpoint(body: UISearchRequest):
    """Search endpoint for the UI — accepts per-request config."""
    _log.info("ui search q=%r mode=%s session=%s", body.query, body.mode, body.session_id)
    cfg = _build_config(body.config)
    conversation = _get_conversation(body.session_id)

    result = await run_search_pipeline(
        query=body.query,
        chat_history=conversation.turns,
        mode=body.mode,
        config=cfg,
    )

    conversation.add_turn(body.query, result.answer)
    return result.model_dump()
```

- [ ] **Step 3: Add `GET /api/search/stream` SSE endpoint**

```python
@app.get("/api/search/stream")
async def ui_stream_endpoint(
    q: str = Query(...),
    mode: str = Query("balanced"),
    session_id: str = Query("default"),
):
    """SSE streaming search for the UI. Config must be set via POST /api/config/{session_id} first."""
    _log.info("ui stream q=%r mode=%s session=%s", q, mode, session_id)

    cfg_entry = _config_cache.get(session_id)
    if cfg_entry:
        cfg = cfg_entry[0]
        # Refresh timestamp
        _config_cache[session_id] = (cfg, time.time())
        # Re-set env vars (they may have been overwritten by another session)
        _build_config(UIConfigRequest(
            serper_api_key=cfg.serper_api_key,
            jina_api_key=cfg.jina_api_key,
            searxng_url=cfg.searxng_url,
            jina_reader_url=cfg.jina_reader_url,
            model=cfg.model,
            fast_model=cfg.fast_model,
            embedding_model=cfg.embedding_model,
        ))
    else:
        cfg = SearchConfig()

    conversation = _get_conversation(session_id)
    history = conversation.turns

    async def event_stream() -> AsyncIterator[str]:
        from .types import PipelineEvent, SearchResponse
        from exo.types import TextEvent

        answer_parts: list[str] = []
        sources_data: list[dict] = []
        suggestions_data: list[str] = []

        async for event in stream_search_pipeline(
            query=q,
            chat_history=history,
            mode=mode,
            config=cfg,
        ):
            if isinstance(event, PipelineEvent):
                yield f"event: status\ndata: {json.dumps({'stage': event.stage, 'status': event.status, 'message': event.message})}\n\n"
            elif isinstance(event, TextEvent):
                answer_parts.append(event.text)
                yield f"event: token\ndata: {json.dumps({'text': event.text})}\n\n"
            elif isinstance(event, SearchResponse):
                sources_data = [
                    {"title": s.title, "url": s.url, "content": s.content}
                    for s in event.sources
                ]
                suggestions_data = event.suggestions
                yield f"event: sources\ndata: {json.dumps({'sources': sources_data})}\n\n"
                yield f"event: suggestions\ndata: {json.dumps({'suggestions': suggestions_data})}\n\n"
                yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"

                # Save to conversation
                answer = "".join(answer_parts)
                conversation.add_turn(q, answer)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

- [ ] **Step 4: Add `DELETE /api/search/{session_id}` and static file mount**

```python
@app.delete("/api/search/{session_id}")
async def ui_clear_session(session_id: str):
    """Clear conversation and config for a session."""
    if session_id in _conversations:
        _conversations[session_id].clear()
    if session_id in _config_cache:
        del _config_cache[session_id]
    return {"status": "cleared", "session_id": session_id}


# Mount static files LAST (catch-all)
_ui_dir = Path(__file__).resolve().parent.parent.parent.parent / "ui"
if _ui_dir.exists():
    app.mount("/", StaticFiles(directory=str(_ui_dir), html=True), name="ui")
```

Note: The `_ui_dir` path resolves from `src/exo/search/server.py` → up 4 levels to `packages/exo-search/` → then into `ui/`. This works for both development (`uv run`) and Docker.

- [ ] **Step 5: Test the server starts and serves the UI**

```bash
cd /home/atg/Github/orbiter-ai
uv run python -c "
from pathlib import Path
ui = Path('packages/exo-search/src/exo/search/server.py').resolve().parent.parent.parent.parent / 'ui'
print(f'UI dir: {ui}')
print(f'Exists: {ui.exists()}')
"
```

Expected: `UI dir: .../packages/exo-search/ui` and `Exists: True` (after Tasks 1+2 are done).

- [ ] **Step 6: Commit**

```bash
git add packages/exo-search/src/exo/search/server.py
git commit -m "feat(exo-search): add UI API endpoints and static file serving"
```

---

### Task 4: Frontend JavaScript (`ui/app.js`)

**Files:**
- Create: `packages/exo-search/ui/app.js`

All interactivity: settings modal, theme toggle, search submission, SSE streaming, answer rendering, chat history.

- [ ] **Step 1: Create `app.js` with settings management (localStorage, modal open/close, radio toggles)**

```javascript
// packages/exo-search/ui/app.js

// === Settings ===
const SETTINGS_KEY = 'exo-search-settings';

function loadSettings() {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY)) || {};
  } catch { return {}; }
}

function saveSettings(settings) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

function getConfig() {
  const s = loadSettings();
  const config = {};
  if (s.searchBackend === 'serper') config.serper_api_key = s.serperApiKey || '';
  else config.searxng_url = s.searxngUrl || '';
  if (s.enrichment === 'jina-cloud') config.jina_api_key = s.jinaApiKey || '';
  else config.jina_reader_url = s.jinaReaderUrl || '';
  if (s.model) config.model = s.model;
  if (s.fastModel) config.fast_model = s.fastModel;
  if (s.embeddingModel) config.embedding_model = s.embeddingModel;
  if (s.llmApiKey) config.api_key = s.llmApiKey;
  if (s.baseUrl) config.base_url = s.baseUrl;
  return config;
}

// Modal logic
const modal = document.getElementById('settingsModal');
function openSettings() { modal.classList.add('open'); populateSettingsForm(); }
function closeSettings() { modal.classList.remove('open'); }

document.getElementById('settingsBtn').addEventListener('click', openSettings);
document.getElementById('settingsBtnSidebar').addEventListener('click', openSettings);
document.getElementById('settingsCancel').addEventListener('click', closeSettings);
modal.addEventListener('click', (e) => { if (e.target === modal) closeSettings(); });

function populateSettingsForm() {
  const s = loadSettings();
  document.getElementById('serperApiKey').value = s.serperApiKey || '';
  document.getElementById('searxngUrl').value = s.searxngUrl || '';
  document.getElementById('jinaApiKey').value = s.jinaApiKey || '';
  document.getElementById('jinaReaderUrl').value = s.jinaReaderUrl || '';
  document.getElementById('modelInput').value = s.model || '';
  document.getElementById('fastModelInput').value = s.fastModel || '';
  document.getElementById('llmApiKey').value = s.llmApiKey || '';
  document.getElementById('baseUrl').value = s.baseUrl || '';
  document.getElementById('embeddingModel').value = s.embeddingModel || '';

  // Set radio states
  setRadioActive('searchBackendToggle', s.searchBackend || 'serper');
  setRadioActive('enrichmentToggle', s.enrichment || 'jina-cloud');
  toggleFieldVisibility('searchBackendToggle');
  toggleFieldVisibility('enrichmentToggle');
}

function setRadioActive(groupId, value) {
  const group = document.getElementById(groupId);
  group.querySelectorAll('.radio-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.value === value);
  });
}

// Radio toggle handlers
document.querySelectorAll('.radio-group').forEach(group => {
  group.addEventListener('click', (e) => {
    const btn = e.target.closest('.radio-btn');
    if (!btn) return;
    group.querySelectorAll('.radio-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    toggleFieldVisibility(group.id);
  });
});

function toggleFieldVisibility(groupId) {
  const group = document.getElementById(groupId);
  const active = group.querySelector('.radio-btn.active')?.dataset.value;
  if (groupId === 'searchBackendToggle') {
    document.getElementById('serperFields').style.display = active === 'serper' ? '' : 'none';
    document.getElementById('searxngFields').style.display = active === 'searxng' ? '' : 'none';
  } else if (groupId === 'enrichmentToggle') {
    document.getElementById('jinaCloudFields').style.display = active === 'jina-cloud' ? '' : 'none';
    document.getElementById('jinaSelfFields').style.display = active === 'jina-self' ? '' : 'none';
  }
}

// Save handler
document.getElementById('settingsSave').addEventListener('click', () => {
  const searchBackend = document.querySelector('#searchBackendToggle .radio-btn.active')?.dataset.value;
  const enrichment = document.querySelector('#enrichmentToggle .radio-btn.active')?.dataset.value;
  saveSettings({
    searchBackend,
    enrichment,
    serperApiKey: document.getElementById('serperApiKey').value,
    searxngUrl: document.getElementById('searxngUrl').value,
    jinaApiKey: document.getElementById('jinaApiKey').value,
    jinaReaderUrl: document.getElementById('jinaReaderUrl').value,
    model: document.getElementById('modelInput').value,
    fastModel: document.getElementById('fastModelInput').value,
    llmApiKey: document.getElementById('llmApiKey').value,
    baseUrl: document.getElementById('baseUrl').value,
    embeddingModel: document.getElementById('embeddingModel').value,
  });
  closeSettings();
});

// Auto-open on first visit
if (!localStorage.getItem(SETTINGS_KEY)) {
  openSettings();
}
```

- [ ] **Step 2: Add theme toggle logic**

```javascript
// === Theme ===
function getTheme() {
  return document.documentElement.getAttribute('data-theme') || 'light';
}

function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('theme', theme);
  updateThemeIcons();
}

function toggleTheme() {
  setTheme(getTheme() === 'light' ? 'dark' : 'light');
}

function updateThemeIcons() {
  const icon = getTheme() === 'light' ? '☀' : '☾';
  document.getElementById('themeToggle').textContent = icon;
  document.getElementById('themeToggleSidebar').textContent = icon;
}

document.getElementById('themeToggle').addEventListener('click', toggleTheme);
document.getElementById('themeToggleSidebar').addEventListener('click', toggleTheme);
updateThemeIcons();
```

- [ ] **Step 3: Add mode selector logic and state management**

```javascript
// === Mode ===
let currentMode = 'balanced';

function initModeSelectors() {
  document.querySelectorAll('.mode-selector').forEach(selector => {
    selector.addEventListener('click', (e) => {
      const btn = e.target.closest('.mode-btn');
      if (!btn) return;
      currentMode = btn.dataset.mode;
      // Sync all mode selectors
      document.querySelectorAll('.mode-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.mode === currentMode);
      });
    });
  });
}
initModeSelectors();

// === UI State ===
let currentSessionId = null;

function showLanding() {
  document.getElementById('landingView').style.display = '';
  document.getElementById('resultsView').classList.remove('active');
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('landingInput').value = '';
  document.getElementById('landingInput').focus();
  currentSessionId = null;
}

function showResults(query) {
  document.getElementById('landingView').style.display = 'none';
  document.getElementById('resultsView').classList.add('active');
  document.getElementById('sidebar').classList.add('open');
  document.getElementById('queryTitle').textContent = query;
  document.getElementById('followUpInput').value = '';
}
```

- [ ] **Step 4: Add chat history management (localStorage)**

```javascript
// === Chat History ===
const HISTORY_KEY = 'exo-search-history';

function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
  catch { return []; }
}

function saveHistory(history) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

function addToHistory(sessionId, query, answer, sources, suggestions, mode) {
  const history = loadHistory();
  let session = history.find(h => h.id === sessionId);
  if (!session) {
    session = {
      id: sessionId,
      title: query.slice(0, 60),
      messages: [],
      mode,
      created_at: new Date().toISOString(),
    };
    history.unshift(session);
  }
  session.messages.push(
    { role: 'user', content: query },
    { role: 'assistant', content: answer, sources, suggestions }
  );
  saveHistory(history);
  renderHistoryList();
}

function renderHistoryList() {
  const list = document.getElementById('historyList');
  const history = loadHistory();
  list.innerHTML = '';

  const now = new Date();
  const today = now.toDateString();
  const yesterday = new Date(now - 86400000).toDateString();

  let currentGroup = '';
  history.forEach(session => {
    const date = new Date(session.created_at);
    let group;
    if (date.toDateString() === today) group = 'Today';
    else if (date.toDateString() === yesterday) group = 'Yesterday';
    else group = 'Older';

    if (group !== currentGroup) {
      currentGroup = group;
      const label = document.createElement('div');
      label.className = 'history-group-label';
      label.textContent = group;
      list.appendChild(label);
    }

    const item = document.createElement('div');
    item.className = 'history-item' + (session.id === currentSessionId ? ' active' : '');
    item.textContent = session.title;
    item.addEventListener('click', () => loadSession(session.id));
    list.appendChild(item);
  });
}
```

- [ ] **Step 5: Add session loading (restore a previous conversation)**

```javascript
function loadSession(sessionId) {
  const history = loadHistory();
  const session = history.find(h => h.id === sessionId);
  if (!session) return;

  currentSessionId = sessionId;
  const firstQuery = session.messages[0]?.content || '';
  showResults(firstQuery);

  const container = document.getElementById('answerContent');
  container.innerHTML = '';

  // Replay all turns
  for (let i = 0; i < session.messages.length; i += 2) {
    const userMsg = session.messages[i];
    const assistantMsg = session.messages[i + 1];
    if (!assistantMsg) break;

    // Turn divider for follow-ups
    if (i > 0) {
      const divider = document.createElement('div');
      divider.className = 'turn-divider';
      divider.textContent = userMsg.content;
      container.appendChild(divider);
    }

    // Source cards
    if (assistantMsg.sources?.length) {
      container.appendChild(renderSourceCards(assistantMsg.sources));
    }

    // Answer prose
    const prose = document.createElement('div');
    prose.className = 'answer-prose';
    prose.innerHTML = renderAnswer(assistantMsg.content, assistantMsg.sources || []);
    container.appendChild(prose);

    // Related section
    if (assistantMsg.suggestions?.length) {
      container.appendChild(renderRelatedSection(assistantMsg.suggestions));
    }
  }

  renderHistoryList();
}
```

- [ ] **Step 6: Add answer rendering helpers (source cards, citations, markdown, related)**

```javascript
// Source colors cycle
const SOURCE_COLORS = ['#6287f5', '#f76f53', '#63f78b', '#6287f5', '#f76f53', '#63f78b'];

function renderSourceCards(sources) {
  const row = document.createElement('div');
  row.className = 'source-cards';
  sources.forEach((source, i) => {
    const card = document.createElement('a');
    card.className = 'source-card';
    card.href = source.url;
    card.target = '_blank';
    card.rel = 'noopener';
    card.id = `source-${i + 1}`;

    const domain = new URL(source.url).hostname.replace('www.', '');
    const color = SOURCE_COLORS[i % SOURCE_COLORS.length];

    card.innerHTML = `
      <span class="source-num" style="background:${color}">${i + 1}</span>
      <img class="favicon" src="https://www.google.com/s2/favicons?domain=${domain}&sz=16" alt="">
      <span class="source-domain">${domain}</span>
      <span class="source-title">${source.title}</span>
    `;
    row.appendChild(card);
  });
  return row;
}

function renderAnswer(markdown, sources) {
  // Render markdown
  let html = marked.parse(markdown);
  // Replace [N] citation patterns with styled badges
  html = html.replace(/\[(\d+)\]/g, (match, num) => {
    const idx = parseInt(num) - 1;
    const source = sources[idx];
    const url = source ? source.url : '#';
    return `<a class="citation" href="${url}" target="_blank" rel="noopener" data-source="${num}" title="${source?.title || ''}">${num}</a>`;
  });
  return html;
}

function renderRelatedSection(suggestions) {
  const section = document.createElement('div');
  section.className = 'related-section';
  section.innerHTML = '<div class="related-label">Related</div>';
  suggestions.forEach(text => {
    const item = document.createElement('div');
    item.className = 'related-item';
    item.innerHTML = `<span>${text}</span><span class="arrow">→</span>`;
    item.addEventListener('click', () => submitSearch(text));
    section.appendChild(item);
  });
  return section;
}
```

- [ ] **Step 7: Add search submission and SSE streaming logic**

```javascript
// === Search ===
let activeEventSource = null;

async function submitSearch(query) {
  if (!query.trim()) return;
  if (activeEventSource) { activeEventSource.close(); activeEventSource = null; }

  // Create or reuse session
  if (!currentSessionId) {
    currentSessionId = crypto.randomUUID();
  }

  showResults(query);

  const container = document.getElementById('answerContent');

  // Add turn divider if follow-up
  if (container.children.length > 0) {
    const divider = document.createElement('div');
    divider.className = 'turn-divider';
    divider.textContent = query;
    container.appendChild(divider);
    document.getElementById('queryTitle').textContent = query;
  }

  // Pipeline status indicator
  const status = document.createElement('div');
  status.className = 'pipeline-status';
  status.innerHTML = '<span class="dot"></span><span id="statusText">Starting...</span>';
  container.appendChild(status);

  // Placeholders for streaming content
  const sourceCardsEl = document.createElement('div');
  const proseEl = document.createElement('div');
  proseEl.className = 'answer-prose';
  const relatedEl = document.createElement('div');

  // Push config first
  const config = getConfig();
  await fetch(`/api/config/${currentSessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });

  // Open SSE stream
  const params = new URLSearchParams({
    q: query,
    mode: currentMode,
    session_id: currentSessionId,
  });

  const es = new EventSource(`/api/search/stream?${params}`);
  activeEventSource = es;
  let answerText = '';
  let sourcesData = [];
  let suggestionsData = [];

  es.addEventListener('status', (e) => {
    const data = JSON.parse(e.data);
    const stage = data.stage;
    const s = data.status;
    const msg = data.message ? ` — ${data.message}` : '';
    document.getElementById('statusText').textContent =
      s === 'started' ? `${capitalize(stage)}...` : `${capitalize(stage)} done${msg}`;
  });

  es.addEventListener('token', (e) => {
    const data = JSON.parse(e.data);
    answerText += data.text;
    // Remove status indicator once writing starts
    if (status.parentNode) {
      status.remove();
      container.appendChild(sourceCardsEl);
      container.appendChild(proseEl);
    }
    proseEl.innerHTML = renderAnswer(answerText, sourcesData);
    // Auto-scroll
    const area = document.getElementById('answerArea');
    area.scrollTop = area.scrollHeight;
  });

  es.addEventListener('sources', (e) => {
    const data = JSON.parse(e.data);
    sourcesData = data.sources;
    sourceCardsEl.replaceWith(renderSourceCards(sourcesData));
    // Re-render answer with citation links now that sources exist
    proseEl.innerHTML = renderAnswer(answerText, sourcesData);
  });

  es.addEventListener('suggestions', (e) => {
    const data = JSON.parse(e.data);
    suggestionsData = data.suggestions;
    const related = renderRelatedSection(suggestionsData);
    container.appendChild(related);
  });

  es.addEventListener('done', () => {
    es.close();
    activeEventSource = null;
    // Save to history
    addToHistory(currentSessionId, query, answerText, sourcesData, suggestionsData, currentMode);
  });

  es.addEventListener('error', () => {
    es.close();
    activeEventSource = null;
    if (status.parentNode) {
      document.getElementById('statusText').textContent = 'Error — check settings and try again';
      status.querySelector('.dot').style.background = 'var(--zen-coral)';
      status.querySelector('.dot').style.animation = 'none';
    }
  });
}

function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}
```

- [ ] **Step 8: Add form handlers and event wiring**

```javascript
// === Form Handlers ===
document.getElementById('landingSearchBar').addEventListener('submit', (e) => {
  e.preventDefault();
  submitSearch(document.getElementById('landingInput').value);
});

document.getElementById('followUpBar').addEventListener('submit', (e) => {
  e.preventDefault();
  submitSearch(document.getElementById('followUpInput').value);
  document.getElementById('followUpInput').value = '';
});

// Suggestion chips
document.querySelectorAll('.suggestion-chip').forEach(chip => {
  chip.addEventListener('click', () => submitSearch(chip.textContent));
});

// New search
document.getElementById('newSearchBtn').addEventListener('click', showLanding);

// Keyboard shortcut — Escape closes modal
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeSettings();
});

// Init
renderHistoryList();
```

- [ ] **Step 9: Test full flow manually**

```bash
cd /home/atg/Github/orbiter-ai
uv run python -m exo.search --serve --port 8000
```

Open `http://localhost:8000`, verify:
1. Settings modal opens on first visit
2. Can configure and save settings
3. Theme toggle works
4. Search bar submits, SSE streams arrive, answer renders with citations
5. Source cards show with favicons
6. Follow-up suggestions work
7. Chat history appears in sidebar
8. "New Search" returns to landing

- [ ] **Step 10: Commit**

```bash
git add packages/exo-search/ui/app.js
git commit -m "feat(exo-search): add search UI frontend logic"
```

---

### Task 5: Dockerfile & docker-compose.yml

**Files:**
- Create: `packages/exo-search/Dockerfile`
- Create: `packages/exo-search/docker-compose.yml`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
# packages/exo-search/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy full workspace (needed for workspace deps)
COPY . .

# Install all workspace packages
RUN uv sync

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "exo.search", "--serve", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Create docker-compose.yml**

```yaml
# packages/exo-search/docker-compose.yml
services:
  exo-search:
    build:
      context: ../..
      dockerfile: packages/exo-search/Dockerfile
    ports:
      - "8000:8000"
    restart: unless-stopped
```

- [ ] **Step 3: Test Docker build**

```bash
cd /home/atg/Github/orbiter-ai
docker build -f packages/exo-search/Dockerfile -t exo-search .
```

Expected: Builds successfully.

- [ ] **Step 4: Test Docker run**

```bash
docker run -p 8000:8000 exo-search
```

Open `http://localhost:8000` and verify UI loads.

- [ ] **Step 5: Commit**

```bash
git add packages/exo-search/Dockerfile packages/exo-search/docker-compose.yml
git commit -m "feat(exo-search): add Dockerfile and docker-compose"
```

---

### Task 6: End-to-End Verification

- [ ] **Step 1: Start the dev server**

```bash
cd /home/atg/Github/orbiter-ai
uv run python -m exo.search --serve --port 8000
```

- [ ] **Step 2: Verify all spec requirements**

Open `http://localhost:8000` and test:

1. **Landing state:** centered search bar, mode selector, suggestion chips, no sidebar
2. **Settings modal:** auto-opens on first visit, search backend toggle (Serper/SearXNG), enrichment toggle (Jina Cloud/Self-hosted), LLM config fields, save/cancel
3. **Search flow:** submit query → pipeline status indicators animate → source cards appear → answer streams in with markdown → citation badges link to sources → "Related" section appears
4. **Follow-up:** typing in bottom input submits follow-up → turn divider + new answer appends below
5. **Sidebar:** slides in after first search, shows chat history grouped by date, "New Search" returns to landing
6. **Session restore:** click a history item → conversation loads back
7. **Theme toggle:** light/dark switches, persists across reload
8. **Settings persist:** reload page → settings survive in localStorage
9. **Existing endpoints:** `curl http://localhost:8000/search?q=test` still works

- [ ] **Step 3: Verify existing tests still pass**

```bash
cd /home/atg/Github/orbiter-ai
uv run pytest packages/exo-search/tests/ -x -q
```

Expected: All tests pass (no regressions from server.py changes).

- [ ] **Step 4: Run lint**

```bash
uv run ruff check packages/exo-search/src/ --fix
uv run ruff format packages/exo-search/src/
```

- [ ] **Step 5: Final commit if any fixes needed**

```bash
git add -A packages/exo-search/
git commit -m "fix(exo-search): lint fixes for search UI"
```
