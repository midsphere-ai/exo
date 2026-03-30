/* ============================================================
   exo-search — Frontend Application
   Vanilla JS, no framework, no build step.
   ============================================================ */

// ============================================================
// 1. Settings Management
// ============================================================

const SETTINGS_KEY = 'exo-search-settings';

function loadSettings() {
  try { return JSON.parse(localStorage.getItem(SETTINGS_KEY)) || {}; }
  catch { return {}; }
}

function saveSettings(settings) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

function getConfig() {
  const s = loadSettings();
  const config = {};
  if (s.searchBackend === 'serper') config.serper_api_key = s.serperApiKey || '';
  else if (s.searchBackend === 'searxng') config.searxng_url = s.searxngUrl || '';
  if (s.enrichment === 'jina-cloud') config.jina_api_key = s.jinaApiKey || '';
  else if (s.enrichment === 'jina-self') config.jina_reader_url = s.jinaReaderUrl || '';
  if (s.model) config.model = s.model;
  if (s.fastModel) config.fast_model = s.fastModel;
  if (s.embeddingModel) config.embedding_model = s.embeddingModel;
  if (s.llmApiKey) config.api_key = s.llmApiKey;
  if (s.baseUrl) config.base_url = s.baseUrl;
  return config;
}

function openSettings() {
  document.getElementById('settingsModal').classList.add('open');
  populateSettingsForm();
}

function closeSettings() {
  document.getElementById('settingsModal').classList.remove('open');
}

function populateSettingsForm() {
  const s = loadSettings();

  // Text inputs
  document.getElementById('serperApiKey').value = s.serperApiKey || '';
  document.getElementById('searxngUrl').value = s.searxngUrl || '';
  document.getElementById('jinaApiKey').value = s.jinaApiKey || '';
  document.getElementById('jinaReaderUrl').value = s.jinaReaderUrl || '';
  document.getElementById('modelInput').value = s.model || '';
  document.getElementById('fastModelInput').value = s.fastModel || '';
  document.getElementById('llmApiKey').value = s.llmApiKey || '';
  document.getElementById('baseUrl').value = s.baseUrl || '';
  document.getElementById('embeddingModel').value = s.embeddingModel || '';

  // Search backend radio
  const searchBackend = s.searchBackend || 'serper';
  setRadioGroupValue('searchBackendToggle', searchBackend);
  toggleFieldVisibility('searchBackendToggle');

  // Enrichment radio
  const enrichment = s.enrichment || 'jina-cloud';
  setRadioGroupValue('enrichmentToggle', enrichment);
  toggleFieldVisibility('enrichmentToggle');
}

function setRadioGroupValue(groupId, value) {
  const group = document.getElementById(groupId);
  group.querySelectorAll('.pill').forEach(function(btn) {
    if (btn.getAttribute('data-value') === value) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });
}

function toggleFieldVisibility(groupId) {
  const group = document.getElementById(groupId);
  const activeBtn = group.querySelector('.pill.active');
  const activeValue = activeBtn ? activeBtn.getAttribute('data-value') : '';

  if (groupId === 'searchBackendToggle') {
    document.getElementById('serperFields').style.display =
      activeValue === 'serper' ? '' : 'none';
    document.getElementById('searxngFields').style.display =
      activeValue === 'searxng' ? '' : 'none';
  } else if (groupId === 'enrichmentToggle') {
    document.getElementById('jinaCloudFields').style.display =
      activeValue === 'jina-cloud' ? '' : 'none';
    document.getElementById('jinaSelfFields').style.display =
      activeValue === 'jina-self' ? '' : 'none';
  }
}

// Radio group click delegation
document.querySelectorAll('.toggle-pills').forEach(function(group) {
  group.addEventListener('click', function(e) {
    var btn = e.target.closest('.pill');
    if (!btn) return;
    group.querySelectorAll('.pill').forEach(function(b) {
      b.classList.remove('active');
    });
    btn.classList.add('active');
    toggleFieldVisibility(group.id);
  });
});

// Settings button handlers
document.getElementById('settingsBtn').addEventListener('click', openSettings);
document.getElementById('settingsBtnSidebar').addEventListener('click', openSettings);
document.getElementById('settingsCancel').addEventListener('click', closeSettings);
document.getElementById('settingsClose').addEventListener('click', closeSettings);

// Click on overlay (outside modal) closes settings
document.getElementById('settingsModal').addEventListener('click', function(e) {
  if (e.target === this) closeSettings();
});

// Save handler
document.getElementById('settingsSave').addEventListener('click', function() {
  var searchGroup = document.getElementById('searchBackendToggle');
  var searchActive = searchGroup.querySelector('.pill.active');
  var enrichGroup = document.getElementById('enrichmentToggle');
  var enrichActive = enrichGroup.querySelector('.pill.active');

  var settings = {
    searchBackend: searchActive ? searchActive.getAttribute('data-value') : 'serper',
    serperApiKey: document.getElementById('serperApiKey').value,
    searxngUrl: document.getElementById('searxngUrl').value,
    enrichment: enrichActive ? enrichActive.getAttribute('data-value') : 'jina-cloud',
    jinaApiKey: document.getElementById('jinaApiKey').value,
    jinaReaderUrl: document.getElementById('jinaReaderUrl').value,
    model: document.getElementById('modelInput').value,
    fastModel: document.getElementById('fastModelInput').value,
    llmApiKey: document.getElementById('llmApiKey').value,
    baseUrl: document.getElementById('baseUrl').value,
    embeddingModel: document.getElementById('embeddingModel').value,
  };

  saveSettings(settings);
  closeSettings();
});

// Auto-open settings on first visit
if (Object.keys(loadSettings()).length === 0) {
  openSettings();
}

// ============================================================
// 2. Theme Toggle
// ============================================================

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
  var icon = getTheme() === 'light' ? '\u2600' : '\u263E'; // sun or moon
  document.getElementById('themeToggle').textContent = icon;
  document.getElementById('themeToggleSidebar').textContent = icon;
}

document.getElementById('themeToggle').addEventListener('click', toggleTheme);
document.getElementById('themeToggleSidebar').addEventListener('click', toggleTheme);

// Set icons on load
updateThemeIcons();

// ============================================================
// 3. Mode Selector (Dropdown)
// ============================================================

var currentMode = 'balanced';

function syncModeSelects(value) {
  currentMode = value;
  document.getElementById('modeSelect').value = value;
  document.getElementById('modeSelectFollowUp').value = value;
  var labels = { speed: '\u26A1 Speed', balanced: '\u2696 Balanced', quality: '\uD83D\uDD2C Quality' };
  document.getElementById('modeBadge').textContent = labels[value] || value;
}

document.getElementById('modeSelect').addEventListener('change', function() {
  syncModeSelects(this.value);
});
document.getElementById('modeSelectFollowUp').addEventListener('change', function() {
  syncModeSelects(this.value);
});

// ============================================================
// 4. UI State Management
// ============================================================

var currentSessionId = null;

function showLanding() {
  document.getElementById('landingView').style.display = '';
  document.getElementById('resultsView').classList.remove('active');
  document.getElementById('sidebar').classList.remove('open');
  var landingInput = document.getElementById('landingInput');
  landingInput.value = '';
  landingInput.focus();
  currentSessionId = null;
}

function showResults(query) {
  document.getElementById('landingView').style.display = 'none';
  document.getElementById('resultsView').classList.add('active');
  document.getElementById('sidebar').classList.add('open');
  document.getElementById('queryTitle').textContent = query;
  document.getElementById('followUpInput').value = '';
}

// ============================================================
// 5. Chat History (localStorage)
// ============================================================

var HISTORY_KEY = 'exo-search-history';

function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
  catch { return []; }
}

function saveHistory(history) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
}

function addToHistory(sessionId, query, answer, sources, suggestions, mode) {
  var history = loadHistory();
  var session = history.find(function(s) { return s.id === sessionId; });
  if (!session) {
    session = {
      id: sessionId,
      title: query.slice(0, 60),
      messages: [],
      mode: mode,
      created_at: new Date().toISOString(),
    };
    history.unshift(session);
  }
  session.messages.push(
    { role: 'user', content: query },
    { role: 'assistant', content: answer, sources: sources, suggestions: suggestions }
  );
  saveHistory(history);
  renderHistoryList();
}

function renderHistoryList() {
  var container = document.getElementById('historyList');
  container.innerHTML = '';
  var history = loadHistory();
  if (history.length === 0) return;

  var now = new Date();
  var todayStr = now.toDateString();
  var yesterday = new Date(now);
  yesterday.setDate(yesterday.getDate() - 1);
  var yesterdayStr = yesterday.toDateString();

  var groups = { 'Today': [], 'Yesterday': [], 'Older': [] };

  history.forEach(function(session) {
    var d = new Date(session.created_at).toDateString();
    if (d === todayStr) groups['Today'].push(session);
    else if (d === yesterdayStr) groups['Yesterday'].push(session);
    else groups['Older'].push(session);
  });

  ['Today', 'Yesterday', 'Older'].forEach(function(label) {
    var items = groups[label];
    if (items.length === 0) return;

    var groupLabel = document.createElement('div');
    groupLabel.className = 'history-group-label';
    groupLabel.textContent = label;
    container.appendChild(groupLabel);

    items.forEach(function(session) {
      var item = document.createElement('div');
      item.className = 'history-item';
      if (session.id === currentSessionId) item.classList.add('active');
      item.textContent = session.title;
      item.addEventListener('click', function() {
        loadSession(session.id);
      });
      container.appendChild(item);
    });
  });
}

// ============================================================
// 6. Session Loading
// ============================================================

function loadSession(sessionId) {
  var history = loadHistory();
  var session = history.find(function(s) { return s.id === sessionId; });
  if (!session) return;

  currentSessionId = sessionId;
  var firstUserMsg = session.messages.find(function(m) { return m.role === 'user'; });
  showResults(firstUserMsg ? firstUserMsg.content : session.title);

  var container = document.getElementById('answerContent');
  container.innerHTML = '';

  // Replay message pairs (user + assistant turns)
  var turnIndex = 0;
  for (var i = 0; i < session.messages.length; i += 2) {
    var userMsg = session.messages[i];
    var assistantMsg = session.messages[i + 1];
    if (!assistantMsg) break;

    // Turn divider for follow-up turns
    if (turnIndex > 0) {
      var divider = document.createElement('div');
      divider.className = 'turn-divider';
      divider.textContent = userMsg.content;
      container.appendChild(divider);
    }

    // Source cards
    if (assistantMsg.sources && assistantMsg.sources.length > 0) {
      container.appendChild(renderSourceCards(assistantMsg.sources));
    }

    // Answer prose
    var prose = document.createElement('div');
    prose.className = 'answer-prose';
    prose.innerHTML = renderAnswer(assistantMsg.content, assistantMsg.sources || []);
    container.appendChild(prose);

    // Related suggestions
    if (assistantMsg.suggestions && assistantMsg.suggestions.length > 0) {
      container.appendChild(renderRelatedSection(assistantMsg.suggestions));
    }

    turnIndex++;
  }

  renderHistoryList();
}

// ============================================================
// 7. Answer Rendering Helpers
// ============================================================

var SOURCE_COLORS = ['#6287f5', '#f76f53', '#63f78b', '#6287f5', '#f76f53', '#63f78b'];

function renderSourceCards(sources) {
  var row = document.createElement('div');
  row.className = 'source-cards';

  sources.forEach(function(source, idx) {
    var a = document.createElement('a');
    a.className = 'source-card';
    a.href = source.url;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';

    var domain = '';
    try { domain = new URL(source.url).hostname.replace('www.', ''); }
    catch { domain = source.url; }

    var color = SOURCE_COLORS[idx % SOURCE_COLORS.length];

    var num = document.createElement('span');
    num.className = 'source-num';
    num.style.background = color;
    num.textContent = String(idx + 1);

    var favicon = document.createElement('img');
    favicon.className = 'favicon';
    favicon.src = 'https://www.google.com/s2/favicons?domain=' + domain + '&sz=16';
    favicon.alt = '';

    var domainSpan = document.createElement('span');
    domainSpan.className = 'source-domain';
    domainSpan.textContent = domain;

    var title = document.createElement('span');
    title.className = 'source-title';
    title.textContent = source.title || domain;

    a.appendChild(num);
    a.appendChild(favicon);
    a.appendChild(domainSpan);
    a.appendChild(title);
    row.appendChild(a);
  });

  return row;
}

function renderAnswer(markdown, sources) {
  if (!markdown) return '';
  var html = marked.parse(markdown);

  // Replace [N] citation patterns with linked badges
  html = html.replace(/\[(\d+)\]/g, function(match, numStr) {
    var num = parseInt(numStr, 10);
    var url = '#';
    if (sources && sources[num - 1] && sources[num - 1].url) {
      url = sources[num - 1].url;
    }
    return '<a class="citation" href="' + url + '" target="_blank" data-source="' + num + '">' + num + '</a>';
  });

  return html;
}

function renderRelatedSection(suggestions) {
  var section = document.createElement('div');
  section.className = 'related-section';

  var label = document.createElement('div');
  label.className = 'related-label';
  label.textContent = 'Related';
  section.appendChild(label);

  var list = document.createElement('div');
  list.className = 'related-list';

  suggestions.forEach(function(text) {
    var item = document.createElement('div');
    item.className = 'related-item';

    var textSpan = document.createElement('span');
    textSpan.textContent = text;

    var arrow = document.createElement('span');
    arrow.className = 'arrow';
    arrow.textContent = '\u2192';

    item.appendChild(textSpan);
    item.appendChild(arrow);

    item.addEventListener('click', function() {
      submitSearch(text);
    });

    list.appendChild(item);
  });

  section.appendChild(list);
  return section;
}

// ============================================================
// 8. Search Submission & SSE Streaming
// ============================================================

var activeEventSource = null;

async function submitSearch(query) {
  if (!query.trim()) return;
  if (activeEventSource) { activeEventSource.close(); activeEventSource = null; }

  if (!currentSessionId) currentSessionId = crypto.randomUUID();

  showResults(query);

  var container = document.getElementById('answerContent');

  // Turn divider for follow-ups
  if (container.children.length > 0) {
    var divider = document.createElement('div');
    divider.className = 'turn-divider';
    divider.textContent = query;
    container.appendChild(divider);
    document.getElementById('queryTitle').textContent = query;
  }

  // Pipeline status indicator
  var status = document.createElement('div');
  status.className = 'pipeline-status';
  status.innerHTML = '<span class="dot"></span><span id="statusText">Starting...</span>';
  container.appendChild(status);

  // Placeholders for source cards and answer prose
  var sourceCardsEl = document.createElement('div');
  var proseEl = document.createElement('div');
  proseEl.className = 'answer-prose';

  // Push config first
  var config = getConfig();
  await fetch('/api/config/' + currentSessionId, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });

  // Open SSE stream
  var params = new URLSearchParams({ q: query, mode: currentMode, session_id: currentSessionId });
  var es = new EventSource('/api/search/stream?' + params.toString());
  activeEventSource = es;
  var answerText = '';
  var sourcesData = [];
  var suggestionsData = [];

  es.addEventListener('status', function(e) {
    var data = JSON.parse(e.data);
    var msg = data.message ? ' \u2014 ' + data.message : '';
    var statusEl = document.getElementById('statusText');
    if (statusEl) {
      statusEl.textContent = data.status === 'started'
        ? capitalize(data.stage) + '...'
        : capitalize(data.stage) + ' done' + msg;
    }
  });

  es.addEventListener('token', function(e) {
    var data = JSON.parse(e.data);
    answerText += data.text;
    // Remove status, insert source cards + prose on first token
    if (status.parentNode) {
      status.remove();
      container.appendChild(sourceCardsEl);
      container.appendChild(proseEl);
    }
    proseEl.innerHTML = renderAnswer(answerText, sourcesData);
    // Auto-scroll
    var answerArea = document.getElementById('answerArea');
    answerArea.scrollTop = answerArea.scrollHeight;
  });

  es.addEventListener('sources', function(e) {
    var data = JSON.parse(e.data);
    sourcesData = data.sources;
    var newCards = renderSourceCards(sourcesData);
    sourceCardsEl.replaceWith(newCards);
    sourceCardsEl = newCards;
    // Re-render answer with citation links now that sources are available
    proseEl.innerHTML = renderAnswer(answerText, sourcesData);
  });

  es.addEventListener('suggestions', function(e) {
    var data = JSON.parse(e.data);
    suggestionsData = data.suggestions;
    container.appendChild(renderRelatedSection(suggestionsData));
  });

  es.addEventListener('done', function() {
    es.close();
    activeEventSource = null;
    addToHistory(currentSessionId, query, answerText, sourcesData, suggestionsData, currentMode);
  });

  es.addEventListener('error', function() {
    es.close();
    activeEventSource = null;
    var statusEl = document.getElementById('statusText');
    if (statusEl) {
      statusEl.textContent = 'Error \u2014 check settings and try again';
    }
    var dot = status.querySelector('.dot');
    if (dot) {
      dot.style.background = 'var(--zen-coral)';
      dot.style.animation = 'none';
    }
  });
}

function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// ============================================================
// 9. Form Handlers & Event Wiring
// ============================================================

// Landing search form
document.getElementById('landingSearchBar').addEventListener('submit', function(e) {
  e.preventDefault();
  submitSearch(document.getElementById('landingInput').value);
});

// Follow-up search form
document.getElementById('followUpBar').addEventListener('submit', function(e) {
  e.preventDefault();
  var input = document.getElementById('followUpInput');
  submitSearch(input.value);
  input.value = '';
});

// Suggestion chips
document.querySelectorAll('.suggestion-chip').forEach(function(chip) {
  chip.addEventListener('click', function() {
    submitSearch(chip.textContent);
  });
});

// New search button
document.getElementById('newSearchBtn').addEventListener('click', showLanding);

// Escape key closes settings
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeSettings();
});

// Render history on load
renderHistoryList();
