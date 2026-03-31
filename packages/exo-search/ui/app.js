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
document.getElementById('settingsCancel').addEventListener('click', closeSettings);
document.getElementById('settingsClose').addEventListener('click', closeSettings);

// Password visibility toggles
document.querySelectorAll('.password-toggle').forEach(function(btn) {
  btn.addEventListener('click', function() {
    var input = document.getElementById(btn.getAttribute('data-target'));
    if (input.type === 'password') {
      input.type = 'text';
      btn.classList.add('visible');
    } else {
      input.type = 'password';
      btn.classList.remove('visible');
    }
  });
});

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
  var sidebarToggle = document.getElementById('themeToggleSidebar');
  if (sidebarToggle) sidebarToggle.textContent = icon;
}

document.getElementById('themeToggle').addEventListener('click', toggleTheme);
var sidebarThemeBtn = document.getElementById('themeToggleSidebar');
if (sidebarThemeBtn) sidebarThemeBtn.addEventListener('click', toggleTheme);

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
  var labels = { speed: '\u26A1 Speed', balanced: '\u2696 Balanced', quality: '\uD83D\uDD2C Quality', deep: '\uD83E\uDDE0 Deep' };
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

function cancelActiveStream() {
  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }
  savePartialResult();
}

function showLanding() {
  cancelActiveStream();
  document.getElementById('landingView').style.display = '';
  document.getElementById('resultsView').classList.remove('active');
  // Keep sidebar open if there's history, otherwise hide it
  var hasHistory = loadHistory().length > 0;
  if (hasHistory) {
    document.getElementById('sidebar').classList.add('open');
    renderHistoryList();
  } else {
    document.getElementById('sidebar').classList.remove('open');
  }
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
  cancelActiveStream();
  var history = loadHistory();
  var session = history.find(function(s) { return s.id === sessionId; });
  if (!session) return;

  currentSessionId = sessionId;
  showResults('');

  var container = document.getElementById('answerContent');
  container.innerHTML = '';

  // Replay message pairs (user + assistant turns)
  for (var i = 0; i < session.messages.length; i += 2) {
    var userMsg = session.messages[i];
    var assistantMsg = session.messages[i + 1];
    if (!assistantMsg) break;

    // User query heading
    var queryEl = document.createElement('div');
    queryEl.className = 'user-query';
    queryEl.textContent = userMsg.content;
    container.appendChild(queryEl);

    // Source cards (collapsible)
    if (assistantMsg.sources && assistantMsg.sources.length > 0) {
      container.appendChild(renderSourcesSection(assistantMsg.sources));
    }

    // Answer prose
    var prose = document.createElement('div');
    prose.className = 'answer-prose';
    prose.innerHTML = renderAnswer(assistantMsg.content, assistantMsg.sources || []);
    container.appendChild(prose);

    // Only show related suggestions for the LAST turn
    if (i + 2 >= session.messages.length && assistantMsg.suggestions && assistantMsg.suggestions.length > 0) {
      container.appendChild(renderRelatedSection(assistantMsg.suggestions));
    }
  }

  renderHistoryList();
}

// ============================================================
// 7. Answer Rendering Helpers
// ============================================================

var SOURCE_COLORS = ['#6287f5', '#f76f53', '#63f78b', '#6287f5', '#f76f53', '#63f78b'];

function renderSourcesSection(sources) {
  var section = document.createElement('div');
  section.className = 'sources-section';

  // Toggle button with count
  var toggle = document.createElement('button');
  toggle.className = 'sources-toggle';
  toggle.innerHTML = '<span class="sources-count">' + sources.length + ' sources</span><span class="chevron">▼</span>';
  toggle.addEventListener('click', function() {
    section.classList.toggle('open');
  });
  section.appendChild(toggle);

  // Source cards (hidden until expanded)
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

    a.appendChild(num);
    a.appendChild(favicon);
    a.appendChild(domainSpan);
    row.appendChild(a);
  });

  section.appendChild(row);
  return section;
}

function renderAnswer(markdown, sources) {
  if (!markdown) return '';

  // Strip the raw "Sources" / "References" section from the bottom —
  // we already show sources in the collapsible UI
  var cleaned = markdown.replace(/\n+#{1,3}\s*(Sources|References)\s*\n[\s\S]*$/i, '');

  var html = marked.parse(cleaned);

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

function renderConfidenceBadge(score) {
  var badge = document.createElement('div');
  badge.className = 'confidence-badge';

  var level, label;
  if (score >= 0.7) {
    level = 'high';
    label = 'High confidence';
  } else if (score >= 0.4) {
    level = 'medium';
    label = 'Medium confidence';
  } else {
    level = 'low';
    label = 'Low confidence';
  }

  badge.classList.add('confidence-' + level);
  var pct = Math.round(score * 100);
  badge.innerHTML = '<span class="confidence-dot"></span>'
    + '<span class="confidence-label">' + label + '</span>'
    + '<span class="confidence-score">' + pct + '%</span>';

  return badge;
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
// 7b. Contradiction Rendering
// ============================================================

function renderContradictions(contradictions) {
  var section = document.createElement('div');
  section.className = 'contradictions-callout';

  var header = document.createElement('div');
  header.className = 'contradictions-header';
  header.innerHTML = '<span class="contradictions-icon">&#9888;</span>'
    + '<span class="contradictions-title">Conflicting information detected</span>';
  section.appendChild(header);

  var desc = document.createElement('div');
  desc.className = 'contradictions-desc';
  desc.textContent = 'Some sources present different facts. Review the details below.';
  section.appendChild(desc);

  contradictions.forEach(function(c) {
    var item = document.createElement('div');
    item.className = 'contradiction-item severity-' + (c.severity || 'moderate');

    var claim = document.createElement('div');
    claim.className = 'contradiction-claim';
    claim.textContent = c.claim;
    item.appendChild(claim);

    var positions = document.createElement('div');
    positions.className = 'contradiction-positions';

    var posA = document.createElement('div');
    posA.className = 'contradiction-pos pos-a';
    var srcA = c.sources_a && c.sources_a.length > 0
      ? ' (Source' + (c.sources_a.length > 1 ? 's ' : ' ') + c.sources_a.join(', ') + ')'
      : '';
    posA.innerHTML = '<strong>Position A' + srcA + ':</strong> ' + c.position_a;

    var posB = document.createElement('div');
    posB.className = 'contradiction-pos pos-b';
    var srcB = c.sources_b && c.sources_b.length > 0
      ? ' (Source' + (c.sources_b.length > 1 ? 's ' : ' ') + c.sources_b.join(', ') + ')'
      : '';
    posB.innerHTML = '<strong>Position B' + srcB + ':</strong> ' + c.position_b;

    positions.appendChild(posA);
    positions.appendChild(posB);
    item.appendChild(positions);
    section.appendChild(item);
  });

  return section;
}

// ============================================================
// 8. Search Submission & SSE Streaming
// ============================================================

var activeEventSource = null;
var pendingResult = null; // tracks in-progress search for saving on switch

function savePartialResult() {
  if (pendingResult && pendingResult.answerText) {
    addToHistory(
      pendingResult.sessionId,
      pendingResult.query,
      pendingResult.answerText,
      pendingResult.sourcesData,
      pendingResult.suggestionsData,
      pendingResult.mode
    );
    pendingResult = null;
  }
}

async function submitSearch(query) {
  if (!query.trim()) return;
  // Save any in-progress result before starting a new search
  savePartialResult();
  cancelActiveStream();

  if (!currentSessionId) currentSessionId = crypto.randomUUID();

  showResults('');

  var container = document.getElementById('answerContent');

  // Remove previous related section (only latest turn gets suggestions)
  var oldRelated = container.querySelector('.related-section');
  if (oldRelated) oldRelated.remove();

  // User query heading
  var queryEl = document.createElement('div');
  queryEl.className = 'user-query';
  queryEl.textContent = query;
  container.appendChild(queryEl);

  // Pipeline tracker (Perplexity-style)
  var tracker = document.createElement('div');
  tracker.className = 'pipeline-tracker';
  var isDeep = currentMode === 'deep';
  var stages = isDeep ? {
    classifier: { label: 'Understanding query', icon: '?' },
    deep_research: { label: 'Sequential research', icon: '\uD83E\uDDE0' },
    enrichment: { label: 'Reading sources', icon: '◉' },
    writer: { label: 'Writing answer', icon: '✎' },
  } : {
    classifier: { label: 'Understanding query', icon: '?' },
    researcher: { label: 'Searching the web', icon: '⌕' },
    enrichment: { label: 'Reading sources', icon: '◉' },
    writer: { label: 'Writing answer', icon: '✎' },
  };
  Object.keys(stages).forEach(function(key) {
    var step = document.createElement('div');
    step.className = 'pipeline-step';
    step.id = 'step-' + key;
    step.innerHTML = '<div class="pipeline-step-icon">' + stages[key].icon + '</div>'
      + '<div><div class="pipeline-step-label">' + stages[key].label + '</div>'
      + '<div class="pipeline-step-detail" id="detail-' + key + '"></div></div>';
    tracker.appendChild(step);
  });
  container.appendChild(tracker);

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
  var searchSessionId = currentSessionId; // capture for closure
  var params = new URLSearchParams({ q: query, mode: currentMode, session_id: searchSessionId });
  var es = new EventSource('/api/search/stream?' + params.toString());
  activeEventSource = es;
  var answerText = '';
  var sourcesData = [];
  var suggestionsData = [];

  // Track in-progress result so it can be saved if user switches away
  pendingResult = {
    sessionId: searchSessionId,
    query: query,
    answerText: '',
    sourcesData: [],
    suggestionsData: [],
    mode: currentMode,
  };

  es.addEventListener('status', function(e) {
    var data = JSON.parse(e.data);
    var stepEl = document.getElementById('step-' + data.stage);
    var detailEl = document.getElementById('detail-' + data.stage);
    if (data.status === 'started') {
      if (stepEl) stepEl.className = 'pipeline-step active';
    } else {
      // completed
      if (stepEl) stepEl.className = 'pipeline-step done';
      if (stepEl) {
        var icon = stepEl.querySelector('.pipeline-step-icon');
        if (icon) icon.textContent = '✓';
      }
      if (detailEl && data.message) detailEl.textContent = data.message;
    }
  });

  es.addEventListener('token', function(e) {
    var data = JSON.parse(e.data);
    answerText += data.text;
    if (pendingResult) pendingResult.answerText = answerText;
    // Remove tracker, insert source cards + prose on first token
    if (tracker.parentNode) {
      tracker.remove();
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
    if (pendingResult) pendingResult.sourcesData = sourcesData;
    var newSection = renderSourcesSection(sourcesData);
    sourceCardsEl.replaceWith(newSection);
    sourceCardsEl = newSection;
    proseEl.innerHTML = renderAnswer(answerText, sourcesData);
  });

  es.addEventListener('suggestions', function(e) {
    var data = JSON.parse(e.data);
    suggestionsData = data.suggestions;
    if (pendingResult) pendingResult.suggestionsData = suggestionsData;
    container.appendChild(renderRelatedSection(suggestionsData));
  });

  // Verified answer replaces streamed text (citations may have been removed)
  es.addEventListener('answer', function(e) {
    var data = JSON.parse(e.data);
    answerText = data.answer;
    if (pendingResult) pendingResult.answerText = answerText;
    proseEl.innerHTML = renderAnswer(answerText, sourcesData);
    // Render confidence badge if present
    if (data.confidence != null) {
      var existingBadge = proseEl.parentNode.querySelector('.confidence-badge');
      if (existingBadge) existingBadge.remove();
      var badge = renderConfidenceBadge(data.confidence);
      proseEl.parentNode.insertBefore(badge, proseEl);
    }
  });

  // Contradiction detection results
  es.addEventListener('contradictions', function(e) {
    var data = JSON.parse(e.data);
    if (data.contradictions && data.contradictions.length > 0) {
      var callout = renderContradictions(data.contradictions);
      // Insert after the answer prose
      proseEl.parentNode.insertBefore(callout, proseEl.nextSibling);
    }
  });

  es.addEventListener('done', function() {
    es.close();
    activeEventSource = null;
    pendingResult = null;
    addToHistory(searchSessionId, query, answerText, sourcesData, suggestionsData, currentMode);
  });

  // Backend sends a typed "error" event with a message when the pipeline fails
  es.addEventListener('error', function(e) {
    es.close();
    activeEventSource = null;
    var msg = 'Connection error \u2014 check settings and try again';
    if (e.data) {
      try {
        var data = JSON.parse(e.data);
        msg = data.error || msg;
      } catch (_) {}
    }
    // Show error in tracker or as fallback text
    if (tracker.parentNode) {
      tracker.innerHTML = '<div class="pipeline-step" style="color:var(--zen-coral)">'
        + '<div class="pipeline-step-icon" style="border-color:var(--zen-coral);color:var(--zen-coral)">!</div>'
        + '<div><div class="pipeline-step-label">Error</div>'
        + '<div class="pipeline-step-detail">' + msg + '</div></div></div>';
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
  // Always start a fresh session from the landing page
  currentSessionId = null;
  document.getElementById('answerContent').innerHTML = '';
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
    currentSessionId = null;
    document.getElementById('answerContent').innerHTML = '';
    submitSearch(chip.textContent);
  });
});

// New search button
document.getElementById('newSearchBtn').addEventListener('click', showLanding);

// Sidebar collapse/open
document.getElementById('sidebarCollapse').addEventListener('click', function() {
  document.getElementById('sidebar').classList.remove('open');
});
document.getElementById('sidebarOpen').addEventListener('click', function() {
  document.getElementById('sidebar').classList.add('open');
});

// Escape key closes settings
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeSettings();
});

// Render history on load and show sidebar if history exists
renderHistoryList();
if (loadHistory().length > 0) {
  document.getElementById('sidebar').classList.add('open');
}
