-- Thread management: threads table for conversation organization
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    first_message_preview TEXT NOT NULL DEFAULT '',
    message_count INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    total_cost REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_threads_user ON threads(user_id);
CREATE INDEX IF NOT EXISTS idx_threads_agent_user ON threads(agent_id, user_id);
CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(user_id, updated_at DESC);

-- Thread messages table
CREATE TABLE IF NOT EXISTS thread_messages (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant')),
    content TEXT NOT NULL DEFAULT '',
    tool_calls_json TEXT,
    usage_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_thread_messages_thread ON thread_messages(thread_id, created_at);
