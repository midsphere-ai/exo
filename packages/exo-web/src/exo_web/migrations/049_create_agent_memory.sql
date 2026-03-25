-- Agent memory persistence: stores memory state per agent/thread combination
CREATE TABLE IF NOT EXISTS agent_memory (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant')),
    content TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_agent_memory_thread ON agent_memory(agent_id, thread_id, created_at);

-- Summary memory: stores running summaries per agent/thread
CREATE TABLE IF NOT EXISTS agent_memory_summary (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    thread_id TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(agent_id, thread_id)
);

CREATE INDEX IF NOT EXISTS idx_agent_memory_summary_lookup ON agent_memory_summary(agent_id, thread_id);
