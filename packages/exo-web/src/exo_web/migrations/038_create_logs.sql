CREATE TABLE IF NOT EXISTS logs (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    level TEXT NOT NULL CHECK (level IN ('debug', 'info', 'warn', 'error')),
    source TEXT NOT NULL CHECK (source IN ('agent', 'tool', 'model', 'system')),
    agent_id TEXT,
    message TEXT NOT NULL,
    metadata_json TEXT,
    user_id TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level);
CREATE INDEX IF NOT EXISTS idx_logs_source ON logs(source);
CREATE INDEX IF NOT EXISTS idx_logs_agent_id ON logs(agent_id);
CREATE INDEX IF NOT EXISTS idx_logs_user_id ON logs(user_id);
