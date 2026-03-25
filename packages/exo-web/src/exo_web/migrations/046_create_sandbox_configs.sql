-- Sandbox configuration per user
CREATE TABLE IF NOT EXISTS sandbox_configs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    allowed_libraries TEXT NOT NULL DEFAULT '["pandas","numpy","matplotlib","json","csv","math","statistics","collections","itertools","functools","re","datetime","io","os.path","pathlib"]',
    timeout_seconds INTEGER NOT NULL DEFAULT 30,
    memory_limit_mb INTEGER NOT NULL DEFAULT 256,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id)
);
