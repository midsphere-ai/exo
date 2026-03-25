CREATE TABLE IF NOT EXISTS plugins (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '0.1.0',
    type TEXT NOT NULL DEFAULT 'extension'
        CHECK (type IN ('model', 'tool', 'strategy', 'extension', 'bundle')),
    manifest_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'installed'
        CHECK (status IN ('installed', 'enabled', 'disabled', 'error')),
    entry_point TEXT NOT NULL DEFAULT '',
    directory TEXT NOT NULL DEFAULT '',
    permissions_json TEXT NOT NULL DEFAULT '[]',
    description TEXT NOT NULL DEFAULT '',
    author TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    installed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_plugins_user_id ON plugins(user_id);
CREATE INDEX IF NOT EXISTS idx_plugins_type ON plugins(type);
CREATE INDEX IF NOT EXISTS idx_plugins_status ON plugins(status);
