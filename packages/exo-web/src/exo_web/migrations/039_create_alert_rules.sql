CREATE TABLE IF NOT EXISTS alert_rules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    condition_type TEXT NOT NULL CHECK (condition_type IN ('error_rate', 'latency', 'cost')),
    condition_threshold REAL NOT NULL,
    action_type TEXT NOT NULL CHECK (action_type IN ('toast', 'email', 'webhook')),
    action_config_json TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_alert_rules_user_id ON alert_rules(user_id);
CREATE INDEX IF NOT EXISTS idx_alert_rules_enabled ON alert_rules(enabled);
