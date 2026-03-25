CREATE TABLE IF NOT EXISTS webhooks (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    hook_id TEXT NOT NULL,
    url_token TEXT NOT NULL UNIQUE,
    enabled INTEGER NOT NULL DEFAULT 1,
    request_log_json TEXT NOT NULL DEFAULT '[]',
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_webhooks_workflow_id ON webhooks(workflow_id);
CREATE INDEX IF NOT EXISTS idx_webhooks_url_token ON webhooks(url_token);
CREATE UNIQUE INDEX IF NOT EXISTS idx_webhooks_workflow_hook ON webhooks(workflow_id, hook_id);

CREATE TABLE IF NOT EXISTS notification_templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('slack', 'discord', 'email')),
    config_json TEXT NOT NULL DEFAULT '{}',
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_notification_templates_user_id ON notification_templates(user_id);
