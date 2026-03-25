CREATE TABLE IF NOT EXISTS deployments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL CHECK (entity_type IN ('agent', 'workflow')),
    entity_id TEXT NOT NULL,
    api_key_hash TEXT NOT NULL,
    rate_limit INTEGER NOT NULL DEFAULT 60,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive')),
    usage_count INTEGER NOT NULL DEFAULT 0,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_deployments_user_id ON deployments(user_id);
CREATE INDEX IF NOT EXISTS idx_deployments_entity ON deployments(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_deployments_api_key_hash ON deployments(api_key_hash);
