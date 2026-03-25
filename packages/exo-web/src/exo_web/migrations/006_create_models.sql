CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    provider_id TEXT NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    context_window INTEGER,
    capabilities TEXT DEFAULT '[]',
    pricing_input REAL,
    pricing_output REAL,
    is_custom INTEGER NOT NULL DEFAULT 0,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_models_provider_id ON models(provider_id);
CREATE INDEX IF NOT EXISTS idx_models_user_id ON models(user_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_models_provider_model ON models(provider_id, model_name);
