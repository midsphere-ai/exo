CREATE TABLE IF NOT EXISTS provider_keys (
    id TEXT PRIMARY KEY,
    provider_id TEXT NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
    encrypted_key TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    strategy_position INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'rate_limited', 'invalid')),
    total_requests INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    last_used TEXT,
    cooldown_until TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_provider_keys_provider_id ON provider_keys(provider_id);

-- Add load_balance_strategy column to providers table
ALTER TABLE providers ADD COLUMN load_balance_strategy TEXT NOT NULL DEFAULT 'round_robin' CHECK (load_balance_strategy IN ('round_robin', 'random', 'least_recently_used'));
