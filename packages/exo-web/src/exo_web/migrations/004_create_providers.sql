CREATE TABLE IF NOT EXISTS providers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider_type TEXT NOT NULL CHECK (provider_type IN ('openai', 'anthropic', 'gemini', 'vertex', 'ollama', 'custom')),
    encrypted_api_key TEXT,
    base_url TEXT,
    max_retries INTEGER NOT NULL DEFAULT 3,
    timeout INTEGER NOT NULL DEFAULT 30,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_providers_user_id ON providers(user_id);
