-- Observability integrations: export traces to external platforms
CREATE TABLE IF NOT EXISTS observability_integrations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    platform TEXT NOT NULL,  -- langfuse, langsmith, datadog, opik, custom_webhook
    display_name TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    endpoint_url TEXT NOT NULL DEFAULT '',
    encrypted_api_key TEXT NOT NULL DEFAULT '',
    project_name TEXT NOT NULL DEFAULT '',
    extra_config_json TEXT NOT NULL DEFAULT '{}',
    last_test_at TEXT,
    last_test_status TEXT,  -- success, error
    last_test_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_observability_integrations_user
    ON observability_integrations(user_id);
