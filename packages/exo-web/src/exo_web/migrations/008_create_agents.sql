CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    instructions TEXT NOT NULL DEFAULT '',
    model_provider TEXT NOT NULL DEFAULT '',
    model_name TEXT NOT NULL DEFAULT '',
    temperature REAL,
    max_tokens INTEGER,
    max_steps INTEGER,
    output_type_json TEXT NOT NULL DEFAULT '{}',
    tools_json TEXT NOT NULL DEFAULT '[]',
    handoffs_json TEXT NOT NULL DEFAULT '[]',
    hooks_json TEXT NOT NULL DEFAULT '{}',
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_agents_project_id ON agents(project_id);
CREATE INDEX IF NOT EXISTS idx_agents_user_id ON agents(user_id);
