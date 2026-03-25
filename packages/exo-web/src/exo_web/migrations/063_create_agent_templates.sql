-- Agent templates: reusable agent configurations.
CREATE TABLE IF NOT EXISTS agent_templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    config_json TEXT NOT NULL DEFAULT '{}',
    tools_required TEXT NOT NULL DEFAULT '[]',
    models_required TEXT NOT NULL DEFAULT '[]',
    version INTEGER NOT NULL DEFAULT 1,
    creator_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_agent_templates_creator ON agent_templates(creator_id);

-- Template versions for tracking changes.
CREATE TABLE IF NOT EXISTS agent_template_versions (
    id TEXT PRIMARY KEY,
    template_id TEXT NOT NULL REFERENCES agent_templates(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL DEFAULT 1,
    config_json TEXT NOT NULL DEFAULT '{}',
    tools_required TEXT NOT NULL DEFAULT '[]',
    models_required TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_agent_template_versions_template ON agent_template_versions(template_id);
