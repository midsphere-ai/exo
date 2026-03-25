-- Generic config version tracking for agents and workflows.
CREATE TABLE IF NOT EXISTS config_versions (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    version_num INTEGER NOT NULL,
    config_json TEXT NOT NULL DEFAULT '{}',
    author TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    tag TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_config_versions_entity ON config_versions(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_config_versions_entity_version ON config_versions(entity_type, entity_id, version_num);
