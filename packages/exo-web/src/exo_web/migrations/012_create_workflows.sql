CREATE TABLE IF NOT EXISTS workflows (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    nodes_json TEXT NOT NULL DEFAULT '[]',
    edges_json TEXT NOT NULL DEFAULT '[]',
    viewport_json TEXT NOT NULL DEFAULT '{"x":0,"y":0,"zoom":1}',
    status TEXT NOT NULL DEFAULT 'draft',
    last_run_at TEXT,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_workflows_project_id ON workflows(project_id);
CREATE INDEX IF NOT EXISTS idx_workflows_user_id ON workflows(user_id);
