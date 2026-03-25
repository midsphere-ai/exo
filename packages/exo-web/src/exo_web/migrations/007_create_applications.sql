CREATE TABLE IF NOT EXISTS applications (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('chatbot', 'chatflow', 'workflow', 'agent', 'text_generator')),
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    config_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'draft',
    last_run_at TEXT,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_applications_project_id ON applications(project_id);
CREATE INDEX IF NOT EXISTS idx_applications_user_id ON applications(user_id);
