-- Crew-based agent orchestration

CREATE TABLE IF NOT EXISTS crews (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    process_type TEXT NOT NULL DEFAULT 'sequential' CHECK (process_type IN ('sequential', 'parallel')),
    config_json TEXT NOT NULL DEFAULT '{}',
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_crews_project ON crews(project_id);
CREATE INDEX IF NOT EXISTS idx_crews_user ON crews(user_id);

CREATE TABLE IF NOT EXISTS crew_tasks (
    id TEXT PRIMARY KEY,
    crew_id TEXT NOT NULL REFERENCES crews(id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    task_description TEXT NOT NULL DEFAULT '',
    expected_output TEXT NOT NULL DEFAULT '',
    task_order INTEGER NOT NULL DEFAULT 0,
    dependencies_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_crew_tasks_crew ON crew_tasks(crew_id);
CREATE INDEX IF NOT EXISTS idx_crew_tasks_agent ON crew_tasks(agent_id);
