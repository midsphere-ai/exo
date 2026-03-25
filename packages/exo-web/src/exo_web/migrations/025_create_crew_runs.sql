-- Crew execution tracking

CREATE TABLE IF NOT EXISTS crew_runs (
    id TEXT PRIMARY KEY,
    crew_id TEXT NOT NULL REFERENCES crews(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'partial', 'failed')),
    process_type TEXT NOT NULL DEFAULT 'sequential',
    input TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_crew_runs_crew ON crew_runs(crew_id);
CREATE INDEX IF NOT EXISTS idx_crew_runs_user ON crew_runs(user_id);

CREATE TABLE IF NOT EXISTS crew_run_tasks (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES crew_runs(id) ON DELETE CASCADE,
    task_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    agent_name TEXT NOT NULL DEFAULT '',
    task_description TEXT NOT NULL DEFAULT '',
    expected_output TEXT NOT NULL DEFAULT '',
    task_order INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    output TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    started_at TEXT,
    completed_at TEXT,
    duration_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_crew_run_tasks_run ON crew_run_tasks(run_id);
