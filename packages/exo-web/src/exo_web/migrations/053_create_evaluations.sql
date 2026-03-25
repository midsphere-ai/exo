CREATE TABLE IF NOT EXISTS evaluations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    test_cases_json TEXT NOT NULL DEFAULT '[]',
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_evaluations_user_id ON evaluations(user_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_agent_id ON evaluations(agent_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_created_at ON evaluations(created_at);
