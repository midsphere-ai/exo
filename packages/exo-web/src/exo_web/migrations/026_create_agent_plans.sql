-- Add autonomous_mode flag to agents table
ALTER TABLE agents ADD COLUMN autonomous_mode INTEGER NOT NULL DEFAULT 0;

-- Execution plans created by the planner agent
CREATE TABLE IF NOT EXISTS agent_plans (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    goal TEXT NOT NULL DEFAULT '',
    version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'superseded', 'completed', 'failed')),
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_agent_plans_agent ON agent_plans(agent_id, status);
CREATE INDEX IF NOT EXISTS idx_agent_plans_user ON agent_plans(user_id, created_at DESC);

-- Individual steps within a plan
CREATE TABLE IF NOT EXISTS agent_plan_steps (
    id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL REFERENCES agent_plans(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    dependencies_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'skipped')),
    executor_output TEXT NOT NULL DEFAULT '',
    verifier_result TEXT NOT NULL DEFAULT '',
    verifier_passed INTEGER,
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_plan_steps_plan ON agent_plan_steps(plan_id, step_number);
