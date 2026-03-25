CREATE TABLE IF NOT EXISTS prompt_optimizations (
    id TEXT PRIMARY KEY,
    agent_id TEXT REFERENCES agents(id) ON DELETE SET NULL,
    template_id TEXT REFERENCES prompt_templates(id) ON DELETE SET NULL,
    original_prompt TEXT NOT NULL,
    optimized_prompt TEXT NOT NULL,
    strategy TEXT NOT NULL DEFAULT 'clarity',
    changes_json TEXT NOT NULL DEFAULT '[]',
    accepted INTEGER NOT NULL DEFAULT 0,
    eval_score_before REAL,
    eval_score_after REAL,
    model_used TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_prompt_optimizations_agent_id ON prompt_optimizations(agent_id);
CREATE INDEX IF NOT EXISTS idx_prompt_optimizations_template_id ON prompt_optimizations(template_id);
CREATE INDEX IF NOT EXISTS idx_prompt_optimizations_user_id ON prompt_optimizations(user_id);
CREATE INDEX IF NOT EXISTS idx_prompt_optimizations_created_at ON prompt_optimizations(created_at);
