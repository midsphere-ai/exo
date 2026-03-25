-- Safety evaluation runs: stores per-run safety test results
CREATE TABLE IF NOT EXISTS safety_runs (
    id TEXT PRIMARY KEY,
    evaluation_id TEXT NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    run_at TEXT NOT NULL DEFAULT (datetime('now')),
    mode TEXT NOT NULL DEFAULT 'preset',
    policy_json TEXT NOT NULL DEFAULT '{}',
    results_json TEXT NOT NULL DEFAULT '[]',
    category_scores_json TEXT NOT NULL DEFAULT '{}',
    overall_score REAL NOT NULL DEFAULT 0.0,
    pass_rate REAL NOT NULL DEFAULT 0.0,
    flagged_count INTEGER NOT NULL DEFAULT 0,
    total_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_safety_runs_evaluation_id ON safety_runs(evaluation_id);
CREATE INDEX IF NOT EXISTS idx_safety_runs_run_at ON safety_runs(run_at);
