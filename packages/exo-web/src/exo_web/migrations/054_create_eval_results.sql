CREATE TABLE IF NOT EXISTS eval_results (
    id TEXT PRIMARY KEY,
    evaluation_id TEXT NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    run_at TEXT NOT NULL DEFAULT (datetime('now')),
    results_json TEXT NOT NULL DEFAULT '[]',
    overall_score REAL NOT NULL DEFAULT 0.0,
    pass_rate REAL NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_eval_results_evaluation_id ON eval_results(evaluation_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_run_at ON eval_results(run_at);
