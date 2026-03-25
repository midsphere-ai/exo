-- Checkpoint save/restore for long-running agent and workflow runs.

CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    step_number INTEGER NOT NULL DEFAULT 0,
    state_blob TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_run_id ON checkpoints(run_id);
