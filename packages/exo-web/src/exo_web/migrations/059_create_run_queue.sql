-- Run queue for concurrency-limited workflow execution.
CREATE TABLE IF NOT EXISTS run_queue (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id),
    status TEXT NOT NULL DEFAULT 'queued',  -- queued, started, cancelled
    trigger_type TEXT NOT NULL DEFAULT 'manual',
    nodes_json TEXT NOT NULL,
    edges_json TEXT NOT NULL,
    queued_at TEXT NOT NULL DEFAULT (datetime('now')),
    started_at TEXT,
    run_id TEXT  -- set when the queued entry starts and a workflow_runs record is created
);

CREATE INDEX IF NOT EXISTS idx_run_queue_status ON run_queue(status);
CREATE INDEX IF NOT EXISTS idx_run_queue_workflow ON run_queue(workflow_id);

-- Workspace-level settings stored in the database (replaces localStorage-only values).
CREATE TABLE IF NOT EXISTS workspace_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Default concurrent run limit.
INSERT OR IGNORE INTO workspace_settings (key, value) VALUES ('concurrent_run_limit', '5');

-- Per-workflow optional concurrency limit.
ALTER TABLE workflows ADD COLUMN max_concurrent INTEGER;
