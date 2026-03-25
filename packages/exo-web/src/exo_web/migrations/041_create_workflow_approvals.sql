-- Workflow approval gates: pause execution until a human approves/rejects.

CREATE TABLE IF NOT EXISTS workflow_approvals (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'timed_out')),
    timeout_minutes INTEGER NOT NULL DEFAULT 60,
    comment TEXT,
    user_id TEXT NOT NULL REFERENCES users(id),
    requested_at TEXT NOT NULL DEFAULT (datetime('now')),
    responded_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_workflow_approvals_run_id ON workflow_approvals(run_id);
CREATE INDEX IF NOT EXISTS idx_workflow_approvals_user_id ON workflow_approvals(user_id);
CREATE INDEX IF NOT EXISTS idx_workflow_approvals_status ON workflow_approvals(status);
