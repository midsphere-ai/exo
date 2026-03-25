-- Cost budgets for spend alerts and limits.

CREATE TABLE IF NOT EXISTS cost_budgets (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL CHECK (scope IN ('workspace', 'agent')),
    scope_id TEXT NOT NULL DEFAULT '',
    budget_amount REAL NOT NULL DEFAULT 0.0,
    period TEXT NOT NULL DEFAULT 'monthly' CHECK (period IN ('daily', 'monthly')),
    alert_threshold REAL NOT NULL DEFAULT 80.0,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_cost_budgets_user_id ON cost_budgets(user_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_cost_budgets_scope ON cost_budgets(user_id, scope, scope_id);
