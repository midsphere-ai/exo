-- Add run-level metadata for history and analytics.

ALTER TABLE workflow_runs ADD COLUMN trigger_type TEXT NOT NULL DEFAULT 'manual';
ALTER TABLE workflow_runs ADD COLUMN input_json TEXT;
ALTER TABLE workflow_runs ADD COLUMN step_count INTEGER;
ALTER TABLE workflow_runs ADD COLUMN total_tokens INTEGER;
ALTER TABLE workflow_runs ADD COLUMN total_cost REAL;
