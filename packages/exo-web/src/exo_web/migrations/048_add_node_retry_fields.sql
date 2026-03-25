-- Add retry tracking fields to workflow_run_logs for per-node retry and error handling.

ALTER TABLE workflow_run_logs ADD COLUMN retry_attempt INTEGER NOT NULL DEFAULT 0;
ALTER TABLE workflow_run_logs ADD COLUMN max_retries INTEGER NOT NULL DEFAULT 0;
ALTER TABLE workflow_run_logs ADD COLUMN on_error TEXT NOT NULL DEFAULT 'fail';
