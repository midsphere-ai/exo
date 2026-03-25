-- Add detailed node execution fields for inspection support.
-- Extends workflow_run_logs with input capture, execution logs, and token usage.

ALTER TABLE workflow_run_logs ADD COLUMN input_json TEXT;
ALTER TABLE workflow_run_logs ADD COLUMN logs_text TEXT;
ALTER TABLE workflow_run_logs ADD COLUMN token_usage_json TEXT;
