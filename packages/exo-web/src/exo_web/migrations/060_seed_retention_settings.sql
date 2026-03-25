-- Seed default retention policy settings into workspace_settings.
INSERT OR IGNORE INTO workspace_settings (key, value) VALUES ('retention_artifacts_days', '90');
INSERT OR IGNORE INTO workspace_settings (key, value) VALUES ('retention_runs_days', '30');
INSERT OR IGNORE INTO workspace_settings (key, value) VALUES ('retention_logs_days', '14');
