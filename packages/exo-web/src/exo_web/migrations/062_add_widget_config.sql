-- Widget configuration for embeddable chat deployments.
ALTER TABLE deployments ADD COLUMN widget_config_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE deployments ADD COLUMN cors_origins TEXT NOT NULL DEFAULT '';
