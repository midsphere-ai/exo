-- Prompt version history: every save creates a version
CREATE TABLE IF NOT EXISTS prompt_versions (
    id TEXT PRIMARY KEY,
    template_id TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    variables_json TEXT NOT NULL DEFAULT '{}',
    version_number INTEGER NOT NULL DEFAULT 1,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (template_id) REFERENCES prompt_templates(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_prompt_versions_template_id ON prompt_versions(template_id);
CREATE INDEX IF NOT EXISTS idx_prompt_versions_user_id ON prompt_versions(user_id);
