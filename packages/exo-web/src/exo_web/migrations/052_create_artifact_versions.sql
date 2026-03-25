CREATE TABLE IF NOT EXISTS artifact_versions (
    id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    file_size INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(artifact_id, version_number)
);

CREATE INDEX IF NOT EXISTS idx_artifact_versions_artifact_id ON artifact_versions(artifact_id);
CREATE INDEX IF NOT EXISTS idx_artifact_versions_created_at ON artifact_versions(created_at);
