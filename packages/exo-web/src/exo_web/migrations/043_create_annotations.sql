-- Annotations table for cached Q&A responses
CREATE TABLE IF NOT EXISTS annotations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    query TEXT NOT NULL,
    original_response TEXT,
    improved_response TEXT NOT NULL,
    similarity_threshold REAL NOT NULL DEFAULT 0.8,
    usage_count INTEGER NOT NULL DEFAULT 0,
    cost_saved REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_annotations_user ON annotations(user_id);

-- FTS5 for annotation query search
CREATE VIRTUAL TABLE IF NOT EXISTS annotations_fts USING fts5(
    query,
    improved_response,
    content='annotations',
    content_rowid='rowid'
);

-- Populate FTS from existing data
INSERT INTO annotations_fts(rowid, query, improved_response)
    SELECT rowid, query, improved_response FROM annotations;

-- Triggers to keep annotations_fts in sync
CREATE TRIGGER IF NOT EXISTS annotations_fts_insert AFTER INSERT ON annotations BEGIN
    INSERT INTO annotations_fts(rowid, query, improved_response)
        VALUES (new.rowid, new.query, new.improved_response);
END;

CREATE TRIGGER IF NOT EXISTS annotations_fts_update AFTER UPDATE ON annotations BEGIN
    INSERT INTO annotations_fts(annotations_fts, rowid, query, improved_response)
        VALUES ('delete', old.rowid, old.query, old.improved_response);
    INSERT INTO annotations_fts(rowid, query, improved_response)
        VALUES (new.rowid, new.query, new.improved_response);
END;

CREATE TRIGGER IF NOT EXISTS annotations_fts_delete AFTER DELETE ON annotations BEGIN
    INSERT INTO annotations_fts(annotations_fts, rowid, query, improved_response)
        VALUES ('delete', old.rowid, old.query, old.improved_response);
END;
