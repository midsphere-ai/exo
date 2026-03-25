-- Track edit history for conversation messages
CREATE TABLE IF NOT EXISTS message_edits (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    original_content TEXT NOT NULL,
    edited_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_message_edits_message ON message_edits(message_id, edited_at);
