CREATE TABLE IF NOT EXISTS tools (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT 'custom'
        CHECK (category IN ('search', 'code', 'file', 'data', 'communication', 'custom')),
    schema_json TEXT NOT NULL DEFAULT '{}',
    code TEXT NOT NULL DEFAULT '',
    tool_type TEXT NOT NULL DEFAULT 'function'
        CHECK (tool_type IN ('function', 'http', 'schema', 'mcp')),
    usage_count INTEGER NOT NULL DEFAULT 0,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tools_project_id ON tools(project_id);
CREATE INDEX IF NOT EXISTS idx_tools_user_id ON tools(user_id);
CREATE INDEX IF NOT EXISTS idx_tools_category ON tools(category);
