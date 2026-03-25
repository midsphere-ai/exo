CREATE TABLE IF NOT EXISTS mcp_servers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    auth_config_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'disconnected'
        CHECK (status IN ('connected', 'disconnected', 'error')),
    last_check_at TEXT,
    error_message TEXT,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_mcp_servers_project_id ON mcp_servers(project_id);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_user_id ON mcp_servers(user_id);
CREATE INDEX IF NOT EXISTS idx_mcp_servers_status ON mcp_servers(status);
