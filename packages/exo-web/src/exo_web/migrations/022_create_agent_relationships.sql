-- Agent relationships: supervisor → sub-agent delegation

CREATE TABLE IF NOT EXISTS agent_relationships (
    id TEXT PRIMARY KEY,
    supervisor_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    sub_agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    relationship_type TEXT NOT NULL DEFAULT 'delegation',
    routing_rule_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_agent_rel_supervisor ON agent_relationships(supervisor_id);
CREATE INDEX IF NOT EXISTS idx_agent_rel_sub_agent ON agent_relationships(sub_agent_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_rel_unique_pair ON agent_relationships(supervisor_id, sub_agent_id);
