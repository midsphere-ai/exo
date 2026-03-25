-- Neuron pipeline configurations for agent context engine
CREATE TABLE IF NOT EXISTS neuron_pipelines (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    name TEXT NOT NULL DEFAULT 'Default Pipeline',
    neurons_json TEXT NOT NULL DEFAULT '[]',
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_neuron_pipelines_agent ON neuron_pipelines(agent_id);
CREATE INDEX IF NOT EXISTS idx_neuron_pipelines_user ON neuron_pipelines(user_id);
