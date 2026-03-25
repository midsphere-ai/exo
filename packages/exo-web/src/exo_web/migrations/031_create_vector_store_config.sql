-- Vector store backend configuration (one active config per user)
CREATE TABLE IF NOT EXISTS vector_store_config (
    id TEXT PRIMARY KEY,
    backend TEXT NOT NULL DEFAULT 'sqlite_vss' CHECK (backend IN ('sqlite_vss', 'milvus', 'qdrant', 'chromadb', 'pinecone')),
    host TEXT DEFAULT '',
    port INTEGER DEFAULT NULL,
    api_key_encrypted TEXT DEFAULT '',
    collection_name TEXT DEFAULT '',
    extra_json TEXT DEFAULT '{}',
    is_active INTEGER DEFAULT 1,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_vector_store_config_user ON vector_store_config (user_id);
