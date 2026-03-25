-- Add retrieval settings to knowledge_bases
ALTER TABLE knowledge_bases ADD COLUMN search_type TEXT NOT NULL DEFAULT 'keyword' CHECK (search_type IN ('semantic', 'keyword', 'hybrid'));
ALTER TABLE knowledge_bases ADD COLUMN top_k INTEGER NOT NULL DEFAULT 5 CHECK (top_k >= 1 AND top_k <= 20);
ALTER TABLE knowledge_bases ADD COLUMN similarity_threshold REAL NOT NULL DEFAULT 0.0 CHECK (similarity_threshold >= 0.0 AND similarity_threshold <= 1.0);
ALTER TABLE knowledge_bases ADD COLUMN reranker_enabled INTEGER NOT NULL DEFAULT 0;

-- Add knowledge_base_ids to agents (JSON array stored as TEXT)
ALTER TABLE agents ADD COLUMN knowledge_base_ids TEXT NOT NULL DEFAULT '[]';
