-- Context configuration columns for Exo's ContextConfig
ALTER TABLE agents ADD COLUMN context_automation_level TEXT NOT NULL DEFAULT 'copilot';
ALTER TABLE agents ADD COLUMN context_max_tokens_per_step INTEGER DEFAULT NULL;
ALTER TABLE agents ADD COLUMN context_max_total_tokens INTEGER DEFAULT NULL;
ALTER TABLE agents ADD COLUMN context_memory_type TEXT NOT NULL DEFAULT 'conversation';
ALTER TABLE agents ADD COLUMN context_workspace_enabled INTEGER NOT NULL DEFAULT 0;
