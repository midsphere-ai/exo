-- FTS5 virtual tables for full-text search across entities

-- Agents: search by name, description, instructions
CREATE VIRTUAL TABLE IF NOT EXISTS agents_fts USING fts5(
    name,
    description,
    instructions,
    content='agents',
    content_rowid='rowid'
);

-- Workflows: search by name, description
CREATE VIRTUAL TABLE IF NOT EXISTS workflows_fts USING fts5(
    name,
    description,
    content='workflows',
    content_rowid='rowid'
);

-- Tools: search by name, description
CREATE VIRTUAL TABLE IF NOT EXISTS tools_fts USING fts5(
    name,
    description,
    content='tools',
    content_rowid='rowid'
);

-- Thread messages: search by content
CREATE VIRTUAL TABLE IF NOT EXISTS thread_messages_fts USING fts5(
    content,
    content='thread_messages',
    content_rowid='rowid'
);

-- Populate FTS tables from existing data
INSERT INTO agents_fts(rowid, name, description, instructions)
    SELECT rowid, name, description, instructions FROM agents;

INSERT INTO workflows_fts(rowid, name, description)
    SELECT rowid, name, description FROM workflows;

INSERT INTO tools_fts(rowid, name, description)
    SELECT rowid, name, description FROM tools;

INSERT INTO thread_messages_fts(rowid, content)
    SELECT rowid, content FROM thread_messages;

-- Triggers to keep agents_fts in sync
CREATE TRIGGER IF NOT EXISTS agents_fts_insert AFTER INSERT ON agents BEGIN
    INSERT INTO agents_fts(rowid, name, description, instructions)
        VALUES (new.rowid, new.name, new.description, new.instructions);
END;

CREATE TRIGGER IF NOT EXISTS agents_fts_update AFTER UPDATE ON agents BEGIN
    INSERT INTO agents_fts(agents_fts, rowid, name, description, instructions)
        VALUES ('delete', old.rowid, old.name, old.description, old.instructions);
    INSERT INTO agents_fts(rowid, name, description, instructions)
        VALUES (new.rowid, new.name, new.description, new.instructions);
END;

CREATE TRIGGER IF NOT EXISTS agents_fts_delete AFTER DELETE ON agents BEGIN
    INSERT INTO agents_fts(agents_fts, rowid, name, description, instructions)
        VALUES ('delete', old.rowid, old.name, old.description, old.instructions);
END;

-- Triggers to keep workflows_fts in sync
CREATE TRIGGER IF NOT EXISTS workflows_fts_insert AFTER INSERT ON workflows BEGIN
    INSERT INTO workflows_fts(rowid, name, description)
        VALUES (new.rowid, new.name, new.description);
END;

CREATE TRIGGER IF NOT EXISTS workflows_fts_update AFTER UPDATE ON workflows BEGIN
    INSERT INTO workflows_fts(workflows_fts, rowid, name, description)
        VALUES ('delete', old.rowid, old.name, old.description);
    INSERT INTO workflows_fts(rowid, name, description)
        VALUES (new.rowid, new.name, new.description);
END;

CREATE TRIGGER IF NOT EXISTS workflows_fts_delete AFTER DELETE ON workflows BEGIN
    INSERT INTO workflows_fts(workflows_fts, rowid, name, description)
        VALUES ('delete', old.rowid, old.name, old.description);
END;

-- Triggers to keep tools_fts in sync
CREATE TRIGGER IF NOT EXISTS tools_fts_insert AFTER INSERT ON tools BEGIN
    INSERT INTO tools_fts(rowid, name, description)
        VALUES (new.rowid, new.name, new.description);
END;

CREATE TRIGGER IF NOT EXISTS tools_fts_update AFTER UPDATE ON tools BEGIN
    INSERT INTO tools_fts(tools_fts, rowid, name, description)
        VALUES ('delete', old.rowid, old.name, old.description);
    INSERT INTO tools_fts(rowid, name, description)
        VALUES (new.rowid, new.name, new.description);
END;

CREATE TRIGGER IF NOT EXISTS tools_fts_delete AFTER DELETE ON tools BEGIN
    INSERT INTO tools_fts(tools_fts, rowid, name, description)
        VALUES ('delete', old.rowid, old.name, old.description);
END;

-- Triggers to keep thread_messages_fts in sync
CREATE TRIGGER IF NOT EXISTS thread_messages_fts_insert AFTER INSERT ON thread_messages BEGIN
    INSERT INTO thread_messages_fts(rowid, content)
        VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS thread_messages_fts_update AFTER UPDATE ON thread_messages BEGIN
    INSERT INTO thread_messages_fts(thread_messages_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
    INSERT INTO thread_messages_fts(rowid, content)
        VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS thread_messages_fts_delete AFTER DELETE ON thread_messages BEGIN
    INSERT INTO thread_messages_fts(thread_messages_fts, rowid, content)
        VALUES ('delete', old.rowid, old.content);
END;
