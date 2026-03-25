-- Dedicated model pricing table for cost estimation.

CREATE TABLE IF NOT EXISTS model_pricing (
    id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    input_price_per_1k REAL NOT NULL DEFAULT 0.0,
    output_price_per_1k REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_model_pricing_model_name ON model_pricing(model_name);

-- Pre-seed pricing for common models.
INSERT OR IGNORE INTO model_pricing (id, model_name, input_price_per_1k, output_price_per_1k) VALUES
    ('mp-gpt4o', 'gpt-4o', 0.0025, 0.01),
    ('mp-gpt4o-mini', 'gpt-4o-mini', 0.00015, 0.0006),
    ('mp-claude35sonnet', 'claude-3.5-sonnet', 0.003, 0.015),
    ('mp-claude3haiku', 'claude-3-haiku', 0.00025, 0.00125),
    ('mp-gemini15pro', 'gemini-1.5-pro', 0.00125, 0.005),
    ('mp-gemini15flash', 'gemini-1.5-flash', 0.000075, 0.0003);
