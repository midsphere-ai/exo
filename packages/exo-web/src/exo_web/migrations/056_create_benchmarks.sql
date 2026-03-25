-- Benchmark suites: group evaluations for multi-agent comparison
CREATE TABLE IF NOT EXISTS benchmarks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    evaluation_id TEXT NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_benchmarks_user_id ON benchmarks(user_id);
CREATE INDEX IF NOT EXISTS idx_benchmarks_created_at ON benchmarks(created_at);

-- Benchmark runs: one row per benchmark execution (compares N agents)
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id TEXT PRIMARY KEY,
    benchmark_id TEXT NOT NULL REFERENCES benchmarks(id) ON DELETE CASCADE,
    agent_ids_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'pending',
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_benchmark_runs_benchmark_id ON benchmark_runs(benchmark_id);

-- Benchmark results: one row per agent per run
CREATE TABLE IF NOT EXISTS benchmark_results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES benchmark_runs(id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    agent_name TEXT NOT NULL DEFAULT '',
    results_json TEXT NOT NULL DEFAULT '[]',
    overall_score REAL NOT NULL DEFAULT 0.0,
    pass_rate REAL NOT NULL DEFAULT 0.0,
    total_latency_ms REAL NOT NULL DEFAULT 0.0,
    avg_latency_ms REAL NOT NULL DEFAULT 0.0,
    estimated_cost REAL NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_benchmark_results_run_id ON benchmark_results(run_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_results_agent_id ON benchmark_results(agent_id);
