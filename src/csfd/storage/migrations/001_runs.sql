CREATE TABLE IF NOT EXISTS runs (
    id                   TEXT PRIMARY KEY,
    phase                TEXT NOT NULL CHECK (phase IN ('phase1', 'phase2', 'full')),
    parent_run_id        TEXT REFERENCES runs(id),
    status               TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'aborted_budget')),
    started_at           TIMESTAMP NOT NULL,
    completed_at         TIMESTAMP,
    run_seed             INTEGER NOT NULL,
    pipeline_version     TEXT NOT NULL,
    git_sha              TEXT,
    config_snapshot_json TEXT NOT NULL,
    stats_json           TEXT,
    error_summary        TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_phase ON runs(phase);
CREATE INDEX IF NOT EXISTS idx_runs_parent ON runs(parent_run_id);
