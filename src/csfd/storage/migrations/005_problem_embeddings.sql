-- Per-problem embedding for commit-time dedup in Phase 1.
-- Vector is float32 little-endian packed; length must equal dim * 4 bytes.

CREATE TABLE IF NOT EXISTS problem_embeddings (
    problem_id  TEXT PRIMARY KEY REFERENCES problems(id) ON DELETE CASCADE,
    run_id      TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    created_at  TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_problem_embeddings_run ON problem_embeddings(run_id);
