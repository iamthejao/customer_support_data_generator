-- Deterministic-pipeline schema.
--
-- Four tables capture one CSFD run end-to-end:
--   * problems          — Problem Database rows (Phase 1 output)
--   * incoming_requests — denormalized self-contained customer requests (Phase 2)
--   * resolutions       — multi-turn agent/customer back-and-forth, one per request
--   * lineage           — explicit (problem -> request -> resolution) traceability
--
-- `runs` lives in 001_runs.sql and `agent_traces` in 003_agent_traces.sql.

CREATE TABLE IF NOT EXISTS problems (
    id                    TEXT PRIMARY KEY,
    run_id                TEXT NOT NULL REFERENCES runs(id),
    title                 TEXT NOT NULL,
    summary               TEXT NOT NULL,
    background            TEXT NOT NULL,
    category              TEXT NOT NULL,
    complexity            TEXT NOT NULL CHECK (complexity IN ('simple', 'medium', 'complex')),
    resolution_hints_json TEXT NOT NULL,
    quality_flag          TEXT,
    created_at            TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_problems_run ON problems(run_id);
CREATE INDEX IF NOT EXISTS idx_problems_category ON problems(category);
CREATE INDEX IF NOT EXISTS idx_problems_complexity ON problems(complexity);

CREATE TABLE IF NOT EXISTS incoming_requests (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_uid     TEXT NOT NULL UNIQUE,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    problem_id      TEXT NOT NULL REFERENCES problems(id),
    ticket_type     TEXT NOT NULL CHECK (ticket_type IN ('docs_request', 'l1', 'l2', 'l3')),
    customer_name   TEXT NOT NULL,
    customer_tier   TEXT NOT NULL,
    customer_tone   TEXT NOT NULL,
    channel         TEXT NOT NULL DEFAULT 'email',
    subject         TEXT NOT NULL,
    body            TEXT NOT NULL,
    quality_flag    TEXT,
    created_at      TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_incoming_run ON incoming_requests(run_id);
CREATE INDEX IF NOT EXISTS idx_incoming_problem ON incoming_requests(problem_id);
CREATE INDEX IF NOT EXISTS idx_incoming_type ON incoming_requests(ticket_type);

CREATE TABLE IF NOT EXISTS resolutions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    resolution_uid       TEXT NOT NULL UNIQUE,
    run_id               TEXT NOT NULL REFERENCES runs(id),
    incoming_request_id  INTEGER NOT NULL UNIQUE REFERENCES incoming_requests(id),
    problem_id           TEXT NOT NULL REFERENCES problems(id),
    ticket_type          TEXT NOT NULL CHECK (ticket_type IN ('docs_request', 'l1', 'l2', 'l3')),
    turns_json           TEXT NOT NULL,
    turn_count           INTEGER NOT NULL,
    resolved             INTEGER NOT NULL DEFAULT 1 CHECK (resolved IN (0, 1)),
    quality_flag         TEXT,
    created_at           TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_resolutions_run ON resolutions(run_id);
CREATE INDEX IF NOT EXISTS idx_resolutions_problem ON resolutions(problem_id);

CREATE TABLE IF NOT EXISTS lineage (
    ticket_uid           TEXT PRIMARY KEY,
    run_id               TEXT NOT NULL REFERENCES runs(id),
    slot_index           INTEGER NOT NULL,
    problem_id           TEXT NOT NULL REFERENCES problems(id),
    ticket_type          TEXT NOT NULL,
    customer_tier        TEXT NOT NULL,
    customer_tone        TEXT NOT NULL,
    incoming_request_id  INTEGER REFERENCES incoming_requests(id),
    resolution_id        INTEGER REFERENCES resolutions(id),
    created_at           TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lineage_run ON lineage(run_id);
CREATE INDEX IF NOT EXISTS idx_lineage_problem ON lineage(problem_id);
