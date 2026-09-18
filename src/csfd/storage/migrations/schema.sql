-- Full CSFD storage schema. Every statement is IF NOT EXISTS, so applying it
-- to an existing database (db-migrate, and every `generate`) drops nothing.
--
-- Tables:
--   * runs               — one row per pipeline run
--   * problems            — Problem Database rows (Phase 1 output)
--   * incoming_requests    — denormalized self-contained customer requests (Phase 2)
--   * resolutions          — multi-turn agent/customer back-and-forth, one per request
--   * lineage              — explicit (problem -> request -> resolution) traceability
--   * agent_traces         — per-agent-call observability records
--   * problem_embeddings   — per-problem embedding for commit-time dedup in Phase 1
--
-- `incoming_requests` / `resolutions` carry `case_uid` + `round_index` so one
-- allocation slot (a case, `lineage.ticket_uid`) can span several contacts
-- ("rounds") — e.g. a caller who calls back; join `resolutions.case_uid =
-- lineage.ticket_uid` to get every round of a case.

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
    created_at            TIMESTAMP NOT NULL,
    symptoms_json         TEXT NOT NULL DEFAULT '[]',
    root_cause_json       TEXT NOT NULL DEFAULT '[]',
    fault_domain          TEXT NOT NULL DEFAULT 'software'
        CHECK (fault_domain IN ('hardware','software','configuration','process','billing','account','integration')),
    customer_impact       TEXT NOT NULL DEFAULT 'degraded'
        CHECK (customer_impact IN ('blocked','degraded','cosmetic','informational')),
    tags_json             TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_problems_run ON problems(run_id);
CREATE INDEX IF NOT EXISTS idx_problems_category ON problems(category);
CREATE INDEX IF NOT EXISTS idx_problems_complexity ON problems(complexity);
CREATE INDEX IF NOT EXISTS idx_problems_fault_domain ON problems(fault_domain);
CREATE INDEX IF NOT EXISTS idx_problems_customer_impact ON problems(customer_impact);

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
    created_at      TIMESTAMP NOT NULL,
    case_uid        TEXT,
    round_index     INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_incoming_run ON incoming_requests(run_id);
CREATE INDEX IF NOT EXISTS idx_incoming_problem ON incoming_requests(problem_id);
CREATE INDEX IF NOT EXISTS idx_incoming_type ON incoming_requests(ticket_type);
CREATE INDEX IF NOT EXISTS idx_incoming_case ON incoming_requests(case_uid, round_index);

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
    created_at           TIMESTAMP NOT NULL,
    channel              TEXT NOT NULL DEFAULT 'email' CHECK (channel IN ('email', 'phone')),
    agent_name           TEXT,
    end_reason           TEXT,
    started_at           TIMESTAMP,
    ended_at             TIMESTAMP,
    duration_s           REAL,
    case_uid             TEXT,
    round_index          INTEGER NOT NULL DEFAULT 1,
    round_count          INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_resolutions_run ON resolutions(run_id);
CREATE INDEX IF NOT EXISTS idx_resolutions_problem ON resolutions(problem_id);
CREATE INDEX IF NOT EXISTS idx_resolutions_channel ON resolutions(channel);
CREATE INDEX IF NOT EXISTS idx_resolutions_case ON resolutions(case_uid, round_index);

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
    created_at            TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lineage_run ON lineage(run_id);
CREATE INDEX IF NOT EXISTS idx_lineage_problem ON lineage(problem_id);

CREATE TABLE IF NOT EXISTS agent_traces (
    id                  TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES runs(id),
    thread_id           TEXT NOT NULL,
    node_name           TEXT NOT NULL,
    agent_role          TEXT NOT NULL,
    artifact_type       TEXT NOT NULL,
    artifact_id         TEXT,
    attempt             INTEGER NOT NULL DEFAULT 0,
    prompt_id           TEXT NOT NULL,
    input_json          TEXT NOT NULL,
    output_json         TEXT,
    verdict             TEXT CHECK (verdict IN ('pass', 'fail')),
    verdict_issues_json TEXT,
    model_provider      TEXT NOT NULL CHECK (model_provider IN ('anthropic', 'openai_compat', 'claude_code_cli', 'fake')),
    model_id            TEXT NOT NULL,
    tokens_in           INTEGER,
    tokens_out          INTEGER,
    cost_usd_estimated  REAL,
    latency_ms          INTEGER,
    parent_trace_id     TEXT REFERENCES agent_traces(id),
    status              TEXT NOT NULL CHECK (status IN ('ok', 'transport_error', 'schema_error', 'budget_error')),
    error_class         TEXT,
    error_message       TEXT,
    created_at          TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_run ON agent_traces(run_id);
CREATE INDEX IF NOT EXISTS idx_traces_artifact ON agent_traces(artifact_type, artifact_id);
CREATE INDEX IF NOT EXISTS idx_traces_role ON agent_traces(agent_role);
CREATE INDEX IF NOT EXISTS idx_traces_prompt ON agent_traces(prompt_id);

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
