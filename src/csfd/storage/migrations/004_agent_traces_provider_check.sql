-- Broaden the agent_traces.model_provider CHECK to include claude_code_cli.
-- SQLite cannot ALTER a CHECK in place, so rebuild the table.

CREATE TABLE agent_traces_new (
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

INSERT INTO agent_traces_new SELECT * FROM agent_traces;
DROP TABLE agent_traces;
ALTER TABLE agent_traces_new RENAME TO agent_traces;

CREATE INDEX IF NOT EXISTS idx_traces_run      ON agent_traces(run_id);
CREATE INDEX IF NOT EXISTS idx_traces_artifact ON agent_traces(artifact_type, artifact_id);
CREATE INDEX IF NOT EXISTS idx_traces_role     ON agent_traces(agent_role);
CREATE INDEX IF NOT EXISTS idx_traces_prompt   ON agent_traces(prompt_id);
