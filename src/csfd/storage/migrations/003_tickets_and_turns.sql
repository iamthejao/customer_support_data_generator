CREATE TABLE IF NOT EXISTS tickets (
    id                     TEXT PRIMARY KEY,
    run_id                 TEXT NOT NULL REFERENCES runs(id),
    problem_id             TEXT NOT NULL REFERENCES problems(id),
    kb_article_id          TEXT REFERENCES kb_articles(id),
    ticket_type            TEXT NOT NULL CHECK (ticket_type IN ('docs_request', 'l1', 'l2', 'l3')),
    priority               TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    status                 TEXT NOT NULL CHECK (status IN ('resolved', 'unresolved', 'escalated')),
    subject                TEXT NOT NULL,
    customer_persona_json  TEXT NOT NULL,
    agent_persona_json     TEXT NOT NULL,
    ground_truth_json      TEXT,
    metadata_json          TEXT,
    quality_flag           TEXT,
    unresolved_issues_json TEXT,
    created_at             TIMESTAMP NOT NULL,
    resolved_at            TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_tickets_run ON tickets(run_id);
CREATE INDEX IF NOT EXISTS idx_tickets_problem ON tickets(problem_id);
CREATE INDEX IF NOT EXISTS idx_tickets_type ON tickets(ticket_type);

CREATE TABLE IF NOT EXISTS turns (
    id                  TEXT PRIMARY KEY,
    ticket_id           TEXT NOT NULL REFERENCES tickets(id),
    turn_index          INTEGER NOT NULL,
    speaker             TEXT NOT NULL CHECK (speaker IN ('customer', 'agent', 'system')),
    speaker_persona     TEXT,
    content             TEXT NOT NULL,
    intent              TEXT,
    kb_references_json  TEXT,
    noise_applied       INTEGER NOT NULL DEFAULT 0 CHECK (noise_applied IN (0, 1)),
    noise_type          TEXT,
    quality_flag        TEXT,
    created_at          TIMESTAMP NOT NULL,
    UNIQUE (ticket_id, turn_index)
);
CREATE INDEX IF NOT EXISTS idx_turns_ticket ON turns(ticket_id);
