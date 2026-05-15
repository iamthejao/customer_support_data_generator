CREATE TABLE IF NOT EXISTS problems (
    id                     TEXT PRIMARY KEY,
    run_id                 TEXT NOT NULL REFERENCES runs(id),
    title                  TEXT NOT NULL,
    description            TEXT NOT NULL,
    category               TEXT NOT NULL,
    severity               TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    has_kb                 INTEGER NOT NULL CHECK (has_kb IN (0, 1)),
    coverage_reasoning     TEXT,
    coverage_confidence    TEXT CHECK (coverage_confidence IN ('low', 'medium', 'high')),
    metadata_json          TEXT,
    quality_flag           TEXT,
    unresolved_issues_json TEXT,
    created_at             TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_problems_run ON problems(run_id);
CREATE INDEX IF NOT EXISTS idx_problems_has_kb ON problems(has_kb);

CREATE TABLE IF NOT EXISTS kb_articles (
    id                         TEXT PRIMARY KEY,
    run_id                     TEXT NOT NULL REFERENCES runs(id),
    problem_id                 TEXT NOT NULL UNIQUE REFERENCES problems(id),
    title                      TEXT NOT NULL,
    content_markdown           TEXT NOT NULL,
    content_hash               TEXT NOT NULL,
    troubleshooting_steps_json TEXT NOT NULL,
    prerequisites_json         TEXT,
    metadata_json              TEXT,
    version                    INTEGER NOT NULL DEFAULT 1,
    quality_flag               TEXT,
    unresolved_issues_json     TEXT,
    created_at                 TIMESTAMP NOT NULL
);
