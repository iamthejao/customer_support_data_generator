-- Add structured problem fields: root cause, classification axes, free-form tags.
--
-- These columns are populated by the Phase 1 problem-brainstorm generator
-- (see prompts/phase1/problem_brainstorm_v2.md.j2). Defaults exist so the
-- ALTER succeeds for any pre-existing rows from older runs; new rows always
-- receive explicit values via ProblemRepo.

ALTER TABLE problems ADD COLUMN symptoms_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE problems ADD COLUMN root_cause_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE problems ADD COLUMN fault_domain TEXT NOT NULL DEFAULT 'software'
    CHECK (fault_domain IN ('hardware','software','configuration','process','billing','account','integration'));
ALTER TABLE problems ADD COLUMN customer_impact TEXT NOT NULL DEFAULT 'degraded'
    CHECK (customer_impact IN ('blocked','degraded','cosmetic','informational'));
ALTER TABLE problems ADD COLUMN tags_json TEXT NOT NULL DEFAULT '[]';

CREATE INDEX IF NOT EXISTS idx_problems_fault_domain ON problems(fault_domain);
CREATE INDEX IF NOT EXISTS idx_problems_customer_impact ON problems(customer_impact);
