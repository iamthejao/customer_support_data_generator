-- Multi-contact cases: one allocation slot (a case, lineage.ticket_uid) can
-- span several contacts ("rounds") — e.g. a caller who calls back.
--
-- Every incoming request / resolution row now names its case and its 1-based
-- position in it. lineage keeps one row per case and links to round 1; join
-- resolutions.case_uid = lineage.ticket_uid to get every round of a case.

ALTER TABLE incoming_requests ADD COLUMN case_uid TEXT;
ALTER TABLE incoming_requests ADD COLUMN round_index INTEGER NOT NULL DEFAULT 1;
ALTER TABLE resolutions ADD COLUMN case_uid TEXT;
ALTER TABLE resolutions ADD COLUMN round_index INTEGER NOT NULL DEFAULT 1;
ALTER TABLE resolutions ADD COLUMN round_count INTEGER NOT NULL DEFAULT 1;

-- Rows written before this migration are single-contact cases linked from lineage.
UPDATE incoming_requests
SET case_uid = (SELECT l.ticket_uid FROM lineage l WHERE l.incoming_request_id = incoming_requests.id)
WHERE case_uid IS NULL;
UPDATE resolutions
SET case_uid = (SELECT l.ticket_uid FROM lineage l WHERE l.resolution_id = resolutions.id)
WHERE case_uid IS NULL;

CREATE INDEX IF NOT EXISTS idx_incoming_case ON incoming_requests(case_uid, round_index);
CREATE INDEX IF NOT EXISTS idx_resolutions_case ON resolutions(case_uid, round_index);
