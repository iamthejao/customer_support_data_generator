-- Contact channel and call timing on resolutions (phone-call transcripts).
--
-- `channel` mirrors incoming_requests.channel so a resolution row is
-- self-describing. `started_at` is the deterministic, seeded contact time
-- (see csfd.calls.schedule_first_contact); `ended_at` / `duration_s` are
-- estimated from the transcript for phone calls and stay NULL for email.
-- Per-utterance offsets live inside turns_json (`start_s` / `end_s`).
-- `end_reason` records how the dialogue loop ended (customer_done /
-- agent_done / cap_hit) for every channel.

ALTER TABLE resolutions ADD COLUMN channel TEXT NOT NULL DEFAULT 'email'
    CHECK (channel IN ('email', 'phone'));
ALTER TABLE resolutions ADD COLUMN agent_name TEXT;
ALTER TABLE resolutions ADD COLUMN end_reason TEXT;
ALTER TABLE resolutions ADD COLUMN started_at TIMESTAMP;
ALTER TABLE resolutions ADD COLUMN ended_at TIMESTAMP;
ALTER TABLE resolutions ADD COLUMN duration_s REAL;

CREATE INDEX IF NOT EXISTS idx_resolutions_channel ON resolutions(channel);
