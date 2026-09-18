-- Contact channel and timing on resolutions (phone calls and email threads).
--
-- `channel` mirrors incoming_requests.channel so a resolution row is
-- self-describing. `started_at` is the deterministic, seeded contact time
-- (see csfd.calls). `ended_at` / `duration_s` are the end of the call or the
-- last email's sent time. Per-message timing lives inside turns_json:
-- `start_s` / `end_s` offsets for phone, `sent_at` timestamps for email.
-- `end_reason` records how the dialogue loop ended (customer_done /
-- agent_done / cap_hit / dropped) for every channel.

ALTER TABLE resolutions ADD COLUMN channel TEXT NOT NULL DEFAULT 'email'
    CHECK (channel IN ('email', 'phone'));
ALTER TABLE resolutions ADD COLUMN agent_name TEXT;
ALTER TABLE resolutions ADD COLUMN end_reason TEXT;
ALTER TABLE resolutions ADD COLUMN started_at TIMESTAMP;
ALTER TABLE resolutions ADD COLUMN ended_at TIMESTAMP;
ALTER TABLE resolutions ADD COLUMN duration_s REAL;

CREATE INDEX IF NOT EXISTS idx_resolutions_channel ON resolutions(channel);
