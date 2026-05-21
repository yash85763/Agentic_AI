-- =============================================================================
-- AgentOS PostgreSQL Schema with Row Level Security
-- =============================================================================
-- Apply after the initial SQLAlchemy table creation.
-- Run as the postgres superuser (or table owner) once per environment.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Non-owner application role (agents connect as this role, not as owner)
-- ---------------------------------------------------------------------------
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT FROM pg_catalog.pg_roles WHERE rolname = 'agentos_app'
  ) THEN
    CREATE ROLE agentos_app WITH LOGIN PASSWORD 'agentos_app_change_me_in_production';
  END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO agentos_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO agentos_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO agentos_app;

-- ---------------------------------------------------------------------------
-- Add user_id and trust columns if not already present
-- (Safe to run multiple times — uses IF NOT EXISTS / DO blocks)
-- ---------------------------------------------------------------------------

DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='jobs' AND column_name='user_id'
  ) THEN
    ALTER TABLE jobs ADD COLUMN user_id TEXT NOT NULL DEFAULT 'system';
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='file_records' AND column_name='user_id'
  ) THEN
    ALTER TABLE file_records ADD COLUMN user_id TEXT NOT NULL DEFAULT 'system';
  END IF;
END $$;

DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='file_records' AND column_name='source_trust_tier'
  ) THEN
    ALTER TABLE file_records
      ADD COLUMN source_trust_tier TEXT NOT NULL DEFAULT 'user-uploaded',
      ADD COLUMN quarantine_status  TEXT NOT NULL DEFAULT 'quarantined',
      ADD COLUMN ingestion_metadata JSONB;
  END IF;
END $$;

-- ---------------------------------------------------------------------------
-- Row Level Security on jobs
-- ---------------------------------------------------------------------------

ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE jobs FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS jobs_user_isolation ON jobs;
CREATE POLICY jobs_user_isolation ON jobs
  USING (
    user_id = current_setting('app.current_user_id', true)::TEXT
    OR current_setting('app.current_user_id', true) = 'system'
  );

-- ---------------------------------------------------------------------------
-- Row Level Security on file_records
-- ---------------------------------------------------------------------------

ALTER TABLE file_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE file_records FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS file_records_user_isolation ON file_records;
CREATE POLICY file_records_user_isolation ON file_records
  USING (
    user_id = current_setting('app.current_user_id', true)::TEXT
    OR current_setting('app.current_user_id', true) = 'system'
  );

-- ---------------------------------------------------------------------------
-- Indexes for performance and strict-mode compatibility
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs (user_id);
CREATE INDEX IF NOT EXISTS idx_file_records_user_id ON file_records (user_id);
CREATE INDEX IF NOT EXISTS idx_file_records_quarantine ON file_records (quarantine_status);

-- ---------------------------------------------------------------------------
-- Audit log table for ingestion events
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS ingestion_audit (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id      TEXT NOT NULL,
  file_id      UUID REFERENCES file_records(id) ON DELETE SET NULL,
  event_type   TEXT NOT NULL,  -- 'ingested' | 'quarantine_lifted' | 'rejected' | 'poisoning_flagged'
  source       TEXT,
  size_bytes   BIGINT,
  content_type TEXT,
  metadata     JSONB,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ingestion_audit_user ON ingestion_audit (user_id);
CREATE INDEX IF NOT EXISTS idx_ingestion_audit_file ON ingestion_audit (file_id);

-- ---------------------------------------------------------------------------
-- Integration test verification query (run this to confirm RLS is active)
-- ---------------------------------------------------------------------------
-- CONNECT AS agentos_app ROLE then run:
--   SELECT set_config('app.current_user_id', 'user_A', true);
--   SELECT COUNT(*) FROM jobs WHERE user_id = 'user_B';  -- must return 0
