-- Migration: 003_multi_tenancy.sql
-- Description: Multi-tenancy support - tenants, quota tracking, agent_id isolation
-- Date: 2026-04-02
-- Depends on: 002_enhanced_knowledge.sql

-- UP

CREATE TABLE IF NOT EXISTS tenants (
    id              VARCHAR(255) PRIMARY KEY,
    agent_id        VARCHAR(255) NOT NULL UNIQUE,
    name            VARCHAR(255) NOT NULL,
    description     TEXT,
    permissions     JSONB NOT NULL DEFAULT '{}',
    quotas          JSONB NOT NULL DEFAULT '{}',
    status          VARCHAR(50) NOT NULL DEFAULT 'active',
    metadata        JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tenants_agent_id ON tenants(agent_id);
CREATE INDEX IF NOT EXISTS idx_tenants_status ON tenants(status);

-- Add tenant_id to session_memories
ALTER TABLE session_memories ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255);
CREATE INDEX IF NOT EXISTS idx_session_memories_tenant_id ON session_memories(tenant_id);
CREATE INDEX IF NOT EXISTS idx_session_memories_tenant_agent ON session_memories(tenant_id, agent_id);

-- Add tenant_id to knowledge_entries
ALTER TABLE knowledge_entries ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255);
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_tenant_id ON knowledge_entries(tenant_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_tenant_domain ON knowledge_entries(tenant_id, domain) WHERE tenant_id IS NOT NULL;

-- Quota Usage Table
CREATE TABLE IF NOT EXISTS quota_usage (
    id                      VARCHAR(255) PRIMARY KEY,
    tenant_id               VARCHAR(255) NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    queries_this_hour       INTEGER NOT NULL DEFAULT 0,
    storage_used_mb         FLOAT NOT NULL DEFAULT 0.0,
    vectors_count           INTEGER NOT NULL DEFAULT 0,
    active_sessions         INTEGER NOT NULL DEFAULT 0,
    total_sessions          INTEGER NOT NULL DEFAULT 0,
    knowledge_entries_count INTEGER NOT NULL DEFAULT 0,
    last_query_at           TIMESTAMPTZ,
    last_reset_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_quota_usage_tenant_id UNIQUE (tenant_id)
);

CREATE INDEX IF NOT EXISTS idx_quota_usage_tenant_id ON quota_usage(tenant_id);
CREATE INDEX IF NOT EXISTS idx_quota_usage_last_reset ON quota_usage(last_reset_at);

-- Helper: auto-update timestamps
CREATE OR REPLACE FUNCTION update_tenant_timestamp()
RETURNS TRIGGER AS $ BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$ LANGUAGE plpgsql;

CREATE TRIGGER trg_tenants_updated_at BEFORE UPDATE ON tenants
    FOR EACH ROW EXECUTE FUNCTION update_tenant_timestamp();
CREATE TRIGGER trg_quota_usage_updated_at BEFORE UPDATE ON quota_usage
    FOR EACH ROW EXECUTE FUNCTION update_tenant_timestamp();

-- Helper: reset hourly quota counters
CREATE OR REPLACE FUNCTION reset_hourly_quota_counters()
RETURNS INTEGER AS $
DECLARE reset_count INTEGER;
BEGIN
    UPDATE quota_usage SET queries_this_hour = 0, last_reset_at = NOW()
    WHERE last_reset_at < NOW() - INTERVAL '1 hour';
    GET DIAGNOSTICS reset_count = ROW_COUNT;
    RETURN reset_count;
END;
$ LANGUAGE plpgsql;

-- DOWN
DROP FUNCTION IF EXISTS reset_hourly_quota_counters();
DROP FUNCTION IF EXISTS update_tenant_timestamp();
DROP TABLE IF EXISTS quota_usage;
ALTER TABLE knowledge_entries DROP COLUMN IF EXISTS tenant_id;
ALTER TABLE session_memories DROP COLUMN IF EXISTS tenant_id;
DROP TABLE IF EXISTS tenants;