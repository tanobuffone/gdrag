-- Migration: 004_agent_registry.sql
-- Description: Agent registry tables for gdrag v3 - agents, heartbeats, GIN indexes
-- Date: 2026-04-02
-- Depends on: 002_enhanced_knowledge.sql

-- UP

-- ─── Agents Table ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS agents (
    agent_id            VARCHAR(255) PRIMARY KEY,
    name                VARCHAR(255) NOT NULL,
    capabilities        TEXT[] DEFAULT '{}',
    endpoints           JSONB DEFAULT '[]',
    metadata            JSONB DEFAULT '{}',
    tags                TEXT[] DEFAULT '{}',
    heartbeat_interval_s INTEGER NOT NULL DEFAULT 60,
    status              VARCHAR(20) NOT NULL DEFAULT 'unknown'
                        CHECK (status IN ('healthy', 'degraded', 'unhealthy', 'unknown')),
    registered_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- GIN index for capabilities array queries (@>, &&)
CREATE INDEX IF NOT EXISTS idx_agents_capabilities
    ON agents USING GIN(capabilities);

-- GIN index for tags array queries
CREATE INDEX IF NOT EXISTS idx_agents_tags
    ON agents USING GIN(tags);

-- GIN index for endpoints JSONB containment queries
CREATE INDEX IF NOT EXISTS idx_agents_endpoints
    ON agents USING GIN(endpoints);

-- Index for status filtering
CREATE INDEX IF NOT EXISTS idx_agents_status
    ON agents(status);

-- Index for registration recency
CREATE INDEX IF NOT EXISTS idx_agents_registered_at
    ON agents(registered_at DESC);

-- ─── Agent Heartbeats Table ────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS agent_heartbeats (
    id                  BIGSERIAL PRIMARY KEY,
    agent_id            VARCHAR(255) NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    timestamp           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status              VARCHAR(20) NOT NULL DEFAULT 'healthy'
                        CHECK (status IN ('healthy', 'degraded', 'unhealthy', 'unknown')),
    metrics             JSONB DEFAULT '{}'
);

-- Index for agent_id lookups
CREATE INDEX IF NOT EXISTS idx_agent_heartbeats_agent_id
    ON agent_heartbeats(agent_id);

-- Composite index for agent + time-range queries
CREATE INDEX IF NOT EXISTS idx_agent_heartbeats_agent_time
    ON agent_heartbeats(agent_id, timestamp DESC);

-- Index for time-based cleanup
CREATE INDEX IF NOT EXISTS idx_agent_heartbeats_timestamp
    ON agent_heartbeats(timestamp);

-- GIN index for metrics JSONB queries
CREATE INDEX IF NOT EXISTS idx_agent_heartbeats_metrics
    ON agent_heartbeats USING GIN(metrics);

-- ─── Trigger: Auto-update updated_at on agents ─────────────────────────────────

CREATE OR REPLACE FUNCTION agents_updated_at_trigger()
RETURNS trigger AS $
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_agents_updated_at ON agents;
CREATE TRIGGER trigger_agents_updated_at
    BEFORE UPDATE ON agents
    FOR EACH ROW EXECUTE FUNCTION agents_updated_at_trigger();

-- ─── Function: Update agent status from latest heartbeat ───────────────────────

CREATE OR REPLACE FUNCTION update_agent_status_from_heartbeat()
RETURNS trigger AS $
BEGIN
    UPDATE agents
    SET status = NEW.status,
        updated_at = NOW()
    WHERE agent_id = NEW.agent_id;
    RETURN NEW;
END;
$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_agent_status ON agent_heartbeats;
CREATE TRIGGER trigger_update_agent_status
    AFTER INSERT ON agent_heartbeats
    FOR EACH ROW EXECUTE FUNCTION update_agent_status_from_heartbeat();

-- ─── Function: Cleanup old heartbeats ──────────────────────────────────────────

CREATE OR REPLACE FUNCTION cleanup_old_heartbeats(retention_hours INTEGER DEFAULT 72)
RETURNS INTEGER AS $
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM agent_heartbeats
    WHERE timestamp < NOW() - (retention_hours || ' hours')::INTERVAL;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$ LANGUAGE plpgsql;

-- ─── Function: Mark stale agents as unhealthy ──────────────────────────────────

CREATE OR REPLACE FUNCTION mark_stale_agents(stale_multiplier FLOAT DEFAULT 3.0)
RETURNS INTEGER AS $
DECLARE
    updated_count INTEGER;
BEGIN
    UPDATE agents
    SET status = 'unhealthy',
        updated_at = NOW()
    WHERE status IN ('healthy', 'degraded')
      AND agent_id NOT IN (
          SELECT agent_id FROM (
              SELECT agent_id, MAX(timestamp) AS last_hb
              FROM agent_heartbeats
              GROUP BY agent_id
          ) latest
          WHERE latest.last_hb > NOW() - (
              (SELECT heartbeat_interval_s FROM agents a WHERE a.agent_id = latest.agent_id)
              * stale_multiplier * INTERVAL '1 second'
          )
      );
    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$ LANGUAGE plpgsql;

-- ─── Views ─────────────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW v_agents_summary AS
SELECT
    a.agent_id,
    a.name,
    a.status,
    a.capabilities,
    a.tags,
    a.registered_at,
    a.heartbeat_interval_s,
    hb.last_heartbeat,
    hb.heartbeat_count,
    CASE
        WHEN hb.last_heartbeat IS NULL THEN 'never_connected'
        WHEN hb.last_heartbeat < NOW() - (a.heartbeat_interval_s * 3 * INTERVAL '1 second')
            THEN 'stale'
        ELSE 'active'
    END AS connectivity
FROM agents a
LEFT JOIN (
    SELECT agent_id,
           MAX(timestamp) AS last_heartbeat,
           COUNT(*) AS heartbeat_count
    FROM agent_heartbeats
    GROUP BY agent_id
) hb ON a.agent_id = hb.agent_id
ORDER BY a.registered_at DESC;

CREATE OR REPLACE VIEW v_recent_heartbeats AS
SELECT
    h.agent_id,
    a.name AS agent_name,
    h.timestamp,
    h.status,
    h.metrics
FROM agent_heartbeats h
JOIN agents a ON h.agent_id = a.agent_id
WHERE h.timestamp > NOW() - INTERVAL '24 hours'
ORDER BY h.timestamp DESC;

-- DOWN
DROP VIEW IF EXISTS v_recent_heartbeats;
DROP VIEW IF EXISTS v_agents_summary;
DROP FUNCTION IF EXISTS mark_stale_agents(FLOAT);
DROP FUNCTION IF EXISTS cleanup_old_heartbeats(INTEGER);
DROP FUNCTION IF EXISTS update_agent_status_from_heartbeat();
DROP FUNCTION IF EXISTS agents_updated_at_trigger();
DROP TABLE IF EXISTS agent_heartbeats;
DROP TABLE IF EXISTS agents;