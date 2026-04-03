-- Migration: 005_state_persistence.sql
-- Description: State persistence tables for gdrag v3 - agent_snapshots, decisions with indexes
-- Date: 2026-04-03
-- Depends on: 004_agent_registry.sql

-- UP

-- ─── Agent Snapshots Table ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS agent_snapshots (
    snapshot_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id            VARCHAR(255) NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    active_sessions     TEXT[] DEFAULT '{}',
    recent_queries      TEXT[] DEFAULT '{}',
    attention_focus     TEXT,
    pending_tasks       TEXT[] DEFAULT '{}',
    metrics             JSONB DEFAULT '{}',
    metadata            JSONB DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for agent_id lookups
CREATE INDEX IF NOT EXISTS idx_agent_snapshots_agent_id
    ON agent_snapshots(agent_id);

-- Composite index for agent + time-range queries (latest snapshot per agent)
CREATE INDEX IF NOT EXISTS idx_agent_snapshots_agent_time
    ON agent_snapshots(agent_id, created_at DESC);

-- Index for time-based cleanup
CREATE INDEX IF NOT EXISTS idx_agent_snapshots_created_at
    ON agent_snapshots(created_at);

-- GIN index for active_sessions array containment queries
CREATE INDEX IF NOT EXISTS idx_agent_snapshots_active_sessions
    ON agent_snapshots USING GIN(active_sessions);

-- GIN index for pending_tasks array queries
CREATE INDEX IF NOT EXISTS idx_agent_snapshots_pending_tasks
    ON agent_snapshots USING GIN(pending_tasks);

-- GIN index for metrics JSONB queries
CREATE INDEX IF NOT EXISTS idx_agent_snapshots_metrics
    ON agent_snapshots USING GIN(metrics);

-- ─── Decisions Table ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS decisions (
    decision_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id            VARCHAR(255) NOT NULL REFERENCES agents(agent_id) ON DELETE CASCADE,
    context             JSONB DEFAULT '{}',
    decision            TEXT NOT NULL,
    reasoning           TEXT NOT NULL,
    outcome             TEXT,
    outcome_metrics     JSONB DEFAULT '{}',
    metadata            JSONB DEFAULT '{}',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at         TIMESTAMPTZ
);

-- Index for agent_id lookups
CREATE INDEX IF NOT EXISTS idx_decisions_agent_id
    ON decisions(agent_id);

-- Composite index for agent + time-range queries
CREATE INDEX IF NOT EXISTS idx_decisions_agent_time
    ON decisions(agent_id, created_at DESC);

-- Index for unresolved decisions (outcome IS NULL)
CREATE INDEX IF NOT EXISTS idx_decisions_unresolved
    ON decisions(agent_id, created_at)
    WHERE outcome IS NULL;

-- Index for time-based queries
CREATE INDEX IF NOT EXISTS idx_decisions_created_at
    ON decisions(created_at);

-- Index for resolved decisions
CREATE INDEX IF NOT EXISTS idx_decisions_resolved_at
    ON decisions(resolved_at)
    WHERE resolved_at IS NOT NULL;

-- GIN index for context JSONB queries
CREATE INDEX IF NOT EXISTS idx_decisions_context
    ON decisions USING GIN(context);

-- GIN index for outcome_metrics JSONB queries
CREATE INDEX IF NOT EXISTS idx_decisions_outcome_metrics
    ON decisions USING GIN(outcome_metrics);

-- Full-text search index on decision text
CREATE INDEX IF NOT EXISTS idx_decisions_fts
    ON decisions USING gin(to_tsvector('english', decision || ' ' || reasoning));

-- ─── Function: Cleanup old snapshots ──────────────────────────────────────────

CREATE OR REPLACE FUNCTION cleanup_old_snapshots(retention_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM agent_snapshots
    WHERE created_at < NOW() - (retention_days || ' days')::INTERVAL;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ─── Function: Get latest snapshot for agent ──────────────────────────────────

CREATE OR REPLACE FUNCTION get_latest_snapshot(p_agent_id VARCHAR(255))
RETURNS SETOF agent_snapshots AS $$
BEGIN
    RETURN QUERY
    SELECT *
    FROM agent_snapshots
    WHERE agent_id = p_agent_id
    ORDER BY created_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- ─── Function: Get unresolved decisions count ─────────────────────────────────

CREATE OR REPLACE FUNCTION get_unresolved_decisions_count(p_agent_id VARCHAR(255))
RETURNS INTEGER AS $$
DECLARE
    count_result INTEGER;
BEGIN
    SELECT COUNT(*)
    INTO count_result
    FROM decisions
    WHERE agent_id = p_agent_id
      AND outcome IS NULL;
    RETURN count_result;
END;
$$ LANGUAGE plpgsql;

-- ─── Views ────────────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW v_agent_state_summary AS
SELECT
    s.agent_id,
    a.name AS agent_name,
    s.snapshot_id,
    s.attention_focus,
    array_length(s.active_sessions, 1) AS active_session_count,
    array_length(s.pending_tasks, 1) AS pending_task_count,
    s.metrics,
    s.created_at AS snapshot_time,
    CASE
        WHEN s.created_at < NOW() - INTERVAL '1 hour' THEN 'stale'
        ELSE 'fresh'
    END AS freshness
FROM agent_snapshots s
JOIN agents a ON s.agent_id = a.agent_id
WHERE s.created_at = (
    SELECT MAX(created_at)
    FROM agent_snapshots s2
    WHERE s2.agent_id = s.agent_id
)
ORDER BY s.created_at DESC;

CREATE OR REPLACE VIEW v_recent_decisions AS
SELECT
    d.decision_id,
    d.agent_id,
    a.name AS agent_name,
    d.decision,
    d.reasoning,
    d.outcome,
    d.created_at,
    d.resolved_at,
    CASE
        WHEN d.outcome IS NULL THEN 'pending'
        ELSE 'resolved'
    END AS status
FROM decisions d
JOIN agents a ON d.agent_id = a.agent_id
WHERE d.created_at > NOW() - INTERVAL '24 hours'
ORDER BY d.created_at DESC;

-- DOWN
DROP VIEW IF EXISTS v_recent_decisions;
DROP VIEW IF EXISTS v_agent_state_summary;
DROP FUNCTION IF EXISTS get_unresolved_decisions_count(VARCHAR);
DROP FUNCTION IF EXISTS get_latest_snapshot(VARCHAR);
DROP FUNCTION IF EXISTS cleanup_old_snapshots(INTEGER);
DROP TABLE IF EXISTS decisions;
DROP TABLE IF EXISTS agent_snapshots;
