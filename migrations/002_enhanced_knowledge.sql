-- Migration: 002_enhanced_knowledge.sql
-- Description: Enhanced FTS indexes, embedding cache, and cleanup functions
-- Date: 2026-03-25
-- Depends on: 001_session_memory.sql

-- UP

-- ─── Full-Text Search ──────────────────────────────────────────────────────────

-- Add tsvector column for FTS on knowledge_entries
ALTER TABLE knowledge_entries
    ADD COLUMN IF NOT EXISTS search_vector tsvector;

-- Populate search_vector from content + title
UPDATE knowledge_entries
    SET search_vector = to_tsvector('english', coalesce(title, '') || ' ' || content);

-- GIN index for fast FTS queries
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_fts
    ON knowledge_entries USING GIN(search_vector);

-- Trigger to auto-update search_vector on insert/update
CREATE OR REPLACE FUNCTION knowledge_entries_search_vector_trigger()
RETURNS trigger AS $$
BEGIN
    NEW.search_vector := to_tsvector('english',
        coalesce(NEW.title, '') || ' ' || NEW.content
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tsvector_update_knowledge_entries ON knowledge_entries;
CREATE TRIGGER tsvector_update_knowledge_entries
    BEFORE INSERT OR UPDATE OF content, title ON knowledge_entries
    FOR EACH ROW EXECUTE FUNCTION knowledge_entries_search_vector_trigger();

-- ─── Embedding Cache ───────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS embedding_cache (
    cache_key        VARCHAR(64) PRIMARY KEY,  -- SHA-256 of (text + model)
    text_hash        VARCHAR(64) NOT NULL,
    model_name       VARCHAR(255) NOT NULL,
    embedding        FLOAT8[] NOT NULL,
    dimensions       INTEGER NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    access_count     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_embedding_cache_model
    ON embedding_cache(model_name);

CREATE INDEX IF NOT EXISTS idx_embedding_cache_last_accessed
    ON embedding_cache(last_accessed);

-- ─── Knowledge Domain Index ────────────────────────────────────────────────────

-- Index for filtering by domain (frequently used in queries)
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_domain
    ON knowledge_entries(domain)
    WHERE domain IS NOT NULL;

-- Index for filtering by source
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_source
    ON knowledge_entries(source)
    WHERE source IS NOT NULL;

-- Composite index for domain + recency queries
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_domain_created
    ON knowledge_entries(domain, created_at DESC)
    WHERE domain IS NOT NULL;

-- ─── Session Query Stats ───────────────────────────────────────────────────────

-- Add avg_processing_time to session_memories for monitoring
ALTER TABLE session_memories
    ADD COLUMN IF NOT EXISTS total_queries INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS avg_processing_ms FLOAT DEFAULT NULL;

-- ─── Cleanup Functions ─────────────────────────────────────────────────────────

-- Purge expired sessions (TTL enforced at application layer, this is a safety net)
CREATE OR REPLACE FUNCTION cleanup_expired_sessions(ttl_hours INTEGER DEFAULT 24)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM session_memories
    WHERE last_active < NOW() - (ttl_hours || ' hours')::INTERVAL;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Purge old embedding cache entries (LRU eviction)
CREATE OR REPLACE FUNCTION cleanup_embedding_cache(max_entries INTEGER DEFAULT 50000)
RETURNS INTEGER AS $$
DECLARE
    total_entries INTEGER;
    deleted_count INTEGER;
    to_delete     INTEGER;
BEGIN
    SELECT COUNT(*) INTO total_entries FROM embedding_cache;
    IF total_entries <= max_entries THEN
        RETURN 0;
    END IF;
    to_delete := total_entries - max_entries;
    DELETE FROM embedding_cache
    WHERE cache_key IN (
        SELECT cache_key FROM embedding_cache
        ORDER BY last_accessed ASC
        LIMIT to_delete
    );
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Purge knowledge entries by domain (admin utility)
CREATE OR REPLACE FUNCTION cleanup_knowledge_by_domain(target_domain TEXT)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM knowledge_entries WHERE domain = target_domain;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ─── Views ─────────────────────────────────────────────────────────────────────

-- Active sessions summary
CREATE OR REPLACE VIEW v_active_sessions AS
SELECT
    s.session_id,
    s.agent_id,
    s.created_at,
    s.last_active,
    s.token_count,
    s.max_tokens,
    s.total_queries,
    s.avg_processing_ms,
    ROUND(s.token_count::NUMERIC / NULLIF(s.max_tokens, 0) * 100, 1) AS token_usage_pct,
    NOW() - s.last_active AS idle_for
FROM session_memories s
WHERE s.last_active > NOW() - INTERVAL '24 hours'
ORDER BY s.last_active DESC;

-- Knowledge base summary by domain
CREATE OR REPLACE VIEW v_knowledge_summary AS
SELECT
    coalesce(domain, 'unclassified') AS domain,
    COUNT(*) AS total_entries,
    SUM(LENGTH(content)) AS total_chars,
    MAX(created_at) AS last_ingested
FROM knowledge_entries
GROUP BY domain
ORDER BY total_entries DESC;
