-- Migration: 001_session_memory.sql
-- Description: Create session memory and knowledge tables for gdrag v2
-- Date: 2026-03-24

-- UP
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Session memories table
CREATE TABLE IF NOT EXISTS session_memories (
    session_id VARCHAR(255) PRIMARY KEY,
    agent_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    context_summary TEXT,
    token_count INTEGER DEFAULT 0,
    max_tokens INTEGER DEFAULT 8000,
    metadata JSONB DEFAULT '{}'
);

-- Create index on agent_id
CREATE INDEX IF NOT EXISTS idx_session_memories_agent_id
    ON session_memories(agent_id);

-- Create index on last_active for cleanup
CREATE INDEX IF NOT EXISTS idx_session_memories_last_active
    ON session_memories(last_active);

-- Query records table (history of queries per session)
CREATE TABLE IF NOT EXISTS query_records (
    query_id VARCHAR(255) PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL REFERENCES session_memories(session_id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    results_ids JSONB DEFAULT '[]',
    relevance_scores JSONB DEFAULT '[]',
    feedback FLOAT,
    processing_time_ms FLOAT
);

-- Create index on session_id
CREATE INDEX IF NOT EXISTS idx_query_records_session_id
    ON query_records(session_id);

-- Create index on timestamp for temporal queries
CREATE INDEX IF NOT EXISTS idx_query_records_timestamp
    ON query_records(timestamp);

-- Knowledge entries table
CREATE TABLE IF NOT EXISTS knowledge_entries (
    id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    domain VARCHAR(100),
    source VARCHAR(500),
    source_type VARCHAR(50) DEFAULT 'document',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create index on domain
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_domain
    ON knowledge_entries(domain);

-- Create full-text search index
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_fts
    ON knowledge_entries USING gin(to_tsvector('english', title || ' ' || content));

-- Create index on created_at
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_created_at
    ON knowledge_entries(created_at);

-- Create index on source_type
CREATE INDEX IF NOT EXISTS idx_knowledge_entries_source_type
    ON knowledge_entries(source_type);

-- Agent configuration table
CREATE TABLE IF NOT EXISTS agent_configs (
    agent_id VARCHAR(255) PRIMARY KEY,
    agent_name VARCHAR(255) NOT NULL,
    permissions JSONB DEFAULT '{}',
    rate_limit INTEGER DEFAULT 100,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Insert default agent configs
INSERT INTO agent_configs (agent_id, agent_name, permissions, rate_limit)
VALUES
    ('cline', 'Cline', '{"query": true, "ingest": true, "session": true}', 100),
    ('claude', 'Claude', '{"query": true, "ingest": true, "session": true}', 100),
    ('gpt', 'GPT', '{"query": true, "ingest": false, "session": true}', 50),
    ('copilot', 'Copilot', '{"query": true, "ingest": false, "session": false}', 30)
ON CONFLICT (agent_id) DO NOTHING;

-- DOWN
-- Drop tables in reverse order
DROP TABLE IF EXISTS agent_configs;
DROP TABLE IF EXISTS query_records;
DROP TABLE IF EXISTS knowledge_entries;
DROP TABLE IF EXISTS session_memories;