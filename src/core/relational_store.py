"""Enhanced PostgreSQL relational store operations for gdrag v3.

Provides session memory persistence, knowledge management, full-text search,
and multi-tenancy support via RLS policies and tenant-aware queries.

v3 Changes:
  - owner_agent_id and visibility columns on knowledge_entries
  - RLS (Row Level Security) policies for PostgreSQL
  - Tenant-filtered queries via set_config / current_setting
  - initialize_schema creates RLS policies automatically
  - Backward-compatible with v2 (agent_id is optional, defaults to no filter)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import psycopg2
from psycopg2.extras import RealDictCursor

from ..core.config import AppConfig
from ..models.schemas import DomainStats, KnowledgeItem, QueryRecord, SessionMemory

logger = logging.getLogger(__name__)


class EnhancedRelationalStore:
    """Enhanced PostgreSQL operations with session, knowledge management, and multi-tenancy.

    v3 adds:
      - owner_agent_id and visibility columns on knowledge_entries.
      - RLS policies that automatically filter rows per tenant.
      - Session variable ``gdrag.current_agent_id`` used by RLS policies.
      - All read/write methods accept optional ``agent_id`` parameter.

    Backward-compatibility:
      - When agent_id is None, no tenant filter is applied (v2 behavior).
      - Existing rows without owner_agent_id are treated as public (legacy).
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._connection = None
        # Track current agent context for RLS
        self._current_agent_id: Optional[str] = None

    def _get_connection(self):
        """Get or create PostgreSQL connection."""
        if self._connection is None or self._connection.closed:
            self._connection = psycopg2.connect(
                host=self.config.database.postgres_host,
                port=self.config.database.postgres_port,
                dbname=self.config.database.postgres_db,
                user=self.config.database.postgres_user,
                password=self.config.database.postgres_password,
            )
            self._connection.autocommit = True
            logger.info(
                f"Connected to PostgreSQL at "
                f"{self.config.database.postgres_host}:{self.config.database.postgres_port}"
            )
        return self._connection

    def close(self) -> None:
        """Close the database connection."""
        if self._connection and not self._connection.closed:
            self._connection.close()
            self._connection = None

    # ------------------------------------------------------------------
    # Agent context for RLS
    # ------------------------------------------------------------------

    def set_agent_context(self, agent_id: Optional[str]) -> None:
        """Set the current agent context for RLS filtering.

        This sets the PostgreSQL session variable ``gdrag.current_agent_id``
        which is used by RLS policies to filter rows.

        Args:
            agent_id: Agent ID to set as context. None to clear.
        """
        self._current_agent_id = agent_id
        if agent_id is not None:
            try:
                conn = self._get_connection()
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT set_config('gdrag.current_agent_id', %s, false)",
                        (agent_id,),
                    )
                logger.debug(f"Set RLS agent context to: {agent_id}")
            except Exception as e:
                logger.warning(f"Could not set agent context: {e}")

    def clear_agent_context(self) -> None:
        """Clear the current agent context."""
        self.set_agent_context(None)

    def _ensure_agent_context(self, agent_id: Optional[str]) -> None:
        """Ensure the agent context is set if agent_id is provided.

        Args:
            agent_id: Agent ID to ensure is set.
        """
        if agent_id is not None and agent_id != self._current_agent_id:
            self.set_agent_context(agent_id)

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def _execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch: bool = True,
    ) -> List[Dict[str, Any]]:
        """Execute a SQL query.

        Args:
            query: SQL query string.
            params: Query parameters.
            fetch: Whether to fetch results.

        Returns:
            List of result dictionaries.
        """
        conn = self._get_connection()

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                if fetch and cursor.description:
                    return [dict(row) for row in cursor.fetchall()]
                return []
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise

    # ========================================================================
    # Session Memory Operations
    # ========================================================================

    def store_session_memory(self, session: SessionMemory) -> str:
        """Persist session memory to PostgreSQL.

        Args:
            session: Session memory to store.

        Returns:
            Session ID.
        """
        query = """
        INSERT INTO session_memories (
            session_id, agent_id, created_at, last_active,
            context_summary, token_count, max_tokens, metadata
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (session_id) DO UPDATE SET
            last_active = EXCLUDED.last_active,
            context_summary = EXCLUDED.context_summary,
            token_count = EXCLUDED.token_count,
            metadata = EXCLUDED.metadata
        RETURNING session_id
        """
        import json

        params = (
            session.session_id,
            session.agent_id,
            session.created_at,
            session.last_active,
            session.context_summary,
            session.token_count,
            session.max_tokens,
            json.dumps(session.metadata),
        )

        result = self._execute_query(query, params)
        return result[0]["session_id"] if result else session.session_id

    def get_session_memory(self, session_id: str, agent_id: Optional[str] = None) -> Optional[SessionMemory]:
        """Retrieve session memory from PostgreSQL.

        Args:
            session_id: Session ID to retrieve.
            agent_id: Optional agent ID for ownership verification.

        Returns:
            Session memory or None if not found.
        """
        if agent_id:
            query = """
            SELECT session_id, agent_id, created_at, last_active,
                   context_summary, token_count, max_tokens, metadata
            FROM session_memories
            WHERE session_id = %s AND agent_id = %s
            """
            results = self._execute_query(query, (session_id, agent_id))
        else:
            query = """
            SELECT session_id, agent_id, created_at, last_active,
                   context_summary, token_count, max_tokens, metadata
            FROM session_memories
            WHERE session_id = %s
            """
            results = self._execute_query(query, (session_id,))

        if not results:
            return None

        row = results[0]
        import json

        return SessionMemory(
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            created_at=row["created_at"],
            last_active=row["last_active"],
            context_summary=row["context_summary"],
            token_count=row["token_count"],
            max_tokens=row["max_tokens"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def store_query_record(
        self,
        session_id: str,
        query_record: QueryRecord,
    ) -> str:
        """Store a query record in session history.

        Args:
            session_id: Parent session ID.
            query_record: Query record to store.

        Returns:
            Query record ID.
        """
        import json

        query = """
        INSERT INTO query_records (
            query_id, session_id, query, timestamp,
            results_ids, relevance_scores, feedback, processing_time_ms
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s
        )
        RETURNING query_id
        """
        params = (
            query_record.query_id,
            session_id,
            query_record.query,
            query_record.timestamp,
            json.dumps(query_record.results_ids),
            json.dumps(query_record.relevance_scores),
            query_record.feedback,
            query_record.processing_time_ms,
        )

        result = self._execute_query(query, params)
        return result[0]["query_id"] if result else query_record.query_id

    def get_session_history(
        self,
        session_id: str,
        limit: int = 50,
    ) -> List[QueryRecord]:
        """Get query history for a session.

        Args:
            session_id: Session ID.
            limit: Maximum records to return.

        Returns:
            List of query records.
        """
        query = """
        SELECT query_id, query, timestamp, results_ids,
               relevance_scores, feedback, processing_time_ms
        FROM query_records
        WHERE session_id = %s
        ORDER BY timestamp DESC
        LIMIT %s
        """
        results = self._execute_query(query, (session_id, limit))

        import json

        records = []
        for row in results:
            record = QueryRecord(
                query_id=row["query_id"],
                query=row["query"],
                timestamp=row["timestamp"],
                results_ids=json.loads(row["results_ids"]) if row["results_ids"] else [],
                relevance_scores=json.loads(row["relevance_scores"]) if row["relevance_scores"] else [],
                feedback=row["feedback"],
                processing_time_ms=row["processing_time_ms"],
            )
            records.append(record)

        return records

    def list_sessions(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[SessionMemory]:
        """List sessions, optionally filtered by agent.

        Args:
            agent_id: Optional agent ID filter.
            limit: Maximum sessions to return.

        Returns:
            List of session memories.
        """
        if agent_id:
            query = """
            SELECT session_id, agent_id, created_at, last_active,
                   context_summary, token_count, max_tokens, metadata
            FROM session_memories
            WHERE agent_id = %s
            ORDER BY last_active DESC
            LIMIT %s
            """
            results = self._execute_query(query, (agent_id, limit))
        else:
            query = """
            SELECT session_id, agent_id, created_at, last_active,
                   context_summary, token_count, max_tokens, metadata
            FROM session_memories
            ORDER BY last_active DESC
            LIMIT %s
            """
            results = self._execute_query(query, (limit,))

        import json

        sessions = []
        for row in results:
            session = SessionMemory(
                session_id=row["session_id"],
                agent_id=row["agent_id"],
                created_at=row["created_at"],
                last_active=row["last_active"],
                context_summary=row["context_summary"],
                token_count=row["token_count"],
                max_tokens=row["max_tokens"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            sessions.append(session)

        return sessions

    def delete_expired_sessions(self, ttl_hours: int = 24, agent_id: Optional[str] = None) -> int:
        """Delete expired sessions.

        Args:
            ttl_hours: Session time-to-live in hours.
            agent_id: Optional agent ID filter.

        Returns:
            Number of sessions deleted.
        """
        if agent_id:
            query = """
            DELETE FROM session_memories
            WHERE last_active < NOW() - INTERVAL '%s hours'
              AND agent_id = %s
            RETURNING session_id
            """
            results = self._execute_query(query, (ttl_hours, agent_id))
        else:
            query = """
            DELETE FROM session_memories
            WHERE last_active < NOW() - INTERVAL '%s hours'
            RETURNING session_id
            """
            results = self._execute_query(query, (ttl_hours,))
        deleted_count = len(results)
        logger.info(f"Deleted {deleted_count} expired sessions")
        return deleted_count

    def delete_session(self, session_id: str, agent_id: Optional[str] = None) -> bool:
        """Delete a session from PostgreSQL.

        Args:
            session_id: Session ID to delete.
            agent_id: Optional agent ID for ownership verification.

        Returns:
            True if deleted, False if not found or not owned by agent.
        """
        if agent_id:
            query = """
            DELETE FROM session_memories
            WHERE session_id = %s AND agent_id = %s
            RETURNING session_id
            """
            results = self._execute_query(query, (session_id, agent_id))
        else:
            query = """
            DELETE FROM session_memories
            WHERE session_id = %s
            RETURNING session_id
            """
            results = self._execute_query(query, (session_id,))
        deleted = len(results) > 0
        if deleted:
            logger.info(f"Deleted session {session_id} from database")
        else:
            logger.warning(f"Session {session_id} not found or not owned by agent for deletion")
        return deleted

    def delete_sessions_by_agent(self, agent_id: str) -> int:
        """Delete all sessions for an agent.

        Args:
            agent_id: Agent ID.

        Returns:
            Number of sessions deleted.
        """
        query = """
        DELETE FROM session_memories
        WHERE agent_id = %s
        RETURNING session_id
        """
        results = self._execute_query(query, (agent_id,))
        deleted_count = len(results)
        logger.info(f"Deleted {deleted_count} sessions for agent {agent_id}")
        return deleted_count

    # ========================================================================
    # Knowledge Operations
    # ========================================================================

    def store_knowledge_entry(
        self,
        item: KnowledgeItem,
        agent_id: Optional[str] = None,
        visibility: str = "private",
    ) -> str:
        """Store a knowledge entry with multi-tenancy support.

        Args:
            item: Knowledge item to store.
            agent_id: Owner agent ID (v3). If None, uses v2 behavior.
            visibility: Visibility level (v3). Defaults to "private".

        Returns:
            Knowledge item ID.
        """
        import json

        # Extract owner from metadata if present (backward compat)
        effective_agent_id = agent_id
        if effective_agent_id is None and "owner_agent_id" in item.metadata:
            effective_agent_id = item.metadata["owner_agent_id"]

        effective_visibility = visibility
        if "visibility" in item.metadata:
            effective_visibility = item.metadata["visibility"]

        query = """
        INSERT INTO knowledge_entries (
            id, title, content, domain, source, source_type,
            metadata, created_at, updated_at,
            owner_agent_id, visibility
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            domain = EXCLUDED.domain,
            metadata = EXCLUDED.metadata,
            updated_at = EXCLUDED.updated_at,
            owner_agent_id = EXCLUDED.owner_agent_id,
            visibility = EXCLUDED.visibility
        RETURNING id
        """
        params = (
            item.id,
            item.title,
            item.content,
            item.domain,
            item.source,
            item.source_type,
            json.dumps(item.metadata),
            item.created_at,
            item.updated_at,
            effective_agent_id or "",
            effective_visibility,
        )

        result = self._execute_query(query, params)
        return result[0]["id"] if result else item.id

    def full_text_search(
        self,
        query_text: str,
        domain: Optional[str] = None,
        limit: int = 10,
        agent_id: Optional[str] = None,
        include_shared: bool = True,
    ) -> List[Dict[str, Any]]:
        """Full-text search in knowledge entries with tenant filtering.

        Args:
            query_text: Search query.
            domain: Optional domain filter.
            limit: Maximum results.
            agent_id: Optional agent ID for tenant filtering (v3).
            include_shared: Whether to include shared/team/public docs.

        Returns:
            List of matching entries.
        """
        # Build visibility condition
        vis_condition = ""
        if agent_id:
            if include_shared:
                vis_condition = """
                    AND (
                        owner_agent_id = %s
                        OR visibility = 'public'
                        OR visibility = 'shared'
                        OR visibility = 'team'
                        OR owner_agent_id = ''
                    )
                """
            else:
                vis_condition = """
                    AND (
                        owner_agent_id = %s
                        OR visibility = 'public'
                        OR owner_agent_id = ''
                    )
                """

        if domain:
            query = f"""
            SELECT id, title, content, domain, source, owner_agent_id, visibility,
                   ts_rank(to_tsvector('english', title || ' ' || content),
                           plainto_tsquery('english', %s)) AS rank
            FROM knowledge_entries
            WHERE domain = %s
              AND to_tsvector('english', title || ' ' || content) @@
                  plainto_tsquery('english', %s)
              {vis_condition}
            ORDER BY rank DESC
            LIMIT %s
            """
            if agent_id:
                params = (query_text, domain, query_text, agent_id, limit)
            else:
                params = (query_text, domain, query_text, limit)
            results = self._execute_query(query, params)
        else:
            query = f"""
            SELECT id, title, content, domain, source, owner_agent_id, visibility,
                   ts_rank(to_tsvector('english', title || ' ' || content),
                           plainto_tsquery('english', %s)) AS rank
            FROM knowledge_entries
            WHERE to_tsvector('english', title || ' ' || content) @@
                  plainto_tsquery('english', %s)
              {vis_condition}
            ORDER BY rank DESC
            LIMIT %s
            """
            if agent_id:
                params = (query_text, query_text, agent_id, limit)
            else:
                params = (query_text, query_text, limit)
            results = self._execute_query(query, params)

        return results

    def get_knowledge_stats(
        self,
        domain: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> DomainStats:
        """Get knowledge statistics with optional tenant filtering.

        Args:
            domain: Optional domain filter.
            agent_id: Optional agent ID for tenant stats (v3).

        Returns:
            Domain statistics.
        """
        where_clauses = []
        params = []

        if domain:
            where_clauses.append("domain = %s")
            params.append(domain)

        if agent_id:
            where_clauses.append(
                "(owner_agent_id = %s OR visibility = 'public' OR owner_agent_id = '')"
            )
            params.append(agent_id)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        query = f"""
        SELECT domain,
               COUNT(*) AS document_count,
               AVG(LENGTH(content)) AS avg_content_length
        FROM knowledge_entries
        {where_sql}
        GROUP BY domain
        """

        results = self._execute_query(query, tuple(params) if params else None)

        if results:
            row = results[0]
            return DomainStats(
                domain=row.get("domain", domain or "all"),
                document_count=row["document_count"],
                chunk_count=0,
                avg_chunk_size=row["avg_content_length"] or 0,
                concepts_count=0,
            )

        return DomainStats(
            domain=domain or "all",
            document_count=0,
            chunk_count=0,
            avg_chunk_size=0,
            concepts_count=0,
        )

    # ------------------------------------------------------------------
    # Tenant-specific queries
    # ------------------------------------------------------------------

    def get_entries_by_owner(
        self,
        agent_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get all knowledge entries owned by a specific agent.

        Args:
            agent_id: Owner agent ID.
            limit: Maximum entries to return.

        Returns:
            List of knowledge entries.
        """
        query = """
        SELECT id, title, content, domain, source, source_type,
               owner_agent_id, visibility, metadata, created_at, updated_at
        FROM knowledge_entries
        WHERE owner_agent_id = %s
        ORDER BY created_at DESC
        LIMIT %s
        """
        return self._execute_query(query, (agent_id, limit))

    def get_entries_by_visibility(
        self,
        visibility: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get knowledge entries by visibility level.

        Args:
            visibility: Visibility level to filter by.
            limit: Maximum entries to return.

        Returns:
            List of knowledge entries.
        """
        query = """
        SELECT id, title, content, domain, source, source_type,
               owner_agent_id, visibility, metadata, created_at, updated_at
        FROM knowledge_entries
        WHERE visibility = %s
        ORDER BY created_at DESC
        LIMIT %s
        """
        return self._execute_query(query, (visibility, limit))

    def update_entry_visibility(
        self,
        entry_id: str,
        new_visibility: str,
    ) -> bool:
        """Update the visibility of a knowledge entry.

        Args:
            entry_id: Knowledge entry ID.
            new_visibility: New visibility level.

        Returns:
            True if updated, False if not found.
        """
        query = """
        UPDATE knowledge_entries
        SET visibility = %s, updated_at = NOW()
        WHERE id = %s
        RETURNING id
        """
        result = self._execute_query(query, (new_visibility, entry_id))
        return len(result) > 0

    def update_entry_owner(
        self,
        entry_id: str,
        new_owner_agent_id: str,
    ) -> bool:
        """Update the owner of a knowledge entry.

        Args:
            entry_id: Knowledge entry ID.
            new_owner_agent_id: New owner agent ID.

        Returns:
            True if updated, False if not found.
        """
        query = """
        UPDATE knowledge_entries
        SET owner_agent_id = %s, updated_at = NOW()
        WHERE id = %s
        RETURNING id
        """
        result = self._execute_query(query, (new_owner_agent_id, entry_id))
        return len(result) > 0

    # ========================================================================
    # Schema Management
    # ========================================================================

    def initialize_schema(self) -> None:
        """Initialize database schema with v3 multi-tenancy support.

        Creates all tables, indexes, and RLS policies.
        Backward-compatible: adds columns to existing tables if they don't exist.
        """
        schema_sql = """
        -- =====================================================================
        -- Extensions
        -- =====================================================================
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

        -- =====================================================================
        -- Session memories table (unchanged from v2)
        -- =====================================================================
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

        CREATE INDEX IF NOT EXISTS idx_session_memories_agent_id
            ON session_memories(agent_id);

        CREATE INDEX IF NOT EXISTS idx_session_memories_last_active
            ON session_memories(last_active);

        -- =====================================================================
        -- Query records table (unchanged from v2)
        -- =====================================================================
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

        CREATE INDEX IF NOT EXISTS idx_query_records_session_id
            ON query_records(session_id);

        -- =====================================================================
        -- Knowledge entries table (v3: with owner_agent_id + visibility)
        -- =====================================================================
        CREATE TABLE IF NOT EXISTS knowledge_entries (
            id VARCHAR(255) PRIMARY KEY,
            title VARCHAR(500) NOT NULL,
            content TEXT NOT NULL,
            domain VARCHAR(100),
            source VARCHAR(500),
            source_type VARCHAR(50) DEFAULT 'document',
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            -- v3 multi-tenancy columns
            owner_agent_id VARCHAR(255) NOT NULL DEFAULT '',
            visibility VARCHAR(50) NOT NULL DEFAULT 'public'
        );

        -- Add columns if they don't exist (migration from v2)
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_entries' AND column_name = 'owner_agent_id'
            ) THEN
                ALTER TABLE knowledge_entries
                    ADD COLUMN owner_agent_id VARCHAR(255) NOT NULL DEFAULT '';
            END IF;

            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'knowledge_entries' AND column_name = 'visibility'
            ) THEN
                ALTER TABLE knowledge_entries
                    ADD COLUMN visibility VARCHAR(50) NOT NULL DEFAULT 'public';
            END IF;
        END $$;

        -- Indexes for knowledge_entries
        CREATE INDEX IF NOT EXISTS idx_knowledge_entries_domain
            ON knowledge_entries(domain);

        CREATE INDEX IF NOT EXISTS idx_knowledge_entries_fts
            ON knowledge_entries USING gin(to_tsvector('english', title || ' ' || content));

        CREATE INDEX IF NOT EXISTS idx_knowledge_entries_created_at
            ON knowledge_entries(created_at);

        CREATE INDEX IF NOT EXISTS idx_knowledge_entries_owner
            ON knowledge_entries(owner_agent_id);

        CREATE INDEX IF NOT EXISTS idx_knowledge_entries_visibility
            ON knowledge_entries(visibility);

        -- =====================================================================
        -- RLS (Row Level Security) for knowledge_entries
        -- =====================================================================

        -- Create schema for session config if not exists
        CREATE SCHEMA IF NOT EXISTS gdrag;

        -- Create session config function for RLS
        -- This allows setting per-request agent context via:
        --   SET LOCAL gdrag.current_agent_id = 'agent_123';
        -- Or: SELECT set_config('gdrag.current_agent_id', 'agent_123', false);

        -- Enable RLS on knowledge_entries
        ALTER TABLE knowledge_entries ENABLE ROW LEVEL SECURITY;

        -- Drop existing policies if they exist (idempotent migration)
        DROP POLICY IF EXISTS knowledge_entries_tenant_policy ON knowledge_entries;

        -- Create RLS policy:
        -- A row is visible when:
        --   1. No agent context is set (v2 backward-compat / admin access)
        --   2. owner_agent_id matches current agent
        --   3. visibility is 'public'
        --   4. visibility is 'shared' (further filtering at app level)
        --   5. visibility is 'team' (further filtering at app level)
        --   6. owner_agent_id is empty (legacy v2 data)
        CREATE POLICY knowledge_entries_tenant_policy ON knowledge_entries
            FOR ALL
            USING (
                current_setting('gdrag.current_agent_id', true) = ''
                OR current_setting('gdrag.current_agent_id', true) IS NULL
                OR owner_agent_id = current_setting('gdrag.current_agent_id', true)
                OR visibility = 'public'
                OR visibility = 'shared'
                OR visibility = 'team'
                OR owner_agent_id = ''
            );

        -- Allow INSERT/UPDATE with proper owner_agent_id
        DROP POLICY IF EXISTS knowledge_entries_insert_policy ON knowledge_entries;

        CREATE POLICY knowledge_entries_insert_policy ON knowledge_entries
            FOR INSERT
            WITH CHECK (
                current_setting('gdrag.current_agent_id', true) = ''
                OR current_setting('gdrag.current_agent_id', true) IS NULL
                OR owner_agent_id = current_setting('gdrag.current_agent_id', true)
                OR owner_agent_id = ''
            );

        -- =====================================================================
        -- Grant permissions for the gdrag schema
        -- =====================================================================
        GRANT USAGE ON SCHEMA gdrag TO PUBLIC;
        """

        conn = self._get_connection()
        with conn.cursor() as cursor:
            cursor.execute(schema_sql)
        logger.info("Database schema initialized with v3 multi-tenancy support")

    def initialize_rls_for_table(self, table_name: str) -> None:
        """Initialize RLS policies for a specific table.

        Useful for enabling RLS on custom tables that follow the same
        owner_agent_id / visibility pattern.

        Args:
            table_name: Name of the table to enable RLS on.
        """
        sql = f"""
        ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY;

        DROP POLICY IF EXISTS {table_name}_tenant_policy ON {table_name};

        CREATE POLICY {table_name}_tenant_policy ON {table_name}
            FOR ALL
            USING (
                current_setting('gdrag.current_agent_id', true) = ''
                OR current_setting('gdrag.current_agent_id', true) IS NULL
                OR owner_agent_id = current_setting('gdrag.current_agent_id', true)
                OR visibility = 'public'
                OR owner_agent_id = ''
            );
        """
        conn = self._get_connection()
        with conn.cursor() as cursor:
            cursor.execute(sql)
        logger.info(f"RLS enabled for table: {table_name}")
