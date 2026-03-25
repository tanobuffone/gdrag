"""Enhanced PostgreSQL relational store operations for gdrag v2.

Provides session memory persistence, knowledge management, and full-text search.
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
    """Enhanced PostgreSQL operations with session and knowledge management."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._connection = None

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

    def get_session_memory(self, session_id: str) -> Optional[SessionMemory]:
        """Retrieve session memory from PostgreSQL.

        Args:
            session_id: Session ID to retrieve.

        Returns:
            Session memory or None if not found.
        """
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

    def delete_expired_sessions(self, ttl_hours: int = 24) -> int:
        """Delete expired sessions.

        Args:
            ttl_hours: Session time-to-live in hours.

        Returns:
            Number of sessions deleted.
        """
        query = """
        DELETE FROM session_memories
        WHERE last_active < NOW() - INTERVAL '%s hours'
        RETURNING session_id
        """
        results = self._execute_query(query, (ttl_hours,))
        deleted_count = len(results)
        logger.info(f"Deleted {deleted_count} expired sessions")
        return deleted_count

    # ========================================================================
    # Knowledge Operations
    # ========================================================================

    def store_knowledge_entry(self, item: KnowledgeItem) -> str:
        """Store a knowledge entry.

        Args:
            item: Knowledge item to store.

        Returns:
            Knowledge item ID.
        """
        import json

        query = """
        INSERT INTO knowledge_entries (
            id, title, content, domain, source, source_type,
            metadata, created_at, updated_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            domain = EXCLUDED.domain,
            metadata = EXCLUDED.metadata,
            updated_at = EXCLUDED.updated_at
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
        )

        result = self._execute_query(query, params)
        return result[0]["id"] if result else item.id

    def full_text_search(
        self,
        query_text: str,
        domain: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Full-text search in knowledge entries.

        Args:
            query_text: Search query.
            domain: Optional domain filter.
            limit: Maximum results.

        Returns:
            List of matching entries.
        """
        if domain:
            query = """
            SELECT id, title, content, domain, source,
                   ts_rank(to_tsvector('english', title || ' ' || content),
                           plainto_tsquery('english', %s)) AS rank
            FROM knowledge_entries
            WHERE domain = %s
              AND to_tsvector('english', title || ' ' || content) @@
                  plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
            """
            results = self._execute_query(query, (query_text, domain, query_text, limit))
        else:
            query = """
            SELECT id, title, content, domain, source,
                   ts_rank(to_tsvector('english', title || ' ' || content),
                           plainto_tsquery('english', %s)) AS rank
            FROM knowledge_entries
            WHERE to_tsvector('english', title || ' ' || content) @@
                  plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
            """
            results = self._execute_query(query, (query_text, query_text, limit))

        return results

    def get_knowledge_stats(
        self,
        domain: Optional[str] = None,
    ) -> DomainStats:
        """Get knowledge statistics.

        Args:
            domain: Optional domain filter.

        Returns:
            Domain statistics.
        """
        if domain:
            query = """
            SELECT domain,
                   COUNT(*) AS document_count,
                   AVG(LENGTH(content)) AS avg_content_length
            FROM knowledge_entries
            WHERE domain = %s
            GROUP BY domain
            """
            results = self._execute_query(query, (domain,))
        else:
            query = """
            SELECT domain,
                   COUNT(*) AS document_count,
                   AVG(LENGTH(content)) AS avg_content_length
            FROM knowledge_entries
            GROUP BY domain
            """
            results = self._execute_query(query)

        if results:
            row = results[0]
            return DomainStats(
                domain=row.get("domain", domain or "all"),
                document_count=row["document_count"],
                chunk_count=0,  # Would need to join with chunks table
                avg_chunk_size=row["avg_content_length"] or 0,
                concepts_count=0,  # Would need to query graph store
            )

        return DomainStats(
            domain=domain or "all",
            document_count=0,
            chunk_count=0,
            avg_chunk_size=0,
            concepts_count=0,
        )

    # ========================================================================
    # Schema Management
    # ========================================================================

    def initialize_schema(self) -> None:
        """Initialize database schema if not exists."""
        schema_sql = """
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

        -- Query records table
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
        """

        conn = self._get_connection()
        with conn.cursor() as cursor:
            cursor.execute(schema_sql)
        logger.info("Database schema initialized")