"""Session memory management for gdrag v2.

Provides session creation, persistence, context retrieval, and compression.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import uuid4

from ..core.config import AppConfig, SessionConfig
from ..core.relational_store import EnhancedRelationalStore
from ..models.schemas import QueryRecord, SessionMemory

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages session memory for agents.

    Handles session lifecycle, query history, context retrieval,
    and automatic compression of old sessions.
    """

    def __init__(
        self,
        config: AppConfig,
        relational_store: Optional[EnhancedRelationalStore] = None,
    ):
        self.config = config
        self.session_config = config.session
        self.relational_store = relational_store
        self._active_sessions: Dict[str, SessionMemory] = {}

    def _get_store(self) -> Optional[EnhancedRelationalStore]:
        """Get relational store if available."""
        return self.relational_store

    def create_session(
        self,
        agent_id: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> SessionMemory:
        """Create a new session.

        Args:
            agent_id: Agent identifier.
            session_id: Optional custom session ID.
            metadata: Optional session metadata.

        Returns:
            Created session memory.
        """
        session = SessionMemory(
            session_id=session_id or str(uuid4()),
            agent_id=agent_id,
            created_at=datetime.utcnow(),
            last_active=datetime.utcnow(),
            query_history=[],
            context_summary=None,
            token_count=0,
            max_tokens=self.session_config.max_tokens,
            metadata=metadata or {},
        )

        # Store in memory
        self._active_sessions[session.session_id] = session

        # Persist to database
        store = self._get_store()
        if store:
            try:
                store.store_session_memory(session)
            except Exception as e:
                logger.warning(f"Failed to persist session: {e}")

        logger.info(f"Created session {session.session_id} for agent {agent_id}")
        return session

    def get_session(self, session_id: str) -> Optional[SessionMemory]:
        """Retrieve a session.

        Args:
            session_id: Session ID to retrieve.

        Returns:
            Session memory or None if not found.
        """
        # Check in-memory cache first
        if session_id in self._active_sessions:
            session = self._active_sessions[session_id]

            # Check if expired
            if session.is_expired(self.session_config.session_ttl_hours):
                self.delete_session(session_id)
                return None

            return session

        # Try to load from database
        store = self._get_store()
        if store:
            try:
                session = store.get_session_memory(session_id)
                if session:
                    # Check if expired
                    if session.is_expired(self.session_config.session_ttl_hours):
                        self.delete_session(session_id)
                        return None

                    # Cache in memory
                    self._active_sessions[session_id] = session
                    return session
            except Exception as e:
                logger.warning(f"Failed to load session: {e}")

        return None

    def update_session(
        self,
        session_id: str,
        query_record: QueryRecord,
    ) -> bool:
        """Add a query record to session history.

        Args:
            session_id: Session ID to update.
            query_record: Query record to add.

        Returns:
            True if successful, False otherwise.
        """
        session = self.get_session(session_id)
        if session is None:
            logger.warning(f"Session {session_id} not found")
            return False

        # Update session
        session.query_history.append(query_record)
        session.last_active = datetime.utcnow()

        # Maintain max history items
        if len(session.query_history) > self.session_config.max_history_items:
            session.query_history = session.query_history[-self.session_config.max_history_items:]

        # Update token count (approximate)
        session.token_count = sum(
            len(q.query.split()) * 1.3 for q in session.query_history
        )

        # Update in-memory cache
        self._active_sessions[session_id] = session

        # Persist to database
        store = self._get_store()
        if store:
            try:
                store.store_session_memory(session)
                store.store_query_record(session_id, query_record)
            except Exception as e:
                logger.warning(f"Failed to persist session update: {e}")

        return True

    def get_session_context(
        self,
        session_id: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Get relevant context from session history.

        Args:
            session_id: Session ID.
            max_tokens: Maximum tokens to include.

        Returns:
            Context string from session history.
        """
        session = self.get_session(session_id)
        if session is None:
            return ""

        max_tok = max_tokens or self.session_config.max_tokens

        # If we have a compressed summary, use it
        if session.context_summary:
            return session.context_summary

        # Build context from recent queries
        context_parts = []
        token_count = 0

        # Start with most recent queries
        for query_record in reversed(session.query_history):
            query_text = f"Q: {query_record.query}"
            query_tokens = len(query_text.split()) * 1.3

            if token_count + query_tokens > max_tok:
                break

            context_parts.insert(0, query_text)
            token_count += query_tokens

        if not context_parts:
            return ""

        return "\n".join(context_parts)

    def compress_session_history(self, session_id: str) -> bool:
        """Compress old session history into a summary.

        Args:
            session_id: Session ID to compress.

        Returns:
            True if successful, False otherwise.
        """
        session = self.get_session(session_id)
        if session is None:
            return False

        if len(session.query_history) < 10:
            # Not enough history to compress
            return False

        # Build summary from old queries
        old_queries = session.query_history[:-10]  # Keep last 10
        if not old_queries:
            return False

        # Create simple summary
        topics = set()
        for q in old_queries:
            # Extract key words (simple approach)
            words = q.query.lower().split()
            significant = [w for w in words if len(w) > 4]
            topics.update(significant[:3])

        summary = f"Previous discussion topics: {', '.join(list(topics)[:10])}"

        # Update session
        session.context_summary = summary
        session.query_history = session.query_history[-10:]  # Keep only recent
        session.token_count = sum(
            len(q.query.split()) * 1.3 for q in session.query_history
        )

        # Update in-memory cache
        self._active_sessions[session_id] = session

        # Persist
        store = self._get_store()
        if store:
            try:
                store.store_session_memory(session)
            except Exception as e:
                logger.warning(f"Failed to persist compressed session: {e}")

        logger.info(f"Compressed session {session_id}")
        return True

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session ID to delete.

        Returns:
            True if successful, False otherwise.
        """
        # Remove from memory
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]

        # Note: Database deletion would require additional SQL
        # For now, sessions expire naturally via TTL

        logger.info(f"Deleted session {session_id}")
        return True

    def list_sessions(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[SessionMemory]:
        """List sessions.

        Args:
            agent_id: Optional agent ID filter.
            limit: Maximum sessions to return.

        Returns:
            List of session memories.
        """
        # Filter in-memory sessions
        sessions = list(self._active_sessions.values())

        if agent_id:
            sessions = [s for s in sessions if s.agent_id == agent_id]

        # Sort by last active
        sessions.sort(key=lambda s: s.last_active, reverse=True)

        # Remove expired sessions
        valid_sessions = []
        for session in sessions:
            if not session.is_expired(self.session_config.session_ttl_hours):
                valid_sessions.append(session)
            else:
                self.delete_session(session.session_id)

        return valid_sessions[:limit]

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions.

        Returns:
            Number of sessions cleaned up.
        """
        expired = []

        for session_id, session in self._active_sessions.items():
            if session.is_expired(self.session_config.session_ttl_hours):
                expired.append(session_id)

        for session_id in expired:
            self.delete_session(session_id)

        # Also clean up in database
        store = self._get_store()
        if store:
            try:
                db_deleted = store.delete_expired_sessions(
                    self.session_config.session_ttl_hours
                )
                logger.info(f"Cleaned up {db_deleted} expired sessions from database")
            except Exception as e:
                logger.warning(f"Failed to clean up database sessions: {e}")

        logger.info(f"Cleaned up {len(expired)} expired sessions from memory")
        return len(expired)

    def get_stats(self) -> Dict:
        """Get session statistics.

        Returns:
            Statistics dictionary.
        """
        active = sum(
            1 for s in self._active_sessions.values()
            if not s.is_expired(self.session_config.session_ttl_hours)
        )

        return {
            "total_sessions": len(self._active_sessions),
            "active_sessions": active,
            "max_tokens_per_session": self.session_config.max_tokens,
            "session_ttl_hours": self.session_config.session_ttl_hours,
        }