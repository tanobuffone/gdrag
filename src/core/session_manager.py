"""Session memory management for gdrag v3.

Provides session creation, persistence, context retrieval, and compression
with multi-tenancy support via agent_id ownership verification.

Publishes events to EventBus when sessions are created or compressed.

Compatibility:
    - API v2: agent_id is optional, no ownership verification
    - API v3: agent_id required, ownership verified on all operations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..core.config import AppConfig, SessionConfig
from ..core.relational_store import EnhancedRelationalStore
from ..models.events import Event, EventType, StreamName
from ..models.schemas import QueryRecord, SessionMemory

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages session memory for agents with multi-tenancy support.

    Handles session lifecycle, query history, context retrieval,
    and automatic compression of old sessions.

    Multi-tenancy (v3):
        When agent_id is provided to get/delete operations, ownership
        is verified. Sessions can only be accessed by their owning agent.

    Compatibility (v2):
        When agent_id is omitted, operations work without ownership
        checks, maintaining backward compatibility with API v2.
    """

    def __init__(
        self,
        config: AppConfig,
        relational_store: Optional[EnhancedRelationalStore] = None,
        event_bus: Optional[Any] = None,
    ):
        self.config = config
        self.session_config = config.session
        self.relational_store = relational_store
        self.event_bus = event_bus
        self._active_sessions: Dict[str, SessionMemory] = {}

    def _get_store(self) -> Optional[EnhancedRelationalStore]:
        """Get relational store if available."""
        return self.relational_store

    def _publish_event(self, stream: str, event: Event) -> None:
        """Publish an event to the EventBus if available.

        Uses a fire-and-forget pattern: if the event bus is unavailable
        or publishing fails, the error is logged but not raised.
        """
        if self.event_bus is None:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.event_bus.publish_safe(stream, event))
            else:
                loop.run_until_complete(self.event_bus.publish(stream, event))
        except RuntimeError:
            try:
                asyncio.run(self.event_bus.publish(stream, event))
            except Exception as exc:
                logger.debug("Could not publish event: %s", exc)
        except Exception as exc:
            logger.debug("Could not publish event: %s", exc)

    def create_session(
        self,
        agent_id: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> SessionMemory:
        """Create a new session for an agent.

        Args:
            agent_id: Agent identifier (required for multi-tenancy).
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

        # Publish session.created event
        self._publish_event(
            StreamName.SESSIONS.value,
            Event(
                event_type=EventType.SESSION_CREATED.value,
                source="session_manager",
                payload={
                    "session_id": session.session_id,
                    "agent_id": agent_id,
                    "max_tokens": self.session_config.max_tokens,
                },
            ),
        )

        logger.info(f"Created session {session.session_id} for agent {agent_id}")
        return session

    def get_session(
        self, session_id: str, agent_id: Optional[str] = None
    ) -> Optional[SessionMemory]:
        """Retrieve a session with optional ownership verification.

        Args:
            session_id: Session ID to retrieve.
            agent_id: Optional agent ID for ownership verification.
                      If provided, only returns session if owned by agent.

        Returns:
            Session memory or None if not found/not owned.
        """
        # Check in-memory cache first
        if session_id in self._active_sessions:
            session = self._active_sessions[session_id]

            # Verify ownership if agent_id provided (v3 behavior)
            if agent_id and session.agent_id != agent_id:
                logger.warning(
                    f"Session {session_id} not owned by agent {agent_id}"
                )
                return None

            # Check if expired
            if session.is_expired(self.session_config.session_ttl_hours):
                self.delete_session(session_id, agent_id=agent_id)
                return None

            return session

        # Try to load from database
        store = self._get_store()
        if store:
            try:
                # Pass agent_id for ownership verification in DB query
                session = store.get_session_memory(session_id, agent_id=agent_id)
                if session:
                    # Check if expired
                    if session.is_expired(self.session_config.session_ttl_hours):
                        self.delete_session(session_id, agent_id=agent_id)
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
        agent_id: Optional[str] = None,
    ) -> bool:
        """Add a query record to session history.

        Args:
            session_id: Session ID to update.
            query_record: Query record to add.
            agent_id: Optional agent ID for ownership verification.

        Returns:
            True if successful, False otherwise.
        """
        session = self.get_session(session_id, agent_id=agent_id)
        if session is None:
            logger.warning(f"Session {session_id} not found or not accessible")
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
        agent_id: Optional[str] = None,
    ) -> str:
        """Get relevant context from session history.

        Args:
            session_id: Session ID.
            max_tokens: Maximum tokens to include.
            agent_id: Optional agent ID for ownership verification.

        Returns:
            Context string from session history.
        """
        session = self.get_session(session_id, agent_id=agent_id)
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

    def compress_session_history(
        self, session_id: str, agent_id: Optional[str] = None
    ) -> bool:
        """Compress old session history into a summary.

        Args:
            session_id: Session ID to compress.
            agent_id: Optional agent ID for ownership verification.

        Returns:
            True if successful, False otherwise.
        """
        session = self.get_session(session_id, agent_id=agent_id)
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

        # Publish session.compressed event
        self._publish_event(
            StreamName.SESSIONS.value,
            Event(
                event_type=EventType.SESSION_COMPRESSED.value,
                source="session_manager",
                payload={
                    "session_id": session_id,
                    "agent_id": session.agent_id,
                    "summary": summary,
                    "queries_retained": len(session.query_history),
                    "queries_compressed": len(old_queries),
                },
            ),
        )

        logger.info(f"Compressed session {session_id}")
        return True

    def delete_session(
        self, session_id: str, agent_id: Optional[str] = None
    ) -> bool:
        """Delete a session with optional ownership verification.

        Args:
            session_id: Session ID to delete.
            agent_id: Optional agent ID for ownership verification.
                      If provided, only deletes if owned by agent.

        Returns:
            True if successful, False if not found/not owned.
        """
        # Check ownership in memory cache before deleting
        if session_id in self._active_sessions:
            session = self._active_sessions[session_id]
            if agent_id and session.agent_id != agent_id:
                logger.warning(
                    f"Cannot delete session {session_id}: not owned by agent {agent_id}"
                )
                return False
            del self._active_sessions[session_id]

        # Delete from database with ownership check
        store = self._get_store()
        if store:
            try:
                deleted = store.delete_session(session_id, agent_id=agent_id)
                if not deleted:
                    logger.warning(
                        f"Session {session_id} not found in database for deletion"
                    )
                    return False
            except Exception as e:
                logger.warning(f"Failed to delete session from database: {e}")
                return False

        logger.info(f"Deleted session {session_id}")
        return True

    def list_sessions(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[SessionMemory]:
        """List sessions, optionally filtered by agent.

        Args:
            agent_id: Optional agent ID filter.
                      For v3 multi-tenancy, this should be provided.
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

        # Also query database for sessions not in memory cache
        store = self._get_store()
        if store:
            try:
                db_sessions = store.list_sessions(agent_id=agent_id, limit=limit)
                # Merge with in-memory sessions (avoid duplicates)
                existing_ids = {s.session_id for s in valid_sessions}
                for db_session in db_sessions:
                    if db_session.session_id not in existing_ids:
                        if not db_session.is_expired(self.session_config.session_ttl_hours):
                            valid_sessions.append(db_session)
                            # Cache in memory
                            self._active_sessions[db_session.session_id] = db_session
            except Exception as e:
                logger.warning(f"Failed to list sessions from database: {e}")

        # Re-sort after merge and apply limit
        valid_sessions.sort(key=lambda s: s.last_active, reverse=True)
        return valid_sessions[:limit]

    def cleanup_expired_sessions(self, agent_id: Optional[str] = None) -> int:
        """Clean up expired sessions.

        Args:
            agent_id: Optional agent ID filter.

        Returns:
            Number of sessions cleaned up.
        """
        expired = []

        for session_id, session in self._active_sessions.items():
            if agent_id and session.agent_id != agent_id:
                continue
            if session.is_expired(self.session_config.session_ttl_hours):
                expired.append(session_id)

        for session_id in expired:
            self.delete_session(session_id)

        # Also clean up in database
        store = self._get_store()
        if store:
            try:
                db_deleted = store.delete_expired_sessions(
                    self.session_config.session_ttl_hours,
                    agent_id=agent_id,
                )
                logger.info(f"Cleaned up {db_deleted} expired sessions from database")
            except Exception as e:
                logger.warning(f"Failed to clean up database sessions: {e}")

        logger.info(f"Cleaned up {len(expired)} expired sessions from memory")
        return len(expired)

    def get_stats(self, agent_id: Optional[str] = None) -> Dict:
        """Get session statistics.

        Args:
            agent_id: Optional agent ID filter for per-agent stats.

        Returns:
            Statistics dictionary.
        """
        sessions = list(self._active_sessions.values())

        if agent_id:
            sessions = [s for s in sessions if s.agent_id == agent_id]

        active = sum(
            1 for s in sessions
            if not s.is_expired(self.session_config.session_ttl_hours)
        )

        return {
            "total_sessions": len(sessions),
            "active_sessions": active,
            "max_tokens_per_session": self.session_config.max_tokens,
            "session_ttl_hours": self.session_config.session_ttl_hours,
            "agent_id": agent_id,
        }
