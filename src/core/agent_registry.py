"""Agent Registry for gdrag v3.

Provides agent registration, heartbeat monitoring, health tracking,
and lifecycle management backed by PostgreSQL.
"""

import json
import logging
from datetime import datetime, timezone
from typing import List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from ..core.config import AppConfig
from ..models.agent import (
    AgentHeartbeat,
    AgentInfo,
    AgentRegistration,
    AgentRegistryStats,
    HealthStatus,
)

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Agent registry backed by PostgreSQL.

    Manages agent registration, heartbeats, health status,
    and stale-agent cleanup using the ``agents`` and
    ``agent_heartbeats`` tables.

    Args:
        config: Application configuration with database settings.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._connection = None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _get_connection(self):
        """Get or create a PostgreSQL connection."""
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
                "AgentRegistry connected to PostgreSQL at %s:%s",
                self.config.database.postgres_host,
                self.config.database.postgres_port,
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
    ) -> List[dict]:
        """Execute a SQL query and optionally return rows.

        Args:
            query: SQL statement.
            params: Query parameters.
            fetch: Whether to fetch and return result rows.

        Returns:
            List of result dictionaries (empty when *fetch* is False).
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                if fetch and cursor.description:
                    return [dict(row) for row in cursor.fetchall()]
                return []
        except Exception:
            logger.exception("AgentRegistry query failed")
            raise

    # ------------------------------------------------------------------
    # Row → model helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_agent_info(row: dict) -> AgentInfo:
        """Convert a database row (or view row) to an ``AgentInfo``."""
        capabilities = row.get("capabilities") or []
        if isinstance(capabilities, str):
            capabilities = json.loads(capabilities)

        endpoints_raw = row.get("endpoints") or []
        if isinstance(endpoints_raw, str):
            endpoints_raw = json.loads(endpoints_raw)

        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        tags = row.get("tags") or []
        if isinstance(tags, str):
            tags = json.loads(tags)

        return AgentInfo(
            agent_id=row["agent_id"],
            name=row["name"],
            status=HealthStatus(row.get("status", "unknown")),
            capabilities=capabilities,
            endpoints=endpoints_raw,
            registered_at=row["registered_at"],
            last_heartbeat=row.get("last_heartbeat"),
            heartbeat_interval_s=row.get("heartbeat_interval_s", 60),
            metadata=metadata,
            tags=tags,
            updated_at=row.get("updated_at", row["registered_at"]),
        )

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def register_agent(self, agent: AgentRegistration) -> str:
        """Register a new agent or update an existing one (upsert).

        Args:
            agent: Registration request data.

        Returns:
            The ``agent_id`` of the registered agent.
        """
        query = """
        INSERT INTO agents (
            agent_id, name, capabilities, endpoints,
            metadata, tags, heartbeat_interval_s, status
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON CONFLICT (agent_id) DO UPDATE SET
            name = EXCLUDED.name,
            capabilities = EXCLUDED.capabilities,
            endpoints = EXCLUDED.endpoints,
            metadata = EXCLUDED.metadata,
            tags = EXCLUDED.tags,
            heartbeat_interval_s = EXCLUDED.heartbeat_interval_s,
            updated_at = NOW()
        RETURNING agent_id
        """
        params = (
            agent.agent_id,
            agent.name,
            agent.capabilities,
            json.dumps([ep.model_dump() for ep in agent.endpoints]),
            json.dumps(agent.metadata),
            agent.tags,
            agent.heartbeat_interval_s,
            HealthStatus.UNKNOWN.value,
        )
        result = self._execute_query(query, params)
        agent_id = result[0]["agent_id"] if result else agent.agent_id
        logger.info("Agent registered: %s", agent_id)
        return agent_id

    def unregister_agent(self, agent_id: str) -> bool:
        """Remove an agent and its heartbeat history.

        Args:
            agent_id: Identifier of the agent to remove.

        Returns:
            ``True`` if the agent existed and was deleted.
        """
        query = """
        DELETE FROM agents
        WHERE agent_id = %s
        RETURNING agent_id
        """
        result = self._execute_query(query, (agent_id,))
        deleted = len(result) > 0
        if deleted:
            logger.info("Agent unregistered: %s", agent_id)
        else:
            logger.warning("Unregister called for unknown agent: %s", agent_id)
        return deleted

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def heartbeat(
        self,
        agent_id: str,
        status: HealthStatus = HealthStatus.HEALTHY,
        metrics: Optional[dict] = None,
    ) -> bool:
        """Record a heartbeat for an agent.

        The database trigger automatically updates the agent's
        ``status`` and ``updated_at`` columns.

        Args:
            agent_id: Agent identifier.
            status: Reported health status.
            metrics: Optional metrics dict (uptime, queue_depth, …).

        Returns:
            ``True`` if the heartbeat was recorded.
        """
        query = """
        INSERT INTO agent_heartbeats (agent_id, status, metrics)
        VALUES (%s, %s, %s)
        RETURNING id
        """
        params = (agent_id, status.value, json.dumps(metrics or {}))
        result = self._execute_query(query, params)
        recorded = len(result) > 0
        if recorded:
            logger.debug("Heartbeat recorded for agent %s", agent_id)
        return recorded

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Retrieve a single agent by ID.

        Args:
            agent_id: Agent identifier.

        Returns:
            ``AgentInfo`` or ``None`` if not found.
        """
        query = """
        SELECT agent_id, name, status, capabilities, endpoints,
               metadata, tags, heartbeat_interval_s,
               registered_at, updated_at,
               (SELECT MAX(timestamp) FROM agent_heartbeats
                WHERE agent_id = a.agent_id) AS last_heartbeat
        FROM agents a
        WHERE agent_id = %s
        """
        rows = self._execute_query(query, (agent_id,))
        if not rows:
            return None
        return self._row_to_agent_info(rows[0])

    def list_agents(
        self,
        status: Optional[HealthStatus | str] = None,
        limit: int = 100,
    ) -> List[AgentInfo]:
        """List registered agents, optionally filtered by status.

        The default *status* value ``"active"`` is treated as a
        convenience filter that returns agents whose status is
        **healthy** or **degraded**.

        Args:
            status: Filter by health status (or ``"active"``).
            limit: Maximum number of agents to return.

        Returns:
            List of ``AgentInfo``.
        """
        if status is None or status == "active":
            query = """
            SELECT agent_id, name, status, capabilities, endpoints,
                   metadata, tags, heartbeat_interval_s,
                   registered_at, updated_at,
                   (SELECT MAX(timestamp) FROM agent_heartbeats
                    WHERE agent_id = a.agent_id) AS last_heartbeat
            FROM agents a
            WHERE status IN ('healthy', 'degraded')
            ORDER BY registered_at DESC
            LIMIT %s
            """
            params: tuple = (limit,)
        else:
            status_val = status.value if isinstance(status, HealthStatus) else status
            query = """
            SELECT agent_id, name, status, capabilities, endpoints,
                   metadata, tags, heartbeat_interval_s,
                   registered_at, updated_at,
                   (SELECT MAX(timestamp) FROM agent_heartbeats
                    WHERE agent_id = a.agent_id) AS last_heartbeat
            FROM agents a
            WHERE status = %s
            ORDER BY registered_at DESC
            LIMIT %s
            """
            params = (status_val, limit)

        rows = self._execute_query(query, params)
        return [self._row_to_agent_info(r) for r in rows]

    def find_agents_by_capability(
        self,
        capability: str,
        limit: int = 100,
    ) -> List[AgentInfo]:
        """Find agents that possess a given capability.

        Uses the GIN index on ``agents.capabilities`` for efficient
        array containment queries.

        Args:
            capability: Capability string to match (e.g. ``"query"``).
            limit: Maximum results.

        Returns:
            List of matching ``AgentInfo``.
        """
        query = """
        SELECT agent_id, name, status, capabilities, endpoints,
               metadata, tags, heartbeat_interval_s,
               registered_at, updated_at,
               (SELECT MAX(timestamp) FROM agent_heartbeats
                WHERE agent_id = a.agent_id) AS last_heartbeat
        FROM agents a
        WHERE capabilities @> ARRAY[%s]::varchar[]
        ORDER BY registered_at DESC
        LIMIT %s
        """
        rows = self._execute_query(query, (capability, limit))
        return [self._row_to_agent_info(r) for r in rows]

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def get_agent_health(self, agent_id: str) -> Optional[HealthStatus]:
        """Get the current health status of an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            ``HealthStatus`` or ``None`` if the agent does not exist.
        """
        query = "SELECT status FROM agents WHERE agent_id = %s"
        rows = self._execute_query(query, (agent_id,))
        if not rows:
            return None
        return HealthStatus(rows[0]["status"])

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_stale_agents(
        self,
        timeout_seconds: int = 300,
    ) -> int:
        """Mark agents whose last heartbeat exceeds *timeout_seconds* as unhealthy.

        Delegates to the ``mark_stale_agents`` database function,
        converting the timeout to a multiplier over the agent's
        configured ``heartbeat_interval_s``.

        Args:
            timeout_seconds: Maximum seconds since last heartbeat
                before an agent is considered stale.

        Returns:
            Number of agents updated.
        """
        query = "SELECT mark_stale_agents(%s::float) AS updated"
        rows = self._execute_query(query, (timeout_seconds / 60.0,))
        updated = rows[0]["updated"] if rows else 0
        if updated:
            logger.info("Marked %d stale agents as unhealthy", updated)
        return updated

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> AgentRegistryStats:
        """Return aggregate registry statistics.

        Returns:
            ``AgentRegistryStats`` with counts per status.
        """
        query = """
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE status = 'healthy') AS healthy,
            COUNT(*) FILTER (WHERE status = 'degraded') AS degraded,
            COUNT(*) FILTER (WHERE status = 'unhealthy') AS unhealthy,
            COUNT(*) FILTER (WHERE status = 'unknown') AS unknown
        FROM agents
        """
        rows = self._execute_query(query)
        row = rows[0] if rows else {}

        hb_query = "SELECT COUNT(*) AS total FROM agent_heartbeats"
        hb_rows = self._execute_query(hb_query)
        hb_total = hb_rows[0]["total"] if hb_rows else 0

        return AgentRegistryStats(
            total_agents=row.get("total", 0),
            healthy_agents=row.get("healthy", 0),
            degraded_agents=row.get("degraded", 0),
            unhealthy_agents=row.get("unhealthy", 0),
            unknown_agents=row.get("unknown", 0),
            total_heartbeats=hb_total,
            last_updated=datetime.now(timezone.utc),
        )
