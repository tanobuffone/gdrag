"""State management for gdrag v3 agents.

Provides async state persistence with Redis (active state cache) and
PostgreSQL (durable snapshots). Decision audit trail is stored in PostgreSQL
with full-text search support.

Architecture:
    - Redis: Fast access to current agent state via key-value cache.
    - PostgreSQL: Durable storage for state snapshots and decision history.
    - asyncio.to_thread: Wraps synchronous psycopg2 calls for async API.

Usage::

    manager = StateManager(redis_url="redis://localhost:6379/0",
                           postgres_dsn="postgresql://gdrag:gdrag@localhost:5432/gdrag")
    # or with an existing pool/connection strings:
    manager = StateManager()
    snapshot_id = await manager.save_snapshot("agent-1", agent_state)
    state = await manager.load_snapshot("agent-1")
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import psycopg2
import redis.asyncio as aioredis

from ..models.state import AgentState, Decision

logger = logging.getLogger(__name__)

# ─── Redis key patterns ──────────────────────────────────────────────────────
STATE_ACTIVE_KEY = "gdrag:state:active:{agent_id}"
DECISIONS_LIST_KEY = "gdrag:decisions:{agent_id}"


class StateManagerError(Exception):
    """Base exception for StateManager errors."""


class SnapshotNotFoundError(StateManagerError):
    """Raised when a snapshot is not found."""


class StateManager:
    """Async state manager backed by Redis and PostgreSQL.

    Manages agent cognitive state snapshots (Redis for hot cache,
    PostgreSQL for durable storage) and records agent decisions for
    audit trails and learning.

    Args:
        redis_url: Redis connection URL.
            Defaults to ``redis://localhost:6379/0``.
        postgres_dsn: PostgreSQL DSN string.
            Defaults to ``postgresql://gdrag:gdrag@localhost:5432/gdrag``.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        postgres_dsn: str = "postgresql://gdrag:gdrag@localhost:5432/gdrag",
    ) -> None:
        self._redis_url = redis_url
        self._postgres_dsn = postgres_dsn
        self._redis: Optional[aioredis.Redis] = None
        self._pgconn = None  # lazy psycopg2 connection
        self._auto_save_tasks: Dict[str, asyncio.Task] = {}

    # ─── Connection helpers ───────────────────────────────────────────────────

    async def _get_redis(self) -> aioredis.Redis:
        """Lazily initialise and return the async Redis client."""
        if self._redis is None:
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    def _get_pgconn(self):
        """Lazily create a synchronous psycopg2 connection (run in thread)."""
        if self._pgconn is None or self._pgconn.closed:
            self._pgconn = psycopg2.connect(self._postgres_dsn)
            self._pgconn.autocommit = True
            logger.debug("StateManager connected to PostgreSQL")
        return self._pgconn

    async def close(self) -> None:
        """Shut down connections and cancel auto-save tasks."""
        for task in self._auto_save_tasks.values():
            task.cancel()
        self._auto_save_tasks.clear()

        if self._redis is not None:
            await self._redis.close()
            self._redis = None

        if self._pgconn is not None and not self._pgconn.closed:
            self._pgconn.close()
            self._pgconn = None

    # ─── Snapshot CRUD ────────────────────────────────────────────────────────

    async def save_snapshot(self, agent_id: str, state: AgentState) -> str:
        """Save an agent state snapshot.

        Stores the state in Redis for fast access and persists it to PostgreSQL.

        Args:
            agent_id: Agent identifier.
            state: AgentState to persist. If ``state.snapshot_id`` is not set,
                a new UUID is generated.

        Returns:
            The snapshot_id string.
        """
        # Ensure snapshot_id
        if not state.snapshot_id:
            state.snapshot_id = str(uuid4())
        # Ensure agent_id consistency
        state.agent_id = agent_id

        state_json = state.model_dump_json()

        # ── Redis: cache active state ─────────────────────────────────────
        redis_client = await self._get_redis()
        state_key = STATE_ACTIVE_KEY.format(agent_id=agent_id)
        await redis_client.set(state_key, state_json, ex=86400)  # 24h TTL

        # ── PostgreSQL: durable snapshot ───────────────────────────────────
        def _pg_save():
            conn = self._get_pgconn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_snapshots
                        (snapshot_id, agent_id, active_sessions,
                         recent_queries, attention_focus,
                         pending_tasks, metrics, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        state.snapshot_id,
                        agent_id,
                        state.active_sessions,
                        state.recent_queries,
                        state.attention_focus,
                        state.pending_tasks,
                        json.dumps(state.metrics),
                        json.dumps(state.metadata),
                        state.created_at.isoformat(),
                    ),
                )

        await asyncio.to_thread(_pg_save)
        logger.info("Saved snapshot %s for agent %s", state.snapshot_id, agent_id)
        return state.snapshot_id

    async def load_snapshot(
        self, agent_id: str, snapshot_id: Optional[str] = None
    ) -> AgentState:
        """Load an agent state snapshot.

        If ``snapshot_id`` is provided, loads that specific snapshot from
        PostgreSQL. Otherwise, tries Redis first (active state) and falls
        back to the latest snapshot in PostgreSQL.

        Args:
            agent_id: Agent identifier.
            snapshot_id: Optional specific snapshot to load.

        Returns:
            An ``AgentState`` instance.

        Raises:
            SnapshotNotFoundError: If no snapshot exists for the agent.
        """
        if snapshot_id is None:
            # ── Try Redis cache first ─────────────────────────────────────
            redis_client = await self._get_redis()
            state_key = STATE_ACTIVE_KEY.format(agent_id=agent_id)
            cached = await redis_client.get(state_key)
            if cached is not None:
                return AgentState.model_validate_json(cached)

        # ── PostgreSQL ─────────────────────────────────────────────────────
        def _pg_load():
            conn = self._get_pgconn()
            with conn.cursor() as cur:
                if snapshot_id is not None:
                    cur.execute(
                        """
                        SELECT snapshot_id, agent_id, active_sessions,
                               recent_queries, attention_focus,
                               pending_tasks, metrics, metadata, created_at
                        FROM agent_snapshots
                        WHERE snapshot_id = %s AND agent_id = %s
                        """,
                        (snapshot_id, agent_id),
                    )
                else:
                    cur.execute(
                        """
                        SELECT snapshot_id, agent_id, active_sessions,
                               recent_queries, attention_focus,
                               pending_tasks, metrics, metadata, created_at
                        FROM agent_snapshots
                        WHERE agent_id = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (agent_id,),
                    )
                return cur.fetchone()

        row = await asyncio.to_thread(_pg_load)

        if row is None:
            raise SnapshotNotFoundError(
                f"No snapshot found for agent {agent_id}"
                + (f" with snapshot_id {snapshot_id}" if snapshot_id else "")
            )

        return AgentState(
            snapshot_id=str(row[0]),
            agent_id=row[1],
            active_sessions=row[2] or [],
            recent_queries=row[3] or [],
            attention_focus=row[4],
            pending_tasks=row[5] or [],
            metrics=row[6] if isinstance(row[6], dict) else json.loads(row[6]) if row[6] else {},
            metadata=row[7] if isinstance(row[7], dict) else json.loads(row[7]) if row[7] else {},
            created_at=(
                row[8].replace(tzinfo=timezone.utc)
                if row[8] and row[8].tzinfo is None
                else row[8]
            ),
        )

    async def list_snapshots(
        self, agent_id: str, limit: int = 10
    ) -> List[AgentState]:
        """List snapshots for an agent, most recent first.

        Args:
            agent_id: Agent identifier.
            limit: Maximum number of snapshots to return.

        Returns:
            List of ``AgentState`` instances.
        """

        def _pg_list():
            conn = self._get_pgconn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT snapshot_id, agent_id, active_sessions,
                           recent_queries, attention_focus,
                           pending_tasks, metrics, metadata, created_at
                    FROM agent_snapshots
                    WHERE agent_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (agent_id, limit),
                )
                return cur.fetchall()

        rows = await asyncio.to_thread(_pg_list)

        snapshots: List[AgentState] = []
        for row in rows:
            snapshots.append(
                AgentState(
                    snapshot_id=str(row[0]),
                    agent_id=row[1],
                    active_sessions=row[2] or [],
                    recent_queries=row[3] or [],
                    attention_focus=row[4],
                    pending_tasks=row[5] or [],
                    metrics=row[6] if isinstance(row[6], dict) else json.loads(row[6]) if row[6] else {},
                    metadata=row[7] if isinstance(row[7], dict) else json.loads(row[7]) if row[7] else {},
                    created_at=(
                        row[8].replace(tzinfo=timezone.utc)
                        if row[8] and row[8].tzinfo is None
                        else row[8]
                    ),
                )
            )
        return snapshots

    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot by its ID.

        Args:
            snapshot_id: Snapshot UUID to delete.

        Returns:
            True if a row was deleted, False otherwise.
        """

        def _pg_delete():
            conn = self._get_pgconn()
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM agent_snapshots WHERE snapshot_id = %s",
                    (snapshot_id,),
                )
                return cur.rowcount > 0

        deleted = await asyncio.to_thread(_pg_delete)
        if deleted:
            logger.info("Deleted snapshot %s", snapshot_id)
        else:
            logger.warning("Snapshot %s not found for deletion", snapshot_id)
        return deleted

    # ─── Auto-save ────────────────────────────────────────────────────────────

    async def auto_save(
        self, agent_id: str, interval_seconds: int = 300
    ) -> asyncio.Task:
        """Start a background task that periodically saves the agent's current state.

        Only one auto-save task runs per agent. Calling this again for the
        same ``agent_id`` cancels the previous task and starts a new one.

        Args:
            agent_id: Agent identifier.
            interval_seconds: Seconds between auto-saves. Default 300 (5 min).

        Returns:
            The asyncio.Task running the auto-save loop.
        """
        # Cancel existing task for this agent
        if agent_id in self._auto_save_tasks:
            self._auto_save_tasks[agent_id].cancel()

        async def _loop():
            logger.info(
                "Auto-save started for %s every %ds", agent_id, interval_seconds
            )
            try:
                while True:
                    await asyncio.sleep(interval_seconds)
                    try:
                        redis_client = await self._get_redis()
                        state_key = STATE_ACTIVE_KEY.format(agent_id=agent_id)
                        cached = await redis_client.get(state_key)
                        if cached is not None:
                            state = AgentState.model_validate_json(cached)
                            await self.save_snapshot(agent_id, state)
                            logger.debug("Auto-saved state for %s", agent_id)
                        else:
                            logger.debug(
                                "No active state in Redis for %s, skipping auto-save",
                                agent_id,
                            )
                    except Exception:
                        logger.exception("Auto-save error for agent %s", agent_id)
            except asyncio.CancelledError:
                logger.info("Auto-save cancelled for agent %s", agent_id)
                raise

        task = asyncio.create_task(_loop())
        self._auto_save_tasks[agent_id] = task
        return task

    # ─── Decision tracking ────────────────────────────────────────────────────

    async def record_decision(self, agent_id: str, decision: Decision) -> str:
        """Record an agent decision.

        Persists the decision to PostgreSQL and caches it in Redis for fast
        recent-decision queries.

        Args:
            agent_id: Agent identifier.
            decision: Decision to record. If ``decision.decision_id`` is not
                set, a new UUID is generated.

        Returns:
            The decision_id string.
        """
        if not decision.decision_id:
            decision.decision_id = str(uuid4())
        decision.agent_id = agent_id

        decision_json = decision.model_dump_json()

        # ── Redis: prepend to recent decisions list ────────────────────────
        redis_client = await self._get_redis()
        decisions_key = DECISIONS_LIST_KEY.format(agent_id=agent_id)
        pipe = redis_client.pipeline()
        pipe.lpush(decisions_key, decision_json)
        pipe.ltrim(decisions_key, 0, 99)  # keep last 100
        await pipe.execute()

        # ── PostgreSQL: durable record ─────────────────────────────────────
        def _pg_record():
            conn = self._get_pgconn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO decisions
                        (decision_id, agent_id, context, decision,
                         reasoning, outcome, outcome_metrics, metadata,
                         created_at, resolved_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        decision.decision_id,
                        agent_id,
                        json.dumps(decision.context),
                        decision.decision,
                        decision.reasoning,
                        decision.outcome,
                        json.dumps(decision.outcome_metrics),
                        json.dumps(decision.metadata),
                        decision.created_at.isoformat(),
                        decision.resolved_at.isoformat() if decision.resolved_at else None,
                    ),
                )

        await asyncio.to_thread(_pg_record)
        logger.info("Recorded decision %s for agent %s", decision.decision_id, agent_id)
        return decision.decision_id

    async def get_decision_history(
        self, agent_id: str, limit: int = 50
    ) -> List[Decision]:
        """Get decision history for an agent, most recent first.

        Args:
            agent_id: Agent identifier.
            limit: Maximum decisions to return.

        Returns:
            List of ``Decision`` instances.
        """

        def _pg_history():
            conn = self._get_pgconn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT decision_id, agent_id, context, decision,
                           reasoning, outcome, outcome_metrics, metadata,
                           created_at, resolved_at
                    FROM decisions
                    WHERE agent_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (agent_id, limit),
                )
                return cur.fetchall()

        rows = await asyncio.to_thread(_pg_history)

        decisions: List[Decision] = []
        for row in rows:
            decisions.append(
                Decision(
                    decision_id=str(row[0]),
                    agent_id=row[1],
                    context=row[2] if isinstance(row[2], dict) else json.loads(row[2]) if row[2] else {},
                    decision=row[3],
                    reasoning=row[4],
                    outcome=row[5],
                    outcome_metrics=row[6] if isinstance(row[6], dict) else json.loads(row[6]) if row[6] else {},
                    metadata=row[7] if isinstance(row[7], dict) else json.loads(row[7]) if row[7] else {},
                    created_at=(
                        row[8].replace(tzinfo=timezone.utc)
                        if row[8] and row[8].tzinfo is None
                        else row[8]
                    ),
                    resolved_at=(
                        row[9].replace(tzinfo=timezone.utc)
                        if row[9] and row[9].tzinfo is None
                        else row[9]
                    ),
                )
            )
        return decisions
