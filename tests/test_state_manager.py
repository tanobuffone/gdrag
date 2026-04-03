"""Unit tests for StateManager – gdrag v3 state persistence.

Tests snapshot save/load, decision history tracking, and auto-save mechanism.
Uses fakeredis for Redis-backed state storage and mocks for async scheduling.

StateManager provides:
- Snapshot save/load of AgentState to Redis (with JSON serialization)
- Decision history tracking (append, query, resolve)
- Auto-save mechanism (periodic flush of dirty state)
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call
from uuid import uuid4

import fakeredis.aioredis
import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# StateManager – minimal implementation for testing.
# Mirrors the patterns in src/models/state.py and src/core/session_manager.py
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


class StateManager:
    """Redis-backed state manager for agent snapshots and decisions.

    Stores AgentState snapshots and Decision records in Redis hashes/lists.
    Supports auto-save with dirty-flag tracking.
    """

    def __init__(self, redis_client, prefix: str = "gdrag:state"):
        self._redis = redis_client
        self._prefix = prefix
        self._dirty_agents: set = set()
        self._auto_save_interval: float = 5.0  # seconds
        self._auto_save_task: Optional[asyncio.Task] = None
        self._running = False

    # ── Key helpers ───────────────────────────────────────────────────────

    def _snapshot_key(self, agent_id: str) -> str:
        return f"{self._prefix}:snapshot:{agent_id}"

    def _decisions_key(self, agent_id: str) -> str:
        return f"{self._prefix}:decisions:{agent_id}"

    def _decision_key(self, agent_id: str, decision_id: str) -> str:
        return f"{self._prefix}:decision:{agent_id}:{decision_id}"

    # ── Snapshot operations ───────────────────────────────────────────────

    async def save_snapshot(self, agent_id: str, state: Dict[str, Any]) -> str:
        """Save an agent state snapshot to Redis.

        Args:
            agent_id: Agent identifier.
            state: State dictionary (must contain at least agent_id).

        Returns:
            Snapshot ID (uuid4).
        """
        snapshot_id = state.get("snapshot_id", str(uuid4()))
        state["snapshot_id"] = snapshot_id
        state["agent_id"] = agent_id
        state["created_at"] = state.get(
            "created_at", datetime.now(timezone.utc).isoformat()
        )

        data = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in state.items()}
        key = self._snapshot_key(agent_id)
        await self._redis.delete(key)
        await self._redis.hset(key, mapping=data)

        # Maintain a sorted-set index by timestamp for time-range queries
        ts = datetime.now(timezone.utc).timestamp()
        await self._redis.zadd(f"{self._prefix}:snapshots:index", {agent_id: ts})

        self._dirty_agents.discard(agent_id)
        return snapshot_id

    async def load_snapshot(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load the latest snapshot for an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            State dictionary or None if not found.
        """
        key = self._snapshot_key(agent_id)
        data = await self._redis.hgetall(key)
        if not data:
            return None

        state: Dict[str, Any] = {}
        for k, v in data.items():
            k_str = k.decode() if isinstance(k, bytes) else k
            v_str = v.decode() if isinstance(v, bytes) else v
            # Try JSON decode for structured fields
            if k_str in ("active_sessions", "recent_queries", "pending_tasks", "metrics", "metadata"):
                try:
                    state[k_str] = json.loads(v_str)
                except (json.JSONDecodeError, TypeError):
                    state[k_str] = v_str
            else:
                state[k_str] = v_str
        return state

    async def delete_snapshot(self, agent_id: str) -> bool:
        """Delete an agent's snapshot.

        Args:
            agent_id: Agent identifier.

        Returns:
            True if a snapshot existed and was deleted.
        """
        key = self._snapshot_key(agent_id)
        deleted = await self._redis.delete(key)
        await self._redis.zrem(f"{self._prefix}:snapshots:index", agent_id)
        return deleted > 0

    async def list_snapshots(self) -> List[str]:
        """List all agent IDs that have snapshots.

        Returns:
            List of agent IDs.
        """
        members = await self._redis.zrange(f"{self._prefix}:snapshots:index", 0, -1)
        return [m.decode() if isinstance(m, bytes) else m for m in members]

    # ── Decision history ──────────────────────────────────────────────────

    async def record_decision(
        self,
        agent_id: str,
        decision: str,
        reasoning: str,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a new decision for an agent.

        Args:
            agent_id: Agent identifier.
            decision: The decision text.
            reasoning: Explanation of the decision.
            context: Contextual information.
            metadata: Additional metadata.

        Returns:
            Decision ID (uuid4).
        """
        decision_id = str(uuid4())
        record = {
            "decision_id": decision_id,
            "agent_id": agent_id,
            "decision": decision,
            "reasoning": reasoning,
            "context": json.dumps(context or {}),
            "outcome": "",
            "outcome_metrics": json.dumps({}),
            "metadata": json.dumps(metadata or {}),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "resolved_at": "",
        }
        d_key = self._decision_key(agent_id, decision_id)
        await self._redis.hset(d_key, mapping=record)

        # Append to the agent's decision list (Redis list)
        l_key = self._decisions_key(agent_id)
        await self._redis.rpush(l_key, decision_id)

        self._dirty_agents.add(agent_id)
        return decision_id

    async def get_decision(self, agent_id: str, decision_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific decision.

        Args:
            agent_id: Agent identifier.
            decision_id: Decision identifier.

        Returns:
            Decision dictionary or None.
        """
        d_key = self._decision_key(agent_id, decision_id)
        data = await self._redis.hgetall(d_key)
        if not data:
            return None

        result: Dict[str, Any] = {}
        for k, v in data.items():
            k_str = k.decode() if isinstance(k, bytes) else k
            v_str = v.decode() if isinstance(v, bytes) else v
            if k_str in ("context", "outcome_metrics", "metadata"):
                try:
                    result[k_str] = json.loads(v_str)
                except (json.JSONDecodeError, TypeError):
                    result[k_str] = v_str
            else:
                result[k_str] = v_str
        return result

    async def resolve_decision(
        self,
        agent_id: str,
        decision_id: str,
        outcome: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Resolve a decision with an outcome.

        Args:
            agent_id: Agent identifier.
            decision_id: Decision identifier.
            outcome: The outcome text.
            metrics: Optional quantitative metrics.

        Returns:
            True if the decision was found and resolved.
        """
        d_key = self._decision_key(agent_id, decision_id)
        exists = await self._redis.exists(d_key)
        if not exists:
            return False

        await self._redis.hset(d_key, "outcome", outcome)
        await self._redis.hset(d_key, "outcome_metrics", json.dumps(metrics or {}))
        await self._redis.hset(
            d_key, "resolved_at", datetime.now(timezone.utc).isoformat()
        )
        self._dirty_agents.add(agent_id)
        return True

    async def get_decision_history(
        self, agent_id: str, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get paginated decision history for an agent.

        Args:
            agent_id: Agent identifier.
            limit: Maximum decisions to return.
            offset: Starting offset.

        Returns:
            List of decision dictionaries in chronological order.
        """
        l_key = self._decisions_key(agent_id)
        decision_ids = await self._redis.lrange(l_key, offset, offset + limit - 1)

        decisions = []
        for did in decision_ids:
            did_str = did.decode() if isinstance(did, bytes) else did
            d = await self.get_decision(agent_id, did_str)
            if d:
                decisions.append(d)
        return decisions

    async def get_unresolved_decisions(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get all unresolved decisions for an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            List of decision dictionaries without an outcome.
        """
        all_decisions = await self.get_decision_history(agent_id, limit=1000)
        return [d for d in all_decisions if not d.get("outcome")]

    # ── Auto-save mechanism ───────────────────────────────────────────────

    def mark_dirty(self, agent_id: str) -> None:
        """Mark an agent's state as dirty (needs saving)."""
        self._dirty_agents.add(agent_id)

    def is_dirty(self, agent_id: str) -> bool:
        """Check if an agent's state is dirty."""
        return agent_id in self._dirty_agents

    async def _auto_save_loop(self, save_fn=None) -> None:
        """Background loop that flushes dirty state periodically.

        Args:
            save_fn: Optional async callable(agent_id) for actual persistence.
        """
        self._running = True
        while self._running:
            await asyncio.sleep(self._auto_save_interval)
            if not self._running:
                break
            dirty = list(self._dirty_agents)
            self._dirty_agents.clear()
            for agent_id in dirty:
                if save_fn:
                    try:
                        await save_fn(agent_id)
                    except Exception:
                        # Re-mark as dirty on failure
                        self._dirty_agents.add(agent_id)

    async def start_auto_save(self, save_fn=None) -> None:
        """Start the auto-save background task."""
        if self._auto_save_task is not None:
            return
        self._running = True
        self._auto_save_task = asyncio.create_task(self._auto_save_loop(save_fn))

    async def stop_auto_save(self) -> None:
        """Stop the auto-save background task."""
        self._running = False
        if self._auto_save_task is not None:
            self._auto_save_task.cancel()
            try:
                await self._auto_save_task
            except asyncio.CancelledError:
                pass
            self._auto_save_task = None

    async def flush(self, save_fn=None) -> int:
        """Immediately flush all dirty state.

        Args:
            save_fn: Optional async callable(agent_id).

        Returns:
            Number of agents flushed.
        """
        dirty = list(self._dirty_agents)
        self._dirty_agents.clear()
        for agent_id in dirty:
            if save_fn:
                try:
                    await save_fn(agent_id)
                except Exception:
                    self._dirty_agents.add(agent_id)
        return len(dirty)

    async def get_stats(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        snapshot_count = await self._redis.zcard(f"{self._prefix}:snapshots:index")
        return {
            "total_snapshots": snapshot_count,
            "dirty_agents": len(self._dirty_agents),
            "auto_save_running": self._running,
            "auto_save_interval_s": self._auto_save_interval,
        }


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest_asyncio.fixture
async def redis_client():
    """Provide a fresh fakeredis client per test."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.flushall()
    await client.aclose()


@pytest_asyncio.fixture
async def state_manager(redis_client):
    """Provide a StateManager instance with fakeredis."""
    return StateManager(redis_client, prefix="test:state")


@pytest.fixture
def sample_agent_state() -> Dict[str, Any]:
    """Sample agent state snapshot for testing."""
    return {
        "agent_id": "agent-001",
        "active_sessions": ["sess-aaa", "sess-bbb"],
        "recent_queries": ["query about python", "query about redis"],
        "attention_focus": "machine-learning",
        "pending_tasks": ["task-10", "task-11"],
        "metrics": {
            "queries_processed": 142,
            "avg_latency_ms": 35.7,
            "error_rate": 0.01,
        },
        "metadata": {"version": "3.0.0", "region": "us-east-1"},
    }


@pytest.fixture
def sample_decision_context() -> Dict[str, Any]:
    """Sample context for a decision record."""
    return {
        "input_tokens": 512,
        "available_tools": ["search", "ingest", "query"],
        "user_intent": "knowledge_retrieval",
    }


# ===========================================================================
# Test: Snapshot Save / Load
# ===========================================================================


class TestSnapshotSaveLoad:
    """Tests for agent state snapshot persistence."""

    @pytest.mark.asyncio
    async def test_save_creates_snapshot(self, state_manager, sample_agent_state):
        """Test that save_snapshot persists state to Redis."""
        snapshot_id = await state_manager.save_snapshot("agent-001", sample_agent_state)

        assert snapshot_id is not None
        assert isinstance(snapshot_id, str)
        assert len(snapshot_id) > 0

    @pytest.mark.asyncio
    async def test_load_returns_saved_state(self, state_manager, sample_agent_state):
        """Test that load_snapshot returns the saved state."""
        await state_manager.save_snapshot("agent-001", sample_agent_state)
        loaded = await state_manager.load_snapshot("agent-001")

        assert loaded is not None
        assert loaded["agent_id"] == "agent-001"
        assert loaded["attention_focus"] == "machine-learning"

    @pytest.mark.asyncio
    async def test_load_preserves_list_fields(self, state_manager, sample_agent_state):
        """Test that list fields are preserved through save/load."""
        await state_manager.save_snapshot("agent-001", sample_agent_state)
        loaded = await state_manager.load_snapshot("agent-001")

        assert loaded["active_sessions"] == ["sess-aaa", "sess-bbb"]
        assert loaded["recent_queries"] == ["query about python", "query about redis"]
        assert loaded["pending_tasks"] == ["task-10", "task-11"]

    @pytest.mark.asyncio
    async def test_load_preserves_dict_fields(self, state_manager, sample_agent_state):
        """Test that dict fields (metrics, metadata) are preserved."""
        await state_manager.save_snapshot("agent-001", sample_agent_state)
        loaded = await state_manager.load_snapshot("agent-001")

        assert loaded["metrics"]["queries_processed"] == 142
        assert loaded["metrics"]["avg_latency_ms"] == 35.7
        assert loaded["metadata"]["version"] == "3.0.0"

    @pytest.mark.asyncio
    async def test_load_nonexistent_returns_none(self, state_manager):
        """Test loading a non-existent snapshot returns None."""
        result = await state_manager.load_snapshot("nonexistent-agent")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_assigns_snapshot_id(self, state_manager):
        """Test that save assigns a snapshot_id if not provided."""
        state = {"agent_id": "agent-x", "attention_focus": "testing"}
        snapshot_id = await state_manager.save_snapshot("agent-x", state)

        loaded = await state_manager.load_snapshot("agent-x")
        assert loaded["snapshot_id"] == snapshot_id

    @pytest.mark.asyncio
    async def test_save_preserves_provided_snapshot_id(self, state_manager):
        """Test that save preserves an existing snapshot_id."""
        custom_id = "custom-snap-123"
        state = {"agent_id": "agent-y", "snapshot_id": custom_id}
        snapshot_id = await state_manager.save_snapshot("agent-y", state)

        assert snapshot_id == custom_id
        loaded = await state_manager.load_snapshot("agent-y")
        assert loaded["snapshot_id"] == custom_id

    @pytest.mark.asyncio
    async def test_save_overwrites_existing_snapshot(self, state_manager):
        """Test that saving again overwrites the previous snapshot."""
        state_v1 = {"agent_id": "agent-z", "attention_focus": "v1-focus"}
        state_v2 = {"agent_id": "agent-z", "attention_focus": "v2-focus"}

        await state_manager.save_snapshot("agent-z", state_v1)
        await state_manager.save_snapshot("agent-z", state_v2)

        loaded = await state_manager.load_snapshot("agent-z")
        assert loaded["attention_focus"] == "v2-focus"

    @pytest.mark.asyncio
    async def test_delete_snapshot(self, state_manager, sample_agent_state):
        """Test deleting a snapshot removes it from Redis."""
        await state_manager.save_snapshot("agent-001", sample_agent_state)
        deleted = await state_manager.delete_snapshot("agent-001")

        assert deleted is True
        loaded = await state_manager.load_snapshot("agent-001")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, state_manager):
        """Test deleting a non-existent snapshot returns False."""
        deleted = await state_manager.delete_snapshot("ghost-agent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_snapshots_empty(self, state_manager):
        """Test listing snapshots when none exist."""
        snapshots = await state_manager.list_snapshots()
        assert snapshots == []

    @pytest.mark.asyncio
    async def test_list_snapshots_returns_all(self, state_manager):
        """Test listing snapshots returns all registered agents."""
        for i in range(3):
            await state_manager.save_snapshot(f"agent-{i}", {"agent_id": f"agent-{i}"})

        snapshots = await state_manager.list_snapshots()
        assert len(snapshots) == 3
        assert "agent-0" in snapshots
        assert "agent-1" in snapshots
        assert "agent-2" in snapshots

    @pytest.mark.asyncio
    async def test_snapshot_includes_created_at(self, state_manager, sample_agent_state):
        """Test that snapshot includes a created_at timestamp."""
        await state_manager.save_snapshot("agent-001", sample_agent_state)
        loaded = await state_manager.load_snapshot("agent-001")

        assert "created_at" in loaded
        # Verify it's a valid ISO timestamp
        parsed = datetime.fromisoformat(loaded["created_at"])
        assert parsed.tzinfo is not None

    @pytest.mark.asyncio
    async def test_save_empty_state(self, state_manager):
        """Test saving a minimal state dict."""
        state = {"agent_id": "minimal-agent"}
        snapshot_id = await state_manager.save_snapshot("minimal-agent", state)

        loaded = await state_manager.load_snapshot("minimal-agent")
        assert loaded is not None
        assert loaded["agent_id"] == "minimal-agent"
        assert loaded["snapshot_id"] == snapshot_id

    @pytest.mark.asyncio
    async def test_multiple_agents_independent(self, state_manager):
        """Test that snapshots for different agents are independent."""
        state_a = {"agent_id": "agent-A", "attention_focus": "focus-A", "metrics": {"count": 1}}
        state_b = {"agent_id": "agent-B", "attention_focus": "focus-B", "metrics": {"count": 2}}

        await state_manager.save_snapshot("agent-A", state_a)
        await state_manager.save_snapshot("agent-B", state_b)

        loaded_a = await state_manager.load_snapshot("agent-A")
        loaded_b = await state_manager.load_snapshot("agent-B")

        assert loaded_a["attention_focus"] == "focus-A"
        assert loaded_b["attention_focus"] == "focus-B"
        assert loaded_a["metrics"]["count"] == 1
        assert loaded_b["metrics"]["count"] == 2


# ===========================================================================
# Test: Decision History
# ===========================================================================


class TestDecisionHistory:
    """Tests for decision recording, retrieval, and resolution."""

    @pytest.mark.asyncio
    async def test_record_decision_returns_id(self, state_manager, sample_decision_context):
        """Test that recording a decision returns a valid ID."""
        decision_id = await state_manager.record_decision(
            agent_id="agent-001",
            decision="Use semantic search",
            reasoning="User query is about technical documentation",
            context=sample_decision_context,
        )

        assert decision_id is not None
        assert isinstance(decision_id, str)
        assert len(decision_id) > 0

    @pytest.mark.asyncio
    async def test_get_decision_returns_record(self, state_manager, sample_decision_context):
        """Test that a recorded decision can be retrieved."""
        decision_id = await state_manager.record_decision(
            agent_id="agent-001",
            decision="Route to knowledge API",
            reasoning="Query matches knowledge domain",
            context=sample_decision_context,
        )

        decision = await state_manager.get_decision("agent-001", decision_id)

        assert decision is not None
        assert decision["decision_id"] == decision_id
        assert decision["agent_id"] == "agent-001"
        assert decision["decision"] == "Route to knowledge API"
        assert decision["reasoning"] == "Query matches knowledge domain"

    @pytest.mark.asyncio
    async def test_decision_preserves_context(self, state_manager, sample_decision_context):
        """Test that decision context is preserved."""
        decision_id = await state_manager.record_decision(
            agent_id="agent-001",
            decision="action",
            reasoning="reason",
            context=sample_decision_context,
        )

        decision = await state_manager.get_decision("agent-001", decision_id)

        assert decision["context"]["input_tokens"] == 512
        assert "search" in decision["context"]["available_tools"]

    @pytest.mark.asyncio
    async def test_decision_starts_unresolved(self, state_manager):
        """Test that a new decision has no outcome."""
        decision_id = await state_manager.record_decision(
            agent_id="agent-001",
            decision="Do something",
            reasoning="Because reasons",
        )

        decision = await state_manager.get_decision("agent-001", decision_id)

        assert decision["outcome"] == ""
        assert decision["resolved_at"] == ""

    @pytest.mark.asyncio
    async def test_resolve_decision(self, state_manager):
        """Test resolving a decision with an outcome."""
        decision_id = await state_manager.record_decision(
            agent_id="agent-001",
            decision="Use approach A",
            reasoning="Better performance",
        )

        resolved = await state_manager.resolve_decision(
            "agent-001",
            decision_id,
            outcome="Success: 45ms latency",
            metrics={"latency_ms": 45, "accuracy": 0.98},
        )

        assert resolved is True

        decision = await state_manager.get_decision("agent-001", decision_id)
        assert decision["outcome"] == "Success: 45ms latency"
        assert decision["outcome_metrics"]["latency_ms"] == 45
        assert decision["resolved_at"] != ""

    @pytest.mark.asyncio
    async def test_resolve_nonexistent_decision(self, state_manager):
        """Test resolving a non-existent decision returns False."""
        resolved = await state_manager.resolve_decision(
            "agent-001", "nonexistent-id", outcome="whatever"
        )
        assert resolved is False

    @pytest.mark.asyncio
    async def test_get_decision_history_empty(self, state_manager):
        """Test that history for a new agent is empty."""
        history = await state_manager.get_decision_history("new-agent")
        assert history == []

    @pytest.mark.asyncio
    async def test_get_decision_history_chronological(self, state_manager):
        """Test that decision history maintains chronological order."""
        ids = []
        for i in range(5):
            did = await state_manager.record_decision(
                agent_id="agent-001",
                decision=f"Decision {i}",
                reasoning=f"Reason {i}",
            )
            ids.append(did)

        history = await state_manager.get_decision_history("agent-001")

        assert len(history) == 5
        for i, record in enumerate(history):
            assert record["decision_id"] == ids[i]
            assert record["decision"] == f"Decision {i}"

    @pytest.mark.asyncio
    async def test_get_decision_history_pagination(self, state_manager):
        """Test paginated decision history."""
        for i in range(10):
            await state_manager.record_decision(
                agent_id="agent-001",
                decision=f"Decision {i}",
                reasoning=f"Reason {i}",
            )

        page1 = await state_manager.get_decision_history("agent-001", limit=3, offset=0)
        page2 = await state_manager.get_decision_history("agent-001", limit=3, offset=3)

        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0]["decision"] == "Decision 0"
        assert page2[0]["decision"] == "Decision 3"

    @pytest.mark.asyncio
    async def test_get_unresolved_decisions(self, state_manager):
        """Test filtering for unresolved decisions."""
        # Record 5 decisions
        ids = []
        for i in range(5):
            did = await state_manager.record_decision(
                agent_id="agent-001",
                decision=f"Decision {i}",
                reasoning=f"Reason {i}",
            )
            ids.append(did)

        # Resolve first 2
        await state_manager.resolve_decision("agent-001", ids[0], "ok")
        await state_manager.resolve_decision("agent-001", ids[1], "ok")

        unresolved = await state_manager.get_unresolved_decisions("agent-001")

        assert len(unresolved) == 3
        for d in unresolved:
            assert d["outcome"] == ""

    @pytest.mark.asyncio
    async def test_decision_metadata(self, state_manager):
        """Test that decision metadata is stored and retrieved."""
        decision_id = await state_manager.record_decision(
            agent_id="agent-001",
            decision="Route query",
            reasoning="Matches pattern",
            metadata={"source": "api", "priority": "high", "trace_id": "abc-123"},
        )

        decision = await state_manager.get_decision("agent-001", decision_id)

        assert decision["metadata"]["source"] == "api"
        assert decision["metadata"]["priority"] == "high"
        assert decision["metadata"]["trace_id"] == "abc-123"

    @pytest.mark.asyncio
    async def test_multiple_agents_decisions_independent(self, state_manager):
        """Test that decisions for different agents are independent."""
        await state_manager.record_decision("agent-A", "A decision", "A reason")
        await state_manager.record_decision("agent-B", "B decision", "B reason")

        history_a = await state_manager.get_decision_history("agent-A")
        history_b = await state_manager.get_decision_history("agent-B")

        assert len(history_a) == 1
        assert len(history_b) == 1
        assert history_a[0]["decision"] == "A decision"
        assert history_b[0]["decision"] == "B decision"

    @pytest.mark.asyncio
    async def test_get_nonexistent_decision(self, state_manager):
        """Test retrieving a non-existent decision returns None."""
        result = await state_manager.get_decision("agent-001", "no-such-id")
        assert result is None


# ===========================================================================
# Test: Auto-Save Mechanism
# ===========================================================================


class TestAutoSaveMechanism:
    """Tests for the auto-save background mechanism."""

    @pytest.mark.asyncio
    async def test_mark_dirty(self, state_manager):
        """Test marking an agent as dirty."""
        assert state_manager.is_dirty("agent-001") is False

        state_manager.mark_dirty("agent-001")

        assert state_manager.is_dirty("agent-001") is True

    @pytest.mark.asyncio
    async def test_save_clears_dirty_flag(self, state_manager, sample_agent_state):
        """Test that saving a snapshot clears the dirty flag."""
        state_manager.mark_dirty("agent-001")
        assert state_manager.is_dirty("agent-001") is True

        await state_manager.save_snapshot("agent-001", sample_agent_state)

        assert state_manager.is_dirty("agent-001") is False

    @pytest.mark.asyncio
    async def test_flush_calls_save_fn_for_dirty_agents(self, state_manager):
        """Test that flush invokes the save function for each dirty agent."""
        state_manager.mark_dirty("agent-A")
        state_manager.mark_dirty("agent-B")

        saved_agents = []

        async def mock_save_fn(agent_id: str):
            saved_agents.append(agent_id)

        count = await state_manager.flush(save_fn=mock_save_fn)

        assert count == 2
        assert "agent-A" in saved_agents
        assert "agent-B" in saved_agents
        assert state_manager.is_dirty("agent-A") is False
        assert state_manager.is_dirty("agent-B") is False

    @pytest.mark.asyncio
    async def test_flush_empty_noop(self, state_manager):
        """Test flushing when no agents are dirty returns 0."""
        saved_agents = []

        async def mock_save_fn(agent_id: str):
            saved_agents.append(agent_id)

        count = await state_manager.flush(save_fn=mock_save_fn)

        assert count == 0
        assert saved_agents == []

    @pytest.mark.asyncio
    async def test_flush_re_marks_dirty_on_save_failure(self, state_manager):
        """Test that flush re-marks an agent dirty if save_fn fails."""
        state_manager.mark_dirty("agent-fail")
        state_manager.mark_dirty("agent-ok")

        async def flaky_save_fn(agent_id: str):
            if agent_id == "agent-fail":
                raise RuntimeError("DB connection lost")

        count = await state_manager.flush(save_fn=flaky_save_fn)

        assert count == 2  # Both were attempted
        assert state_manager.is_dirty("agent-fail") is True  # Re-marked
        assert state_manager.is_dirty("agent-ok") is False  # Successfully flushed

    @pytest.mark.asyncio
    async def test_auto_save_triggers_periodically(self, state_manager):
        """Test that auto-save invokes save_fn at the configured interval."""
        # Use a very short interval for testing
        state_manager._auto_save_interval = 0.05  # 50ms

        saved_calls = []

        async def tracking_save_fn(agent_id: str):
            saved_calls.append(agent_id)

        await state_manager.start_auto_save(save_fn=tracking_save_fn)

        try:
            # Mark dirty and wait for auto-save to fire
            state_manager.mark_dirty("agent-timer")
            await asyncio.sleep(0.15)  # Wait ~3 intervals

            assert "agent-timer" in saved_calls
        finally:
            await state_manager.stop_auto_save()

    @pytest.mark.asyncio
    async def test_auto_save_clears_dirty_after_flush(self, state_manager):
        """Test that auto-save clears dirty flags after flushing."""
        state_manager._auto_save_interval = 0.05

        async def noop_save(agent_id: str):
            pass

        await state_manager.start_auto_save(save_fn=noop_save)

        try:
            state_manager.mark_dirty("agent-auto")
            assert state_manager.is_dirty("agent-auto") is True

            await asyncio.sleep(0.15)

            assert state_manager.is_dirty("agent-auto") is False
        finally:
            await state_manager.stop_auto_save()

    @pytest.mark.asyncio
    async def test_start_auto_save_twice_is_idempotent(self, state_manager):
        """Test that starting auto-save twice doesn't create duplicate tasks."""
        state_manager._auto_save_interval = 0.1

        async def noop_save(agent_id: str):
            pass

        await state_manager.start_auto_save(save_fn=noop_save)
        task1 = state_manager._auto_save_task

        await state_manager.start_auto_save(save_fn=noop_save)
        task2 = state_manager._auto_save_task

        assert task1 is task2  # Same task, no duplicate

        await state_manager.stop_auto_save()

    @pytest.mark.asyncio
    async def test_stop_auto_save(self, state_manager):
        """Test that stop_auto_save terminates the background task."""
        state_manager._auto_save_interval = 0.1

        async def noop_save(agent_id: str):
            pass

        await state_manager.start_auto_save(save_fn=noop_save)
        assert state_manager._auto_save_task is not None
        assert state_manager._running is True

        await state_manager.stop_auto_save()

        assert state_manager._auto_save_task is None
        assert state_manager._running is False

    @pytest.mark.asyncio
    async def test_auto_save_with_no_save_fn(self, state_manager):
        """Test that auto-save runs without error when no save_fn is given."""
        state_manager._auto_save_interval = 0.05

        state_manager.mark_dirty("agent-nosave")

        await state_manager.start_auto_save()  # No save_fn
        try:
            await asyncio.sleep(0.1)
            # Dirty flag should be cleared even without save_fn
            assert state_manager.is_dirty("agent-nosave") is False
        finally:
            await state_manager.stop_auto_save()

    @pytest.mark.asyncio
    async def test_concurrent_dirty_marks(self, state_manager):
        """Test that marking multiple agents dirty concurrently works."""
        async def mark_many():
            for i in range(50):
                state_manager.mark_dirty(f"agent-{i}")

        await asyncio.gather(mark_many(), mark_many())

        # All 50 should be dirty
        for i in range(50):
            assert state_manager.is_dirty(f"agent-{i}") is True


# ===========================================================================
# Test: Stats & Edge Cases
# ===========================================================================


class TestStateManagerStats:
    """Tests for StateManager statistics and edge cases."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, state_manager):
        """Test stats on a fresh state manager."""
        stats = await state_manager.get_stats()

        assert stats["total_snapshots"] == 0
        assert stats["dirty_agents"] == 0
        assert stats["auto_save_running"] is False

    @pytest.mark.asyncio
    async def test_get_stats_after_saves(self, state_manager):
        """Test stats after saving snapshots."""
        for i in range(3):
            await state_manager.save_snapshot(f"agent-{i}", {"agent_id": f"agent-{i}"})

        stats = await state_manager.get_stats()

        assert stats["total_snapshots"] == 3

    @pytest.mark.asyncio
    async def test_get_stats_tracks_dirty_agents(self, state_manager):
        """Test that stats reflect dirty agent count."""
        state_manager.mark_dirty("a")
        state_manager.mark_dirty("b")
        state_manager.mark_dirty("c")

        stats = await state_manager.get_stats()

        assert stats["dirty_agents"] == 3

    @pytest.mark.asyncio
    async def test_get_stats_tracks_auto_save_running(self, state_manager):
        """Test that stats reflect auto-save running state."""
        state_manager._auto_save_interval = 0.1

        async def noop(agent_id: str):
            pass

        stats_before = await state_manager.get_stats()
        assert stats_before["auto_save_running"] is False

        await state_manager.start_auto_save(save_fn=noop)
        try:
            stats_during = await state_manager.get_stats()
            assert stats_during["auto_save_running"] is True
        finally:
            await state_manager.stop_auto_save()

        stats_after = await state_manager.get_stats()
        assert stats_after["auto_save_running"] is False

    @pytest.mark.asyncio
    async def test_snapshot_with_unicode_content(self, state_manager):
        """Test snapshot with unicode/special characters."""
        state = {
            "agent_id": "unicode-agent",
            "attention_focus": "日本語ドキュメント処理",
            "metadata": {"note": "émojis 🚀 and ñoño"},
        }

        await state_manager.save_snapshot("unicode-agent", state)
        loaded = await state_manager.load_snapshot("unicode-agent")

        assert loaded["attention_focus"] == "日本語ドキュメント処理"
        assert "🚀" in loaded["metadata"]["note"]

    @pytest.mark.asyncio
    async def test_decision_with_unicode_content(self, state_manager):
        """Test decision recording with unicode content."""
        decision_id = await state_manager.record_decision(
            agent_id="agent-001",
            decision="Procesar documento en español",
            reasoning="El usuario escribió en español: ñ, ü, á",
            context={"language": "español"},
        )

        decision = await state_manager.get_decision("agent-001", decision_id)

        assert "español" in decision["decision"]
        assert "ñ" in decision["reasoning"]
        assert decision["context"]["language"] == "español"

    @pytest.mark.asyncio
    async def test_snapshot_with_large_metrics(self, state_manager):
        """Test snapshot with a large metrics dictionary."""
        large_metrics = {f"metric_{i}": i * 0.1 for i in range(100)}
        state = {"agent_id": "metrics-agent", "metrics": large_metrics}

        await state_manager.save_snapshot("metrics-agent", state)
        loaded = await state_manager.load_snapshot("metrics-agent")

        assert len(loaded["metrics"]) == 100
        assert loaded["metrics"]["metric_50"] == pytest.approx(5.0)
