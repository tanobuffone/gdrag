"""Tests for AgentRegistry module (gdrag v3).

Tests agent registration, heartbeat monitoring, capability discovery,
stale agent cleanup, and registry statistics.
Uses unittest.mock for PostgreSQL isolation following existing test patterns.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.core.agent_registry import AgentRegistry
from src.models.agent import (
    AgentEndpoint,
    AgentHeartbeat,
    AgentInfo,
    AgentRegistration,
    AgentRegistryStats,
    HealthStatus,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_config():
    """Provide a mock AppConfig with database settings."""
    config = MagicMock()
    config.database.postgres_host = "localhost"
    config.database.postgres_port = 5432
    config.database.postgres_db = "gdrag_test"
    config.database.postgres_user = "test"
    config.database.postgres_password = "test"
    return config


@pytest.fixture
def mock_connection():
    """Provide a mock psycopg2 connection with cursor context manager."""
    conn = MagicMock()
    conn.closed = False
    conn.autocommit = True

    cursor = MagicMock()
    cursor.description = True
    cursor.fetchall.return_value = []
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)

    conn.cursor.return_value = cursor
    return conn


@pytest.fixture
def registry(mock_config, mock_connection):
    """Provide an AgentRegistry with mocked PostgreSQL connection."""
    reg = AgentRegistry(config=mock_config)
    reg._connection = mock_connection
    return reg


@pytest.fixture
def sample_agent_row():
    """Provide a sample database row dict for an agent."""
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return {
        "agent_id": "agent-alpha",
        "name": "Alpha Agent",
        "status": "healthy",
        "capabilities": ["query", "ingest", "session"],
        "endpoints": json.dumps([
            {"url": "http://alpha:8080", "protocol": "http", "path": "/api", "timeout_ms": 30000, "metadata": {}}
        ]),
        "metadata": json.dumps({"version": "1.0.0"}),
        "tags": ["prod", "primary"],
        "heartbeat_interval_s": 60,
        "registered_at": now,
        "updated_at": now,
        "last_heartbeat": now,
    }


@pytest.fixture
def sample_registration():
    """Provide a sample AgentRegistration model."""
    return AgentRegistration(
        agent_id="agent-new",
        name="New Agent",
        capabilities=["query", "ingest"],
        endpoints=[
            AgentEndpoint(url="http://new:8080", protocol="http", path="/api")
        ],
        metadata={"version": "2.0.0"},
        heartbeat_interval_s=30,
        tags=["dev"],
    )


def _make_agent_info(
    agent_id: str = "agent-alpha",
    name: str = "Alpha Agent",
    status: HealthStatus = HealthStatus.HEALTHY,
    capabilities: list = None,
    last_heartbeat: datetime = ...,  # sentinel: use 'now' only when not overridden
    heartbeat_interval_s: int = 60,
) -> AgentInfo:
    """Helper to build AgentInfo quickly."""
    now = datetime.now(timezone.utc)
    _hb = now if last_heartbeat is ... else last_heartbeat
    return AgentInfo(
        agent_id=agent_id,
        name=name,
        status=status,
        capabilities=capabilities or ["query", "ingest"],
        endpoints=[],
        registered_at=now,
        last_heartbeat=_hb,
        heartbeat_interval_s=heartbeat_interval_s,
        metadata={},
        tags=[],
        updated_at=now,
    )


# ===========================================================================
# Test: Agent Registration
# ===========================================================================


class TestAgentRegistration:
    """Tests for registering and unregistering agents."""

    def test_register_agent_returns_id(self, registry, mock_connection):
        """Test that register_agent returns the agent_id."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"agent_id": "agent-new"}]

        result = registry.register_agent(AgentRegistration(
            agent_id="agent-new",
            name="New Agent",
        ))

        assert result == "agent-new"

    def test_register_agent_executes_upsert(self, registry, mock_connection, sample_registration):
        """Test that register_agent executes an INSERT ... ON CONFLICT query."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"agent_id": "agent-new"}]

        registry.register_agent(sample_registration)

        call_args = str(cursor.execute.call_args)
        assert "INSERT" in call_args.upper()
        assert "ON CONFLICT" in call_args.upper()

    def test_register_agent_with_capabilities(self, registry, mock_connection, sample_registration):
        """Test registering an agent with specific capabilities."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"agent_id": "agent-new"}]

        registry.register_agent(sample_registration)

        # Verify capabilities were passed as a parameter
        params = cursor.execute.call_args[0][1]
        assert "query" in params[2]
        assert "ingest" in params[2]

    def test_register_agent_with_endpoints(self, registry, mock_connection, sample_registration):
        """Test registering an agent with endpoints."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"agent_id": "agent-new"}]

        registry.register_agent(sample_registration)

        params = cursor.execute.call_args[0][1]
        endpoints_json = params[3]
        endpoints = json.loads(endpoints_json)
        assert len(endpoints) == 1
        assert endpoints[0]["url"] == "http://new:8080"

    def test_register_agent_with_tags(self, registry, mock_connection, sample_registration):
        """Test registering an agent with tags."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"agent_id": "agent-new"}]

        registry.register_agent(sample_registration)

        params = cursor.execute.call_args[0][1]
        assert "dev" in params[5]

    def test_register_agent_custom_heartbeat_interval(self, registry, mock_connection, sample_registration):
        """Test registering an agent with custom heartbeat interval."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"agent_id": "agent-new"}]

        registry.register_agent(sample_registration)

        params = cursor.execute.call_args[0][1]
        assert params[6] == 30  # heartbeat_interval_s

    def test_register_agent_default_status_unknown(self, registry, mock_connection):
        """Test that newly registered agents start with UNKNOWN status."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"agent_id": "a1"}]

        registry.register_agent(AgentRegistration(agent_id="a1", name="A1"))

        params = cursor.execute.call_args[0][1]
        assert params[7] == "unknown"

    def test_register_agent_fallback_id(self, registry, mock_connection):
        """Test that register_agent falls back to input agent_id when RETURNING is empty."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []  # Empty RETURNING result

        result = registry.register_agent(AgentRegistration(agent_id="fallback", name="FB"))

        assert result == "fallback"

    def test_unregister_agent_success(self, registry, mock_connection):
        """Test unregistering an existing agent."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"agent_id": "agent-alpha"}]

        deleted = registry.unregister_agent("agent-alpha")

        assert deleted is True

    def test_unregister_agent_not_found(self, registry, mock_connection):
        """Test unregistering a non-existent agent returns False."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        deleted = registry.unregister_agent("nonexistent")

        assert deleted is False

    def test_unregister_agent_executes_delete(self, registry, mock_connection):
        """Test that unregister_agent executes a DELETE query."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"agent_id": "a1"}]

        registry.unregister_agent("a1")

        call_args = str(cursor.execute.call_args)
        assert "DELETE" in call_args.upper()

    def test_register_upsert_updates_existing(self, registry, mock_connection):
        """Test that registering an existing agent updates its data (upsert)."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"agent_id": "existing"}]

        reg = AgentRegistration(
            agent_id="existing",
            name="Updated Name",
            capabilities=["new_cap"],
        )
        result = registry.register_agent(reg)

        assert result == "existing"
        call_args = str(cursor.execute.call_args)
        assert "DO UPDATE" in call_args.upper()


# ===========================================================================
# Test: Heartbeat Monitoring
# ===========================================================================


class TestHeartbeatMonitoring:
    """Tests for agent heartbeat recording and health tracking."""

    def test_heartbeat_records_successfully(self, registry, mock_connection):
        """Test that heartbeat returns True when recorded."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"id": 1}]

        recorded = registry.heartbeat("agent-alpha")

        assert recorded is True

    def test_heartbeat_inserts_into_heartbeats_table(self, registry, mock_connection):
        """Test that heartbeat executes INSERT into agent_heartbeats."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"id": 1}]

        registry.heartbeat("agent-alpha")

        call_args = str(cursor.execute.call_args)
        assert "agent_heartbeats" in call_args
        assert "INSERT" in call_args.upper()

    def test_heartbeat_with_healthy_status(self, registry, mock_connection):
        """Test heartbeat with HEALTHY status."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"id": 1}]

        registry.heartbeat("agent-alpha", status=HealthStatus.HEALTHY)

        params = cursor.execute.call_args[0][1]
        assert params[1] == "healthy"

    def test_heartbeat_with_degraded_status(self, registry, mock_connection):
        """Test heartbeat with DEGRADED status."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"id": 1}]

        registry.heartbeat("agent-alpha", status=HealthStatus.DEGRADED)

        params = cursor.execute.call_args[0][1]
        assert params[1] == "degraded"

    def test_heartbeat_with_unhealthy_status(self, registry, mock_connection):
        """Test heartbeat with UNHEALTHY status."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"id": 1}]

        registry.heartbeat("agent-alpha", status=HealthStatus.UNHEALTHY)

        params = cursor.execute.call_args[0][1]
        assert params[1] == "unhealthy"

    def test_heartbeat_with_metrics(self, registry, mock_connection):
        """Test heartbeat includes metrics in the insert."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"id": 1}]

        metrics = {"uptime_s": 3600, "queue_depth": 5, "memory_mb": 256}
        registry.heartbeat("agent-alpha", metrics=metrics)

        params = cursor.execute.call_args[0][1]
        stored_metrics = json.loads(params[2])
        assert stored_metrics["uptime_s"] == 3600
        assert stored_metrics["queue_depth"] == 5

    def test_heartbeat_without_metrics_defaults_empty(self, registry, mock_connection):
        """Test heartbeat without metrics stores empty dict."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"id": 1}]

        registry.heartbeat("agent-alpha")

        params = cursor.execute.call_args[0][1]
        stored_metrics = json.loads(params[2])
        assert stored_metrics == {}

    def test_heartbeat_failed_returns_false(self, registry, mock_connection):
        """Test that heartbeat returns False when INSERT fails to return id."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        recorded = registry.heartbeat("agent-alpha")

        assert recorded is False

    def test_heartbeat_default_status_is_healthy(self, registry, mock_connection):
        """Test that default heartbeat status is HEALTHY."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"id": 1}]

        registry.heartbeat("agent-alpha")

        params = cursor.execute.call_args[0][1]
        assert params[1] == "healthy"

    def test_get_agent_health(self, registry, mock_connection):
        """Test getting current health status of an agent."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"status": "degraded"}]

        health = registry.get_agent_health("agent-alpha")

        assert health == HealthStatus.DEGRADED

    def test_get_agent_health_not_found(self, registry, mock_connection):
        """Test get_agent_health returns None for unknown agent."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        health = registry.get_agent_health("nonexistent")

        assert health is None

    def test_get_agent_health_all_statuses(self, registry, mock_connection):
        """Test get_agent_health for all possible status values."""
        cursor = mock_connection.cursor.return_value

        for status in ["healthy", "degraded", "unhealthy", "unknown"]:
            cursor.fetchall.return_value = [{"status": status}]
            health = registry.get_agent_health("agent")
            assert health == HealthStatus(status)


# ===========================================================================
# Test: Capability Discovery
# ===========================================================================


class TestCapabilityDiscovery:
    """Tests for finding agents by capability."""

    def test_find_agents_by_capability(self, registry, mock_connection, sample_agent_row):
        """Test finding agents with a specific capability."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [sample_agent_row]

        agents = registry.find_agents_by_capability("query")

        assert len(agents) == 1
        assert agents[0].agent_id == "agent-alpha"
        assert "query" in agents[0].capabilities

    def test_find_agents_by_capability_uses_gin_query(self, registry, mock_connection):
        """Test that capability search uses PostgreSQL @> containment operator."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        registry.find_agents_by_capability("ingest")

        call_args = str(cursor.execute.call_args)
        assert "@>" in call_args

    def test_find_agents_by_capability_empty_result(self, registry, mock_connection):
        """Test finding agents when no match exists."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        agents = registry.find_agents_by_capability("nonexistent_cap")

        assert agents == []

    def test_find_agents_by_capability_respects_limit(self, registry, mock_connection, sample_agent_row):
        """Test that capability search passes limit to query."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [sample_agent_row]

        registry.find_agents_by_capability("query", limit=5)

        params = cursor.execute.call_args[0][1]
        assert 5 in params

    def test_find_agents_by_capability_multiple_results(self, registry, mock_connection):
        """Test finding multiple agents with the same capability."""
        cursor = mock_connection.cursor.return_value
        now = datetime.now(timezone.utc)
        rows = [
            {
                "agent_id": f"agent-{i}",
                "name": f"Agent {i}",
                "status": "healthy",
                "capabilities": ["query", "ingest"],
                "endpoints": "[]",
                "metadata": "{}",
                "tags": "[]",
                "heartbeat_interval_s": 60,
                "registered_at": now,
                "updated_at": now,
                "last_heartbeat": now,
            }
            for i in range(3)
        ]
        cursor.fetchall.return_value = rows

        agents = registry.find_agents_by_capability("query")

        assert len(agents) == 3
        assert all("query" in a.capabilities for a in agents)

    def test_list_agents(self, registry, mock_connection, sample_agent_row):
        """Test listing all active agents."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [sample_agent_row]

        agents = registry.list_agents()

        assert len(agents) == 1
        assert agents[0].agent_id == "agent-alpha"

    def test_list_agents_filter_by_status(self, registry, mock_connection, sample_agent_row):
        """Test listing agents filtered by specific status."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [sample_agent_row]

        registry.list_agents(status=HealthStatus.HEALTHY)

        call_args = str(cursor.execute.call_args)
        assert "healthy" in call_args

    def test_list_agents_filter_active(self, registry, mock_connection, sample_agent_row):
        """Test listing active agents (healthy or degraded)."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [sample_agent_row]

        registry.list_agents(status="active")

        call_args = str(cursor.execute.call_args)
        assert "healthy" in call_args
        assert "degraded" in call_args

    def test_list_agents_respects_limit(self, registry, mock_connection, sample_agent_row):
        """Test that list_agents passes limit to query."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [sample_agent_row]

        registry.list_agents(limit=10)

        params = cursor.execute.call_args[0][1]
        assert 10 in params

    def test_list_agents_empty(self, registry, mock_connection):
        """Test listing agents when none exist."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        agents = registry.list_agents()

        assert agents == []

    def test_get_agent_found(self, registry, mock_connection, sample_agent_row):
        """Test retrieving a single agent by ID."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [sample_agent_row]

        agent = registry.get_agent("agent-alpha")

        assert agent is not None
        assert agent.agent_id == "agent-alpha"
        assert agent.name == "Alpha Agent"
        assert HealthStatus.HEALTHY == agent.status

    def test_get_agent_not_found(self, registry, mock_connection):
        """Test get_agent returns None for unknown agent."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        agent = registry.get_agent("nonexistent")

        assert agent is None

    def test_get_agent_parses_json_fields(self, registry, mock_connection, sample_agent_row):
        """Test that get_agent correctly parses JSON fields."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [sample_agent_row]

        agent = registry.get_agent("agent-alpha")

        assert isinstance(agent.capabilities, list)
        assert "query" in agent.capabilities
        assert isinstance(agent.metadata, dict)
        assert agent.metadata["version"] == "1.0.0"

    def test_get_agent_parses_string_json_fields(self, registry, mock_connection):
        """Test that get_agent handles string-encoded JSON fields."""
        now = datetime.now(timezone.utc)
        row = {
            "agent_id": "str-agent",
            "name": "String Agent",
            "status": "healthy",
            "capabilities": '["cap1", "cap2"]',  # string, not list
            "endpoints": "[]",
            "metadata": '{"key": "value"}',
            "tags": '["tag1"]',
            "heartbeat_interval_s": 60,
            "registered_at": now,
            "updated_at": now,
            "last_heartbeat": now,
        }
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [row]

        agent = registry.get_agent("str-agent")

        assert agent.capabilities == ["cap1", "cap2"]
        assert agent.metadata == {"key": "value"}
        assert agent.tags == ["tag1"]


# ===========================================================================
# Test: Cleanup Stale Agents
# ===========================================================================


class TestCleanupStaleAgents:
    """Tests for marking and cleaning up stale agents."""

    def test_cleanup_stale_agents_calls_db_function(self, registry, mock_connection):
        """Test that cleanup_stale_agents calls the mark_stale_agents DB function."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"updated": 0}]

        registry.cleanup_stale_agents()

        call_args = str(cursor.execute.call_args)
        assert "mark_stale_agents" in call_args

    def test_cleanup_stale_agents_returns_count(self, registry, mock_connection):
        """Test cleanup_stale_agents returns the number of updated agents."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"updated": 3}]

        count = registry.cleanup_stale_agents(timeout_seconds=300)

        assert count == 3

    def test_cleanup_stale_agents_zero_when_none_stale(self, registry, mock_connection):
        """Test cleanup_stale_agents returns 0 when no agents are stale."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"updated": 0}]

        count = registry.cleanup_stale_agents()

        assert count == 0

    def test_cleanup_stale_agents_default_timeout(self, registry, mock_connection):
        """Test cleanup_stale_agents uses 300s default timeout."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"updated": 0}]

        registry.cleanup_stale_agents()

        params = cursor.execute.call_args[0][1]
        # 300 seconds / 60 = 5.0 minutes
        assert params[0] == 5.0

    def test_cleanup_stale_agents_custom_timeout(self, registry, mock_connection):
        """Test cleanup_stale_agents with custom timeout."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"updated": 1}]

        registry.cleanup_stale_agents(timeout_seconds=600)

        params = cursor.execute.call_args[0][1]
        # 600 seconds / 60 = 10.0 minutes
        assert params[0] == 10.0

    def test_cleanup_stale_agents_empty_result(self, registry, mock_connection):
        """Test cleanup_stale_agents handles empty query result."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        count = registry.cleanup_stale_agents()

        assert count == 0

    def test_agent_info_is_stale_no_heartbeat(self):
        """Test AgentInfo.is_stale returns True when last_heartbeat is None."""
        agent = _make_agent_info(last_heartbeat=None)

        assert agent.is_stale() is True

    def test_agent_info_is_stale_past_threshold(self):
        """Test AgentInfo.is_stale returns True when past threshold."""
        old_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        agent = _make_agent_info(
            last_heartbeat=old_time,
            heartbeat_interval_s=60,
        )

        # Default multiplier is 3.0, so threshold = 3 * 60 = 180s = 3min
        # 10 minutes > 3 minutes, so stale
        assert agent.is_stale() is True

    def test_agent_info_is_not_stale_recent_heartbeat(self):
        """Test AgentInfo.is_stale returns False for recent heartbeat."""
        recent = datetime.now(timezone.utc) - timedelta(seconds=10)
        agent = _make_agent_info(
            last_heartbeat=recent,
            heartbeat_interval_s=60,
        )

        assert agent.is_stale() is False

    def test_agent_info_is_stale_custom_multiplier(self):
        """Test AgentInfo.is_stale with custom multiplier."""
        # Heartbeat 5 minutes ago
        hb_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        agent = _make_agent_info(
            last_heartbeat=hb_time,
            heartbeat_interval_s=60,
        )

        # multiplier=10 → threshold = 600s = 10min → not stale
        assert agent.is_stale(multiplier=10.0) is False
        # multiplier=1 → threshold = 60s = 1min → stale
        assert agent.is_stale(multiplier=1.0) is True

    def test_agent_info_is_healthy(self):
        """Test AgentInfo.is_healthy checks both status and staleness."""
        recent = datetime.now(timezone.utc) - timedelta(seconds=10)
        healthy = _make_agent_info(
            status=HealthStatus.HEALTHY,
            last_heartbeat=recent,
        )
        unhealthy = _make_agent_info(
            status=HealthStatus.UNHEALTHY,
            last_heartbeat=recent,
        )

        assert healthy.is_healthy() is True
        assert unhealthy.is_healthy() is False

    def test_agent_info_is_healthy_stale_agent(self):
        """Test AgentInfo.is_healthy returns False for stale healthy agent."""
        old_time = datetime.now(timezone.utc) - timedelta(hours=1)
        stale_healthy = _make_agent_info(
            status=HealthStatus.HEALTHY,
            last_heartbeat=old_time,
            heartbeat_interval_s=60,
        )

        assert stale_healthy.is_healthy() is False


# ===========================================================================
# Test: Registry Statistics
# ===========================================================================


class TestRegistryStats:
    """Tests for registry statistics aggregation."""

    def test_get_stats_returns_all_fields(self, registry, mock_connection):
        """Test that get_stats returns AgentRegistryStats with all fields."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.side_effect = [
            [{"total": 10, "healthy": 5, "degraded": 2, "unhealthy": 2, "unknown": 1}],
            [{"total": 50}],
        ]

        stats = registry.get_stats()

        assert isinstance(stats, AgentRegistryStats)
        assert stats.total_agents == 10
        assert stats.healthy_agents == 5
        assert stats.degraded_agents == 2
        assert stats.unhealthy_agents == 2
        assert stats.unknown_agents == 1
        assert stats.total_heartbeats == 50

    def test_get_stats_empty_registry(self, registry, mock_connection):
        """Test get_stats with empty registry."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.side_effect = [
            [{"total": 0, "healthy": 0, "degraded": 0, "unhealthy": 0, "unknown": 0}],
            [{"total": 0}],
        ]

        stats = registry.get_stats()

        assert stats.total_agents == 0
        assert stats.total_heartbeats == 0

    def test_get_stats_queries_agents_table(self, registry, mock_connection):
        """Test that get_stats queries the agents table with FILTER clauses."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.side_effect = [
            [{"total": 1, "healthy": 1, "degraded": 0, "unhealthy": 0, "unknown": 0}],
            [{"total": 5}],
        ]

        registry.get_stats()

        # First call should query agents
        first_call = str(cursor.execute.call_args_list[0])
        assert "agents" in first_call

    def test_get_stats_queries_heartbeats_table(self, registry, mock_connection):
        """Test that get_stats also queries agent_heartbeats for count."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.side_effect = [
            [{"total": 1, "healthy": 1, "degraded": 0, "unhealthy": 0, "unknown": 0}],
            [{"total": 5}],
        ]

        registry.get_stats()

        # Second call should query agent_heartbeats
        second_call = str(cursor.execute.call_args_list[1])
        assert "agent_heartbeats" in second_call

    def test_get_stats_includes_timestamp(self, registry, mock_connection):
        """Test that get_stats includes last_updated timestamp."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.side_effect = [
            [{"total": 0, "healthy": 0, "degraded": 0, "unhealthy": 0, "unknown": 0}],
            [{"total": 0}],
        ]

        stats = registry.get_stats()

        assert stats.last_updated is not None


# ===========================================================================
# Test: Connection Management
# ===========================================================================


class TestConnectionManagement:
    """Tests for database connection lifecycle."""

    def test_close_connection(self, mock_config, mock_connection):
        """Test that close() closes the database connection."""
        reg = AgentRegistry(config=mock_config)
        reg._connection = mock_connection

        reg.close()

        mock_connection.close.assert_called_once()
        assert reg._connection is None

    def test_close_already_closed(self, mock_config, mock_connection):
        """Test that close() is idempotent when connection already closed."""
        reg = AgentRegistry(config=mock_config)
        mock_connection.closed = True
        reg._connection = mock_connection

        reg.close()

        mock_connection.close.assert_not_called()

    def test_close_no_connection(self, mock_config):
        """Test that close() with no connection does not raise."""
        reg = AgentRegistry(config=mock_config)
        reg._connection = None

        reg.close()  # Should not raise

    def test_get_connection_creates_new(self, mock_config):
        """Test that _get_connection creates a new connection when none exists."""
        with patch("src.core.agent_registry.psycopg2") as mock_psycopg2:
            mock_conn = MagicMock()
            mock_conn.closed = False
            mock_psycopg2.connect.return_value = mock_conn

            reg = AgentRegistry(config=mock_config)
            conn = reg._get_connection()

            mock_psycopg2.connect.assert_called_once_with(
                host="localhost",
                port=5432,
                dbname="gdrag_test",
                user="test",
                password="test",
            )
            assert conn == mock_conn
            assert conn.autocommit is True

    def test_get_connection_reuses_existing(self, mock_config, mock_connection):
        """Test that _get_connection reuses an open connection."""
        reg = AgentRegistry(config=mock_config)
        reg._connection = mock_connection

        with patch("src.core.agent_registry.psycopg2") as mock_psycopg2:
            conn = reg._get_connection()

            mock_psycopg2.connect.assert_not_called()
            assert conn is mock_connection

    def test_get_connection_recreates_closed(self, mock_config, mock_connection):
        """Test that _get_connection creates new when existing is closed."""
        mock_connection.closed = True
        reg = AgentRegistry(config=mock_config)
        reg._connection = mock_connection

        with patch("src.core.agent_registry.psycopg2") as mock_psycopg2:
            mock_new_conn = MagicMock()
            mock_new_conn.closed = False
            mock_psycopg2.connect.return_value = mock_new_conn

            conn = reg._get_connection()

            mock_psycopg2.connect.assert_called_once()
            assert conn is mock_new_conn


# ===========================================================================
# Test: AgentInfo Model
# ===========================================================================


class TestAgentInfoModel:
    """Tests for the AgentInfo Pydantic model."""

    def test_agent_info_defaults(self):
        """Test AgentInfo has sensible defaults."""
        now = datetime.now(timezone.utc)
        info = AgentInfo(
            agent_id="test",
            name="Test Agent",
            registered_at=now,
            updated_at=now,
        )
        assert info.status == HealthStatus.UNKNOWN
        assert info.capabilities == []
        assert info.heartbeat_interval_s == 60

    def test_health_status_enum_values(self):
        """Test HealthStatus enum has expected values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_agent_registration_defaults(self):
        """Test AgentRegistration has sensible defaults."""
        reg = AgentRegistration(agent_id="a1", name="Agent 1")
        assert reg.capabilities == []
        assert reg.endpoints == []
        assert reg.heartbeat_interval_s == 60
        assert reg.tags == []

    def test_agent_endpoint_defaults(self):
        """Test AgentEndpoint has sensible defaults."""
        ep = AgentEndpoint(url="http://localhost:8080")
        assert ep.protocol == "http"
        assert ep.timeout_ms == 30000
        assert ep.path is None

    def test_agent_heartbeat_defaults(self):
        """Test AgentHeartbeat has sensible defaults."""
        hb = AgentHeartbeat(agent_id="a1")
        assert hb.status == HealthStatus.HEALTHY
        assert hb.metrics == {}
        assert hb.timestamp is not None

    def test_agent_registry_stats_fields(self):
        """Test AgentRegistryStats has all required fields."""
        now = datetime.now(timezone.utc)
        stats = AgentRegistryStats(
            total_agents=10,
            healthy_agents=5,
            degraded_agents=2,
            unhealthy_agents=2,
            unknown_agents=1,
            total_heartbeats=50,
            last_updated=now,
        )
        assert stats.total_agents == 10
        assert stats.total_heartbeats == 50
        assert stats.last_updated == now
