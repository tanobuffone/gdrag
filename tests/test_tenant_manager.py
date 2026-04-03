"""Tests for TenantManager module (gdrag v3).

Tests multi-tenancy CRUD operations, permission enforcement,
quota limits, and Row Level Security context management.
Uses unittest.mock for PostgreSQL isolation following existing test patterns.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.core.tenant_manager import (
    TenantManager,
    TenantAccessDeniedError,
    QuotaExceededError,
    TenantNotFoundError,
)
from src.models.tenant import (
    QuotaUsage,
    Tenant,
    TenantPermissions,
    TenantQuotas,
    TenantStatus,
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
    conn.autocommit = False

    cursor = MagicMock()
    cursor.description = True
    cursor.fetchall.return_value = []
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)

    conn.cursor.return_value = cursor
    return conn


@pytest.fixture
def tenant_manager(mock_config, mock_connection):
    """Provide a TenantManager with mocked PostgreSQL connection."""
    manager = TenantManager(config=mock_config)
    manager._connection = mock_connection
    return manager


@pytest.fixture
def sample_permissions():
    """Provide sample tenant permissions."""
    return TenantPermissions(
        domains=["software", "finance"],
        can_read_shared=True,
        can_write_shared=False,
        max_sessions=10,
        allowed_operations=["query", "ingest", "session"],
    )


@pytest.fixture
def sample_quotas():
    """Provide sample tenant quotas."""
    return TenantQuotas(
        max_queries_per_hour=1000,
        max_storage_mb=500,
        max_vectors=100000,
        max_sessions=50,
        max_knowledge_entries=10000,
        max_concurrent_sessions=10,
    )


@pytest.fixture
def sample_tenant_row():
    """Provide a sample database row dict for a tenant."""
    return {
        "id": "tenant-uuid-001",
        "agent_id": "agent-alpha",
        "name": "Alpha Agent",
        "description": "Test tenant",
        "permissions": json.dumps({
            "domains": ["software"],
            "can_read_shared": True,
            "can_write_shared": False,
            "max_sessions": 10,
            "allowed_operations": ["query", "ingest", "session"],
            "custom_permissions": {},
        }),
        "quotas": json.dumps({
            "max_queries_per_hour": 1000,
            "max_storage_mb": 500,
            "max_vectors": 100000,
            "max_sessions": 50,
            "max_knowledge_entries": 10000,
            "max_concurrent_sessions": 10,
            "custom_quotas": {},
        }),
        "status": "active",
        "metadata": json.dumps({"env": "test"}),
        "created_at": datetime(2026, 1, 1, 0, 0, 0),
        "updated_at": datetime(2026, 1, 1, 0, 0, 0),
    }


@pytest.fixture
def sample_quota_usage_row():
    """Provide a sample database row dict for quota usage."""
    now = datetime.utcnow()
    return {
        "id": "quota-uuid-001",
        "tenant_id": "tenant-uuid-001",
        "queries_this_hour": 42,
        "storage_used_mb": 120.5,
        "vectors_count": 5000,
        "active_sessions": 3,
        "total_sessions": 15,
        "knowledge_entries_count": 250,
        "last_query_at": now,
        "last_reset_at": now,
        "updated_at": now,
    }


def _make_tenant(
    agent_id: str = "agent-alpha",
    name: str = "Alpha Agent",
    status: TenantStatus = TenantStatus.ACTIVE,
    permissions: TenantPermissions = None,
    quotas: TenantQuotas = None,
) -> Tenant:
    """Helper to build a Tenant model quickly."""
    return Tenant(
        id="tenant-uuid-001",
        agent_id=agent_id,
        name=name,
        permissions=permissions or TenantPermissions(),
        quotas=quotas or TenantQuotas(),
        status=status,
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 1),
    )


# ===========================================================================
# Test: Tenant CRUD
# ===========================================================================


class TestTenantCRUD:
    """Tests for tenant create / read / update / delete operations."""

    def test_create_tenant_inserts_row(self, tenant_manager, mock_connection):
        """Test that create_tenant executes an INSERT query and commits."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        tenant = tenant_manager.create_tenant(
            agent_id="agent-new",
            name="New Agent",
            description="Brand new tenant",
        )

        assert tenant.agent_id == "agent-new"
        assert tenant.name == "New Agent"
        assert tenant.status == TenantStatus.ACTIVE
        mock_connection.commit.assert_called()

    def test_create_tenant_with_custom_permissions(self, tenant_manager, mock_connection, sample_permissions):
        """Test creating a tenant with explicit permissions."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        tenant = tenant_manager.create_tenant(
            agent_id="agent-perm",
            name="Perm Agent",
            permissions=sample_permissions,
        )

        assert tenant.permissions.domains == ["software", "finance"]
        assert tenant.permissions.can_write_shared is False

    def test_create_tenant_with_custom_quotas(self, tenant_manager, mock_connection, sample_quotas):
        """Test creating a tenant with explicit quotas."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        tenant = tenant_manager.create_tenant(
            agent_id="agent-quota",
            name="Quota Agent",
            quotas=sample_quotas,
        )

        assert tenant.quotas.max_queries_per_hour == 1000
        assert tenant.quotas.max_storage_mb == 500

    def test_create_duplicate_tenant_raises_value_error(self, tenant_manager, mock_connection):
        """Test that creating a tenant with existing agent_id raises ValueError."""
        import psycopg2
        cursor = mock_connection.cursor.return_value
        cursor.execute.side_effect = psycopg2.errors.UniqueViolation("duplicate key")

        with pytest.raises(ValueError, match="already exists"):
            tenant_manager.create_tenant(agent_id="dup-agent", name="Dup")

        mock_connection.rollback.assert_called()

    def test_get_tenant_found(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test retrieving an existing tenant by agent_id."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [sample_tenant_row]

        tenant = tenant_manager.get_tenant("agent-alpha")

        assert tenant is not None
        assert tenant.agent_id == "agent-alpha"
        assert tenant.name == "Alpha Agent"
        assert tenant.status == TenantStatus.ACTIVE

    def test_get_tenant_not_found(self, tenant_manager, mock_connection):
        """Test that get_tenant returns None for unknown agent_id."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        tenant = tenant_manager.get_tenant("nonexistent")

        assert tenant is None

    def test_get_tenant_by_id(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test retrieving a tenant by internal UUID."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [sample_tenant_row]

        tenant = tenant_manager.get_tenant_by_id("tenant-uuid-001")

        assert tenant is not None
        assert tenant.id == "tenant-uuid-001"

    def test_get_tenant_by_id_not_found(self, tenant_manager, mock_connection):
        """Test get_tenant_by_id returns None for unknown id."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        tenant = tenant_manager.get_tenant_by_id("bad-uuid")

        assert tenant is None

    def test_update_permissions(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test updating permissions for an existing tenant."""
        cursor = mock_connection.cursor.return_value

        # First call: get_tenant lookup
        # Second call: UPDATE ... RETURNING
        updated_row = dict(sample_tenant_row)
        updated_row["permissions"] = json.dumps({
            "domains": ["finance"],
            "can_read_shared": False,
            "can_write_shared": True,
            "max_sessions": 20,
            "allowed_operations": ["query"],
            "custom_permissions": {},
        })
        cursor.fetchall.side_effect = [[sample_tenant_row], [updated_row]]

        new_perms = TenantPermissions(
            domains=["finance"],
            can_read_shared=False,
            can_write_shared=True,
            max_sessions=20,
            allowed_operations=["query"],
        )
        tenant = tenant_manager.update_permissions("agent-alpha", new_perms)

        assert tenant.permissions.domains == ["finance"]
        assert tenant.permissions.can_write_shared is True
        mock_connection.commit.assert_called()

    def test_update_permissions_tenant_not_found(self, tenant_manager, mock_connection):
        """Test that updating permissions for unknown tenant raises TenantNotFoundError."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        with pytest.raises(TenantNotFoundError):
            tenant_manager.update_permissions("ghost", TenantPermissions())

    def test_update_quotas(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test updating quotas for an existing tenant."""
        cursor = mock_connection.cursor.return_value

        updated_row = dict(sample_tenant_row)
        updated_row["quotas"] = json.dumps({
            "max_queries_per_hour": 500,
            "max_storage_mb": 200,
            "max_vectors": 50000,
            "max_sessions": 25,
            "max_knowledge_entries": 5000,
            "max_concurrent_sessions": 5,
            "custom_quotas": {},
        })
        cursor.fetchall.side_effect = [[sample_tenant_row], [updated_row]]

        new_quotas = TenantQuotas(
            max_queries_per_hour=500,
            max_storage_mb=200,
            max_vectors=50000,
            max_sessions=25,
            max_knowledge_entries=5000,
            max_concurrent_sessions=5,
        )
        tenant = tenant_manager.update_quotas("agent-alpha", new_quotas)

        assert tenant.quotas.max_queries_per_hour == 500
        mock_connection.commit.assert_called()

    def test_update_quotas_tenant_not_found(self, tenant_manager, mock_connection):
        """Test that updating quotas for unknown tenant raises TenantNotFoundError."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        with pytest.raises(TenantNotFoundError):
            tenant_manager.update_quotas("ghost", TenantQuotas())

    def test_delete_tenant_success(self, tenant_manager, mock_connection):
        """Test deleting an existing tenant."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [{"id": "tenant-uuid-001"}]

        deleted = tenant_manager.delete_tenant("agent-alpha")

        assert deleted is True
        mock_connection.commit.assert_called()

    def test_delete_tenant_not_found(self, tenant_manager, mock_connection):
        """Test deleting a non-existent tenant returns False."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        deleted = tenant_manager.delete_tenant("nonexistent")

        assert deleted is False

    def test_list_tenants_no_filter(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test listing all tenants without status filter."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [sample_tenant_row]

        tenants = tenant_manager.list_tenants()

        assert len(tenants) == 1
        assert tenants[0].agent_id == "agent-alpha"

    def test_list_tenants_with_status_filter(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test listing tenants filtered by status."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [sample_tenant_row]

        tenants = tenant_manager.list_tenants(status=TenantStatus.ACTIVE)

        assert len(tenants) == 1
        # Verify the query included the status parameter
        call_args = cursor.execute.call_args
        assert "active" in str(call_args)

    def test_list_tenants_empty(self, tenant_manager, mock_connection):
        """Test listing tenants when none exist."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        tenants = tenant_manager.list_tenants()

        assert tenants == []

    def test_list_tenants_respects_limit(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test that list_tenants passes limit to the query."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [sample_tenant_row]

        tenant_manager.list_tenants(limit=5)

        call_args = cursor.execute.call_args
        assert 5 in call_args[0][1] or call_args[0][1] == (5,)

    def test_rollback_on_create_failure(self, tenant_manager, mock_connection):
        """Test that a failed create_tenant rolls back the transaction."""
        cursor = mock_connection.cursor.return_value
        cursor.execute.side_effect = Exception("db error")

        with pytest.raises(Exception, match="db error"):
            tenant_manager.create_tenant(agent_id="fail-agent", name="Fail")

        mock_connection.rollback.assert_called()


# ===========================================================================
# Test: Permission Enforcement
# ===========================================================================


class TestPermissionEnforcement:
    """Tests for tenant access control and permission checks."""

    def test_check_access_granted(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test that check_access returns True for allowed operation."""
        cursor = mock_connection.cursor.return_value
        # Make sure allowed_operations includes 'read' and can_read_shared is True
        row = dict(sample_tenant_row)
        row["permissions"] = json.dumps({
            "domains": ["software"],
            "can_read_shared": True,
            "can_write_shared": False,
            "max_sessions": 10,
            "allowed_operations": ["query", "ingest", "session", "read", "write"],
            "custom_permissions": {},
        })
        cursor.fetchall.return_value = [row]

        granted = tenant_manager.check_access("agent-alpha", "knowledge", "read")

        assert granted is True

    def test_check_access_unknown_tenant(self, tenant_manager, mock_connection):
        """Test check_access returns False for unknown tenant."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        granted = tenant_manager.check_access("ghost", "knowledge", "read")

        assert granted is False

    def test_check_access_suspended_tenant(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test check_access denies access for suspended tenant."""
        suspended_row = dict(sample_tenant_row)
        suspended_row["status"] = "suspended"
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [suspended_row]

        granted = tenant_manager.check_access("agent-alpha", "knowledge", "read")

        assert granted is False

    def test_check_access_operation_not_allowed(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test check_access denies when operation not in allowed_operations."""
        restricted_row = dict(sample_tenant_row)
        restricted_row["permissions"] = json.dumps({
            "domains": [],
            "can_read_shared": True,
            "can_write_shared": True,
            "max_sessions": 10,
            "allowed_operations": ["query"],  # 'ingest' not allowed
            "custom_permissions": {},
        })
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [restricted_row]

        granted = tenant_manager.check_access("agent-alpha", "knowledge", "ingest")

        assert granted is False

    def test_check_access_write_shared_denied(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test check_access denies write to shared when can_write_shared is False."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [sample_tenant_row]

        # sample_tenant_row has can_write_shared=False
        granted = tenant_manager.check_access("agent-alpha", "knowledge", "write")

        assert granted is False

    def test_check_access_write_shared_granted(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test check_access grants write to shared when can_write_shared is True."""
        row = dict(sample_tenant_row)
        row["permissions"] = json.dumps({
            "domains": [],
            "can_read_shared": True,
            "can_write_shared": True,
            "max_sessions": 10,
            "allowed_operations": ["query", "ingest", "session", "write"],
            "custom_permissions": {},
        })
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [row]

        granted = tenant_manager.check_access("agent-alpha", "knowledge", "write")

        assert granted is True

    def test_check_access_read_shared_denied(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test check_access denies read shared when can_read_shared is False."""
        row = dict(sample_tenant_row)
        row["permissions"] = json.dumps({
            "domains": [],
            "can_read_shared": False,
            "can_write_shared": False,
            "max_sessions": 10,
            "allowed_operations": ["query", "read"],
            "custom_permissions": {},
        })
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [row]

        granted = tenant_manager.check_access("agent-alpha", "knowledge", "read")

        assert granted is False

    def test_require_access_granted(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test require_access does not raise when access is granted."""
        cursor = mock_connection.cursor.return_value
        row = dict(sample_tenant_row)
        row["permissions"] = json.dumps({
            "domains": ["software"],
            "can_read_shared": True,
            "can_write_shared": False,
            "max_sessions": 10,
            "allowed_operations": ["query", "ingest", "session", "read"],
            "custom_permissions": {},
        })
        cursor.fetchall.return_value = [row]

        # Should not raise
        tenant_manager.require_access("agent-alpha", "knowledge", "read")

    def test_require_access_raises_on_denied(self, tenant_manager, mock_connection):
        """Test require_access raises TenantAccessDeniedError when access is denied."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        with pytest.raises(TenantAccessDeniedError) as exc_info:
            tenant_manager.require_access("ghost", "knowledge", "read")

        assert exc_info.value.agent_id == "ghost"
        assert exc_info.value.resource == "knowledge"
        assert exc_info.value.action == "read"

    def test_require_access_suspended_raises(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test require_access raises for suspended tenant."""
        suspended_row = dict(sample_tenant_row)
        suspended_row["status"] = "suspended"
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = [suspended_row]

        with pytest.raises(TenantAccessDeniedError):
            tenant_manager.require_access("agent-alpha", "session", "create")

    def test_check_access_session_operations(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test check_access for session resource operations."""
        cursor = mock_connection.cursor.return_value
        row = dict(sample_tenant_row)
        row["permissions"] = json.dumps({
            "domains": ["software"],
            "can_read_shared": True,
            "can_write_shared": False,
            "max_sessions": 10,
            "allowed_operations": ["query", "ingest", "session", "create", "read"],
            "custom_permissions": {},
        })
        cursor.fetchall.return_value = [row]

        assert tenant_manager.check_access("agent-alpha", "session", "create") is True
        assert tenant_manager.check_access("agent-alpha", "session", "read") is True
        assert tenant_manager.check_access("agent-alpha", "session", "query") is True


# ===========================================================================
# Test: Quota Limits
# ===========================================================================


class TestQuotaLimits:
    """Tests for quota tracking, enforcement, and usage management."""

    def test_get_quota_usage(self, tenant_manager, mock_connection, sample_tenant_row, sample_quota_usage_row):
        """Test retrieving quota usage for a tenant."""
        cursor = mock_connection.cursor.return_value
        # First call: get_tenant, second call: quota_usage query
        cursor.fetchall.side_effect = [[sample_tenant_row], [sample_quota_usage_row]]

        usage = tenant_manager.get_quota_usage("agent-alpha")

        assert usage is not None
        assert usage.queries_this_hour == 42
        assert usage.storage_used_mb == 120.5
        assert usage.vectors_count == 5000

    def test_get_quota_usage_tenant_not_found(self, tenant_manager, mock_connection):
        """Test get_quota_usage returns None for unknown tenant."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        usage = tenant_manager.get_quota_usage("unknown-tenant")

        assert usage is None

    def test_get_quota_usage_no_usage_record(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test get_quota_usage returns None when no usage record exists."""
        cursor = mock_connection.cursor.return_value
        # First call: get_tenant succeeds, second call: no quota_usage row
        cursor.fetchall.side_effect = [[sample_tenant_row], []]

        usage = tenant_manager.get_quota_usage("agent-alpha")

        assert usage is None

    def test_enforce_quota_within_limits(self, tenant_manager, mock_connection, sample_tenant_row, sample_quota_usage_row):
        """Test enforce_quota returns True when usage is within limits."""
        cursor = mock_connection.cursor.return_value
        # First: get_tenant, second: get_quota_usage -> get_tenant (again), third: quota_usage query
        cursor.fetchall.side_effect = [
            [sample_tenant_row],      # enforce_quota -> get_tenant
            [sample_tenant_row],      # get_quota_usage -> get_tenant
            [sample_quota_usage_row], # get_quota_usage -> quota query
        ]

        within = tenant_manager.enforce_quota("agent-alpha", "query")

        assert within is True

    def test_enforce_quota_exceeded(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test enforce_quota raises QuotaExceededError when limit is hit."""
        cursor = mock_connection.cursor.return_value
        now = datetime.utcnow()

        exceeded_usage = {
            "id": "q1",
            "tenant_id": "tenant-uuid-001",
            "queries_this_hour": 1000,  # at max limit
            "storage_used_mb": 0,
            "vectors_count": 0,
            "active_sessions": 0,
            "total_sessions": 0,
            "knowledge_entries_count": 0,
            "last_query_at": None,
            "last_reset_at": now,
            "updated_at": now,
        }
        cursor.fetchall.side_effect = [
            [sample_tenant_row],  # enforce_quota -> get_tenant
            [sample_tenant_row],  # get_quota_usage -> get_tenant
            [exceeded_usage],     # get_quota_usage -> quota query
        ]

        with pytest.raises(QuotaExceededError) as exc_info:
            tenant_manager.enforce_quota("agent-alpha", "query")

        assert exc_info.value.agent_id == "agent-alpha"
        assert exc_info.value.quota_name == "queries_per_hour"

    def test_enforce_quota_unknown_tenant_raises(self, tenant_manager, mock_connection):
        """Test enforce_quota raises TenantNotFoundError for unknown tenant."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        with pytest.raises(TenantNotFoundError):
            tenant_manager.enforce_quota("ghost", "query")

    def test_enforce_quota_session_operation(self, tenant_manager, mock_connection, sample_tenant_row, sample_quota_usage_row):
        """Test enforce_quota for session operation within limits."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.side_effect = [
            [sample_tenant_row],      # enforce_quota -> get_tenant
            [sample_tenant_row],      # get_quota_usage -> get_tenant
            [sample_quota_usage_row], # get_quota_usage -> quota query
        ]

        result = tenant_manager.enforce_quota("agent-alpha", "session")

        assert result is True

    def test_enforce_quota_no_usage_record(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test enforce_quota returns True when no quota usage record exists."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.side_effect = [
            [sample_tenant_row],  # enforce_quota -> get_tenant
            [sample_tenant_row],  # get_quota_usage -> get_tenant
            [],                   # get_quota_usage -> no quota row
        ]

        result = tenant_manager.enforce_quota("agent-alpha", "query")

        assert result is True

    def test_increment_query_count(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test incrementing the query counter for a tenant."""
        cursor = mock_connection.cursor.return_value
        # get_tenant returns the tenant
        cursor.fetchall.side_effect = [[sample_tenant_row], []]

        tenant_manager.increment_query_count("agent-alpha")

        # Verify commit was called (UPDATE was executed)
        mock_connection.commit.assert_called()

    def test_increment_query_count_unknown_tenant(self, tenant_manager, mock_connection):
        """Test increment_query_count is a no-op for unknown tenant."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        tenant_manager.increment_query_count("ghost")

        # No commit should be called
        mock_connection.commit.assert_not_called()

    def test_reset_quotas(self, tenant_manager, mock_connection, sample_tenant_row):
        """Test resetting quota counters for a tenant."""
        cursor = mock_connection.cursor.return_value
        reset_row = {
            "id": "quota-uuid-001",
            "tenant_id": "tenant-uuid-001",
            "queries_this_hour": 0,
            "storage_used_mb": 120.5,
            "vectors_count": 5000,
            "active_sessions": 3,
            "total_sessions": 15,
            "knowledge_entries_count": 250,
            "last_query_at": datetime(2026, 1, 1, 12, 0, 0),
            "last_reset_at": datetime(2026, 1, 1, 12, 0, 0),
            "updated_at": datetime(2026, 1, 1, 12, 0, 0),
        }
        cursor.fetchall.side_effect = [[sample_tenant_row], [reset_row]]

        usage = tenant_manager.reset_quotas("agent-alpha")

        assert usage is not None
        assert usage.queries_this_hour == 0
        mock_connection.commit.assert_called()

    def test_reset_quotas_tenant_not_found(self, tenant_manager, mock_connection):
        """Test reset_quotas returns None for unknown tenant."""
        cursor = mock_connection.cursor.return_value
        cursor.fetchall.return_value = []

        usage = tenant_manager.reset_quotas("ghost")

        assert usage is None

    def test_quota_usage_within_quotas_model(self):
        """Test QuotaUsage.is_within_quotas model method."""
        usage = QuotaUsage(
            tenant_id="t1",
            queries_this_hour=500,
            storage_used_mb=200,
            vectors_count=10000,
            active_sessions=5,
            total_sessions=20,
            knowledge_entries_count=1000,
        )
        quotas = TenantQuotas(
            max_queries_per_hour=1000,
            max_storage_mb=500,
            max_vectors=100000,
            max_concurrent_sessions=10,
            max_sessions=50,
            max_knowledge_entries=10000,
        )

        result = usage.is_within_quotas(quotas)

        assert result["queries_per_hour"] is True
        assert result["storage"] is True
        assert result["vectors"] is True
        assert result["active_sessions"] is True

    def test_quota_usage_get_violations_empty(self):
        """Test QuotaUsage.get_violations returns empty when within limits."""
        usage = QuotaUsage(
            tenant_id="t1",
            queries_this_hour=100,
            storage_used_mb=50,
            vectors_count=1000,
            active_sessions=2,
            total_sessions=10,
            knowledge_entries_count=100,
        )
        quotas = TenantQuotas(
            max_queries_per_hour=1000,
            max_storage_mb=500,
            max_vectors=100000,
            max_concurrent_sessions=10,
            max_sessions=50,
            max_knowledge_entries=10000,
        )

        violations = usage.get_violations(quotas)

        assert violations == []

    def test_quota_usage_get_violations_detected(self):
        """Test QuotaUsage.get_violations returns violated quotas."""
        usage = QuotaUsage(
            tenant_id="t1",
            queries_this_hour=1000,  # at limit
            storage_used_mb=600,     # exceeds 500
            vectors_count=1000,
            active_sessions=10,      # at limit
            total_sessions=10,
            knowledge_entries_count=100,
        )
        quotas = TenantQuotas(
            max_queries_per_hour=1000,
            max_storage_mb=500,
            max_vectors=100000,
            max_concurrent_sessions=10,
            max_sessions=50,
            max_knowledge_entries=10000,
        )

        violations = usage.get_violations(quotas)

        assert "queries_per_hour" in violations
        assert "storage" in violations
        assert "active_sessions" in violations


# ===========================================================================
# Test: RLS Context Management
# ===========================================================================


class TestRLSContext:
    """Tests for Row Level Security context and policy setup."""

    def test_set_tenant_context(self, tenant_manager, mock_connection):
        """Test setting the RLS tenant context variable."""
        cursor = mock_connection.cursor.return_value

        tenant_manager.set_tenant_context("agent-alpha")

        # Verify SET app.current_tenant was called
        call_args = str(cursor.execute.call_args)
        assert "app.current_tenant" in call_args
        assert "agent-alpha" in call_args

    def test_clear_tenant_context(self, tenant_manager, mock_connection):
        """Test clearing the RLS tenant context variable."""
        cursor = mock_connection.cursor.return_value

        tenant_manager.clear_tenant_context()

        call_args = str(cursor.execute.call_args)
        assert "RESET" in call_args.upper()

    def test_setup_rls_policies(self, tenant_manager, mock_connection):
        """Test that setup_rls_policies creates policies for RLS tables."""
        cursor = mock_connection.cursor.return_value

        tenant_manager.setup_rls_policies()

        # Should have called commit after creating policies
        mock_connection.commit.assert_called()

        # Verify multiple CREATE POLICY calls (one per table × 4 operations)
        all_calls = [str(c) for c in cursor.execute.call_args_list]
        policy_calls = [c for c in all_calls if "POLICY" in c.upper()]
        # 2 tables × 4 policies (SELECT, INSERT, UPDATE, DELETE) = 8
        assert len(policy_calls) == 8

    def test_setup_rls_policies_handles_duplicate(self, tenant_manager, mock_connection):
        """Test that setup_rls_policies handles existing policies gracefully."""
        import psycopg2
        cursor = mock_connection.cursor.return_value
        cursor.execute.side_effect = psycopg2.errors.DuplicateObject("already exists")

        # Should not raise — rolls back and logs
        tenant_manager.setup_rls_policies()

        mock_connection.rollback.assert_called()

    def test_rls_tables_constant(self):
        """Test that RLS_TABLES includes expected tables."""
        assert "knowledge_entries" in TenantManager.RLS_TABLES
        assert "session_memories" in TenantManager.RLS_TABLES


# ===========================================================================
# Test: Tenant Model
# ===========================================================================


class TestTenantModel:
    """Tests for the Tenant Pydantic model."""

    def test_tenant_defaults(self):
        """Test Tenant has sensible defaults."""
        tenant = Tenant(
            id="t1",
            agent_id="agent-1",
            name="Test",
        )
        assert tenant.status == TenantStatus.ACTIVE
        assert tenant.permissions is not None
        assert tenant.quotas is not None

    def test_tenant_is_active(self):
        """Test is_active returns True only for ACTIVE status."""
        active = _make_tenant(status=TenantStatus.ACTIVE)
        suspended = _make_tenant(status=TenantStatus.SUSPENDED)

        assert active.is_active() is True
        assert suspended.is_active() is False

    def test_tenant_can_access_domain_unrestricted(self):
        """Test can_access_domain returns True when domains list is empty."""
        tenant = _make_tenant(permissions=TenantPermissions(domains=[]))

        assert tenant.can_access_domain("any_domain") is True
        assert tenant.can_access_domain(None) is True

    def test_tenant_can_access_domain_restricted(self):
        """Test can_access_domain respects domain whitelist."""
        tenant = _make_tenant(
            permissions=TenantPermissions(domains=["software", "finance"])
        )

        assert tenant.can_access_domain("software") is True
        assert tenant.can_access_domain("finance") is True
        assert tenant.can_access_domain("medical") is False

    def test_tenant_can_access_domain_inactive(self):
        """Test can_access_domain returns False for inactive tenant."""
        tenant = _make_tenant(
            status=TenantStatus.SUSPENDED,
            permissions=TenantPermissions(domains=["software"]),
        )

        assert tenant.can_access_domain("software") is False

    def test_tenant_can_perform_operation(self):
        """Test can_perform_operation checks allowed_operations."""
        tenant = _make_tenant(
            permissions=TenantPermissions(allowed_operations=["query", "ingest"])
        )

        assert tenant.can_perform_operation("query") is True
        assert tenant.can_perform_operation("ingest") is True
        assert tenant.can_perform_operation("delete") is False

    def test_tenant_can_perform_operation_inactive(self):
        """Test can_perform_operation returns False for inactive tenant."""
        tenant = _make_tenant(
            status=TenantStatus.SUSPENDED,
            permissions=TenantPermissions(allowed_operations=["query"]),
        )

        assert tenant.can_perform_operation("query") is False

    def test_tenant_status_enum_values(self):
        """Test TenantStatus enum has expected values."""
        assert TenantStatus.ACTIVE.value == "active"
        assert TenantStatus.SUSPENDED.value == "suspended"
        assert TenantStatus.PENDING.value == "pending"
        assert TenantStatus.DEACTIVATED.value == "deactivated"


# ===========================================================================
# Test: Error Classes
# ===========================================================================


class TestTenantErrors:
    """Tests for custom exception classes."""

    def test_tenant_access_denied_error(self):
        """Test TenantAccessDeniedError carries context."""
        err = TenantAccessDeniedError("agent-1", "knowledge", "write")

        assert err.agent_id == "agent-1"
        assert err.resource == "knowledge"
        assert err.action == "write"
        assert "denied" in str(err).lower()

    def test_quota_exceeded_error(self):
        """Test QuotaExceededError carries context."""
        err = QuotaExceededError("agent-1", "queries_per_hour", 1001, 1000)

        assert err.agent_id == "agent-1"
        assert err.quota_name == "queries_per_hour"
        assert err.current == 1001
        assert err.limit == 1000
        assert "quota" in str(err).lower()

    def test_tenant_not_found_error(self):
        """Test TenantNotFoundError carries agent_id."""
        err = TenantNotFoundError("missing-agent")

        assert err.agent_id == "missing-agent"
        assert "not found" in str(err).lower()
