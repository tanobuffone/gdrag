"""Multi-tenant isolation manager for gdrag v3.

Provides tenant lifecycle management, Row Level Security (RLS) enforcement,
permission controls, and quota tracking for multi-tenant deployments.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import psycopg2
from psycopg2.extras import RealDictCursor

from ..core.config import AppConfig
from ..models.tenant import (
    QuotaUsage,
    Tenant,
    TenantPermissions,
    TenantQuotas,
    TenantStatus,
)

logger = logging.getLogger(__name__)


class TenantAccessDeniedError(Exception):
    """Raised when a tenant attempts an unauthorized operation."""

    def __init__(self, agent_id: str, resource: str, action: str):
        self.agent_id = agent_id
        self.resource = resource
        self.action = action
        super().__init__(
            f"Tenant '{agent_id}' denied: {action} on {resource}"
        )


class QuotaExceededError(Exception):
    """Raised when a tenant exceeds a quota limit."""

    def __init__(self, agent_id: str, quota_name: str, current: Any, limit: Any):
        self.agent_id = agent_id
        self.quota_name = quota_name
        self.current = current
        self.limit = limit
        super().__init__(
            f"Tenant '{agent_id}' quota exceeded: {quota_name} "
            f"({current}/{limit})"
        )


class TenantNotFoundError(Exception):
    """Raised when a tenant is not found."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        super().__init__(f"Tenant not found: {agent_id}")


class TenantManager:
    """Manages tenant isolation, permissions, quotas, and RLS.

    Provides CRUD operations for tenants, access control enforcement,
    quota tracking, and PostgreSQL Row Level Security policy management.

    Attributes:
        config: Application configuration with database settings.
    """

    # RLS-enabled tables for tenant isolation
    RLS_TABLES = ("knowledge_entries", "session_memories")

    # Default resource-action permission mappings
    _RESOURCE_PERMISSION_MAP = {
        ("knowledge", "read"): "can_read_shared",
        ("knowledge", "write"): "can_write_shared",
        ("knowledge", "ingest"): "allowed_operations",
        ("session", "create"): "allowed_operations",
        ("session", "read"): "allowed_operations",
        ("session", "query"): "allowed_operations",
    }

    def __init__(self, config: AppConfig):
        self.config = config
        self._connection = None

    # ========================================================================
    # Connection Management
    # ========================================================================

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
            self._connection.autocommit = False
            logger.info(
                "TenantManager connected to PostgreSQL at "
                f"{self.config.database.postgres_host}:{self.config.database.postgres_port}"
            )
        return self._connection

    def close(self) -> None:
        """Close the database connection."""
        if self._connection and not self._connection.closed:
            self._connection.close()
            self._connection = None

    def _execute_in_transaction(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch: bool = True,
    ) -> List[Dict[str, Any]]:
        """Execute a query within the current transaction.

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
            logger.error(f"Transaction query error: {e}")
            raise

    def _commit(self) -> None:
        """Commit the current transaction."""
        conn = self._get_connection()
        conn.commit()

    def _rollback(self) -> None:
        """Rollback the current transaction."""
        conn = self._get_connection()
        try:
            if not conn.closed:
                conn.rollback()
        except Exception as e:
            logger.warning(f"Rollback error: {e}")

    # ========================================================================
    # Row Level Security
    # ========================================================================

    def setup_rls_policies(self) -> None:
        """Create RLS policies for tenant data isolation.

        Enables Row Level Security on tenant-scoped tables and creates
        policies that restrict access to rows matching the current
        tenant context set via ``SET app.current_tenant``.

        Call once during migration or startup to ensure policies exist.
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                for table in self.RLS_TABLES:
                    # Enable RLS on the table
                    cursor.execute(
                        f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY"
                    )

                    # Create SELECT policy
                    cursor.execute(f"""
                        CREATE POLICY tenant_isolation_select ON {table}
                        FOR SELECT
                        USING (
                            tenant_id IS NULL
                            OR tenant_id = current_setting(
                                'app.current_tenant', true
                            )
                        )
                    """)

                    # Create INSERT policy
                    cursor.execute(f"""
                        CREATE POLICY tenant_isolation_insert ON {table}
                        FOR INSERT
                        WITH CHECK (
                            tenant_id IS NULL
                            OR tenant_id = current_setting(
                                'app.current_tenant', true
                            )
                        )
                    """)

                    # Create UPDATE policy
                    cursor.execute(f"""
                        CREATE POLICY tenant_isolation_update ON {table}
                        FOR UPDATE
                        USING (
                            tenant_id IS NULL
                            OR tenant_id = current_setting(
                                'app.current_tenant', true
                            )
                        )
                    """)

                    # Create DELETE policy
                    cursor.execute(f"""
                        CREATE POLICY tenant_isolation_delete ON {table}
                        FOR DELETE
                        USING (
                            tenant_id IS NULL
                            OR tenant_id = current_setting(
                                'app.current_tenant', true
                            )
                        )
                    """)

                    logger.info(f"RLS policies created for table: {table}")

            conn.commit()
        except psycopg2.errors.DuplicateObject:
            conn.rollback()
            logger.info("RLS policies already exist, skipping creation")
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to setup RLS policies: {e}")
            raise

    def set_tenant_context(self, agent_id: str) -> None:
        """Set the current tenant context for RLS enforcement.

        After calling this method, all subsequent queries in the same
        connection will be filtered by the tenant's row-level security
        policies.

        Args:
            agent_id: The agent/tenant identifier to set as context.
        """
        conn = self._get_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "SET app.current_tenant = %s", (agent_id,)
            )
        logger.debug(f"Tenant context set to: {agent_id}")

    def clear_tenant_context(self) -> None:
        """Clear the current tenant context."""
        conn = self._get_connection()
        with conn.cursor() as cursor:
            cursor.execute("RESET app.current_tenant")
        logger.debug("Tenant context cleared")

    # ========================================================================
    # Tenant CRUD Operations
    # ========================================================================

    def create_tenant(
        self,
        agent_id: str,
        name: str,
        permissions: Optional[TenantPermissions] = None,
        quotas: Optional[TenantQuotas] = None,
        description: Optional[str] = None,
    ) -> Tenant:
        """Create a new tenant with the given configuration.

        Args:
            agent_id: Unique agent identifier.
            name: Display name for the tenant.
            permissions: Permission configuration (defaults if None).
            quotas: Resource quota limits (defaults if None).
            description: Optional description.

        Returns:
            The created Tenant model.

        Raises:
            ValueError: If a tenant with the agent_id already exists.
        """
        tenant_id = str(uuid4())
        now = datetime.utcnow()
        perms = permissions or TenantPermissions()
        qtas = quotas or TenantQuotas()

        tenant = Tenant(
            id=tenant_id,
            agent_id=agent_id,
            name=name,
            description=description,
            permissions=perms,
            quotas=qtas,
            status=TenantStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        )

        query = """
            INSERT INTO tenants (
                id, agent_id, name, description,
                permissions, quotas, status, metadata,
                created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """
        params = (
            tenant.id,
            tenant.agent_id,
            tenant.name,
            tenant.description,
            json.dumps(perms.model_dump()),
            json.dumps(qtas.model_dump()),
            tenant.status.value,
            json.dumps(tenant.metadata),
            tenant.created_at,
            tenant.updated_at,
        )

        try:
            self._execute_in_transaction(query, params, fetch=False)
            self._commit()
        except psycopg2.errors.UniqueViolation:
            self._rollback()
            raise ValueError(f"Tenant with agent_id '{agent_id}' already exists")
        except Exception as e:
            self._rollback()
            logger.error(f"Failed to create tenant: {e}")
            raise

        # Initialize quota usage tracking
        self._create_quota_usage(tenant_id)

        logger.info(
            f"Created tenant '{name}' (agent_id={agent_id}, id={tenant_id})"
        )
        return tenant

    def get_tenant(self, agent_id: str) -> Optional[Tenant]:
        """Retrieve a tenant by agent_id.

        Args:
            agent_id: Agent identifier to look up.

        Returns:
            Tenant model or None if not found.
        """
        query = """
            SELECT id, agent_id, name, description,
                   permissions, quotas, status, metadata,
                   created_at, updated_at
            FROM tenants
            WHERE agent_id = %s
        """
        results = self._execute_in_transaction(query, (agent_id,))

        if not results:
            return None

        row = results[0]
        return self._row_to_tenant(row)

    def get_tenant_by_id(self, tenant_id: str) -> Optional[Tenant]:
        """Retrieve a tenant by internal id.

        Args:
            tenant_id: Internal tenant UUID.

        Returns:
            Tenant model or None if not found.
        """
        query = """
            SELECT id, agent_id, name, description,
                   permissions, quotas, status, metadata,
                   created_at, updated_at
            FROM tenants
            WHERE id = %s
        """
        results = self._execute_in_transaction(query, (tenant_id,))

        if not results:
            return None

        row = results[0]
        return self._row_to_tenant(row)

    def update_permissions(
        self,
        agent_id: str,
        permissions: TenantPermissions,
    ) -> Tenant:
        """Update the permissions for an existing tenant.

        Args:
            agent_id: Agent identifier of the tenant.
            permissions: New permission configuration.

        Returns:
            Updated Tenant model.

        Raises:
            TenantNotFoundError: If tenant does not exist.
        """
        tenant = self.get_tenant(agent_id)
        if tenant is None:
            raise TenantNotFoundError(agent_id)

        now = datetime.utcnow()
        query = """
            UPDATE tenants
            SET permissions = %s, updated_at = %s
            WHERE agent_id = %s
            RETURNING id, agent_id, name, description,
                      permissions, quotas, status, metadata,
                      created_at, updated_at
        """
        params = (
            json.dumps(permissions.model_dump()),
            now,
            agent_id,
        )

        try:
            results = self._execute_in_transaction(query, params)
            self._commit()
        except Exception as e:
            self._rollback()
            logger.error(f"Failed to update permissions for {agent_id}: {e}")
            raise

        updated = self._row_to_tenant(results[0])
        logger.info(f"Updated permissions for tenant agent_id={agent_id}")
        return updated

    def update_quotas(
        self,
        agent_id: str,
        quotas: TenantQuotas,
    ) -> Tenant:
        """Update the quotas for an existing tenant.

        Args:
            agent_id: Agent identifier of the tenant.
            quotas: New quota configuration.

        Returns:
            Updated Tenant model.

        Raises:
            TenantNotFoundError: If tenant does not exist.
        """
        tenant = self.get_tenant(agent_id)
        if tenant is None:
            raise TenantNotFoundError(agent_id)

        now = datetime.utcnow()
        query = """
            UPDATE tenants
            SET quotas = %s, updated_at = %s
            WHERE agent_id = %s
            RETURNING id, agent_id, name, description,
                      permissions, quotas, status, metadata,
                      created_at, updated_at
        """
        params = (json.dumps(quotas.model_dump()), now, agent_id)

        try:
            results = self._execute_in_transaction(query, params)
            self._commit()
        except Exception as e:
            self._rollback()
            logger.error(f"Failed to update quotas for {agent_id}: {e}")
            raise

        updated = self._row_to_tenant(results[0])
        logger.info(f"Updated quotas for tenant agent_id={agent_id}")
        return updated

    def delete_tenant(self, agent_id: str) -> bool:
        """Delete a tenant and all associated data.

        Cascading deletes handle quota_usage, and tenant_id references
        on knowledge_entries and session_memories.

        Args:
            agent_id: Agent identifier of the tenant to delete.

        Returns:
            True if the tenant was deleted, False if it was not found.
        """
        query = "DELETE FROM tenants WHERE agent_id = %s RETURNING id"
        try:
            results = self._execute_in_transaction(query, (agent_id,))
            self._commit()
        except Exception as e:
            self._rollback()
            logger.error(f"Failed to delete tenant {agent_id}: {e}")
            raise

        deleted = len(results) > 0
        if deleted:
            logger.info(f"Deleted tenant agent_id={agent_id}")
        return deleted

    def list_tenants(
        self,
        status: Optional[TenantStatus] = None,
        limit: int = 100,
    ) -> List[Tenant]:
        """List tenants, optionally filtered by status.

        Args:
            status: Optional status filter.
            limit: Maximum tenants to return.

        Returns:
            List of Tenant models.
        """
        if status:
            query = """
                SELECT id, agent_id, name, description,
                       permissions, quotas, status, metadata,
                       created_at, updated_at
                FROM tenants
                WHERE status = %s
                ORDER BY created_at DESC
                LIMIT %s
            """
            results = self._execute_in_transaction(query, (status.value, limit))
        else:
            query = """
                SELECT id, agent_id, name, description,
                       permissions, quotas, status, metadata,
                       created_at, updated_at
                FROM tenants
                ORDER BY created_at DESC
                LIMIT %s
            """
            results = self._execute_in_transaction(query, (limit,))

        return [self._row_to_tenant(row) for row in results]

    # ========================================================================
    # Access Control
    # ========================================================================

    def check_access(
        self,
        agent_id: str,
        resource: str,
        action: str,
    ) -> bool:
        """Check whether a tenant is allowed to perform an action on a resource.

        Verifies:
        - Tenant exists and is active.
        - Operation is in the tenant's allowed_operations.
        - Domain-specific permission checks (e.g. can_write_shared).

        Args:
            agent_id: Agent identifier.
            resource: Resource type (e.g. 'knowledge', 'session').
            action: Action to perform (e.g. 'read', 'write', 'query').

        Returns:
            True if access is granted, False otherwise.
        """
        tenant = self.get_tenant(agent_id)
        if tenant is None:
            logger.warning(f"Access check for unknown tenant: {agent_id}")
            return False

        if not tenant.is_active():
            logger.warning(
                f"Access denied: tenant {agent_id} status={tenant.status.value}"
            )
            return False

        # Check operation is allowed
        if not tenant.can_perform_operation(action):
            logger.info(
                f"Access denied: tenant {agent_id} cannot perform '{action}'"
            )
            return False

        # Check domain-specific permissions
        perm_key = self._RESOURCE_PERMISSION_MAP.get((resource, action))
        if perm_key is not None:
            if perm_key == "can_read_shared":
                if not tenant.permissions.can_read_shared:
                    logger.info(
                        f"Access denied: tenant {agent_id} cannot read shared"
                    )
                    return False
            elif perm_key == "can_write_shared":
                if not tenant.permissions.can_write_shared:
                    logger.info(
                        f"Access denied: tenant {agent_id} cannot write shared"
                    )
                    return False

        return True

    def require_access(
        self,
        agent_id: str,
        resource: str,
        action: str,
    ) -> None:
        """Enforce access control, raising if access is denied.

        Args:
            agent_id: Agent identifier.
            resource: Resource type.
            action: Action to perform.

        Raises:
            TenantAccessDeniedError: If access is not permitted.
        """
        if not self.check_access(agent_id, resource, action):
            raise TenantAccessDeniedError(agent_id, resource, action)

    # ========================================================================
    # Quota Management
    # ========================================================================

    def get_quota_usage(self, agent_id: str) -> Optional[QuotaUsage]:
        """Get current quota usage for a tenant.

        Resets hourly counters if more than one hour has elapsed since
        the last reset.

        Args:
            agent_id: Agent identifier.

        Returns:
            QuotaUsage model or None if tenant not found.
        """
        tenant = self.get_tenant(agent_id)
        if tenant is None:
            return None

        query = """
            SELECT id, tenant_id,
                   queries_this_hour, storage_used_mb, vectors_count,
                   active_sessions, total_sessions, knowledge_entries_count,
                   last_query_at, last_reset_at, updated_at
            FROM quota_usage
            WHERE tenant_id = %s
        """
        results = self._execute_in_transaction(query, (tenant.id,))

        if not results:
            return None

        row = results[0]

        # Check if hourly reset is needed
        if row["last_reset_at"] is not None:
            elapsed = datetime.utcnow() - row["last_reset_at"]
            if elapsed.total_seconds() >= 3600:
                self._reset_hourly_quota(tenant.id)
                row["queries_this_hour"] = 0

        return QuotaUsage(
            id=row["id"],
            tenant_id=row["tenant_id"],
            queries_this_hour=row["queries_this_hour"],
            storage_used_mb=float(row["storage_used_mb"]),
            vectors_count=row["vectors_count"],
            active_sessions=row["active_sessions"],
            total_sessions=row["total_sessions"],
            knowledge_entries_count=row["knowledge_entries_count"],
            last_query_at=row["last_query_at"],
            last_reset_at=row["last_reset_at"],
            updated_at=row["updated_at"],
        )

    def enforce_quota(
        self,
        agent_id: str,
        operation: str,
    ) -> bool:
        """Enforce quota limits before performing an operation.

        Checks the tenant's quotas against current usage and raises if
        the operation would exceed a limit.

        Args:
            agent_id: Agent identifier.
            operation: Operation type ('query', 'ingest', 'session').

        Returns:
            True if the operation is within quota limits.

        Raises:
            TenantNotFoundError: If tenant does not exist.
            QuotaExceededError: If a quota limit is exceeded.
        """
        tenant = self.get_tenant(agent_id)
        if tenant is None:
            raise TenantNotFoundError(agent_id)

        usage = self.get_quota_usage(agent_id)
        if usage is None:
            return True

        violations = usage.get_violations(tenant.quotas)
        if not violations:
            return True

        # Map operation to the relevant quota key
        operation_quota_map = {
            "query": "queries_per_hour",
            "ingest": "knowledge_entries",
            "session": "active_sessions",
        }

        relevant = operation_quota_map.get(operation)
        if relevant and relevant in violations:
            limit_value = getattr(tenant.quotas, f"max_{relevant}", None)
            usage_value = getattr(usage, relevant.rstrip("s") + (
                "_this_hour" if relevant == "queries_per_hour" else
                "_count" if relevant == "knowledge_entries" else
                "_sessions" if relevant == "active_sessions" else ""
            ), None)
            raise QuotaExceededError(
                agent_id=agent_id,
                quota_name=relevant,
                current=usage_value,
                limit=limit_value,
            )

        return True

    def increment_query_count(self, agent_id: str) -> None:
        """Increment the query counter for a tenant.

        Call this after a successful query to track hourly usage.

        Args:
            agent_id: Agent identifier.
        """
        tenant = self.get_tenant(agent_id)
        if tenant is None:
            return

        query = """
            UPDATE quota_usage
            SET queries_this_hour = queries_this_hour + 1,
                last_query_at = NOW(),
                updated_at = NOW()
            WHERE tenant_id = %s
        """
        try:
            self._execute_in_transaction(query, (tenant.id,), fetch=False)
            self._commit()
        except Exception as e:
            self._rollback()
            logger.warning(f"Failed to increment query count for {agent_id}: {e}")

    def increment_session_count(self, agent_id: str) -> None:
        """Increment session counters for a tenant.

        Args:
            agent_id: Agent identifier.
        """
        tenant = self.get_tenant(agent_id)
        if tenant is None:
            return

        query = """
            UPDATE quota_usage
            SET active_sessions = active_sessions + 1,
                total_sessions = total_sessions + 1,
                updated_at = NOW()
            WHERE tenant_id = %s
        """
        try:
            self._execute_in_transaction(query, (tenant.id,), fetch=False)
            self._commit()
        except Exception as e:
            self._rollback()
            logger.warning(f"Failed to increment session count for {agent_id}: {e}")

    def decrement_session_count(self, agent_id: str) -> None:
        """Decrement the active session counter for a tenant.

        Args:
            agent_id: Agent identifier.
        """
        tenant = self.get_tenant(agent_id)
        if tenant is None:
            return

        query = """
            UPDATE quota_usage
            SET active_sessions = GREATEST(active_sessions - 1, 0),
                updated_at = NOW()
            WHERE tenant_id = %s
        """
        try:
            self._execute_in_transaction(query, (tenant.id,), fetch=False)
            self._commit()
        except Exception as e:
            self._rollback()
            logger.warning(f"Failed to decrement session count for {agent_id}: {e}")

    def update_storage_usage(
        self,
        agent_id: str,
        delta_mb: float,
    ) -> None:
        """Update storage usage for a tenant.

        Args:
            agent_id: Agent identifier.
            delta_mb: Change in storage (positive = add, negative = remove).
        """
        tenant = self.get_tenant(agent_id)
        if tenant is None:
            return

        query = """
            UPDATE quota_usage
            SET storage_used_mb = GREATEST(storage_used_mb + %s, 0),
                updated_at = NOW()
            WHERE tenant_id = %s
        """
        try:
            self._execute_in_transaction(query, (delta_mb, tenant.id,), fetch=False)
            self._commit()
        except Exception as e:
            self._rollback()
            logger.warning(f"Failed to update storage for {agent_id}: {e}")

    def increment_knowledge_entries(self, agent_id: str, count: int = 1) -> None:
        """Increment the knowledge entry counter for a tenant.

        Args:
            agent_id: Agent identifier.
            count: Number of entries to add.
        """
        tenant = self.get_tenant(agent_id)
        if tenant is None:
            return

        query = """
            UPDATE quota_usage
            SET knowledge_entries_count = knowledge_entries_count + %s,
                updated_at = NOW()
            WHERE tenant_id = %s
        """
        try:
            self._execute_in_transaction(query, (count, tenant.id,), fetch=False)
            self._commit()
        except Exception as e:
            self._rollback()
            logger.warning(
                f"Failed to increment knowledge entries for {agent_id}: {e}"
            )

    def reset_quotas(self, agent_id: str) -> Optional[QuotaUsage]:
        """Reset all quota counters for a tenant.

        Args:
            agent_id: Agent identifier.

        Returns:
            Reset QuotaUsage or None if tenant not found.
        """
        tenant = self.get_tenant(agent_id)
        if tenant is None:
            return None

        query = """
            UPDATE quota_usage
            SET queries_this_hour = 0,
                last_reset_at = NOW(),
                updated_at = NOW()
            WHERE tenant_id = %s
            RETURNING id, tenant_id,
                      queries_this_hour, storage_used_mb, vectors_count,
                      active_sessions, total_sessions, knowledge_entries_count,
                      last_query_at, last_reset_at, updated_at
        """
        try:
            results = self._execute_in_transaction(query, (tenant.id,))
            self._commit()
        except Exception as e:
            self._rollback()
            logger.error(f"Failed to reset quotas for {agent_id}: {e}")
            raise

        if not results:
            return None
        row = results[0]
        return QuotaUsage(
            id=row["id"],
            tenant_id=row["tenant_id"],
            queries_this_hour=row["queries_this_hour"],
            storage_used_mb=float(row["storage_used_mb"]),
            vectors_count=row["vectors_count"],
            active_sessions=row["active_sessions"],
            total_sessions=row["total_sessions"],
            knowledge_entries_count=row["knowledge_entries_count"],
            last_query_at=row["last_query_at"],
            last_reset_at=row["last_reset_at"],
            updated_at=row["updated_at"],
        )

    # ========================================================================
    # Tenant Context Manager
    # ========================================================================

    def tenant_context(self, agent_id: str):
        """Context manager that sets tenant context for RLS.

        Usage::

            with tm.tenant_context("agent_123"):
                # All DB queries here are filtered by RLS
                results = store.full_text_search("query")

        Args:
            agent_id: Agent identifier.

        Yields:
            None

        Raises:
            TenantNotFoundError: If tenant does not exist.
        """
        return _TenantContext(self, agent_id)

    # ========================================================================
    # Private Helpers
    # ========================================================================

    def _row_to_tenant(self, row: Dict[str, Any]) -> Tenant:
        """Convert a database row to a Tenant model."""
        permissions_data = row["permissions"]
        quotas_data = row["quotas"]
        metadata_data = row["metadata"]

        return Tenant(
            id=row["id"],
            agent_id=row["agent_id"],
            name=row["name"],
            description=row.get("description"),
            permissions=TenantPermissions(
                **permissions_data if isinstance(permissions_data, dict)
                else json.loads(permissions_data)
            ),
            quotas=TenantQuotas(
                **quotas_data if isinstance(quotas_data, dict)
                else json.loads(quotas_data)
            ),
            status=TenantStatus(row["status"]),
            metadata=(
                metadata_data if isinstance(metadata_data, dict)
                else json.loads(metadata_data)
            ) if metadata_data else {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _create_quota_usage(self, tenant_id: str) -> None:
        """Initialize quota usage record for a new tenant."""
        quota_id = str(uuid4())
        now = datetime.utcnow()

        query = """
            INSERT INTO quota_usage (
                id, tenant_id,
                queries_this_hour, storage_used_mb, vectors_count,
                active_sessions, total_sessions, knowledge_entries_count,
                last_reset_at, updated_at
            ) VALUES (
                %s, %s, 0, 0.0, 0, 0, 0, 0, %s, %s
            )
        """
        try:
            self._execute_in_transaction(
                query, (quota_id, tenant_id, now, now), fetch=False
            )
            self._commit()
        except Exception as e:
            self._rollback()
            logger.warning(f"Failed to create quota usage for tenant {tenant_id}: {e}")

    def _reset_hourly_quota(self, tenant_id: str) -> None:
        """Reset hourly query counter for a tenant."""
        query = """
            UPDATE quota_usage
            SET queries_this_hour = 0,
                last_reset_at = NOW(),
                updated_at = NOW()
            WHERE tenant_id = %s
        """
        try:
            self._execute_in_transaction(query, (tenant_id,), fetch=False)
            self._commit()
        except Exception as e:
            self._rollback()
            logger.warning(f"Failed to reset hourly quota for {tenant_id}: {e}")


class _TenantContext:
    """Context manager for tenant-scoped database operations."""

    def __init__(self, manager: TenantManager, agent_id: str):
        self._manager = manager
        self._agent_id = agent_id

    def __enter__(self):
        tenant = self._manager.get_tenant(self._agent_id)
        if tenant is None:
            raise TenantNotFoundError(self._agent_id)
        self._manager.set_tenant_context(self._agent_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._manager.clear_tenant_context()
        return False
