"""Multi-Tenancy models for gdrag v3.

Defines tenant isolation, permissions, quotas, and usage tracking
for multi-tenant deployment support.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class TenantStatus(str, Enum):
    """Status of a tenant account."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"
    DEACTIVATED = "deactivated"


class TenantPermissions(BaseModel):
    """Permissions configuration for a tenant.
    
    Controls what domains a tenant can access and whether
    they can read/write shared knowledge.
    """
    domains: List[str] = Field(
        default_factory=list,
        description="Allowed knowledge domains (empty = all)"
    )
    can_read_shared: bool = Field(
        default=True,
        description="Can read shared/public knowledge"
    )
    can_write_shared: bool = Field(
        default=False,
        description="Can write to shared/public knowledge"
    )
    max_sessions: int = Field(
        default=10, ge=1,
        description="Max concurrent sessions"
    )
    allowed_operations: List[str] = Field(
        default_factory=lambda: ["query", "ingest", "session"],
        description="Allowed operations"
    )
    custom_permissions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom permission flags"
    )


class TenantQuotas(BaseModel):
    """Resource quotas for a tenant.
    
    Defines the maximum resource consumption limits
    for rate limiting and capacity planning.
    """
    max_queries_per_hour: int = Field(
        default=1000, ge=0,
        description="Max queries per hour"
    )
    max_storage_mb: int = Field(
        default=500, ge=0,
        description="Max storage in megabytes"
    )
    max_vectors: int = Field(
        default=100000, ge=0,
        description="Max vector embeddings"
    )
    max_sessions: int = Field(
        default=50, ge=0,
        description="Max total sessions (lifetime)"
    )
    max_knowledge_entries: int = Field(
        default=10000, ge=0,
        description="Max knowledge entries"
    )
    max_concurrent_sessions: int = Field(
        default=10, ge=1,
        description="Max concurrent active sessions"
    )
    custom_quotas: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom quota limits"
    )


class QuotaUsage(BaseModel):
    """Current quota usage tracking for a tenant.
    
    Tracks real-time consumption of resources against
    the configured tenant quotas.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    tenant_id: str = Field(description="Associated tenant ID")
    queries_this_hour: int = Field(default=0, ge=0)
    storage_used_mb: float = Field(default=0.0, ge=0.0)
    vectors_count: int = Field(default=0, ge=0)
    active_sessions: int = Field(default=0, ge=0)
    total_sessions: int = Field(default=0, ge=0)
    knowledge_entries_count: int = Field(default=0, ge=0)
    last_query_at: Optional[datetime] = Field(default=None)
    last_reset_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def is_within_quotas(self, quotas: TenantQuotas) -> Dict[str, bool]:
        """Check if current usage is within configured quotas."""
        return {
            "queries_per_hour": self.queries_this_hour < quotas.max_queries_per_hour,
            "storage": self.storage_used_mb < quotas.max_storage_mb,
            "vectors": self.vectors_count < quotas.max_vectors,
            "active_sessions": self.active_sessions < quotas.max_concurrent_sessions,
            "total_sessions": self.total_sessions < quotas.max_sessions,
            "knowledge_entries": self.knowledge_entries_count < quotas.max_knowledge_entries,
        }

    def get_violations(self, quotas: TenantQuotas) -> List[str]:
        """Get list of quota violations."""
        return [k for k, v in self.is_within_quotas(quotas).items() if not v]


class Tenant(BaseModel):
    """Main tenant model for multi-tenancy support.
    
    Represents a tenant (agent/organization) in the system with
    its configuration, permissions, and quotas.
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str = Field(description="Unique agent identifier")
    name: str = Field(min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=1000)
    permissions: TenantPermissions = Field(default_factory=TenantPermissions)
    quotas: TenantQuotas = Field(default_factory=TenantQuotas)
    status: TenantStatus = Field(default=TenantStatus.ACTIVE)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def is_active(self) -> bool:
        """Check if tenant is active."""
        return self.status == TenantStatus.ACTIVE

    def can_access_domain(self, domain: Optional[str]) -> bool:
        """Check if tenant can access a specific domain."""
        if not self.is_active():
            return False
        if not self.permissions.domains:
            return True
        return domain in self.permissions.domains if domain else True

    def can_perform_operation(self, operation: str) -> bool:
        """Check if tenant can perform a specific operation."""
        if not self.is_active():
            return False
        return operation in self.permissions.allowed_operations


class TenantCreateRequest(BaseModel):
    """Request model for creating a new tenant."""
    agent_id: str = Field(min_length=1, max_length=255)
    name: str = Field(min_length=1, max_length=255)
    description: Optional[str] = Field(default=None)
    permissions: Optional[TenantPermissions] = Field(default=None)
    quotas: Optional[TenantQuotas] = Field(default=None)


class TenantUpdateRequest(BaseModel):
    """Request model for updating a tenant."""
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = Field(default=None)
    permissions: Optional[TenantPermissions] = Field(default=None)
    quotas: Optional[TenantQuotas] = Field(default=None)
    status: Optional[TenantStatus] = Field(default=None)


class TenantResponse(BaseModel):
    """Response model for tenant data."""
    tenant: Tenant
    quota_usage: Optional[QuotaUsage] = Field(default=None)