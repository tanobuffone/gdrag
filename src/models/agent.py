"""Agent Registry models for gdrag v3.

Defines data structures for agent registration, health monitoring,
and lifecycle management.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Health Status Enum
# ============================================================================

class HealthStatus(str, Enum):
    """Health status of a registered agent."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# ============================================================================
# Agent Registration Models
# ============================================================================

class AgentEndpoint(BaseModel):
    """Endpoint configuration for an agent."""
    url: str = Field(description="Endpoint URL")
    protocol: str = Field(
        default="http",
        description="Protocol (http, https, grpc, ws, wss)"
    )
    path: Optional[str] = Field(
        default=None,
        description="API path suffix (e.g., /api/v1/query)"
    )
    timeout_ms: int = Field(
        default=30000,
        ge=1000,
        le=120000,
        description="Request timeout in milliseconds"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional endpoint metadata"
    )


class AgentRegistration(BaseModel):
    """Agent registration request model.

    Used to register a new agent in the registry.
    """
    agent_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique agent identifier (e.g., 'cline', 'claude-prod')"
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable agent name"
    )
    capabilities: List[str] = Field(
        default_factory=list,
        description="List of agent capabilities (e.g., ['query', 'ingest', 'session'])"
    )
    endpoints: List[AgentEndpoint] = Field(
        default_factory=list,
        description="Agent endpoint configurations"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional agent metadata (version, owner, tags, etc.)"
    )
    heartbeat_interval_s: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Expected heartbeat interval in seconds"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization and filtering"
    )


# ============================================================================
# Agent Info Models
# ============================================================================

class AgentInfo(BaseModel):
    """Registered agent information with status.

    Represents the current state of a registered agent.
    """
    agent_id: str = Field(description="Unique agent identifier")
    name: str = Field(description="Human-readable agent name")
    status: HealthStatus = Field(
        default=HealthStatus.UNKNOWN,
        description="Current health status"
    )
    capabilities: List[str] = Field(
        default_factory=list,
        description="Registered agent capabilities"
    )
    endpoints: List[AgentEndpoint] = Field(
        default_factory=list,
        description="Configured endpoints"
    )
    registered_at: datetime = Field(
        description="Timestamp when agent was registered"
    )
    last_heartbeat: Optional[datetime] = Field(
        default=None,
        description="Timestamp of last heartbeat received"
    )
    heartbeat_interval_s: int = Field(
        default=60,
        description="Expected heartbeat interval in seconds"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional agent metadata"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Agent tags"
    )
    updated_at: datetime = Field(
        description="Last update timestamp"
    )

    def is_stale(self, multiplier: float = 3.0) -> bool:
        """Check if agent heartbeat is stale (missed N intervals)."""
        if self.last_heartbeat is None:
            return True
        from datetime import timedelta
        threshold = self.last_heartbeat + timedelta(
            seconds=self.heartbeat_interval_s * multiplier
        )
        return datetime.now(timezone.utc) > threshold

    def is_healthy(self) -> bool:
        """Check if agent is currently healthy."""
        return self.status == HealthStatus.HEALTHY and not self.is_stale()


class AgentHeartbeat(BaseModel):
    """Agent heartbeat record.

    Represents a single heartbeat event from an agent.
    """
    agent_id: str = Field(description="Agent identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Heartbeat timestamp"
    )
    status: HealthStatus = Field(
        default=HealthStatus.HEALTHY,
        description="Reported health status"
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent metrics (uptime, queue_depth, memory_usage, etc.)"
    )


class AgentRegistryStats(BaseModel):
    """Statistics for the agent registry."""
    total_agents: int = Field(description="Total registered agents")
    healthy_agents: int = Field(description="Agents with healthy status")
    degraded_agents: int = Field(description="Agents with degraded status")
    unhealthy_agents: int = Field(description="Agents with unhealthy status")
    unknown_agents: int = Field(description="Agents with unknown status")
    total_heartbeats: int = Field(description="Total heartbeat records")
    last_updated: datetime = Field(description="Stats generation timestamp")
