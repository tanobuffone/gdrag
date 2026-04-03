"""API v3 – Agent Registry endpoints.

Provides CRUD operations for agent registration, heartbeats,
health monitoring, and capability-based discovery.

Routes are prefixed with ``/api/v3/agents``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ...core.agent_registry import AgentRegistry
from ...models.agent import (
    AgentHeartbeat,
    AgentInfo,
    AgentRegistration,
    HealthStatus,
)
from .dependencies import get_agent_registry


# ============================================================================
# Request / Response schemas
# ============================================================================

class HeartbeatRequest(BaseModel):
    """POST heartbeat request body."""
    status: HealthStatus = Field(
        default=HealthStatus.HEALTHY,
        description="Reported health status",
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Agent metrics (uptime, queue_depth, …)",
    )


class RegisterResponse(BaseModel):
    """Response after successful agent registration."""
    agent_id: str
    status: str = "registered"


class UnregisterResponse(BaseModel):
    """Response after successful agent deletion."""
    agent_id: str
    status: str = "unregistered"


class HeartbeatResponse(BaseModel):
    """Response after successful heartbeat recording."""
    agent_id: str
    status: str = "ok"


class HealthResponse(BaseModel):
    """Response model for agent health endpoint."""
    agent_id: str
    health_status: Optional[str] = None


class AgentListResponse(BaseModel):
    """Response model for listing agents."""
    agents: List[AgentInfo]
    count: int


class CapabilitiesResponse(BaseModel):
    """Response model for capability search."""
    capability: str
    agents: List[AgentInfo]
    count: int


# ============================================================================
# Router
# ============================================================================

agents_router = APIRouter(
    prefix="/api/v3/agents",
    tags=["agents-v3"],
)


# ─── Endpoints ───────────────────────────────────────────────────────────────


@agents_router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=201,
    summary="Register a new agent",
    description="Register an agent in the registry (upsert semantics).",
)
async def register_agent(
    registration: AgentRegistration,
    registry: AgentRegistry = Depends(get_agent_registry),
) -> RegisterResponse:
    try:
        agent_id = registry.register_agent(registration)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register agent: {exc}",
        )
    return RegisterResponse(agent_id=agent_id)


@agents_router.delete(
    "/{agent_id}",
    response_model=UnregisterResponse,
    summary="Unregister (delete) an agent",
    description="Remove an agent and its heartbeat history from the registry.",
)
async def unregister_agent(
    agent_id: str,
    registry: AgentRegistry = Depends(get_agent_registry),
) -> UnregisterResponse:
    try:
        deleted = registry.unregister_agent(agent_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unregister agent: {exc}",
        )
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found",
        )
    return UnregisterResponse(agent_id=agent_id)


@agents_router.post(
    "/{agent_id}/heartbeat",
    response_model=HeartbeatResponse,
    summary="Record agent heartbeat",
    description="Post a heartbeat signal from an agent to update its health status.",
)
async def agent_heartbeat(
    agent_id: str,
    body: HeartbeatRequest,
    registry: AgentRegistry = Depends(get_agent_registry),
) -> HeartbeatResponse:
    try:
        recorded = registry.heartbeat(
            agent_id=agent_id,
            status=body.status,
            metrics=body.metrics,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record heartbeat: {exc}",
        )
    if not recorded:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found",
        )
    return HeartbeatResponse(agent_id=agent_id)


@agents_router.get(
    "",
    response_model=AgentListResponse,
    summary="List registered agents",
    description="Return all registered agents, optionally filtered by status.",
)
async def list_agents(
    status: Optional[str] = Query(
        default=None,
        description="Filter by health status (healthy, degraded, unhealthy, unknown) or 'active'",
    ),
    limit: int = Query(default=100, ge=1, le=1000),
    registry: AgentRegistry = Depends(get_agent_registry),
) -> AgentListResponse:
    try:
        agents = registry.list_agents(status=status, limit=limit)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list agents: {exc}",
        )
    return AgentListResponse(agents=agents, count=len(agents))


@agents_router.get(
    "/{agent_id}",
    response_model=AgentInfo,
    summary="Get agent details",
    description="Retrieve full details of a single registered agent.",
)
async def get_agent(
    agent_id: str,
    registry: AgentRegistry = Depends(get_agent_registry),
) -> AgentInfo:
    try:
        agent = registry.get_agent(agent_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve agent: {exc}",
        )
    if agent is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found",
        )
    return agent


@agents_router.get(
    "/{agent_id}/health",
    response_model=HealthResponse,
    summary="Get agent health status",
    description="Return the current health status of a specific agent.",
)
async def get_agent_health(
    agent_id: str,
    registry: AgentRegistry = Depends(get_agent_registry),
) -> HealthResponse:
    try:
        health = registry.get_agent_health(agent_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve health: {exc}",
        )
    if health is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found",
        )
    return HealthResponse(
        agent_id=agent_id,
        health_status=health.value,
    )


@agents_router.get(
    "/capabilities/{capability}",
    response_model=CapabilitiesResponse,
    summary="Find agents by capability",
    description="Return all agents that possess the specified capability.",
)
async def find_agents_by_capability(
    capability: str,
    limit: int = Query(default=100, ge=1, le=1000),
    registry: AgentRegistry = Depends(get_agent_registry),
) -> CapabilitiesResponse:
    try:
        agents = registry.find_agents_by_capability(capability, limit=limit)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search by capability: {exc}",
        )
    return CapabilitiesResponse(
        capability=capability,
        agents=agents,
        count=len(agents),
    )
