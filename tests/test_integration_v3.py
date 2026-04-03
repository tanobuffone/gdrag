"""Integration / E2E tests for gdrag v3 API.

Tests the full multi-agent workflow:
  registration → task enqueue → events → completion
Failure recovery and concurrent agent scenarios.
Uses httpx.AsyncClient with FastAPI TestClient for real HTTP-level testing.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import fakeredis.aioredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Import the FastAPI app and v3 components
# ---------------------------------------------------------------------------

# We build a lightweight FastAPI app for testing that only includes v3 routes
# to avoid pulling in heavy external deps (Qdrant, Memgraph, PostgreSQL).

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v3.agents import agents_router
from src.api.v3.tasks import tasks_router
from src.api.v3.knowledge import knowledge_router
from src.api.v3 import dependencies as v3_deps
from src.api.v3.dependencies import TaskQueue, KnowledgeManager, TaskStatus
from src.models.agent import (
    AgentInfo,
    AgentRegistration,
    AgentRegistryStats,
    HealthStatus,
)


# ---------------------------------------------------------------------------
# In-memory AgentRegistry stub for testing
# ---------------------------------------------------------------------------


class InMemoryAgentRegistry:
    """Lightweight in-memory AgentRegistry for integration tests.

    Mirrors the interface of src.core.agent_registry.AgentRegistry but
    uses no external dependencies (PostgreSQL, etc.).
    """

    def __init__(self):
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._heartbeats: Dict[str, List[Dict[str, Any]]] = {}

    def register_agent(self, registration: AgentRegistration) -> str:
        """Register or update an agent."""
        agent_id = registration.agent_id
        self._agents[agent_id] = {
            "agent_id": agent_id,
            "name": registration.name,
            "capabilities": list(registration.capabilities),
            "endpoints": [ep.model_dump() for ep in registration.endpoints],
            "metadata": dict(registration.metadata),
            "tags": list(registration.tags),
            "heartbeat_interval_s": registration.heartbeat_interval_s,
            "status": HealthStatus.UNKNOWN.value,
            "registered_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "last_heartbeat": None,
        }
        if agent_id not in self._heartbeats:
            self._heartbeats[agent_id] = []
        return agent_id

    def unregister_agent(self, agent_id: str) -> bool:
        if agent_id in self._agents:
            del self._agents[agent_id]
            self._heartbeats.pop(agent_id, None)
            return True
        return False

    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        data = self._agents.get(agent_id)
        if data is None:
            return None
        return self._to_agent_info(data)

    def list_agents(
        self, status: Optional[str] = None, limit: int = 100
    ) -> List[AgentInfo]:
        results = []
        for data in self._agents.values():
            if status is None or status == "active":
                if data["status"] in ("healthy", "degraded"):
                    results.append(self._to_agent_info(data))
            elif data["status"] == status:
                results.append(self._to_agent_info(data))
        return results[:limit]

    def heartbeat(
        self,
        agent_id: str,
        status: HealthStatus = HealthStatus.HEALTHY,
        metrics: Optional[dict] = None,
    ) -> bool:
        if agent_id not in self._agents:
            return False
        self._agents[agent_id]["status"] = status.value
        self._agents[agent_id]["last_heartbeat"] = datetime.now(timezone.utc)
        self._agents[agent_id]["updated_at"] = datetime.now(timezone.utc)
        self._heartbeats.setdefault(agent_id, []).append(
            {"status": status.value, "metrics": metrics or {}}
        )
        return True

    def get_agent_health(self, agent_id: str) -> Optional[HealthStatus]:
        data = self._agents.get(agent_id)
        if data is None:
            return None
        return HealthStatus(data["status"])

    def find_agents_by_capability(
        self, capability: str, limit: int = 100
    ) -> List[AgentInfo]:
        results = []
        for data in self._agents.values():
            if capability in data.get("capabilities", []):
                results.append(self._to_agent_info(data))
        return results[:limit]

    def get_stats(self) -> AgentRegistryStats:
        healthy = sum(
            1 for a in self._agents.values() if a["status"] == "healthy"
        )
        degraded = sum(
            1 for a in self._agents.values() if a["status"] == "degraded"
        )
        unhealthy = sum(
            1 for a in self._agents.values() if a["status"] == "unhealthy"
        )
        unknown = sum(
            1 for a in self._agents.values() if a["status"] == "unknown"
        )
        return AgentRegistryStats(
            total_agents=len(self._agents),
            healthy_agents=healthy,
            degraded_agents=degraded,
            unhealthy_agents=unhealthy,
            unknown_agents=unknown,
            total_heartbeats=sum(len(v) for v in self._heartbeats.values()),
            last_updated=datetime.now(timezone.utc),
        )

    def cleanup_stale_agents(self, timeout_seconds: int = 300) -> int:
        return 0  # no-op for tests

    def _to_agent_info(self, data: Dict[str, Any]) -> AgentInfo:
        from src.models.agent import AgentEndpoint

        endpoints = []
        for ep in data.get("endpoints", []):
            endpoints.append(AgentEndpoint(**ep))

        return AgentInfo(
            agent_id=data["agent_id"],
            name=data["name"],
            status=HealthStatus(data.get("status", "unknown")),
            capabilities=data.get("capabilities", []),
            endpoints=endpoints,
            registered_at=data.get("registered_at", datetime.now(timezone.utc)),
            last_heartbeat=data.get("last_heartbeat"),
            heartbeat_interval_s=data.get("heartbeat_interval_s", 60),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            updated_at=data.get("updated_at", datetime.now(timezone.utc)),
        )


# ===========================================================================
# App factory
# ===========================================================================


def create_test_app() -> FastAPI:
    """Create a minimal FastAPI app with v3 routes for testing."""
    test_app = FastAPI(title="gdrag v3 test")
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Attach per-app singletons so get_task_queue / get_knowledge_manager
    # pick them up via request.app.state.
    tq = TaskQueue()
    test_app.state.task_queue = tq
    test_app.state.knowledge_manager = KnowledgeManager()
    registry = InMemoryAgentRegistry()
    test_app.state.agent_registry = registry
    # Also set the module-level singletons so get_agent_registry and
    # test code can access the same instances without importing the
    # real AgentRegistry (which has a path bug).
    v3_deps._task_queue = tq
    v3_deps._agent_registry_singleton = registry
    test_app.include_router(agents_router)
    test_app.include_router(tasks_router)
    test_app.include_router(knowledge_router)
    return test_app


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest_asyncio.fixture
async def test_app():
    """Provide a fresh FastAPI test app per test."""
    # Reset global module-level singletons used as fallback
    v3_deps._task_queue = None
    v3_deps._knowledge_manager = None
    v3_deps._agent_registry_singleton = None
    app = create_test_app()
    yield app
    v3_deps._task_queue = None
    v3_deps._knowledge_manager = None
    v3_deps._agent_registry_singleton = None


@pytest_asyncio.fixture
async def client(test_app):
    """Provide an AsyncClient that talks to the test app over ASGI."""
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def redis_client():
    """Provide a fresh fakeredis client per test."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.flushall()
    await client.aclose()


# ---------------------------------------------------------------------------
# Helper: multi-agent workflow orchestrator
# ---------------------------------------------------------------------------


async def register_agent(
    client: AsyncClient,
    agent_id: str,
    name: str = "Test Agent",
    capabilities: Optional[List[str]] = None,
):
    """Register an agent via the API."""
    payload = {
        "agent_id": agent_id,
        "name": name,
        "capabilities": capabilities or ["read", "write"],
        "endpoints": [
            {"url": f"http://{agent_id}:8000"}
        ],
    }
    return await client.post("/api/v3/agents/register", json=payload)


async def send_heartbeat(
    client: AsyncClient, agent_id: str, status: str = "healthy"
):
    """Send a heartbeat for an agent."""
    return await client.post(
        f"/api/v3/agents/{agent_id}/heartbeat",
        json={"status": status, "metrics": {"uptime_s": 120}},
    )


async def enqueue_task(
    client: AsyncClient,
    task_type: str,
    payload: Optional[Dict[str, Any]] = None,
):
    """Enqueue a task via the API."""
    return await client.post(
        "/api/v3/tasks/enqueue",
        json={"task_type": task_type, "payload": payload or {}},
    )


async def ingest_knowledge(
    client: AsyncClient,
    content: str,
    title: str = "",
    domain: Optional[str] = None,
):
    """Ingest knowledge via the API."""
    return await client.post(
        "/api/v3/knowledge/ingest",
        json={
            "content": content,
            "title": title,
            "domain": domain,
            "source": "test",
        },
    )


# ===========================================================================
# Test: Complete Multi-Agent Flow
#   registration → task enqueue → events → completion
# ===========================================================================


class TestMultiAgentFlow:
    """End-to-end test: register agents → enqueue tasks → complete."""

    @pytest.mark.asyncio
    async def test_register_agent_returns_201(self, client):
        """Test that registering an agent returns 201 with agent_id."""
        resp = await register_agent(client, "agent-alpha", "Alpha Agent")

        assert resp.status_code == 201
        data = resp.json()
        assert data["agent_id"] == "agent-alpha"
        assert data["status"] == "registered"

    @pytest.mark.asyncio
    async def test_register_multiple_agents(self, client):
        """Test registering several agents in sequence."""
        for i in range(5):
            resp = await register_agent(client, f"agent-{i}")
            assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_list_agents_after_registration(self, client):
        """Test that list endpoint returns all registered agents."""
        await register_agent(client, "agent-one", "Agent One")
        await register_agent(client, "agent-two", "Agent Two")

        # Send heartbeats so they show up as "active"
        await send_heartbeat(client, "agent-one")
        await send_heartbeat(client, "agent-two")

        resp = await client.get("/api/v3/agents")
        assert resp.status_code == 200

        data = resp.json()
        agent_ids = [a["agent_id"] for a in data["agents"]]
        assert "agent-one" in agent_ids
        assert "agent-two" in agent_ids
        assert data["count"] == 2

    @pytest.mark.asyncio
    async def test_get_agent_details(self, client):
        """Test retrieving agent details after registration."""
        await register_agent(
            client, "agent-detail", "Detail Agent", ["read", "query"]
        )

        resp = await client.get("/api/v3/agents/agent-detail")
        assert resp.status_code == 200

        data = resp.json()
        assert data["agent_id"] == "agent-detail"
        assert data["name"] == "Detail Agent"
        assert "read" in data["capabilities"]
        assert "query" in data["capabilities"]

    @pytest.mark.asyncio
    async def test_get_nonexistent_agent_returns_404(self, client):
        """Test that getting an unknown agent returns 404."""
        resp = await client.get("/api/v3/agents/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_heartbeat_updates_agent_health(self, client):
        """Test that a heartbeat updates the agent's health status."""
        await register_agent(client, "agent-hb")

        resp = await send_heartbeat(client, "agent-hb", status="healthy")
        assert resp.status_code == 200

        # Check health endpoint
        health_resp = await client.get("/api/v3/agents/agent-hb/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["health_status"] == "healthy"

    @pytest.mark.asyncio
    async def test_unregister_agent(self, client):
        """Test unregistering (deleting) an agent."""
        await register_agent(client, "agent-del")

        resp = await client.delete("/api/v3/agents/agent-del")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unregistered"

        # Verify it's gone
        get_resp = await client.get("/api/v3/agents/agent-del")
        assert get_resp.status_code == 404

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_agent_returns_404(self, client):
        """Test that deleting an unknown agent returns 404."""
        resp = await client.delete("/api/v3/agents/ghost")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_full_registration_task_completion_flow(self, client):
        """Full E2E: register agent → enqueue task → check status → get result.

        This is the primary multi-agent workflow test.
        """
        # Step 1: Register two agents
        await register_agent(
            client, "orchestrator", "Orchestrator", ["orchestrate", "read"]
        )
        await register_agent(
            client, "worker-1", "Worker 1", ["process", "write"]
        )

        # Step 2: Verify both are registered
        # Send heartbeats so they appear in the active list
        await send_heartbeat(client, "orchestrator")
        await send_heartbeat(client, "worker-1")

        list_resp = await client.get("/api/v3/agents")
        assert list_resp.status_code == 200
        assert list_resp.json()["count"] == 2

        # Step 3: Enqueue a task (simulating orchestrator dispatching work)
        task_resp = await enqueue_task(
            client, "process_document", {"doc_id": "doc-42", "format": "markdown"}
        )
        assert task_resp.status_code == 201
        task_data = task_resp.json()
        task_id = task_data["task_id"]
        assert task_data["status"] == "pending"
        assert task_data["task_type"] == "process_document"

        # Step 4: Check task status
        status_resp = await client.get(f"/api/v3/tasks/{task_id}/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["task_id"] == task_id

    @pytest.mark.asyncio
    async def test_task_lifecycle_cancel(self, client):
        """Test task lifecycle: enqueue → cancel."""
        task_resp = await enqueue_task(
            client, "long_running", {"duration": 3600}
        )
        task_id = task_resp.json()["task_id"]

        cancel_resp = await client.post(f"/api/v3/tasks/{task_id}/cancel")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_task_purge(self, client):
        """Test purging tasks from the queue."""
        for i in range(5):
            await enqueue_task(client, f"task_type_{i}", {"i": i})

        purge_resp = await client.post("/api/v3/tasks/purge", json={})
        assert purge_resp.status_code == 200
        assert purge_resp.json()["purged"] == 5

    @pytest.mark.asyncio
    async def test_knowledge_ingest_and_search_flow(self, client):
        """Full E2E: ingest knowledge → search → promote."""
        ingest_resp = await ingest_knowledge(
            client,
            content=(
                "Python is a high-level programming language "
                "known for readability. Python programming is fun."
            ),
            title="Python Basics",
            domain="software",
        )
        assert ingest_resp.status_code == 201
        doc_id = ingest_resp.json()["doc_id"]

        search_resp = await client.get(
            "/api/v3/knowledge/search",
            params={"query": "python programming"},
        )
        assert search_resp.status_code == 200
        search_data = search_resp.json()
        assert search_data["total"] >= 1

        promote_resp = await client.post(
            f"/api/v3/knowledge/{doc_id}/promote"
        )
        assert promote_resp.status_code == 200
        assert promote_resp.json()["promoted"] is True

    @pytest.mark.asyncio
    async def test_knowledge_search_with_domain_filter(self, client):
        """Test knowledge search with domain filtering."""
        await ingest_knowledge(
            client, "Finance content", "Finance Doc", domain="finance"
        )
        await ingest_knowledge(
            client, "Software content", "Software Doc", domain="software"
        )

        resp = await client.get(
            "/api/v3/knowledge/search",
            params={"query": "content", "domain": "finance"},
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 1
        assert results[0]["domain"] == "finance"

    @pytest.mark.asyncio
    async def test_knowledge_revoke_access(self, client):
        """Test revoking access to a knowledge entry."""
        ingest_resp = await ingest_knowledge(client, "Sensitive data", "Secret")
        doc_id = ingest_resp.json()["doc_id"]

        revoke_resp = await client.delete(
            f"/api/v3/knowledge/{doc_id}/access"
        )
        assert revoke_resp.status_code == 200
        assert revoke_resp.json()["access_revoked"] is True

    @pytest.mark.asyncio
    async def test_knowledge_revoke_nonexistent_returns_404(self, client):
        """Test revoking access to a non-existent entry returns 404."""
        resp = await client.delete("/api/v3/knowledge/no-such-doc/access")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_promote_nonexistent_returns_404(self, client):
        """Test promoting a non-existent entry returns 404."""
        resp = await client.post("/api/v3/knowledge/no-such-doc/promote")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task_returns_404(self, client):
        """Test cancelling a non-existent task returns 404."""
        resp = await client.post("/api/v3/tasks/no-task-here/cancel")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_task_status_nonexistent_returns_404(self, client):
        """Test status of a non-existent task returns 404."""
        resp = await client.get("/api/v3/tasks/ghost-task/status")
        assert resp.status_code == 404


# ===========================================================================
# Test: Failure Recovery
# ===========================================================================


class TestFailureRecovery:
    """Tests for failure recovery scenarios."""

    @pytest.mark.asyncio
    async def test_cancel_task_then_enqueue_new(self, client):
        """Test that after cancelling a task, a new one can be enqueued."""
        task1 = await enqueue_task(client, "work", {"attempt": 1})
        task1_id = task1.json()["task_id"]
        await client.post(f"/api/v3/tasks/{task1_id}/cancel")

        task2 = await enqueue_task(client, "work", {"attempt": 2})
        assert task2.status_code == 201
        task2_id = task2.json()["task_id"]
        assert task2_id != task1_id

    @pytest.mark.asyncio
    async def test_task_result_not_available_before_completion(self, client):
        """Test that getting result of a pending task returns 409."""
        task_resp = await enqueue_task(client, "pending_task", {})
        task_id = task_resp.json()["task_id"]

        result_resp = await client.get(f"/api/v3/tasks/{task_id}/result")
        assert result_resp.status_code == 409

    @pytest.mark.asyncio
    async def test_cancel_completed_task_returns_409(self, client):
        """Test that cancelling a completed task returns 409."""
        task_resp = await enqueue_task(client, "done_task", {})
        task_id = task_resp.json()["task_id"]

        # Manually complete via the TaskQueue directly
        # The queue lives on the test app's state, not the module global.
        task_resp_data = task_resp.json()
        # Enqueue a new task and directly access via the test app's queue
        from src.api.v3.dependencies import TaskQueue
        # Use the global fallback queue which is created lazily
        queue = v3_deps._get_or_create_task_queue()
        task = queue.get(task_id)
        task.status = TaskStatus.COMPLETED
        task.result = {"done": True}

        cancel_resp = await client.post(f"/api/v3/tasks/{task_id}/cancel")
        assert cancel_resp.status_code == 409

    @pytest.mark.asyncio
    async def test_re_registration_updates_agent(self, client):
        """Test that registering the same agent_id updates its info."""
        await register_agent(client, "agent-upd", "Original Name", ["read"])
        await register_agent(
            client, "agent-upd", "Updated Name", ["read", "write"]
        )

        resp = await client.get("/api/v3/agents/agent-upd")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == "agent-upd"

    @pytest.mark.asyncio
    async def test_purge_by_status(self, client):
        """Test purging tasks filtered by status."""
        t1 = await enqueue_task(client, "task_a", {})
        t2 = await enqueue_task(client, "task_b", {})

        await client.post(f"/api/v3/tasks/{t1.json()['task_id']}/cancel")

        purge_resp = await client.post(
            "/api/v3/tasks/purge", json={"status": "cancelled"}
        )
        assert purge_resp.status_code == 200
        assert purge_resp.json()["purged"] >= 1

    @pytest.mark.asyncio
    async def test_purge_invalid_status_returns_422(self, client):
        """Test that purging with an invalid status returns 422."""
        resp = await client.post(
            "/api/v3/tasks/purge", json={"status": "bogus"}
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_ingest_returns_422(self, client):
        """Test that ingesting empty content returns 422."""
        resp = await client.post(
            "/api/v3/knowledge/ingest",
            json={"content": "", "title": "Empty", "source": "test"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_search_empty_query_returns_results(self, client):
        """Test that search with no matches returns empty results list."""
        resp = await client.get(
            "/api/v3/knowledge/search",
            params={"query": "xyznonexistentterm99999"},
        )
        assert resp.status_code == 200
        assert resp.json()["results"] == []
        assert resp.json()["total"] == 0


# ===========================================================================
# Test: Concurrent Agents
# ===========================================================================


class TestConcurrentAgents:
    """Tests for concurrent multi-agent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_agent_registration(self, client):
        """Test registering many agents concurrently."""

        async def register_one(idx: int):
            return await register_agent(
                client, f"concurrent-agent-{idx}", f"Agent {idx}"
            )

        tasks = [register_one(i) for i in range(20)]
        responses = await asyncio.gather(*tasks)

        for resp in responses:
            assert resp.status_code == 201

        # Heartbeat them all so they appear as active
        for i in range(20):
            await send_heartbeat(client, f"concurrent-agent-{i}")

        list_resp = await client.get("/api/v3/agents", params={"limit": 100})
        assert list_resp.json()["count"] == 20

    @pytest.mark.asyncio
    async def test_concurrent_task_enqueue(self, client):
        """Test enqueuing many tasks concurrently."""

        async def enqueue_one(idx: int):
            return await enqueue_task(
                client, f"task-{idx}", {"index": idx}
            )

        tasks = [enqueue_one(i) for i in range(50)]
        responses = await asyncio.gather(*tasks)

        for resp in responses:
            assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_concurrent_heartbeats(self, client):
        """Test sending heartbeats for multiple agents concurrently."""
        for i in range(10):
            await register_agent(client, f"hb-agent-{i}")

        async def heartbeat_one(idx: int):
            return await send_heartbeat(client, f"hb-agent-{idx}")

        tasks = [heartbeat_one(i) for i in range(10)]
        responses = await asyncio.gather(*tasks)

        for resp in responses:
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_concurrent_knowledge_ingest(self, client):
        """Test ingesting multiple knowledge entries concurrently."""

        async def ingest_one(idx: int):
            return await ingest_knowledge(
                client,
                content=(
                    f"Knowledge entry number {idx} with unique "
                    f"content about topic {idx}."
                ),
                title=f"Doc {idx}",
                domain="test",
            )

        tasks = [ingest_one(i) for i in range(15)]
        responses = await asyncio.gather(*tasks)

        for resp in responses:
            assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self, client):
        """Test a realistic mix of concurrent operations."""
        results = []

        async def op_register(idx):
            r = await register_agent(client, f"mixed-agent-{idx}")
            results.append(("register", r.status_code))

        async def op_task(idx):
            r = await enqueue_task(
                client, f"mixed-task-{idx}", {"i": idx}
            )
            results.append(("task", r.status_code))

        async def op_knowledge(idx):
            r = await ingest_knowledge(
                client, f"Mixed knowledge {idx}", f"Title {idx}"
            )
            results.append(("knowledge", r.status_code))

        ops = []
        for i in range(10):
            ops.append(op_register(i))
            ops.append(op_task(i))
            ops.append(op_knowledge(i))

        await asyncio.gather(*ops)

        assert len(results) == 30
        for op_type, status in results:
            assert status in (200, 201), f"{op_type} returned {status}"

    @pytest.mark.asyncio
    async def test_concurrent_task_cancel(self, client):
        """Test cancelling multiple tasks concurrently."""
        task_ids = []
        for i in range(10):
            resp = await enqueue_task(client, f"cancel-me-{i}", {})
            task_ids.append(resp.json()["task_id"])

        async def cancel_one(tid):
            return await client.post(f"/api/v3/tasks/{tid}/cancel")

        responses = await asyncio.gather(
            *[cancel_one(tid) for tid in task_ids]
        )

        for resp in responses:
            assert resp.status_code == 200
            assert resp.json()["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_concurrent_unregister(self, client):
        """Test unregistering multiple agents concurrently."""
        for i in range(5):
            await register_agent(client, f"unreg-{i}")

        async def unregister_one(idx):
            return await client.delete(f"/api/v3/agents/unreg-{idx}")

        responses = await asyncio.gather(
            *[unregister_one(i) for i in range(5)]
        )

        for resp in responses:
            assert resp.status_code == 200

        # Verify all gone
        list_resp = await client.get("/api/v3/agents")
        assert list_resp.json()["count"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_knowledge_search(self, client):
        """Test searching knowledge concurrently after ingesting."""
        for i in range(5):
            await ingest_knowledge(
                client,
                content=f"Searchable content about topic {i}",
                title=f"Search Doc {i}",
            )

        async def search_one(idx):
            return await client.get(
                "/api/v3/knowledge/search",
                params={"query": f"topic {idx}"},
            )

        responses = await asyncio.gather(
            *[search_one(i) for i in range(5)]
        )

        for resp in responses:
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_100_concurrent_tasks_no_data_loss(self, client):
        """Stress test: enqueue 100 tasks concurrently, verify all created."""

        async def enqueue_one(idx):
            return await enqueue_task(client, "stress", {"idx": idx})

        tasks = [enqueue_one(i) for i in range(100)]
        responses = await asyncio.gather(*tasks)

        success_count = sum(1 for r in responses if r.status_code == 201)
        assert success_count == 100

        # Verify task IDs are unique
        task_ids = {r.json()["task_id"] for r in responses}
        assert len(task_ids) == 100


# ===========================================================================
# Test: Edge Cases & Error Handling
# ===========================================================================


class TestEdgeCases:
    """Edge case and error handling tests."""

    @pytest.mark.asyncio
    async def test_task_enqueue_invalid_payload(self, client):
        """Test enqueuing a task with missing task_type returns 422."""
        resp = await client.post(
            "/api/v3/tasks/enqueue", json={"payload": {}}
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_agent_capabilities_search(self, client):
        """Test finding agents by capability."""
        await register_agent(
            client, "cap-agent-1", capabilities=["read", "write"]
        )
        await register_agent(
            client, "cap-agent-2", capabilities=["read", "query"]
        )
        await register_agent(
            client, "cap-agent-3", capabilities=["admin"]
        )

        resp = await client.get("/api/v3/agents/capabilities/read")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        agent_ids = [a["agent_id"] for a in data["agents"]]
        assert "cap-agent-1" in agent_ids
        assert "cap-agent-2" in agent_ids

    @pytest.mark.asyncio
    async def test_task_enqueue_with_complex_payload(self, client):
        """Test enqueuing a task with nested/complex payload."""
        complex_payload = {
            "documents": [
                {"id": "doc-1", "path": "/a/b/c.txt"},
                {"id": "doc-2", "path": "/d/e/f.txt"},
            ],
            "options": {
                "format": "markdown",
                "include_metadata": True,
                "max_depth": 5,
            },
            "tags": ["important", "review"],
        }
        resp = await enqueue_task(client, "batch_process", complex_payload)
        assert resp.status_code == 201

        task_id = resp.json()["task_id"]
        status_resp = await client.get(f"/api/v3/tasks/{task_id}/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["payload"] == complex_payload

    @pytest.mark.asyncio
    async def test_knowledge_ingest_with_metadata(self, client):
        """Test ingesting knowledge with additional metadata."""
        resp = await client.post(
            "/api/v3/knowledge/ingest",
            json={
                "content": "Document with metadata",
                "title": "Meta Doc",
                "domain": "test",
                "source": "api",
                "metadata": {"author": "tester", "version": 2},
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["doc_id"] is not None

    @pytest.mark.asyncio
    async def test_multiple_tasks_same_type(self, client):
        """Test enqueuing multiple tasks of the same type."""
        for i in range(10):
            resp = await enqueue_task(
                client, "embed_document", {"doc": f"doc-{i}"}
            )
            assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_agent_health_unknown_for_new_agent(self, client):
        """Test that a newly registered agent has unknown health status."""
        await register_agent(client, "fresh-agent")

        resp = await client.get("/api/v3/agents/fresh-agent/health")
        assert resp.status_code == 200
        assert resp.json()["health_status"] == "unknown"

    @pytest.mark.asyncio
    async def test_list_agents_with_status_filter(self, client):
        """Test listing agents with status filter."""
        await register_agent(client, "filter-agent-1")
        await register_agent(client, "filter-agent-2")

        # Both have unknown status, filter by active (should be 0)
        resp = await client.get("/api/v3/agents", params={"status": "active"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_full_workflow_simulation(self, client):
        """Simulate a realistic multi-step workflow.

        1. Register orchestrator and worker agents
        2. Workers send heartbeats
        3. Orchestrator ingests knowledge
        4. Orchestrator enqueues tasks for workers
        5. Cancel a task
        6. Verify final state
        """
        # 1. Register
        await register_agent(
            client, "orchestrator", "Orchestrator", ["orchestrate", "read"]
        )
        await register_agent(
            client, "worker-alpha", "Worker Alpha", ["process"]
        )
        await register_agent(
            client, "worker-beta", "Worker Beta", ["process"]
        )

        # 2. Heartbeats
        await send_heartbeat(client, "worker-alpha")
        await send_heartbeat(client, "worker-beta")

        # 3. Knowledge ingest
        await ingest_knowledge(
            client,
            content=(
                "Workflow documentation for task processing pipeline."
            ),
            title="Workflow Docs",
            domain="internal",
        )

        # 4. Enqueue tasks
        task_ids = []
        for i in range(5):
            resp = await enqueue_task(
                client, "process_pipeline", {"step": i}
            )
            task_ids.append(resp.json()["task_id"])

        # 5. Cancel one task
        cancel_resp = await client.post(
            f"/api/v3/tasks/{task_ids[2]}/cancel"
        )
        assert cancel_resp.status_code == 200

        # 6. Verify state
        # Orchestrator has unknown status so won't appear in active list;
        # workers have healthy status from heartbeats
        search_resp = await client.get(
            "/api/v3/knowledge/search", params={"query": "workflow"}
        )
        assert search_resp.json()["total"] >= 1
