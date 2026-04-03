"""Tests for AgentCommunication module (gdrag v3).

Tests inter-agent communication patterns:
- Request-Response: context requests between agents
- Broadcast: one-to-many context sharing
- Handoff: task delegation between agents
- Collaborative: multi-agent collaboration

Uses pytest-asyncio and unittest.mock for isolated testing.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call
from uuid import uuid4

import pytest
import pytest_asyncio

from src.core.agent_comm import (
    AgentCommunication,
    InMemoryEventBus,
    SimpleTenantManager,
    EventBus,
    TenantManager,
)
from src.models.communication import (
    CollaborationRequest,
    CollaborationResponse,
    ContextRequest,
    ContextResponse,
    SharedContext,
)
from src.models.tasks import Task, TaskHandoff, TaskStatus, TaskPriority


# ============================================================================
# Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def event_bus():
    """Provide a fresh InMemoryEventBus per test."""
    bus = InMemoryEventBus()
    yield bus


@pytest_asyncio.fixture
async def tenant_manager():
    """Provide a SimpleTenantManager with registered agents."""
    tm = SimpleTenantManager(default_tenant="tenant-1")
    tm.register_agent("agent-alpha", "tenant-1")
    tm.register_agent("agent-beta", "tenant-1")
    tm.register_agent("agent-gamma", "tenant-2")
    yield tm


@pytest_asyncio.fixture
async def comm(event_bus, tenant_manager):
    """Provide an AgentCommunication instance (NOT initialized)."""
    agent_comm = AgentCommunication(
        event_bus=event_bus,
        tenant_manager=tenant_manager,
    )
    yield agent_comm


@pytest.fixture
def sample_task() -> Task:
    """Create a sample task for handoff testing."""
    return Task(
        task_id="task-001",
        title="Analyze dataset",
        description="Run analysis on the training dataset",
        status=TaskStatus.PENDING,
        priority=TaskPriority.MEDIUM,
        assigned_to="agent-alpha",
        created_by="agent-alpha",
        context="Dataset contains 10k samples",
        metadata={"handoff_reason": "needs ML expertise"},
    )


@pytest.fixture
def sample_shared_context() -> SharedContext:
    """Create a sample shared context."""
    return SharedContext(
        from_agent="agent-alpha",
        to_agent="agent-beta",
        content="Key findings from the analysis pipeline",
        domain="data-science",
        tags=["analysis", "findings"],
    )


def _make_context_response_dict(request_id, from_agent, to_agent, context, relevance=1.0):
    """Build a serializable dict matching ContextResponse schema."""
    resp = ContextResponse(
        request_id=request_id,
        from_agent=from_agent,
        to_agent=to_agent,
        context=context,
        relevance_score=relevance,
        token_count=int(len(context.split()) * 1.3),
        source="test",
    )
    return resp.model_dump()


# ============================================================================
# Test: Initialization and Lifecycle
# ============================================================================


class TestAgentCommLifecycle:
    """Tests for AgentCommunication initialization and lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize_subscribes_events(self, comm):
        """Test that initialize registers event subscriptions."""
        await comm.initialize()
        assert "context.request" in comm._subscriptions
        assert "context.share" in comm._subscriptions
        assert "task.handoff" in comm._subscriptions

    @pytest.mark.asyncio
    async def test_shutdown_clears_subscriptions(self, comm):
        """Test that shutdown removes all subscriptions."""
        await comm.initialize()
        assert len(comm._subscriptions) == 3
        await comm.shutdown()
        assert len(comm._subscriptions) == 0

    @pytest.mark.asyncio
    async def test_stats_initial_state(self, comm):
        """Test that stats reflect empty initial state."""
        stats = await comm.get_stats()
        assert stats["shared_contexts"] == 0
        assert stats["active_tasks"] == 0
        assert stats["total_tasks"] == 0
        assert stats["handoff_history_size"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_operations(self, comm, sample_task, sample_shared_context):
        """Test that stats are updated after operations."""
        await comm.share_context(
            from_agent="agent-alpha", to_agent="agent-beta",
            context=sample_shared_context,
        )
        await comm.register_task(sample_task)
        stats = await comm.get_stats()
        assert stats["shared_contexts"] == 1
        assert stats["total_tasks"] == 1
        assert stats["active_tasks"] == 1


# ============================================================================
# Test: Request-Response Pattern
# ============================================================================


class TestContextRequests:
    """Tests for request-response context sharing.

    We mock event_bus.request() because InMemoryEventBus.request() generates
    an internal correlation_id that doesn't match the request_id used in
    ContextResponse emission.
    """

    @pytest.mark.asyncio
    async def test_request_context_returns_response(self, comm):
        """Test successful context request between agents."""
        async def mock_request(event_type, payload, target, timeout=30.0):
            req = ContextRequest(**payload)
            return _make_context_response_dict(
                request_id=req.request_id,
                from_agent=target,
                to_agent=payload["from_agent"],
                context="Python optimization tips for async code",
            )
        comm.event_bus.request = mock_request

        response = await comm.request_context(
            from_agent="agent-alpha", to_agent="agent-beta",
            query="optimization", max_tokens=1000,
        )

        assert isinstance(response, ContextResponse)
        assert response.from_agent == "agent-beta"
        assert response.to_agent == "agent-alpha"
        assert "optimization" in response.context.lower()
        assert response.relevance_score > 0

    @pytest.mark.asyncio
    async def test_request_context_permission_denied(self, comm):
        """Test that cross-tenant access is blocked."""
        comm.tenant_manager._allow_cross_tenant = False
        with pytest.raises(PermissionError, match="Cross-tenant access denied"):
            await comm.request_context(
                from_agent="agent-alpha", to_agent="agent-gamma", query="some data",
            )

    @pytest.mark.asyncio
    async def test_request_context_timeout_returns_empty(self, comm):
        """Test that timeout yields an empty ContextResponse."""
        comm.event_bus.request = AsyncMock(return_value=None)

        response = await comm.request_context(
            from_agent="agent-alpha", to_agent="agent-beta", query="anything",
        )

        assert response.context == ""
        assert response.relevance_score == 0.0
        assert response.source == "timeout"

    @pytest.mark.asyncio
    async def test_request_context_sends_correct_payload(self, comm):
        """Test that request_context sends valid ContextRequest to event_bus."""
        captured = []

        async def mock_request(event_type, payload, target, timeout):
            captured.append((event_type, payload, target))
            req = ContextRequest(**payload)
            return _make_context_response_dict(
                request_id=req.request_id, from_agent=target,
                to_agent=payload["from_agent"], context="ok",
            )
        comm.event_bus.request = mock_request

        await comm.request_context(
            from_agent="agent-alpha", to_agent="agent-beta", query="test query",
        )

        assert len(captured) == 1
        evt_type, payload, target = captured[0]
        assert evt_type == "context.request"
        assert target == "agent-beta"
        assert payload["from_agent"] == "agent-alpha"
        assert payload["to_agent"] == "agent-beta"
        assert payload["query"] == "test query"
        assert "request_id" in payload

    @pytest.mark.asyncio
    async def test_request_context_max_tokens_in_payload(self, comm):
        """Test that max_tokens is passed to event_bus.request."""
        captured = []

        async def mock_request(event_type, payload, target, timeout):
            captured.append(payload)
            req = ContextRequest(**payload)
            return _make_context_response_dict(
                request_id=req.request_id, from_agent=target,
                to_agent=payload["from_agent"], context="word " * 50,
            )
        comm.event_bus.request = mock_request

        await comm.request_context(
            from_agent="agent-alpha", to_agent="agent-beta",
            query="word", max_tokens=42,
        )

        assert captured[0]["max_tokens"] == 42

    @pytest.mark.asyncio
    async def test_multiple_context_requests_sequential(self, comm):
        """Test multiple sequential context requests."""
        async def mock_request(event_type, payload, target, timeout):
            req = ContextRequest(**payload)
            ctx = "First context about testing" if "testing" in req.query \
                else "Second context about deployment"
            return _make_context_response_dict(
                request_id=req.request_id, from_agent=target,
                to_agent=payload["from_agent"], context=ctx,
            )
        comm.event_bus.request = mock_request

        resp1 = await comm.request_context(
            from_agent="agent-alpha", to_agent="agent-beta", query="testing",
        )
        resp2 = await comm.request_context(
            from_agent="agent-alpha", to_agent="agent-beta", query="deployment",
        )

        assert "testing" in resp1.context.lower()
        assert "deployment" in resp2.context.lower()


# ============================================================================
# Test: Broadcast Pattern
# ============================================================================


class TestBroadcastPattern:
    """Tests for broadcast context sharing."""

    @pytest.mark.asyncio
    async def test_share_context_stored_locally(self, comm, sample_shared_context):
        """Test that shared context is stored in local memory."""
        await comm.share_context(
            from_agent="agent-alpha", to_agent="agent-beta",
            context=sample_shared_context,
        )
        contexts = comm._shared_contexts.get("agent-alpha", [])
        assert len(contexts) == 1
        assert contexts[0].content == sample_shared_context.content

    @pytest.mark.asyncio
    async def test_share_context_permission_denied(self, comm):
        """Test that cross-tenant share is blocked when denied."""
        comm.tenant_manager._allow_cross_tenant = False
        ctx = SharedContext(from_agent="agent-alpha", content="secret")
        with pytest.raises(PermissionError, match="Cross-tenant access denied"):
            await comm.share_context(
                from_agent="agent-alpha", to_agent="agent-gamma", context=ctx,
            )

    @pytest.mark.asyncio
    async def test_share_context_emits_event(self, comm, event_bus):
        """Test that share_context emits a context.share event via event_bus."""
        emitted = []
        original_emit = event_bus.emit

        async def tracking_emit(event_type, payload, target=None, correlation_id=None):
            emitted.append({"event_type": event_type, "payload": payload, "target": target})
            return await original_emit(event_type, payload, target, correlation_id)

        event_bus.emit = tracking_emit

        ctx = SharedContext(
            from_agent="agent-alpha", to_agent="agent-beta",
            content="event test content",
        )
        await comm.share_context(
            from_agent="agent-alpha", to_agent="agent-beta", context=ctx,
        )

        share_events = [e for e in emitted if e["event_type"] == "context.share"]
        assert len(share_events) >= 1
        assert share_events[0]["target"] == "agent-beta"
        assert share_events[0]["payload"]["content"] == "event test content"

    @pytest.mark.asyncio
    async def test_broadcast_context_to_tenant_agents(self, comm, event_bus, tenant_manager):
        """Test broadcasting context to all agents in the same tenant."""
        emitted = []
        original_emit = event_bus.emit

        async def tracking_emit(event_type, payload, target=None, correlation_id=None):
            emitted.append({"event_type": event_type, "payload": payload, "target": target})
            return await original_emit(event_type, payload, target, correlation_id)

        event_bus.emit = tracking_emit

        ctx = SharedContext(
            from_agent="agent-alpha", content="Announcement for team", tags=["team"],
        )
        count = await comm.broadcast_context(
            from_agent="agent-alpha", context=ctx, tenant_only=True,
        )

        share_events = [e for e in emitted if e["event_type"] == "context.share"]
        assert count == 1
        assert len(share_events) == 1
        assert share_events[0]["target"] == "agent-beta"
        assert share_events[0]["payload"]["content"] == "Announcement for team"

    @pytest.mark.asyncio
    async def test_broadcast_excludes_sender(self, comm, event_bus, tenant_manager):
        """Test that broadcast does not send to the sender."""
        tenant_manager.register_agent("agent-alpha", "tenant-1")
        emitted = []
        original_emit = event_bus.emit

        async def tracking_emit(event_type, payload, target=None, correlation_id=None):
            emitted.append({"event_type": event_type, "target": target})
            return await original_emit(event_type, payload, target, correlation_id)

        event_bus.emit = tracking_emit

        ctx = SharedContext(from_agent="agent-alpha", content="test broadcast")
        await comm.broadcast_context(from_agent="agent-alpha", context=ctx, tenant_only=True)

        share_targets = [e["target"] for e in emitted if e["event_type"] == "context.share"]
        assert "agent-alpha" not in share_targets

    @pytest.mark.asyncio
    async def test_broadcast_global_returns_negative_one(self, comm, event_bus):
        """Test global broadcast returns -1 for unknown count."""
        ctx = SharedContext(from_agent="agent-alpha", content="global message")
        count = await comm.broadcast_context(
            from_agent="agent-alpha", context=ctx, tenant_only=False,
        )
        assert count == -1

    @pytest.mark.asyncio
    async def test_share_context_broadcast_to_none(self, comm, event_bus):
        """Test sharing context with to_agent=None acts as broadcast."""
        await event_bus.subscribe("context.share", lambda p: None)
        ctx = SharedContext(from_agent="agent-alpha", content="broadcast content")
        await comm.share_context(from_agent="agent-alpha", to_agent=None, context=ctx)
        assert "agent-alpha" in comm._shared_contexts

    @pytest.mark.asyncio
    async def test_broadcast_no_tenant_agents(self, comm, event_bus, tenant_manager):
        """Test broadcast when sender has no tenant agents in their tenant."""
        # agent-new gets default tenant with no other agents registered
        tm_isolated = SimpleTenantManager(default_tenant="isolated")
        comm.tenant_manager = tm_isolated

        ctx = SharedContext(from_agent="agent-new", content="lonely broadcast")
        count = await comm.broadcast_context(
            from_agent="agent-new", context=ctx, tenant_only=True,
        )
        assert count == 0


# ============================================================================
# Test: Handoff Pattern
# ============================================================================


class TestTaskHandoff:
    """Tests for task handoff between agents."""

    @pytest.mark.asyncio
    async def test_handoff_task_transfers_ownership(self, comm, sample_task):
        """Test that handoff updates task assigned_to and status."""
        handoff_id = await comm.handoff_task(
            from_agent="agent-alpha", to_agent="agent-beta", task=sample_task,
        )
        assert sample_task.assigned_to == "agent-beta"
        assert sample_task.status == TaskStatus.ASSIGNED
        assert sample_task.metadata["previous_owner"] == "agent-alpha"
        assert sample_task.metadata["handoff_id"] == handoff_id
        assert handoff_id is not None

    @pytest.mark.asyncio
    async def test_handoff_records_history(self, comm, sample_task):
        """Test that handoff is recorded in history."""
        await comm.handoff_task(
            from_agent="agent-alpha", to_agent="agent-beta", task=sample_task,
        )
        history = await comm.get_handoff_history()
        assert len(history) == 1
        assert history[0].from_agent == "agent-alpha"
        assert history[0].to_agent == "agent-beta"
        assert history[0].task_id == sample_task.task_id

    @pytest.mark.asyncio
    async def test_handoff_completed_task_raises(self, comm):
        """Test that handing off a completed task raises ValueError."""
        task = Task(task_id="done", title="Done", status=TaskStatus.COMPLETED)
        with pytest.raises(ValueError, match="Cannot handoff task in status completed"):
            await comm.handoff_task(from_agent="agent-alpha", to_agent="agent-beta", task=task)

    @pytest.mark.asyncio
    async def test_handoff_cancelled_task_raises(self, comm):
        """Test that handing off a cancelled task raises ValueError."""
        task = Task(task_id="cancelled", title="Cancelled", status=TaskStatus.CANCELLED)
        with pytest.raises(ValueError, match="Cannot handoff task in status cancelled"):
            await comm.handoff_task(from_agent="agent-alpha", to_agent="agent-beta", task=task)

    @pytest.mark.asyncio
    async def test_handoff_failed_task_raises(self, comm):
        """Test that handing off a failed task raises ValueError."""
        task = Task(task_id="failed", title="Failed", status=TaskStatus.FAILED)
        with pytest.raises(ValueError, match="Cannot handoff task in status failed"):
            await comm.handoff_task(from_agent="agent-alpha", to_agent="agent-beta", task=task)

    @pytest.mark.asyncio
    async def test_handoff_permission_denied(self, comm):
        """Test that cross-tenant handoff is blocked when denied."""
        comm.tenant_manager._allow_cross_tenant = False
        task = Task(task_id="t1", title="task", status=TaskStatus.PENDING)
        with pytest.raises(PermissionError, match="Cross-tenant access denied"):
            await comm.handoff_task(
                from_agent="agent-alpha", to_agent="agent-gamma", task=task,
            )

    @pytest.mark.asyncio
    async def test_handoff_emits_event(self, comm, sample_task, event_bus):
        """Test that handoff emits a task.handoff event via event_bus."""
        emitted = []
        original_emit = event_bus.emit

        async def tracking_emit(event_type, payload, target=None, correlation_id=None):
            emitted.append({"event_type": event_type, "payload": payload, "target": target})
            return await original_emit(event_type, payload, target, correlation_id)

        event_bus.emit = tracking_emit

        await comm.handoff_task(
            from_agent="agent-alpha", to_agent="agent-beta", task=sample_task,
        )

        handoff_events = [e for e in emitted if e["event_type"] == "task.handoff"]
        assert len(handoff_events) >= 1
        payload = handoff_events[0]["payload"]
        assert payload["handoff"]["from_agent"] == "agent-alpha"
        assert payload["handoff"]["to_agent"] == "agent-beta"
        assert payload["task"]["task_id"] == sample_task.task_id

    @pytest.mark.asyncio
    async def test_handoff_stores_task_in_active(self, comm, sample_task):
        """Test that handed-off task is stored in active tasks."""
        await comm.handoff_task(
            from_agent="agent-alpha", to_agent="agent-beta", task=sample_task,
        )
        stored = await comm.get_task(sample_task.task_id)
        assert stored is not None
        assert stored.assigned_to == "agent-beta"

    @pytest.mark.asyncio
    async def test_multiple_handoffs_chained(self, comm):
        """Test chaining handoffs: alpha -> beta -> alpha."""
        task = Task(
            task_id="chain", title="Chain", status=TaskStatus.PENDING,
            assigned_to="agent-alpha",
        )

        await comm.handoff_task(from_agent="agent-alpha", to_agent="agent-beta", task=task)
        assert task.assigned_to == "agent-beta"

        await comm.handoff_task(from_agent="agent-beta", to_agent="agent-alpha", task=task)
        assert task.assigned_to == "agent-alpha"

        history = await comm.get_handoff_history(task_id=task.task_id)
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_get_handoff_history_filter_by_agent(self, comm):
        """Test filtering handoff history by agent ID."""
        t1 = Task(task_id="t1", title="t1", status=TaskStatus.PENDING)
        t2 = Task(task_id="t2", title="t2", status=TaskStatus.PENDING)

        await comm.handoff_task(from_agent="agent-alpha", to_agent="agent-beta", task=t1)
        await comm.handoff_task(from_agent="agent-beta", to_agent="agent-alpha", task=t2)

        beta_hist = await comm.get_handoff_history(agent_id="agent-beta")
        assert len(beta_hist) == 2
        alpha_hist = await comm.get_handoff_history(agent_id="agent-alpha")
        assert len(alpha_hist) == 2

    @pytest.mark.asyncio
    async def test_get_handoff_history_limit(self, comm):
        """Test handoff history respects limit parameter."""
        for i in range(10):
            t = Task(task_id=f"t-{i}", title=f"t-{i}", status=TaskStatus.PENDING)
            await comm.handoff_task(from_agent="agent-alpha", to_agent="agent-beta", task=t)
        limited = await comm.get_handoff_history(limit=3)
        assert len(limited) == 3


# ============================================================================
# Test: Shared Context Retrieval
# ============================================================================


class TestSharedContextRetrieval:
    """Tests for get_shared_context and context lifecycle."""

    @pytest.mark.asyncio
    async def test_get_shared_context_returns_matching(self, comm):
        """Test that get_shared_context returns relevant contexts."""
        ctx = SharedContext(
            from_agent="agent-beta", to_agent="agent-alpha",
            content="Machine learning best practices", tags=["ml"],
        )
        await comm.share_context(from_agent="agent-beta", to_agent="agent-alpha", context=ctx)
        results = await comm.get_shared_context("agent-alpha", "machine learning")
        assert len(results) >= 1
        assert any("machine learning" in r.content.lower() for r in results)

    @pytest.mark.asyncio
    async def test_get_shared_context_filters_expired(self, comm):
        """Test that expired contexts are excluded."""
        ctx = SharedContext(
            from_agent="agent-beta", to_agent="agent-alpha",
            content="This will expire",
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        await comm.share_context(from_agent="agent-beta", to_agent="agent-alpha", context=ctx)
        results = await comm.get_shared_context("agent-alpha", "expire")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_shared_context_no_match(self, comm):
        """Test that non-matching queries return empty list."""
        ctx = SharedContext(
            from_agent="agent-beta", to_agent="agent-alpha",
            content="Python optimization tips", tags=["python"],
        )
        await comm.share_context(from_agent="agent-beta", to_agent="agent-alpha", context=ctx)
        results = await comm.get_shared_context("agent-alpha", "quantum physics")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_shared_context_by_tags(self, comm):
        """Test context matching via tags."""
        ctx = SharedContext(
            from_agent="agent-beta", to_agent="agent-alpha",
            content="Some content", tags=["deployment", "kubernetes"],
        )
        await comm.share_context(from_agent="agent-beta", to_agent="agent-alpha", context=ctx)
        results = await comm.get_shared_context("agent-alpha", "kubernetes")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_get_context_by_id(self, comm):
        """Test retrieving context by its ID."""
        ctx = SharedContext(from_agent="agent-alpha", content="Find me by ID")
        await comm.share_context(from_agent="agent-alpha", to_agent="agent-beta", context=ctx)
        found = await comm.get_context_by_id(ctx.context_id)
        assert found is not None
        assert found.content == "Find me by ID"

    @pytest.mark.asyncio
    async def test_get_context_by_id_not_found(self, comm):
        """Test that nonexistent context ID returns None."""
        found = await comm.get_context_by_id("nonexistent-id")
        assert found is None

    @pytest.mark.asyncio
    async def test_cleanup_expired_contexts(self, comm):
        """Test cleanup removes expired contexts."""
        expired = SharedContext(
            from_agent="agent-alpha", content="expired",
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        valid = SharedContext(
            from_agent="agent-alpha", content="still valid",
            expires_at=datetime.utcnow() + timedelta(hours=1),
        )
        await comm.share_context(from_agent="agent-alpha", to_agent="agent-beta", context=expired)
        await comm.share_context(from_agent="agent-alpha", to_agent="agent-beta", context=valid)
        removed = await comm.cleanup_expired_contexts()
        assert removed >= 1
        remaining = comm._shared_contexts.get("agent-alpha", [])
        assert any(c.content == "still valid" for c in remaining)

    @pytest.mark.asyncio
    async def test_shared_context_sorted_by_recency(self, comm):
        """Test that results are sorted most recent first."""
        old = SharedContext(from_agent="agent-beta", content="Old analysis result", tags=["analysis"])
        await comm.share_context(from_agent="agent-beta", to_agent="agent-alpha", context=old)
        await asyncio.sleep(0.02)
        new = SharedContext(from_agent="agent-beta", content="New analysis result", tags=["analysis"])
        await comm.share_context(from_agent="agent-beta", to_agent="agent-alpha", context=new)
        results = await comm.get_shared_context("agent-alpha", "analysis")
        assert len(results) >= 2
        assert results[0].created_at >= results[1].created_at


# ============================================================================
# Test: Task Management Helpers
# ============================================================================


class TestTaskManagement:
    """Tests for task registration, status updates, and retrieval."""

    @pytest.mark.asyncio
    async def test_register_and_get_task(self, comm, sample_task):
        """Test registering and retrieving a task."""
        await comm.register_task(sample_task)
        retrieved = await comm.get_task(sample_task.task_id)
        assert retrieved is not None
        assert retrieved.title == sample_task.title

    @pytest.mark.asyncio
    async def test_get_task_not_found(self, comm):
        """Test that nonexistent task returns None."""
        assert await comm.get_task("nope") is None

    @pytest.mark.asyncio
    async def test_update_task_status(self, comm, sample_task):
        """Test updating task status."""
        await comm.register_task(sample_task)
        updated = await comm.update_task_status(
            sample_task.task_id, TaskStatus.COMPLETED, result="Done",
        )
        assert updated is True
        task = await comm.get_task(sample_task.task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "Done"
        assert task.completed_at is not None

    @pytest.mark.asyncio
    async def test_update_task_status_not_found(self, comm):
        """Test updating nonexistent task returns False."""
        assert await comm.update_task_status("nonexistent", TaskStatus.COMPLETED) is False

    @pytest.mark.asyncio
    async def test_update_task_status_with_error(self, comm, sample_task):
        """Test updating task with error message."""
        await comm.register_task(sample_task)
        await comm.update_task_status(
            sample_task.task_id, TaskStatus.FAILED, error="OOM",
        )
        task = await comm.get_task(sample_task.task_id)
        assert task.status == TaskStatus.FAILED
        assert task.error == "OOM"

    @pytest.mark.asyncio
    async def test_get_agent_tasks(self, comm):
        """Test retrieving tasks assigned to an agent."""
        t1 = Task(task_id="t1", title="T1", assigned_to="agent-alpha", status=TaskStatus.PENDING)
        t2 = Task(task_id="t2", title="T2", assigned_to="agent-alpha", status=TaskStatus.COMPLETED)
        t3 = Task(task_id="t3", title="T3", assigned_to="agent-beta", status=TaskStatus.PENDING)
        for t in [t1, t2, t3]:
            await comm.register_task(t)
        alpha_tasks = await comm.get_agent_tasks("agent-alpha")
        assert len(alpha_tasks) == 2
        assert all(t.assigned_to == "agent-alpha" for t in alpha_tasks)

    @pytest.mark.asyncio
    async def test_get_agent_tasks_filtered_by_status(self, comm):
        """Test filtering agent tasks by status."""
        t1 = Task(task_id="t1", title="T1", assigned_to="agent-alpha", status=TaskStatus.PENDING)
        t2 = Task(task_id="t2", title="T2", assigned_to="agent-alpha", status=TaskStatus.COMPLETED)
        for t in [t1, t2]:
            await comm.register_task(t)
        pending = await comm.get_agent_tasks("agent-alpha", status=TaskStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].task_id == "t1"

    @pytest.mark.asyncio
    async def test_get_agent_tasks_sorted_by_priority(self, comm):
        """Test that tasks are sorted by priority (critical first)."""
        for p, tid in [(TaskPriority.LOW, "low"), (TaskPriority.CRITICAL, "crit"), (TaskPriority.HIGH, "high")]:
            await comm.register_task(Task(task_id=tid, title=tid, assigned_to="agent-alpha", priority=p))
        tasks = await comm.get_agent_tasks("agent-alpha")
        assert tasks[0].priority == TaskPriority.CRITICAL
        assert tasks[1].priority == TaskPriority.HIGH
        assert tasks[2].priority == TaskPriority.LOW

    @pytest.mark.asyncio
    async def test_get_agent_tasks_limit(self, comm):
        """Test that agent tasks respect limit."""
        for i in range(10):
            await comm.register_task(Task(task_id=f"t-{i}", title=f"T{i}", assigned_to="agent-alpha"))
        assert len(await comm.get_agent_tasks("agent-alpha", limit=3)) == 3


# ============================================================================
# Test: InMemoryEventBus
# ============================================================================


class TestInMemoryEventBus:
    """Tests for the InMemoryEventBus implementation.

    NOTE: InMemoryEventBus.request() generates its own internal correlation_id.
    emit() with a matching correlation_id resolves the pending future directly
    (skipping handlers).
    """

    @pytest.mark.asyncio
    async def test_emit_to_subscribers(self, event_bus):
        """Test that emit notifies matching subscribers."""
        received = []
        await event_bus.subscribe("evt", lambda p: received.append(p))
        await event_bus.emit("evt", {"data": "hello"})
        assert received == [{"data": "hello"}]

    @pytest.mark.asyncio
    async def test_emit_with_target(self, event_bus):
        """Test targeted emit only reaches matching agent."""
        received = []
        await event_bus.subscribe("evt", lambda p: received.append(p), agent_id="agent-1")
        await event_bus.emit("evt", {"data": "for-1"}, target="agent-1")
        await event_bus.emit("evt", {"data": "for-2"}, target="agent-2")
        assert len(received) == 1
        assert received[0]["data"] == "for-1"

    @pytest.mark.asyncio
    async def test_emit_without_target_notifies_all(self, event_bus):
        """Test emit without target notifies all subscribers."""
        received = []
        await event_bus.subscribe("evt", lambda p: received.append("a"), agent_id="a1")
        await event_bus.subscribe("evt", lambda p: received.append("b"), agent_id="a2")
        await event_bus.emit("evt", {"x": 1})
        assert set(received) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_correlated_emit_resolves_future(self, event_bus):
        """Test emit with correlation_id resolves pending future directly."""
        future = asyncio.get_event_loop().create_future()
        corr_id = "test-corr-123"
        async with event_bus._lock:
            event_bus._pending_requests[corr_id] = future

        await event_bus.emit("resp", {"result": "ok"}, correlation_id=corr_id)
        assert future.done()
        assert future.result() == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_correlated_emit_skips_handlers(self, event_bus):
        """Test correlated emit does NOT call event handlers."""
        called = []
        await event_bus.subscribe("resp", lambda p: called.append(p))

        future = asyncio.get_event_loop().create_future()
        corr_id = "skip-h"
        async with event_bus._lock:
            event_bus._pending_requests[corr_id] = future

        await event_bus.emit("resp", {"x": 1}, correlation_id=corr_id)
        assert future.done()
        assert len(called) == 0


    @pytest.mark.asyncio
    async def test_unsubscribe(self, event_bus):
        """Test unsubscribing removes the handler."""
        received = []
        sub_id = await event_bus.subscribe("evt", lambda p: received.append(p))
        await event_bus.emit("evt", {"p": 1})
        assert len(received) == 1
        await event_bus.unsubscribe(sub_id)
        await event_bus.emit("evt", {"p": 2})
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_returns_false_for_unknown(self, event_bus):
        """Test unsubscribing nonexistent ID returns False."""
        assert await event_bus.unsubscribe("nope") is False

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_event(self, event_bus):
        """Test multiple subscribers all fire for the same event."""
        a, b = [], []
        await event_bus.subscribe("m", lambda p: a.append(p))
        await event_bus.subscribe("m", lambda p: b.append(p))
        await event_bus.emit("m", {"d": 1})
        assert len(a) == 1 and len(b) == 1

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_break_others(self, event_bus):
        """Test handler exception doesn't prevent other handlers from running."""
        received = []
        await event_bus.subscribe("e", lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
        await event_bus.subscribe("e", lambda p: received.append(p))
        await event_bus.emit("e", {"data": "ok"})
        assert len(received) == 1


# ============================================================================
# Test: SimpleTenantManager
# ============================================================================


class TestSimpleTenantManager:
    """Tests for SimpleTenantManager."""

    @pytest.mark.asyncio
    async def test_register_agent(self):
        tm = SimpleTenantManager()
        tm.register_agent("agent-1", "tenant-a")
        assert await tm.get_tenant_id("agent-1") == "tenant-a"

    @pytest.mark.asyncio
    async def test_default_tenant_for_unregistered(self):
        tm = SimpleTenantManager(default_tenant="default")
        assert await tm.get_tenant_id("unknown") == "default"

    @pytest.mark.asyncio
    async def test_get_tenant_agents(self):
        tm = SimpleTenantManager()
        tm.register_agent("a1", "team-1")
        tm.register_agent("a2", "team-1")
        tm.register_agent("a3", "team-2")
        agents = await tm.get_tenant_agents("team-1")
        assert set(agents) == {"a1", "a2"}

    @pytest.mark.asyncio
    async def test_get_tenant_agents_empty(self):
        tm = SimpleTenantManager()
        assert await tm.get_tenant_agents("nonexistent") == []

    @pytest.mark.asyncio
    async def test_validate_access_always_true(self):
        tm = SimpleTenantManager()
        assert await tm.validate_access("a1", "r1", "read") is True

    @pytest.mark.asyncio
    async def test_validate_cross_tenant_allowed(self):
        tm = SimpleTenantManager()
        assert await tm.validate_cross_tenant("a1", "a2") is True

    @pytest.mark.asyncio
    async def test_validate_cross_tenant_blocked(self):
        tm = SimpleTenantManager()
        tm._allow_cross_tenant = False
        assert await tm.validate_cross_tenant("a1", "a2") is False
