"""Agent Communication module for gdrag v3.

Provides inter-agent communication with support for multiple patterns:
- Request-Response: Direct context requests between agents
- Broadcast: One-to-many context sharing
- Handoff: Task delegation between agents
- Collaborative: Multi-agent collaborative work

Integrates with EventBus for message routing and TenantManager for multi-tenancy.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from ..models.communication import (
    CollaborationRequest,
    CollaborationResponse,
    CommunicationMessage,
    CommunicationPattern,
    ContextRequest,
    ContextResponse,
    MessageType,
    SharedContext,
)
from ..models.tasks import Task, TaskHandoff, TaskStatus

logger = logging.getLogger(__name__)


# ============================================================================
# Abstract Interfaces
# ============================================================================

class EventBus(ABC):
    """Abstract event bus for message routing between agents.

    Implementations should support publish/subscribe pattern with
    topic-based routing and correlation IDs for request-response.
    """

    @abstractmethod
    async def emit(
        self,
        event_type: str,
        payload: Dict[str, Any],
        target: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Emit an event to the bus.

        Args:
            event_type: Type of event (e.g., 'context.request').
            payload: Event data.
            target: Optional target agent ID for directed messages.
            correlation_id: Optional correlation ID for request-response.
        """
        ...

    @abstractmethod
    async def subscribe(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], Any],
        agent_id: Optional[str] = None,
    ) -> str:
        """Subscribe to events.

        Args:
            event_type: Event type pattern to subscribe to.
            handler: Async callable to handle events.
            agent_id: Optional agent ID for targeted subscriptions.

        Returns:
            Subscription ID.
        """
        ...

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events.

        Args:
            subscription_id: Subscription ID returned from subscribe().

        Returns:
            True if successfully unsubscribed.
        """
        ...

    @abstractmethod
    async def request(
        self,
        event_type: str,
        payload: Dict[str, Any],
        target: str,
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """Send a request and wait for response.

        Args:
            event_type: Event type for the request.
            payload: Request data.
            target: Target agent ID.
            timeout: Response timeout in seconds.

        Returns:
            Response payload or None on timeout.
        """
        ...


class TenantManager(ABC):
    """Abstract tenant manager for multi-tenancy support.

    Handles tenant isolation, access validation, and tenant-scoped operations.
    """

    @abstractmethod
    async def validate_access(
        self,
        agent_id: str,
        resource_id: str,
        action: str,
    ) -> bool:
        """Validate agent access to a resource.

        Args:
            agent_id: Agent requesting access.
            resource_id: Target resource ID.
            action: Action being performed (read, write, etc.).

        Returns:
            True if access is allowed.
        """
        ...

    @abstractmethod
    async def get_tenant_id(self, agent_id: str) -> Optional[str]:
        """Get tenant ID for an agent.

        Args:
            agent_id: Agent ID.

        Returns:
            Tenant ID or None if agent not found.
        """
        ...

    @abstractmethod
    async def get_tenant_agents(self, tenant_id: str) -> List[str]:
        """Get all agent IDs in a tenant.

        Args:
            tenant_id: Tenant ID.

        Returns:
            List of agent IDs in the tenant.
        """
        ...

    @abstractmethod
    async def validate_cross_tenant(
        self,
        from_agent: str,
        to_agent: str,
    ) -> bool:
        """Validate cross-tenant communication.

        Args:
            from_agent: Source agent ID.
            to_agent: Target agent ID.

        Returns:
            True if cross-tenant communication is allowed.
        """
        ...


# ============================================================================
# Default Implementations
# ============================================================================

class InMemoryEventBus(EventBus):
    """In-memory event bus implementation for development/testing."""

    def __init__(self):
        self._subscribers: Dict[str, List[Dict[str, Any]]] = {}
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    async def emit(
        self,
        event_type: str,
        payload: Dict[str, Any],
        target: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Emit an event to all matching subscribers."""
        async with self._lock:
            handlers = self._subscribers.get(event_type, [])

            # Check for correlated response
            if correlation_id and correlation_id in self._pending_requests:
                future = self._pending_requests.pop(correlation_id)
                if not future.done():
                    future.set_result(payload)
                return

            # Notify matching handlers
            for sub in handlers:
                if target is None or sub.get("agent_id") == target:
                    try:
                        if asyncio.iscoroutinefunction(sub["handler"]):
                            await sub["handler"](payload)
                        else:
                            sub["handler"](payload)
                    except Exception as e:
                        logger.error(f"Error in event handler: {e}")

    async def subscribe(
        self,
        event_type: str,
        handler: Callable[[Dict[str, Any]], Any],
        agent_id: Optional[str] = None,
    ) -> str:
        """Subscribe to events of a specific type."""
        sub_id = str(uuid4())
        async with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append({
                "id": sub_id,
                "handler": handler,
                "agent_id": agent_id,
            })
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a subscription by ID."""
        async with self._lock:
            for event_type, subs in self._subscribers.items():
                for i, sub in enumerate(subs):
                    if sub["id"] == subscription_id:
                        subs.pop(i)
                        return True
        return False

    async def request(
        self,
        event_type: str,
        payload: Dict[str, Any],
        target: str,
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """Send request and wait for response."""
        correlation_id = str(uuid4())
        future = asyncio.get_event_loop().create_future()

        async with self._lock:
            self._pending_requests[correlation_id] = future

        # Emit the request
        await self.emit(
            event_type,
            payload,
            target=target,
            correlation_id=correlation_id,
        )

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            async with self._lock:
                self._pending_requests.pop(correlation_id, None)
            return None


class SimpleTenantManager(TenantManager):
    """Simple tenant manager for single-tenant or basic multi-tenant setups."""

    def __init__(self, default_tenant: str = "default"):
        self._default_tenant = default_tenant
        self._agent_tenants: Dict[str, str] = {}
        self._tenant_agents: Dict[str, Set[str]] = {default_tenant: set()}
        self._allow_cross_tenant: bool = True

    async def validate_access(
        self,
        agent_id: str,
        resource_id: str,
        action: str,
    ) -> bool:
        """Always allow access in simple implementation."""
        return True

    async def get_tenant_id(self, agent_id: str) -> Optional[str]:
        """Get tenant for an agent."""
        return self._agent_tenants.get(agent_id, self._default_tenant)

    async def get_tenant_agents(self, tenant_id: str) -> List[str]:
        """Get all agents in a tenant."""
        return list(self._tenant_agents.get(tenant_id, set()))

    async def validate_cross_tenant(
        self,
        from_agent: str,
        to_agent: str,
    ) -> bool:
        """Check if cross-tenant communication is allowed."""
        return self._allow_cross_tenant

    def register_agent(self, agent_id: str, tenant_id: Optional[str] = None) -> None:
        """Register an agent to a tenant."""
        tid = tenant_id or self._default_tenant
        self._agent_tenants[agent_id] = tid
        if tid not in self._tenant_agents:
            self._tenant_agents[tid] = set()
        self._tenant_agents[tid].add(agent_id)


# ============================================================================
# AgentCommunication Class
# ============================================================================

class AgentCommunication:
    """Handles inter-agent communication for gdrag v3.

    Supports four communication patterns:
    1. Request-Response: Agent A requests context from Agent B
    2. Broadcast: Agent shares context with multiple agents
    3. Handoff: Agent delegates a task to another agent
    4. Collaborative: Multiple agents work together on a task

    Args:
        event_bus: Event bus implementation for message routing.
        tenant_manager: Tenant manager for multi-tenancy support.
    """

    def __init__(
        self,
        event_bus: EventBus,
        tenant_manager: TenantManager,
    ):
        self.event_bus = event_bus
        self.tenant_manager = tenant_manager

        # In-memory stores
        self._shared_contexts: Dict[str, List[SharedContext]] = {}
        self._handoff_history: List[TaskHandoff] = []
        self._active_tasks: Dict[str, Task] = {}
        self._subscriptions: Dict[str, str] = {}

        # Lock for thread safety
        self._lock = asyncio.Lock()

        logger.info("AgentCommunication initialized")

    async def initialize(self) -> None:
        """Initialize subscriptions and event handlers."""
        # Subscribe to context requests
        sub_id = await self.event_bus.subscribe(
            "context.request",
            self._handle_context_request,
        )
        self._subscriptions["context.request"] = sub_id

        # Subscribe to context shares
        sub_id = await self.event_bus.subscribe(
            "context.share",
            self._handle_context_share,
        )
        self._subscriptions["context.share"] = sub_id

        # Subscribe to task handoffs
        sub_id = await self.event_bus.subscribe(
            "task.handoff",
            self._handle_task_handoff,
        )
        self._subscriptions["task.handoff"] = sub_id

        logger.info("AgentCommunication subscriptions initialized")

    async def shutdown(self) -> None:
        """Clean up subscriptions and resources."""
        for sub_id in self._subscriptions.values():
            await self.event_bus.unsubscribe(sub_id)
        self._subscriptions.clear()
        logger.info("AgentCommunication shut down")

    # ========================================================================
    # Request-Response Pattern
    # ========================================================================

    async def request_context(
        self,
        from_agent: str,
        to_agent: str,
        query: str,
        max_tokens: int = 2000,
    ) -> ContextResponse:
        """Request context from another agent (request-response pattern).

        Sends a context request to a specific agent and waits for the response.
        Validates tenant permissions before sending.

        Args:
            from_agent: Requesting agent ID.
            to_agent: Target agent ID to request context from.
            query: Query describing the context needed.
            max_tokens: Maximum tokens in the response.

        Returns:
            ContextResponse with the requested context.

        Raises:
            PermissionError: If cross-tenant access is denied.
            TimeoutError: If target agent does not respond.
        """
        # Validate cross-tenant access
        if not await self.tenant_manager.validate_cross_tenant(from_agent, to_agent):
            raise PermissionError(
                f"Cross-tenant access denied: {from_agent} -> {to_agent}"
            )

        # Create request
        request = ContextRequest(
            from_agent=from_agent,
            to_agent=to_agent,
            query=query,
            max_tokens=max_tokens,
        )

        logger.info(
            f"Context request: {from_agent} -> {to_agent}, query: {query[:50]}..."
        )

        # Send request via event bus
        response_payload = await self.event_bus.request(
            "context.request",
            request.model_dump(),
            target=to_agent,
            timeout=30.0,
        )

        if response_payload is None:
            logger.warning(f"Context request timeout: {from_agent} -> {to_agent}")
            # Return empty response on timeout
            return ContextResponse(
                request_id=request.request_id,
                from_agent=to_agent,
                to_agent=from_agent,
                context="",
                relevance_score=0.0,
                token_count=0,
                source="timeout",
            )

        return ContextResponse(**response_payload)

    async def _handle_context_request(self, payload: Dict[str, Any]) -> None:
        """Handle incoming context request."""
        request = ContextRequest(**payload)

        # Validate access
        if not await self.tenant_manager.validate_access(
            request.to_agent,
            request.from_agent,
            "read",
        ):
            logger.warning(f"Access denied for context request: {request.request_id}")
            return

        # Retrieve context for the requesting agent
        contexts = await self.get_shared_context(
            request.to_agent,
            request.query,
        )

        # Build response context
        response_context = ""
        relevance_score = 0.0

        if contexts:
            # Combine relevant contexts within token limit
            token_count = 0
            context_parts = []
            for ctx in contexts:
                ctx_tokens = len(ctx.content.split()) * 1.3
                if token_count + ctx_tokens > request.max_tokens:
                    break
                context_parts.append(ctx.content)
                token_count += ctx_tokens
                relevance_score = max(relevance_score, 1.0)

            response_context = "\n\n".join(context_parts)

        # Create and emit response
        response = ContextResponse(
            request_id=request.request_id,
            from_agent=request.to_agent,
            to_agent=request.from_agent,
            context=response_context,
            relevance_score=relevance_score,
            token_count=int(len(response_context.split()) * 1.3),
            source="shared_context",
        )

        await self.event_bus.emit(
            "context.response",
            response.model_dump(),
            target=request.from_agent,
            correlation_id=request.request_id,
        )

    # ========================================================================
    # Broadcast Pattern
    # ========================================================================

    async def share_context(
        self,
        from_agent: str,
        to_agent: str,
        context: SharedContext,
    ) -> None:
        """Share context with another agent or broadcast to all.

        Supports both targeted sharing (to_agent specified) and broadcast
        (to_agent is None). Validates tenant permissions.

        Args:
            from_agent: Sharing agent ID.
            to_agent: Target agent ID, or None for broadcast.
            context: SharedContext to share.

        Raises:
            PermissionError: If cross-tenant access is denied.
        """
        # Update context metadata
        context.from_agent = from_agent
        context.to_agent = to_agent

        # Validate cross-tenant access for targeted share
        if to_agent is not None:
            if not await self.tenant_manager.validate_cross_tenant(from_agent, to_agent):
                raise PermissionError(
                    f"Cross-tenant access denied: {from_agent} -> {to_agent}"
                )

        # Store context locally
        async with self._lock:
            if from_agent not in self._shared_contexts:
                self._shared_contexts[from_agent] = []
            self._shared_contexts[from_agent].append(context)

        logger.info(
            f"Context shared: {from_agent} -> "
            f"{to_agent or 'broadcast'}, id: {context.context_id}"
        )

        # Emit context share event
        await self.event_bus.emit(
            "context.share",
            context.model_dump(),
            target=to_agent,
        )

    async def _handle_context_share(self, payload: Dict[str, Any]) -> None:
        """Handle incoming shared context."""
        context = SharedContext(**payload)

        # If targeted share, validate access
        if context.to_agent is not None:
            if not await self.tenant_manager.validate_access(
                context.to_agent,
                context.from_agent,
                "read",
            ):
                logger.warning(f"Access denied for shared context: {context.context_id}")
                return

        # Store received context
        async with self._lock:
            target = context.to_agent or "__broadcast__"
            if target not in self._shared_contexts:
                self._shared_contexts[target] = []
            self._shared_contexts[target].append(context)

        logger.debug(f"Stored shared context: {context.context_id}")

    async def broadcast_context(
        self,
        from_agent: str,
        context: SharedContext,
        tenant_only: bool = True,
    ) -> int:
        """Broadcast context to all agents (or tenant agents only).

        Args:
            from_agent: Broadcasting agent ID.
            context: SharedContext to broadcast.
            tenant_only: If True, broadcast only to agents in same tenant.

        Returns:
            Number of agents that received the broadcast.
        """
        context.from_agent = from_agent
        context.to_agent = None

        if tenant_only:
            tenant_id = await self.tenant_manager.get_tenant_id(from_agent)
            if tenant_id:
                agents = await self.tenant_manager.get_tenant_agents(tenant_id)
                # Exclude sender
                agents = [a for a in agents if a != from_agent]
            else:
                agents = []
        else:
            # Global broadcast - emit without target
            await self.event_bus.emit(
                "context.broadcast",
                context.model_dump(),
            )
            return -1  # Unknown count for global broadcast

        # Send to each agent individually
        count = 0
        for agent_id in agents:
            try:
                await self.event_bus.emit(
                    "context.share",
                    context.model_dump(),
                    target=agent_id,
                )
                count += 1
            except Exception as e:
                logger.error(f"Failed to broadcast to {agent_id}: {e}")

        logger.info(f"Broadcast from {from_agent} to {count} agents")
        return count

    # ========================================================================
    # Handoff Pattern
    # ========================================================================

    async def handoff_task(
        self,
        from_agent: str,
        to_agent: str,
        task: Task,
    ) -> str:
        """Hand off a task to another agent (handoff pattern).

        Transfers task ownership and responsibility to another agent.
        Records the handoff in history for audit purposes.

        Args:
            from_agent: Agent handing off the task.
            to_agent: Agent receiving the task.
            task: Task to hand off.

        Returns:
            Handoff ID for tracking.

        Raises:
            PermissionError: If cross-tenant access is denied.
            ValueError: If task status does not allow handoff.
        """
        # Validate task status
        if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED):
            raise ValueError(
                f"Cannot handoff task in status {task.status.value}"
            )

        # Validate cross-tenant access
        if not await self.tenant_manager.validate_cross_tenant(from_agent, to_agent):
            raise PermissionError(
                f"Cross-tenant access denied: {from_agent} -> {to_agent}"
            )

        # Create handoff record
        handoff = TaskHandoff(
            task_id=task.task_id,
            from_agent=from_agent,
            to_agent=to_agent,
            reason=task.metadata.get("handoff_reason", ""),
            context=task.context,
        )

        # Update task
        task.assigned_to = to_agent
        task.status = TaskStatus.ASSIGNED
        task.updated_at = datetime.utcnow()
        task.metadata["previous_owner"] = from_agent
        task.metadata["handoff_id"] = handoff.handoff_id

        # Store in history
        async with self._lock:
            self._handoff_history.append(handoff)
            self._active_tasks[task.task_id] = task

        logger.info(
            f"Task handoff: {task.task_id} from {from_agent} to {to_agent}, "
            f"handoff_id: {handoff.handoff_id}"
        )

        # Emit handoff event
        await self.event_bus.emit(
            "task.handoff",
            {
                "handoff": handoff.model_dump(),
                "task": task.model_dump(),
            },
            target=to_agent,
        )

        return handoff.handoff_id

    async def _handle_task_handoff(self, payload: Dict[str, Any]) -> None:
        """Handle incoming task handoff."""
        handoff = TaskHandoff(**payload["handoff"])
        task = Task(**payload["task"])

        # Validate access
        if not await self.tenant_manager.validate_access(
            handoff.to_agent,
            task.task_id,
            "write",
        ):
            logger.warning(f"Access denied for task handoff: {handoff.handoff_id}")
            return

        # Store task locally
        async with self._lock:
            self._active_tasks[task.task_id] = task
            self._handoff_history.append(handoff)

        logger.info(f"Received task handoff: {handoff.handoff_id}")

    async def get_handoff_history(
        self,
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[TaskHandoff]:
        """Get task handoff history with optional filters.

        Args:
            task_id: Filter by task ID.
            agent_id: Filter by agent ID (from or to).
            limit: Maximum records to return.

        Returns:
            List of TaskHandoff records.
        """
        async with self._lock:
            history = self._handoff_history.copy()

        # Apply filters
        if task_id:
            history = [h for h in history if h.task_id == task_id]

        if agent_id:
            history = [
                h for h in history
                if h.from_agent == agent_id or h.to_agent == agent_id
            ]

        # Sort by timestamp descending
        history.sort(key=lambda h: h.timestamp, reverse=True)

        return history[:limit]

    # ========================================================================
    # Collaborative Pattern
    # ========================================================================

    async def request_collaboration(
        self,
        from_agent: str,
        to_agents: List[str],
        objective: str,
        context: str = "",
        max_tokens: int = 2000,
    ) -> List[CollaborationResponse]:
        """Request collaboration from multiple agents.

        Sends a collaboration request to multiple agents and collects
        their responses. Useful for tasks requiring diverse expertise.

        Args:
            from_agent: Initiating agent ID.
            to_agents: List of agent IDs to collaborate with.
            objective: The collaboration objective.
            context: Shared context for all collaborators.
            max_tokens: Maximum tokens per response.

        Returns:
            List of CollaborationResponse from participating agents.
        """
        # Create collaboration request
        request = CollaborationRequest(
            from_agent=from_agent,
            to_agents=to_agents,
            objective=objective,
            context=context,
            max_tokens=max_tokens,
        )

        logger.info(
            f"Collaboration request: {from_agent} -> {len(to_agents)} agents, "
            f"objective: {objective[:50]}..."
        )

        # Send requests to all agents in parallel
        tasks = []
        for agent_id in to_agents:
            # Validate access first
            if await self.tenant_manager.validate_cross_tenant(from_agent, agent_id):
                task = self._request_single_collaboration(request, agent_id)
                tasks.append(task)
            else:
                logger.warning(
                    f"Skipping collaboration with {agent_id}: cross-tenant denied"
                )

        # Gather responses with timeout
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful responses
        valid_responses = []
        for resp in responses:
            if isinstance(resp, CollaborationResponse):
                valid_responses.append(resp)
            elif isinstance(resp, Exception):
                logger.error(f"Collaboration error: {resp}")

        logger.info(
            f"Collaboration complete: {len(valid_responses)}/{len(to_agents)} responses"
        )

        return valid_responses

    async def _request_single_collaboration(
        self,
        request: CollaborationRequest,
        agent_id: str,
    ) -> CollaborationResponse:
        """Request collaboration from a single agent."""
        response_payload = await self.event_bus.request(
            "collaboration.request",
            {
                "request": request.model_dump(),
                "target_agent": agent_id,
            },
            target=agent_id,
            timeout=30.0,
        )

        if response_payload is None:
            # Return empty response on timeout
            return CollaborationResponse(
                request_id=request.request_id,
                from_agent=agent_id,
                contribution="",
                status="timeout",
            )

        return CollaborationResponse(**response_payload)

    async def respond_collaboration(
        self,
        request_id: str,
        from_agent: str,
        contribution: str,
        status: str = "completed",
    ) -> None:
        """Respond to a collaboration request.

        Args:
            request_id: Original collaboration request ID.
            from_agent: Responding agent ID.
            contribution: The agent's contribution to the collaboration.
            status: Response status (completed, partial, error).
        """
        response = CollaborationResponse(
            request_id=request_id,
            from_agent=from_agent,
            contribution=contribution,
            status=status,
        )

        await self.event_bus.emit(
            "collaboration.response",
            response.model_dump(),
            correlation_id=request_id,
        )

        logger.info(f"Collaboration response from {from_agent}: {status}")

    # ========================================================================
    # Shared Context Retrieval
    # ========================================================================

    async def get_shared_context(
        self,
        agent_id: str,
        query: str,
    ) -> List[SharedContext]:
        """Get shared context relevant to a query.

        Retrieves shared contexts that are relevant to the agent,
        filtering by query terms and checking expiration.

        Args:
            agent_id: Agent requesting context.
            query: Query to match against context content.

        Returns:
            List of relevant SharedContext objects.
        """
        now = datetime.utcnow()
        relevant_contexts = []

        async with self._lock:
            # Collect contexts from all sources
            all_contexts = []

            # Contexts shared by this agent
            if agent_id in self._shared_contexts:
                all_contexts.extend(self._shared_contexts[agent_id])

            # Contexts shared to this agent
            for contexts in self._shared_contexts.values():
                for ctx in contexts:
                    if ctx.to_agent == agent_id:
                        all_contexts.append(ctx)

            # Broadcast contexts
            if "__broadcast__" in self._shared_contexts:
                all_contexts.extend(self._shared_contexts["__broadcast__"])

        # Filter by expiration and relevance
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for ctx in all_contexts:
            # Check expiration
            if ctx.expires_at and ctx.expires_at < now:
                continue

            # Simple relevance check: match query terms or tags
            content_lower = ctx.content.lower()
            tags_lower = {t.lower() for t in ctx.tags}

            # Check if any query term appears in content or tags
            content_matches = any(term in content_lower for term in query_terms)
            tag_matches = bool(query_terms & tags_lower)

            if content_matches or tag_matches:
                relevant_contexts.append(ctx)

        # Sort by creation time (most recent first)
        relevant_contexts.sort(key=lambda c: c.created_at, reverse=True)

        logger.debug(
            f"Found {len(relevant_contexts)} relevant contexts for agent {agent_id}"
        )

        return relevant_contexts

    async def get_context_by_id(self, context_id: str) -> Optional[SharedContext]:
        """Get a specific shared context by ID.

        Args:
            context_id: Context ID to retrieve.

        Returns:
            SharedContext if found, None otherwise.
        """
        async with self._lock:
            for contexts in self._shared_contexts.values():
                for ctx in contexts:
                    if ctx.context_id == context_id:
                        return ctx
        return None

    async def cleanup_expired_contexts(self) -> int:
        """Remove expired shared contexts.

        Returns:
            Number of contexts removed.
        """
        now = datetime.utcnow()
        removed = 0

        async with self._lock:
            for agent_id in list(self._shared_contexts.keys()):
                contexts = self._shared_contexts[agent_id]
                before = len(contexts)
                self._shared_contexts[agent_id] = [
                    ctx for ctx in contexts
                    if ctx.expires_at is None or ctx.expires_at > now
                ]
                removed += before - len(self._shared_contexts[agent_id])

        logger.info(f"Cleaned up {removed} expired contexts")
        return removed

    # ========================================================================
    # Task Management Helpers
    # ========================================================================

    async def register_task(self, task: Task) -> None:
        """Register a task for tracking.

        Args:
            task: Task to register.
        """
        async with self._lock:
            self._active_tasks[task.task_id] = task
        logger.debug(f"Registered task: {task.task_id}")

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID.

        Args:
            task_id: Task ID to retrieve.

        Returns:
            Task if found, None otherwise.
        """
        async with self._lock:
            return self._active_tasks.get(task_id)

    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> bool:
        """Update task status.

        Args:
            task_id: Task to update.
            status: New status.
            result: Optional result data.
            error: Optional error message.

        Returns:
            True if updated successfully.
        """
        async with self._lock:
            task = self._active_tasks.get(task_id)
            if task is None:
                return False

            task.status = status
            task.updated_at = datetime.utcnow()

            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            if status == TaskStatus.COMPLETED:
                task.completed_at = datetime.utcnow()

        logger.info(f"Task {task_id} status updated to {status.value}")
        return True

    async def get_agent_tasks(
        self,
        agent_id: str,
        status: Optional[TaskStatus] = None,
        limit: int = 100,
    ) -> List[Task]:
        """Get tasks assigned to an agent.

        Args:
            agent_id: Agent ID to get tasks for.
            status: Optional status filter.
            limit: Maximum tasks to return.

        Returns:
            List of tasks assigned to the agent.
        """
        async with self._lock:
            tasks = [
                task for task in self._active_tasks.values()
                if task.assigned_to == agent_id
            ]

        if status:
            tasks = [t for t in tasks if t.status == status]

        # Sort by priority and creation time
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        tasks.sort(
            key=lambda t: (priority_order.get(t.priority.value, 99), t.created_at)
        )

        return tasks[:limit]

    # ========================================================================
    # Statistics and Monitoring
    # ========================================================================

    async def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics.

        Returns:
            Dictionary with statistics.
        """
        async with self._lock:
            total_contexts = sum(
                len(ctxs) for ctxs in self._shared_contexts.values()
            )
            active_tasks = sum(
                1 for t in self._active_tasks.values()
                if t.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            )

        return {
            "shared_contexts": total_contexts,
            "active_tasks": active_tasks,
            "total_tasks": len(self._active_tasks),
            "handoff_history_size": len(self._handoff_history),
            "subscriptions": len(self._subscriptions),
        }
