"""Enhanced MCP server for gdrag v3.

Provides MCP tools for RAG operations (v2), multi-agent orchestration (v3),
session management, attention focusing, context compression, agent registry,
task queue, inter-agent communication, and state persistence.

All v2 tools are preserved. v3 tools added:
  agent_register, agent_heartbeat, agent_list,
  task_enqueue, task_status, task_cancel,
  knowledge_promote, context_request, context_share,
  state_save, state_load
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mcp.server import FastMCP

from ...core.agent_comm import (
    AgentCommunication,
    InMemoryEventBus,
    SimpleTenantManager,
)
from ...core.agent_registry import AgentRegistry
from ...core.config import load_config, AppConfig
from ...core.pipeline import QueryPipeline
from ...core.task_queue import TaskQueue
from ...models.agent import AgentEndpoint, AgentRegistration, HealthStatus
from ...models.communication import SharedContext
from ...models.schemas import EnhancedQueryRequest
from ...models.state import AgentState
from ...models.tasks import Task, TaskPriority, TaskStatus

logger = logging.getLogger(__name__)

# ============================================================================
# Singleton instances (lazy-initialized)
# ============================================================================

_pipeline: Optional[QueryPipeline] = None
_config: Optional[AppConfig] = None
_agent_registry: Optional[AgentRegistry] = None
_task_queue: Optional[TaskQueue] = None
_agent_comm: Optional[AgentCommunication] = None


def _get_config() -> AppConfig:
    """Get or create the application config."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_pipeline() -> QueryPipeline:
    """Get or create the query pipeline instance."""
    global _pipeline
    if _pipeline is None:
        config = _get_config()
        _pipeline = QueryPipeline(config)
    return _pipeline


def _get_agent_registry() -> AgentRegistry:
    """Get or create the AgentRegistry singleton."""
    global _agent_registry
    if _agent_registry is None:
        config = _get_config()
        _agent_registry = AgentRegistry(config)
    return _agent_registry


def _get_task_queue() -> TaskQueue:
    """Get or create the TaskQueue singleton."""
    global _task_queue
    if _task_queue is None:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        _task_queue = TaskQueue(redis_url=redis_url)
    return _task_queue


def _get_agent_comm() -> AgentCommunication:
    """Get or create the AgentCommunication singleton."""
    global _agent_comm
    if _agent_comm is None:
        event_bus = InMemoryEventBus()
        tenant_mgr = SimpleTenantManager()
        _agent_comm = AgentCommunication(
            event_bus=event_bus,
            tenant_manager=tenant_mgr,
        )
    return _agent_comm


# ============================================================================
# State persistence helpers (PostgreSQL-backed)
# ============================================================================

_STATE_TABLE = "agent_states"


def _ensure_state_table() -> None:
    """Ensure the agent_states table exists."""
    import psycopg2

    config = _get_config()
    conn = psycopg2.connect(
        host=config.database.postgres_host,
        port=config.database.postgres_port,
        dbname=config.database.postgres_db,
        user=config.database.postgres_user,
        password=config.database.postgres_password,
    )
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {_STATE_TABLE} (
                    agent_id    VARCHAR(255) NOT NULL,
                    snapshot_id VARCHAR(255) NOT NULL,
                    state_data  JSONB NOT NULL,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (agent_id, snapshot_id)
                )
            """)
    finally:
        conn.close()


def _save_state_pg(agent_state: AgentState) -> str:
    """Persist an AgentState snapshot to PostgreSQL."""
    import psycopg2

    _ensure_state_table()
    config = _get_config()
    conn = psycopg2.connect(
        host=config.database.postgres_host,
        port=config.database.postgres_port,
        dbname=config.database.postgres_db,
        user=config.database.postgres_user,
        password=config.database.postgres_password,
    )
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {_STATE_TABLE} (agent_id, snapshot_id, state_data, created_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (agent_id, snapshot_id) DO UPDATE SET
                    state_data = EXCLUDED.state_data,
                    created_at = EXCLUDED.created_at
                """,
                (
                    agent_state.agent_id,
                    agent_state.snapshot_id,
                    agent_state.model_dump_json(),
                    agent_state.created_at,
                ),
            )
    finally:
        conn.close()
    return agent_state.snapshot_id


def _load_state_pg(agent_id: str, snapshot_id: Optional[str] = None) -> Optional[AgentState]:
    """Load the latest (or specific) AgentState snapshot from PostgreSQL."""
    import psycopg2

    _ensure_state_table()
    config = _get_config()
    conn = psycopg2.connect(
        host=config.database.postgres_host,
        port=config.database.postgres_port,
        dbname=config.database.postgres_db,
        user=config.database.postgres_user,
        password=config.database.postgres_password,
    )
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            if snapshot_id:
                cur.execute(
                    f"SELECT state_data FROM {_STATE_TABLE} WHERE agent_id = %s AND snapshot_id = %s",
                    (agent_id, snapshot_id),
                )
            else:
                cur.execute(
                    f"SELECT state_data FROM {_STATE_TABLE} WHERE agent_id = %s ORDER BY created_at DESC LIMIT 1",
                    (agent_id,),
                )
            row = cur.fetchone()
            if row is None:
                return None
            return AgentState.model_validate_json(row[0])
    finally:
        conn.close()


# ============================================================================
# Async helper -- bridge sync MCP tools to async core modules
# ============================================================================

def _run_async(coro):
    """Run an async coroutine from synchronous MCP tool code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    return asyncio.run(coro)


# ========================================================================
# Server factory
# ========================================================================

def create_mcp_server() -> FastMCP:
    """Create and configure the MCP server with v2 + v3 tools.

    Returns:
        Configured FastMCP server instance.
    """
    mcp = FastMCP("gdrag-v3")

    # ====================================================================
    # V2  -- Query Tools
    # ====================================================================

    @mcp.tool()
    def rag_query(
        query: str,
        domain: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10,
        compress: bool = False,
    ) -> Dict[str, Any]:
        """Execute an enhanced RAG query.

        Performs semantic search across all knowledge stores with optional
        session context, re-ranking, and compression.

        Args:
            query: Search query text.
            domain: Optional domain filter (software, finance, academic, print3d).
            session_id: Optional session ID for context.
            limit: Maximum results to return (1-100).
            compress: Whether to compress results.

        Returns:
            Dictionary with query results and metadata.
        """
        try:
            pipeline = get_pipeline()

            request = EnhancedQueryRequest(
                query=query,
                domain=domain,
                session_id=session_id,
                use_context=bool(session_id),
                compress_results=compress,
                attention_focus=True,
                limit=limit,
            )

            response = pipeline.execute(request)

            return {
                "query": response.query,
                "results": [
                    {
                        "content": r.content,
                        "title": r.title,
                        "domain": r.domain,
                        "score": r.final_score,
                        "doc_id": r.doc_id,
                    }
                    for r in response.results
                ],
                "session_context_used": response.session_context_used,
                "compressed_context": response.compressed_context,
                "metadata": {
                    "total_results": response.metadata.total_results,
                    "processing_time_ms": response.metadata.processing_time_ms,
                    "reranked": response.metadata.reranked,
                    "compressed": response.metadata.compressed,
                },
            }
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return {"error": str(e)}

    @mcp.tool()
    def semantic_search(
        query: str,
        domain: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Perform semantic search in vector store.

        Searches for similar content using embeddings.

        Args:
            query: Search query text.
            domain: Optional domain filter.
            limit: Maximum results.

        Returns:
            Dictionary with search results.
        """
        try:
            pipeline = get_pipeline()
            results = pipeline.vector_store.search(
                query=query,
                limit=limit,
                domain=domain,
            )

            return {
                "query": query,
                "results": [
                    {
                        "content": r.content,
                        "title": r.title,
                        "domain": r.domain,
                        "score": r.semantic_score,
                        "doc_id": r.doc_id,
                    }
                    for r in results
                ],
                "count": len(results),
            }
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return {"error": str(e)}

    @mcp.tool()
    def graph_search(
        query: str,
        depth: int = 2,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Search knowledge graph for related concepts.

        Finds concepts and their relationships in the graph database.

        Args:
            query: Search query or concept name.
            depth: Graph traversal depth.
            limit: Maximum results.

        Returns:
            Dictionary with graph search results.
        """
        try:
            pipeline = get_pipeline()

            # Extract concepts from query
            concepts = pipeline.graph_store.extract_concepts(query)

            if not concepts:
                return {"query": query, "results": [], "count": 0}

            # Find related concepts
            related = pipeline.graph_store.find_related_concepts(
                concepts[:5], depth=depth, limit=limit
            )

            return {
                "query": query,
                "concepts_extracted": concepts[:10],
                "results": [
                    {
                        "source": r.source_concept,
                        "target": r.target_concept,
                        "relation": r.relation_type,
                        "weight": r.weight,
                    }
                    for r in related
                ],
                "count": len(related),
            }
        except Exception as e:
            logger.error(f"Graph search error: {e}")
            return {"error": str(e)}

    @mcp.tool()
    def structured_query(
        query: str,
        domain: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Perform structured search in PostgreSQL.

        Full-text search across knowledge entries.

        Args:
            query: Search query text.
            domain: Optional domain filter.
            limit: Maximum results.

        Returns:
            Dictionary with query results.
        """
        try:
            pipeline = get_pipeline()
            results = pipeline.relational_store.full_text_search(
                query_text=query,
                domain=domain,
                limit=limit,
            )

            return {
                "query": query,
                "results": results,
                "count": len(results),
            }
        except Exception as e:
            logger.error(f"Structured query error: {e}")
            return {"error": str(e)}

    # ====================================================================
    # V2  -- Session Tools
    # ====================================================================

    @mcp.tool()
    def session_create(
        agent_id: str = "default",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new agent session.

        Sessions track query history and provide context for future queries.

        Args:
            agent_id: Agent identifier.
            session_id: Optional custom session ID.

        Returns:
            Dictionary with session information.
        """
        try:
            pipeline = get_pipeline()
            session = pipeline.session_manager.create_session(
                agent_id=agent_id,
                session_id=session_id,
            )

            return {
                "session_id": session.session_id,
                "agent_id": session.agent_id,
                "created_at": session.created_at.isoformat(),
                "max_tokens": session.max_tokens,
            }
        except Exception as e:
            logger.error(f"Session creation error: {e}")
            return {"error": str(e)}

    @mcp.tool()
    def session_context(
        session_id: str,
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """Get context from session history.

        Retrieves relevant context from previous queries in the session.

        Args:
            session_id: Session ID.
            max_tokens: Maximum tokens to include.

        Returns:
            Dictionary with session context.
        """
        try:
            pipeline = get_pipeline()
            context = pipeline.session_manager.get_session_context(
                session_id=session_id,
                max_tokens=max_tokens,
            )

            session = pipeline.session_manager.get_session(session_id)

            return {
                "session_id": session_id,
                "context": context,
                "query_count": len(session.query_history) if session else 0,
                "token_count": session.token_count if session else 0,
            }
        except Exception as e:
            logger.error(f"Session context error: {e}")
            return {"error": str(e)}

    @mcp.tool()
    def session_compress(session_id: str) -> Dict[str, Any]:
        """Compress session history.

        Summarizes old queries to save tokens while preserving context.

        Args:
            session_id: Session ID to compress.

        Returns:
            Dictionary with compression result.
        """
        try:
            pipeline = get_pipeline()
            success = pipeline.session_manager.compress_session_history(session_id)

            return {
                "session_id": session_id,
                "compressed": success,
                "message": "Session compressed successfully" if success else "Not enough history to compress",
            }
        except Exception as e:
            logger.error(f"Session compression error: {e}")
            return {"error": str(e)}

    # ====================================================================
    # V2  -- Ingestion Tools
    # ====================================================================

    @mcp.tool()
    def ingest_knowledge(
        content: str,
        title: str = "",
        domain: Optional[str] = None,
        source: str = "mcp",
    ) -> Dict[str, Any]:
        """Ingest knowledge into the system.

        Chunks content, extracts concepts, and stores in all databases.

        Args:
            content: Document content to ingest.
            title: Document title.
            domain: Knowledge domain.
            source: Source identifier.

        Returns:
            Dictionary with ingestion statistics.
        """
        try:
            pipeline = get_pipeline()
            stats = pipeline.ingest_document(
                content=content,
                title=title,
                domain=domain,
                source=source,
            )

            return {
                "status": "success",
                "doc_id": stats["doc_id"],
                "chunks_created": stats["chunks_created"],
                "concepts_extracted": stats["concepts_extracted"],
            }
        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            return {"error": str(e)}

    # ====================================================================
    # V2  -- Health & Stats Tools
    # ====================================================================

    @mcp.tool()
    def get_health() -> Dict[str, Any]:
        """Get system health status.

        Returns health of all components.

        Returns:
            Dictionary with health status.
        """
        try:
            pipeline = get_pipeline()
            return pipeline.get_health()
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {"status": "unhealthy", "error": str(e)}

    @mcp.tool()
    def get_stats() -> Dict[str, Any]:
        """Get system statistics.

        Returns statistics from all components.

        Returns:
            Dictionary with statistics.
        """
        try:
            pipeline = get_pipeline()

            vector_stats = pipeline.vector_store.get_collection_stats()
            session_stats = pipeline.session_manager.get_stats()
            attention_stats = pipeline.attention_manager.get_stats()

            return {
                "vector_store": {
                    "documents": vector_stats.total_documents,
                    "chunks": vector_stats.total_chunks,
                    "domains": vector_stats.domains,
                },
                "sessions": session_stats,
                "attention": attention_stats,
            }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {"error": str(e)}

    # ====================================================================
    # V3  -- Agent Registry Tools
    # ====================================================================

    @mcp.tool()
    def agent_register(
        agent_id: str,
        name: str,
        capabilities: Optional[List[str]] = None,
        endpoints: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        heartbeat_interval_s: int = 60,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Register a new agent in the multi-agent registry.

        Creates or updates (upserts) an agent entry in PostgreSQL.
        Other agents can then discover and communicate with this agent.

        Args:
            agent_id: Unique agent identifier (e.g., 'cline', 'claude-prod').
            name: Human-readable agent name.
            capabilities: Agent capabilities (e.g., ['query', 'ingest']).
            endpoints: Endpoint configurations as list of dicts with 'url' key.
            metadata: Additional metadata dict.
            heartbeat_interval_s: Expected heartbeat interval (10-600 seconds).
            tags: Tags for categorization.

        Returns:
            Dictionary with registered agent_id and status.
        """
        try:
            registry = _get_agent_registry()

            ep_list = []
            if endpoints:
                for ep in endpoints:
                    if isinstance(ep, dict):
                        ep_list.append(AgentEndpoint(**ep))
                    else:
                        ep_list.append(ep)

            registration = AgentRegistration(
                agent_id=agent_id,
                name=name,
                capabilities=capabilities or [],
                endpoints=ep_list,
                metadata=metadata or {},
                heartbeat_interval_s=heartbeat_interval_s,
                tags=tags or [],
            )

            registered_id = registry.register_agent(registration)

            return {
                "status": "registered",
                "agent_id": registered_id,
                "name": name,
            }
        except Exception as e:
            logger.error(f"Agent registration error: {e}")
            return {"error": str(e)}

    @mcp.tool()
    def agent_heartbeat(
        agent_id: str,
        status: str = "healthy",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a heartbeat for a registered agent.

        Updates the agent's last heartbeat timestamp and health status.
        Agents should call this periodically to signal liveness.

        Args:
            agent_id: Agent identifier.
            status: Health status ('healthy', 'degraded', 'unhealthy').
            metrics: Optional metrics dict (uptime, queue_depth, etc.).

        Returns:
            Dictionary with heartbeat confirmation.
        """
        try:
            registry = _get_agent_registry()

            success = registry.heartbeat(
                agent_id=agent_id,
                status=HealthStatus(status),
                metrics=metrics or {},
            )

            return {
                "agent_id": agent_id,
                "heartbeat_recorded": success,
                "status": status,
            }
        except Exception as e:
            logger.error(f"Agent heartbeat error: {e}")
            return {"error": str(e)}

    @mcp.tool()
    def agent_list(
        status_filter: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """List registered agents.

        Returns information about all registered agents with optional status filter.

        Args:
            status_filter: Filter by health status ('healthy', 'degraded', 'unhealthy', 'active').
            limit: Maximum agents to return.

        Returns:
            Dictionary with list of agent info dicts.
        """
        try:
            registry = _get_agent_registry()

            status_val = None
            if status_filter:
                status_val = HealthStatus(status_filter) if status_filter != "active" else "active"

            agents = registry.list_agents(
                status=status_val,
                limit=limit,
            )

            return {
                "agents": [
                    {
                        "agent_id": a.agent_id,
                        "name": a.name,
                        "status": a.status.value,
                        "capabilities": a.capabilities,
                        "tags": a.tags,
                        "registered_at": a.registered_at.isoformat(),
                        "last_heartbeat": a.last_heartbeat.isoformat() if a.last_heartbeat else None,
                    }
                    for a in agents
                ],
                "count": len(agents),
            }
        except Exception as e:
            logger.error(f"Agent list error: {e}")
            return {"error": str(e)}

    # ====================================================================
    # V3  -- Task Queue Tools
    # ====================================================================

    @mcp.tool()
    def task_enqueue(
        name: str,
        payload: Optional[Dict[str, Any]] = None,
        queue: str = "default",
        priority: str = "medium",
        title: str = "",
        description: str = "",
        created_by: str = "mcp",
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """Enqueue a new task for asynchronous processing.

        Adds a task to the Redis-backed priority queue. Workers can
        dequeue and process tasks in priority order.

        Args:
            name: Task name / function identifier.
            payload: Task execution data.
            queue: Queue name (default: 'default').
            priority: Task priority ('low', 'medium', 'high', 'critical').
            title: Human-readable task title.
            description: Detailed task description.
            created_by: Agent that created the task.
            max_retries: Maximum retry attempts.

        Returns:
            Dictionary with task_id and queue status.
        """
        try:
            task_queue = _get_task_queue()

            priority_map = {
                "low": 1,
                "medium": 5,
                "high": 10,
                "critical": 20,
            }

            task = Task(
                name=name,
                title=title or name,
                description=description,
                payload=payload or {},
                queue=queue,
                priority=TaskPriority(priority),
                priority_int=priority_map.get(priority, 5),
                created_by=created_by,
                max_retries=max_retries,
            )

            task_id = _run_async(task_queue.enqueue(task))

            return {
                "status": "enqueued",
                "task_id": task_id,
                "queue": queue,
                "priority": priority,
            }
        except Exception as e:
            logger.error(f"Task enqueue error: {e}")
            return {"error": str(e)}

    @mcp.tool()
    def task_status(
        task_id: str,
    ) -> Dict[str, Any]:
        """Get the current status of a task.

        Retrieves full task details including status, result, and error.

        Args:
            task_id: Task identifier.

        Returns:
            Dictionary with task details.
        """
        try:
            task_queue = _get_task_queue()

            task = _run_async(task_queue.get_task(task_id))

            if task is None:
                return {"error": f"Task {task_id} not found"}

            return {
                "task_id": task.task_id,
                "name": task.name,
                "status": task.status.value,
                "queue": task.queue,
                "priority": task.priority.value,
                "created_by": task.created_by,
                "assigned_to": task.assigned_to,
                "worker_id": task.worker_id,
                "retries": task.retries,
                "max_retries": task.max_retries,
                "result": task.result,
                "error": task.error,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            }
        except Exception as e:
            logger.error(f"Task status error: {e}")
            return {"error": str(e)}

    @mcp.tool()
    def task_cancel(
        task_id: str,
    ) -> Dict[str, Any]:
        """Cancel a pending or queued task.

        Only tasks in PENDING, QUEUED, or RETRYING status can be cancelled.

        Args:
            task_id: Task identifier.

        Returns:
            Dictionary with cancellation result.
        """
        try:
            task_queue = _get_task_queue()

            cancelled = _run_async(task_queue.cancel(task_id))

            return {
                "task_id": task_id,
                "cancelled": cancelled,
                "message": "Task cancelled" if cancelled else "Task could not be cancelled (not found or already running)",
            }
        except Exception as e:
            logger.error(f"Task cancel error: {e}")
            return {"error": str(e)}

    # ====================================================================
    # V3  -- Knowledge Promotion Tool
    # ====================================================================

    @mcp.tool()
    def knowledge_promote(
        content: str,
        title: str = "",
        domain: Optional[str] = None,
        source: str = "promotion",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Promote knowledge to the permanent knowledge base.

        Ingests content into all knowledge stores (vector, graph, relational)
        with a 'promotion' source marker, making it available for RAG queries
        across all agents and sessions.

        Args:
            content: Knowledge content to promote.
            title: Knowledge title.
            domain: Knowledge domain (software, finance, academic, print3d).
            source: Source identifier (default: 'promotion').
            tags: Optional tags for categorization.

        Returns:
            Dictionary with promotion statistics.
        """
        try:
            pipeline = get_pipeline()
            stats = pipeline.ingest_document(
                content=content,
                title=title,
                domain=domain,
                source=source,
            )

            return {
                "status": "promoted",
                "doc_id": stats["doc_id"],
                "chunks_created": stats["chunks_created"],
                "concepts_extracted": stats["concepts_extracted"],
                "domain": domain,
                "tags": tags or [],
            }
        except Exception as e:
            logger.error(f"Knowledge promote error: {e}")
            return {"error": str(e)}

    # ====================================================================
    # V3  -- Inter-Agent Communication Tools
    # ====================================================================

    @mcp.tool()
    def context_request(
        from_agent: str,
        to_agent: str,
        query: str,
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """Request context from another agent.

        Sends a context request to a target agent and waits for the response.
        Uses the request-response communication pattern.

        Args:
            from_agent: Requesting agent ID.
            to_agent: Target agent ID to request context from.
            query: Query describing the context needed.
            max_tokens: Maximum tokens in the response.

        Returns:
            Dictionary with the context response.
        """
        try:
            agent_comm = _get_agent_comm()

            response = _run_async(
                agent_comm.request_context(
                    from_agent=from_agent,
                    to_agent=to_agent,
                    query=query,
                    max_tokens=max_tokens,
                )
            )

            return {
                "request_id": response.request_id,
                "from_agent": response.from_agent,
                "to_agent": response.to_agent,
                "context": response.context,
                "relevance_score": response.relevance_score,
                "token_count": response.token_count,
                "source": response.source,
            }
        except Exception as e:
            logger.error(f"Context request error: {e}")
            return {"error": str(e)}

    @mcp.tool()
    def context_share(
        from_agent: str,
        content: str,
        to_agent: Optional[str] = None,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
        expires_hours: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Share context with another agent or broadcast to all.

        Stores shared context accessible by other agents. If to_agent is
        specified, shares only with that agent. If omitted, broadcasts.

        Args:
            from_agent: Sharing agent ID.
            content: Context content to share.
            to_agent: Target agent ID (None for broadcast).
            domain: Knowledge domain.
            tags: Tags for categorization.
            expires_hours: Hours until context expires (None = no expiry).

        Returns:
            Dictionary with share confirmation.
        """
        try:
            agent_comm = _get_agent_comm()

            expires_at = None
            if expires_hours is not None:
                from datetime import timedelta
                expires_at = datetime.utcnow() + timedelta(hours=expires_hours)

            shared_ctx = SharedContext(
                from_agent=from_agent,
                to_agent=to_agent,
                content=content,
                domain=domain,
                tags=tags or [],
                expires_at=expires_at,
            )

            _run_async(
                agent_comm.share_context(
                    from_agent=from_agent,
                    to_agent=to_agent,
                    context=shared_ctx,
                )
            )

            return {
                "status": "shared",
                "context_id": shared_ctx.context_id,
                "from_agent": from_agent,
                "to_agent": to_agent or "broadcast",
                "domain": domain,
            }
        except Exception as e:
            logger.error(f"Context share error: {e}")
            return {"error": str(e)}

    # ====================================================================
    # V3  -- State Persistence Tools
    # ====================================================================

    @mcp.tool()
    def state_save(
        agent_id: str,
        active_sessions: Optional[List[str]] = None,
        recent_queries: Optional[List[str]] = None,
        attention_focus: Optional[str] = None,
        pending_tasks: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Save an agent's cognitive state snapshot.

        Persists the agent's active sessions, query history, attention focus,
        pending tasks, and metrics to PostgreSQL for recovery and continuity.

        Args:
            agent_id: Agent identifier.
            active_sessions: List of active session IDs.
            recent_queries: Recent query texts.
            attention_focus: Current area of focus.
            pending_tasks: List of pending task IDs.
            metrics: Performance metrics dict.
            metadata: Additional state metadata.

        Returns:
            Dictionary with snapshot_id and save confirmation.
        """
        try:
            state = AgentState(
                agent_id=agent_id,
                active_sessions=active_sessions or [],
                recent_queries=recent_queries or [],
                attention_focus=attention_focus,
                pending_tasks=pending_tasks or [],
                metrics=metrics or {},
                metadata=metadata or {},
            )

            snapshot_id = _save_state_pg(state)

            return {
                "status": "saved",
                "agent_id": agent_id,
                "snapshot_id": snapshot_id,
                "created_at": state.created_at.isoformat(),
            }
        except Exception as e:
            logger.error(f"State save error: {e}")
            return {"error": str(e)}

    @mcp.tool()
    def state_load(
        agent_id: str,
        snapshot_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load an agent's cognitive state snapshot.

        Retrieves the latest (or specific) state snapshot for an agent
        from PostgreSQL.

        Args:
            agent_id: Agent identifier.
            snapshot_id: Specific snapshot ID (None = latest).

        Returns:
            Dictionary with state data or error if not found.
        """
        try:
            state = _load_state_pg(agent_id, snapshot_id)

            if state is None:
                return {
                    "error": f"No state found for agent {agent_id}"
                    + (f" snapshot {snapshot_id}" if snapshot_id else ""),
                    "agent_id": agent_id,
                }

            return {
                "agent_id": state.agent_id,
                "snapshot_id": state.snapshot_id,
                "active_sessions": state.active_sessions,
                "recent_queries": state.recent_queries,
                "attention_focus": state.attention_focus,
                "pending_tasks": state.pending_tasks,
                "metrics": state.metrics,
                "metadata": state.metadata,
                "created_at": state.created_at.isoformat(),
            }
        except Exception as e:
            logger.error(f"State load error: {e}")
            return {"error": str(e)}

    return mcp


if __name__ == "__main__":
    server = create_mcp_server()
    server.run()
