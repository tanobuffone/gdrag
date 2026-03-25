"""Enhanced API router for gdrag v2.

Provides REST endpoints for query, ingestion, sessions, and health.
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from ...core.config import AppConfig, load_config
from ...core.pipeline import QueryPipeline
from ...models.schemas import EnhancedQueryRequest, EnhancedQueryResponse

logger = logging.getLogger(__name__)

# Create router
api_v2_router = APIRouter(prefix="/api/v2", tags=["gdrag v2"])


# ============================================================================
# Request/Response Models
# ============================================================================

class IngestRequest(BaseModel):
    """Request to ingest a document."""
    content: str = Field(..., description="Document content")
    title: str = Field(default="", description="Document title")
    domain: Optional[str] = Field(default=None, description="Knowledge domain")
    source: str = Field(default="api", description="Source identifier")
    doc_id: Optional[str] = Field(default=None, description="Custom document ID")


class IngestResponse(BaseModel):
    """Response from document ingestion."""
    doc_id: str
    chunks_created: int
    concepts_extracted: int
    status: str = "success"


class SessionCreateRequest(BaseModel):
    """Request to create a session."""
    agent_id: str = Field(default="default", description="Agent identifier")
    session_id: Optional[str] = Field(default=None, description="Custom session ID")
    metadata: Optional[Dict] = Field(default=None, description="Session metadata")


class SessionResponse(BaseModel):
    """Session information response."""
    session_id: str
    agent_id: str
    created_at: str
    last_active: str
    query_count: int
    token_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str = "2.0.0"
    components: Dict


# ============================================================================
# Dependencies
# ============================================================================

# Global pipeline instance (initialized on startup)
_pipeline: Optional[QueryPipeline] = None


def get_pipeline() -> QueryPipeline:
    """Get or create the query pipeline instance."""
    global _pipeline
    if _pipeline is None:
        config = load_config()
        _pipeline = QueryPipeline(config)
    return _pipeline


def get_agent_id(
    x_agent_id: Optional[str] = Header(default=None, alias="X-Agent-ID")
) -> str:
    """Extract agent ID from header."""
    return x_agent_id or "default"


# ============================================================================
# Endpoints
# ============================================================================

@api_v2_router.post("/query", response_model=EnhancedQueryResponse)
async def query(
    request: EnhancedQueryRequest,
    agent_id: str = Depends(get_agent_id),
    pipeline: QueryPipeline = Depends(get_pipeline),
) -> EnhancedQueryResponse:
    """Execute an enhanced query with session and attention support.

    This is the main query endpoint that orchestrates the full RAG pipeline:
    - Semantic search across vector store
    - Re-ranking with cross-encoder
    - Attention-focused retrieval
    - Optional context compression
    - Session memory integration
    """
    try:
        response = pipeline.execute(request)
        return response
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_v2_router.post("/query/session", response_model=EnhancedQueryResponse)
async def query_with_session(
    query: str,
    session_id: str,
    domain: Optional[str] = None,
    limit: int = 10,
    agent_id: str = Depends(get_agent_id),
    pipeline: QueryPipeline = Depends(get_pipeline),
) -> EnhancedQueryResponse:
    """Execute a query within a session context.

    Automatically manages session creation and context integration.
    """
    try:
        response = pipeline.execute_with_session(
            query=query,
            session_id=session_id,
            agent_id=agent_id,
            domain=domain,
            limit=limit,
        )
        return response
    except Exception as e:
        logger.error(f"Session query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_v2_router.post("/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    pipeline: QueryPipeline = Depends(get_pipeline),
) -> IngestResponse:
    """Ingest a document into the knowledge base.

    Chunks the document, extracts concepts, and stores in all databases.
    """
    try:
        stats = pipeline.ingest_document(
            content=request.content,
            title=request.title,
            domain=request.domain,
            source=request.source,
            doc_id=request.doc_id,
        )
        return IngestResponse(
            doc_id=stats["doc_id"],
            chunks_created=stats["chunks_created"],
            concepts_extracted=stats["concepts_extracted"],
        )
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_v2_router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: SessionCreateRequest,
    pipeline: QueryPipeline = Depends(get_pipeline),
) -> SessionResponse:
    """Create a new agent session.

    Sessions track query history and provide context for future queries.
    """
    try:
        session = pipeline.session_manager.create_session(
            agent_id=request.agent_id,
            session_id=request.session_id,
            metadata=request.metadata,
        )
        return SessionResponse(
            session_id=session.session_id,
            agent_id=session.agent_id,
            created_at=session.created_at.isoformat(),
            last_active=session.last_active.isoformat(),
            query_count=len(session.query_history),
            token_count=session.token_count,
        )
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_v2_router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    pipeline: QueryPipeline = Depends(get_pipeline),
) -> SessionResponse:
    """Get session information.

    Returns details about a specific session.
    """
    session = pipeline.session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionResponse(
        session_id=session.session_id,
        agent_id=session.agent_id,
        created_at=session.created_at.isoformat(),
        last_active=session.last_active.isoformat(),
        query_count=len(session.query_history),
        token_count=session.token_count,
    )


@api_v2_router.get("/sessions", response_model=List[SessionResponse])
async def list_sessions(
    agent_id: Optional[str] = None,
    limit: int = 100,
    pipeline: QueryPipeline = Depends(get_pipeline),
) -> List[SessionResponse]:
    """List sessions.

    Optionally filter by agent ID.
    """
    sessions = pipeline.session_manager.list_sessions(
        agent_id=agent_id,
        limit=limit,
    )
    return [
        SessionResponse(
            session_id=s.session_id,
            agent_id=s.agent_id,
            created_at=s.created_at.isoformat(),
            last_active=s.last_active.isoformat(),
            query_count=len(s.query_history),
            token_count=s.token_count,
        )
        for s in sessions
    ]


@api_v2_router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    pipeline: QueryPipeline = Depends(get_pipeline),
) -> Dict:
    """Delete a session.

    Removes session data from memory and database.
    """
    success = pipeline.session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "deleted", "session_id": session_id}


@api_v2_router.get("/health", response_model=HealthResponse)
async def health(
    pipeline: QueryPipeline = Depends(get_pipeline),
) -> HealthResponse:
    """Health check endpoint.

    Returns status of all system components.
    """
    health_data = pipeline.get_health()
    return HealthResponse(
        status=health_data["status"],
        components=health_data["components"],
    )


@api_v2_router.get("/stats")
async def stats(
    pipeline: QueryPipeline = Depends(get_pipeline),
) -> Dict:
    """Get system statistics.

    Returns statistics from all components.
    """
    return {
        "vector_store": pipeline.vector_store.get_collection_stats().dict(),
        "sessions": pipeline.session_manager.get_stats(),
        "attention": pipeline.attention_manager.get_stats(),
    }