"""gdrag v3 Knowledge API.

Provides endpoints for knowledge management:
- POST   /knowledge/ingest     - Ingest new knowledge
- POST   /knowledge/{id}/promote - Promote entry to production
- GET    /knowledge/search      - Search knowledge entries
- DELETE /knowledge/{id}/access  - Revoke access to an entry
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from .dependencies import KnowledgeManager, get_knowledge_manager

logger = logging.getLogger(__name__)

knowledge_router = APIRouter(prefix="/api/v3", tags=["gdrag v3 - Knowledge"])


# ============================================================================
# Request / Response models
# ============================================================================

class IngestRequest(BaseModel):
    """Request body to ingest knowledge."""
    content: str = Field(..., min_length=1, description="Document content to ingest")
    title: str = Field(default="", description="Document title")
    domain: Optional[str] = Field(default=None, description="Knowledge domain (software, finance, academic, print3d)")
    source: str = Field(default="api", description="Source identifier")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class IngestResponse(BaseModel):
    """Response after ingesting knowledge."""
    doc_id: str
    title: str
    domain: Optional[str] = None
    source: str
    promoted: bool
    created_at: str
    status: str = "ingested"


class PromoteResponse(BaseModel):
    """Response after promoting knowledge."""
    doc_id: str
    promoted: bool
    status: str = "ok"


class KnowledgeResult(BaseModel):
    """A single knowledge search result."""
    doc_id: str
    title: str
    domain: Optional[str] = None
    source: str
    promoted: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str


class SearchResponse(BaseModel):
    """Response from knowledge search."""
    query: str
    domain: Optional[str] = None
    results: List[KnowledgeResult]
    total: int


class AccessRevokeResponse(BaseModel):
    """Response after revoking access."""
    doc_id: str
    access_revoked: bool
    status: str = "ok"


# ============================================================================
# Endpoints
# ============================================================================

@knowledge_router.post(
    "/knowledge/ingest",
    response_model=IngestResponse,
    status_code=201,
)
async def ingest_knowledge(
    request: IngestRequest,
    manager: KnowledgeManager = Depends(get_knowledge_manager),
) -> IngestResponse:
    """Ingest new knowledge into the system.

    Chunks the content, extracts concepts, and stores in all databases.
    Returns the created entry with its unique ID.
    """
    entry = manager.ingest(
        content=request.content,
        title=request.title,
        domain=request.domain,
        source=request.source,
        metadata=request.metadata,
    )
    logger.info("Knowledge ingested: %s (title=%s, domain=%s)", entry.doc_id, entry.title, entry.domain)
    return IngestResponse(
        doc_id=entry.doc_id,
        title=entry.title,
        domain=entry.domain,
        source=entry.source,
        promoted=entry.promoted,
        created_at=entry.created_at.isoformat(),
    )


@knowledge_router.post(
    "/knowledge/{doc_id}/promote",
    response_model=PromoteResponse,
)
async def promote_knowledge(
    doc_id: str,
    manager: KnowledgeManager = Depends(get_knowledge_manager),
) -> PromoteResponse:
    """Promote a knowledge entry to production.

    Marks the entry as promoted, making it available for production queries.
    Returns 404 if the entry does not exist.
    """
    if not manager.promote(doc_id):
        raise HTTPException(status_code=404, detail=f"Knowledge entry '{doc_id}' not found")
    return PromoteResponse(doc_id=doc_id, promoted=True)


@knowledge_router.get(
    "/knowledge/search",
    response_model=SearchResponse,
)
async def search_knowledge(
    query: str = Query(..., min_length=1, description="Search query text"),
    domain: Optional[str] = Query(default=None, description="Filter by domain"),
    limit: int = Query(default=10, ge=1, le=100, description="Max results"),
    manager: KnowledgeManager = Depends(get_knowledge_manager),
) -> SearchResponse:
    """Search knowledge entries.

    Performs semantic / full-text search across stored knowledge.
    Optionally filter by domain.
    """
    entries = manager.search(query=query, domain=domain, limit=limit)
    results = [
        KnowledgeResult(
            doc_id=e.doc_id,
            title=e.title,
            domain=e.domain,
            source=e.source,
            promoted=e.promoted,
            metadata=e.metadata,
            created_at=e.created_at.isoformat(),
        )
        for e in entries
    ]
    return SearchResponse(
        query=query,
        domain=domain,
        results=results,
        total=len(results),
    )


@knowledge_router.delete(
    "/knowledge/{doc_id}/access",
    response_model=AccessRevokeResponse,
)
async def revoke_knowledge_access(
    doc_id: str,
    manager: KnowledgeManager = Depends(get_knowledge_manager),
) -> AccessRevokeResponse:
    """Revoke access to a knowledge entry.

    Marks the entry as non-public so it is excluded from queries unless
    explicitly authorised. Returns 404 if the entry does not exist.
    """
    if not manager.revoke_access(doc_id):
        raise HTTPException(status_code=404, detail=f"Knowledge entry '{doc_id}' not found")
    return AccessRevokeResponse(doc_id=doc_id, access_revoked=True)
