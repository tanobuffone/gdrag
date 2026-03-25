"""Pydantic models for gdrag v2.

Defines all data structures used throughout the application,
including request/response models, database models, and internal types.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ============================================================================
# Embedding Models
# ============================================================================

class EmbeddingResult(BaseModel):
    """Result of an embedding operation."""
    text: str = Field(description="Original text")
    embedding: List[float] = Field(description="Generated embedding vector")
    provider: str = Field(description="Embedding provider used")
    model: str = Field(description="Model used for embedding")
    dimensions: int = Field(description="Embedding dimensions")
    cached: bool = Field(default=False, description="Whether result was cached")
    processing_time_ms: float = Field(description="Processing time in milliseconds")


# ============================================================================
# Chunk Models
# ============================================================================

class Chunk(BaseModel):
    """A chunk of a document."""
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    doc_id: str = Field(description="Parent document ID")
    content: str = Field(description="Chunk text content")
    chunk_index: int = Field(description="Position in document")
    start_char: int = Field(description="Start character position")
    end_char: int = Field(description="End character position")
    token_count: int = Field(description="Approximate token count")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = Field(default=None)


# ============================================================================
# Knowledge Models
# ============================================================================

class KnowledgeItem(BaseModel):
    """A knowledge item stored in the system."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(description="Item title")
    content: str = Field(description="Item content")
    domain: Optional[str] = Field(default=None, description="Knowledge domain")
    source: str = Field(description="Source of the knowledge")
    source_type: Literal["document", "url", "api", "manual"] = Field(
        default="document"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ConceptRelation(BaseModel):
    """A relation between concepts in the knowledge graph."""
    source_concept: str = Field(description="Source concept name")
    target_concept: str = Field(description="Target concept name")
    relation_type: str = Field(description="Type of relation")
    weight: float = Field(default=1.0, description="Relation strength")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Session Models
# ============================================================================

class QueryRecord(BaseModel):
    """Record of a query in a session."""
    query_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str = Field(description="Original query text")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    results_ids: List[str] = Field(default_factory=list)
    relevance_scores: List[float] = Field(default_factory=list)
    feedback: Optional[float] = Field(
        default=None,
        description="User/agent feedback (0-1)"
    )
    processing_time_ms: Optional[float] = Field(default=None)


class SessionMemory(BaseModel):
    """Session memory for an agent."""
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str = Field(description="Agent identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    query_history: List[QueryRecord] = Field(default_factory=list)
    context_summary: Optional[str] = Field(
        default=None,
        description="Compressed summary of session context"
    )
    token_count: int = Field(default=0, description="Current token count")
    max_tokens: int = Field(default=8000, description="Maximum tokens allowed")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_expired(self, ttl_hours: int = 24) -> bool:
        """Check if session has expired."""
        from datetime import timedelta
        expiry_time = self.last_active + timedelta(hours=ttl_hours)
        return datetime.utcnow() > expiry_time


# ============================================================================
# Query Models
# ============================================================================

class EnhancedQueryRequest(BaseModel):
    """Enhanced query request with session and attention support."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Query text"
    )
    domain: Optional[str] = Field(
        default=None,
        description="Filter by knowledge domain"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for context"
    )
    use_context: bool = Field(
        default=True,
        description="Include session context in results"
    )
    compress_results: bool = Field(
        default=False,
        description="Compress result context"
    )
    attention_focus: bool = Field(
        default=True,
        description="Use attention-focused retrieval"
    )
    embedding_provider: Optional[Literal["local", "openai", "both"]] = Field(
        default=None,
        description="Override embedding provider"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum results to return"
    )


class RankedResult(BaseModel):
    """A ranked search result."""
    content: str = Field(description="Result content")
    title: str = Field(description="Result title")
    domain: Optional[str] = Field(default=None)
    source: str = Field(description="Source identifier")
    semantic_score: float = Field(description="Semantic similarity score")
    rerank_score: float = Field(default=0.0, description="Re-ranking score")
    attention_score: float = Field(default=0.0, description="Attention score")
    final_score: float = Field(description="Final combined score")
    chunk_index: int = Field(default=0)
    doc_id: str = Field(description="Document ID")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResponseMetadata(BaseModel):
    """Metadata for query response."""
    total_results: int = Field(description="Total results found")
    reranked: bool = Field(description="Whether results were re-ranked")
    compressed: bool = Field(description="Whether context was compressed")
    session_id: Optional[str] = Field(default=None)
    embedding_provider: str = Field(description="Provider used for embeddings")
    processing_time_ms: float = Field(description="Total processing time")
    tokens_saved: int = Field(default=0, description="Tokens saved by compression")


class EnhancedQueryResponse(BaseModel):
    """Enhanced query response with full metadata."""
    query: str = Field(description="Original query")
    results: List[RankedResult] = Field(description="Ranked results")
    compressed_context: Optional[str] = Field(
        default=None,
        description="Compressed context if requested"
    )
    session_context_used: bool = Field(
        default=False,
        description="Whether session context was used"
    )
    attention_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Attention scores per result"
    )
    metadata: ResponseMetadata = Field(description="Response metadata")


# ============================================================================
# Statistics Models
# ============================================================================

class CollectionStats(BaseModel):
    """Statistics for a knowledge collection."""
    collection_name: str
    total_documents: int
    total_chunks: int
    total_embeddings: int
    domains: List[str]
    last_updated: datetime


class DomainStats(BaseModel):
    """Statistics for a knowledge domain."""
    domain: str
    document_count: int
    chunk_count: int
    avg_chunk_size: float
    concepts_count: int