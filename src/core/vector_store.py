"""Enhanced Qdrant vector store operations for gdrag v3.

Provides chunked storage, attention-focused search, re-ranking integration,
and multi-tenancy support with namespace isolation and visibility controls.

v3 Changes:
  - Per-tenant namespace support (tenant_{agent_id}_knowledge)
  - owner_agent_id and visibility stored in chunk payloads
  - Search methods accept agent_id for tenant-aware filtering
  - Backward-compatible with v2 (agent_id is optional, defaults to legacy behavior)
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from ..core.config import AppConfig, RerankConfig
from ..core.embeddings import embed_query, embed_texts
from ..models.schemas import Chunk, CollectionStats, RankedResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Visibility enum (re-exported from schemas for convenience)
# ---------------------------------------------------------------------------

class VisibilityLevel(str, Enum):
    """Visibility levels for knowledge documents.
    
    Mirrors src.models.schemas.VisibilityLevel to avoid circular imports
    when vector_store is used standalone.
    """
    PRIVATE = "private"
    TEAM = "team"
    SHARED = "shared"
    PUBLIC = "public"


class EnhancedVectorStore:
    """Enhanced Qdrant operations with chunked storage, attention support, and multi-tenancy.

    v3 adds:
      - Namespace isolation per tenant (configurable via use_namespaces).
      - owner_agent_id + visibility stored in every chunk payload.
      - Search methods accept ``agent_id`` to enforce tenant scoping.

    Backward-compatibility:
      - When ``agent_id`` is omitted the store behaves like v2 (no tenant filter).
      - Legacy collections without owner_agent_id payloads continue to work.
    """

    # Default collection name used in v2
    _DEFAULT_COLLECTION = "gdrag_knowledge"

    def __init__(
        self,
        config: AppConfig,
        use_namespaces: bool = False,
    ):
        """Initialize the vector store.

        Args:
            config: Application configuration.
            use_namespaces: If True, each tenant gets its own Qdrant collection
                named ``tenant_{agent_id}_knowledge``.  If False (default), a
                single shared collection is used with payload-based filtering.
        """
        self.config = config
        self._client: Optional[QdrantClient] = None
        self._base_collection = config.database.qdrant_collection
        self.use_namespaces = use_namespaces
        # Cache of ensured collections to avoid repeated API calls
        self._ensured_collections: set = set()

    # ------------------------------------------------------------------
    # Client helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> QdrantClient:
        """Get or create Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                host=self.config.database.qdrant_host,
                port=self.config.database.qdrant_port,
            )
            logger.info(
                f"Connected to Qdrant at "
                f"{self.config.database.qdrant_host}:{self.config.database.qdrant_port}"
            )
        return self._client

    # ------------------------------------------------------------------
    # Namespace helpers
    # ------------------------------------------------------------------

    def _get_collection_name(self, agent_id: Optional[str] = None) -> str:
        """Return the collection name for the given agent.

        If namespaces are enabled and agent_id is provided, returns
        ``tenant_{agent_id}_knowledge``.  Otherwise returns the base
        (legacy) collection name.
        """
        if self.use_namespaces and agent_id:
            safe_id = agent_id.replace("/", "_").replace("\\", "_")
            return f"tenant_{safe_id}_knowledge"
        return self._base_collection

    def _get_legacy_collection(self) -> str:
        """Return the legacy (v2) collection name."""
        return self._base_collection

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(
        self,
        vector_size: int,
        agent_id: Optional[str] = None,
    ) -> None:
        """Ensure collection exists with correct configuration.

        Args:
            vector_size: Size of embedding vectors.
            agent_id: Optional agent ID for namespace-specific collection.
        """
        collection_name = self._get_collection_name(agent_id)

        # Skip if already ensured in this session
        cache_key = f"{collection_name}:{vector_size}"
        if cache_key in self._ensured_collections:
            return

        client = self._get_client()

        try:
            collection_info = client.get_collection(collection_name)
            current_size = collection_info.config.params.vectors.size

            if current_size != vector_size:
                logger.warning(
                    f"Collection {collection_name} vector size mismatch: "
                    f"{current_size} vs {vector_size}. Recreating."
                )
                client.delete_collection(collection_name)
                self._create_collection(vector_size, agent_id)
        except Exception:
            logger.info(f"Creating collection: {collection_name}")
            self._create_collection(vector_size, agent_id)

        self._ensured_collections.add(cache_key)

    def _create_collection(
        self,
        vector_size: int,
        agent_id: Optional[str] = None,
    ) -> None:
        """Create Qdrant collection.

        Args:
            vector_size: Size of embedding vectors.
            agent_id: Optional agent ID for namespace-specific collection.
        """
        client = self._get_client()
        collection_name = self._get_collection_name(agent_id)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            f"Collection '{collection_name}' created with vector size: {vector_size}"
        )

    # ------------------------------------------------------------------
    # Filter builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_visibility_filter(
        agent_id: Optional[str],
        include_shared: bool = True,
    ) -> Optional[Filter]:
        """Build a Qdrant filter for multi-tenancy and visibility.

        Rules:
          - If agent_id is None, no filter (v2 backward-compat).
          - If include_shared is True:
              owner == agent_id  OR  visibility == "public"
              OR visibility == "shared"
              OR visibility == "team"
          - If include_shared is False:
              owner == agent_id  OR  visibility == "public"

        Note: TEAM visibility resolution (checking same-team membership)
        is done at the KnowledgeManager level.  At the vector store level
        we simply include TEAM docs and let the upper layer filter further.
        """
        if agent_id is None:
            return None

        must_conditions = []

        if include_shared:
            # Owner's own docs + public + shared + team
            should_conditions = [
                FieldCondition(
                    key="owner_agent_id",
                    match=MatchValue(value=agent_id),
                ),
                FieldCondition(
                    key="visibility",
                    match=MatchValue(value=VisibilityLevel.PUBLIC.value),
                ),
                FieldCondition(
                    key="visibility",
                    match=MatchValue(value=VisibilityLevel.SHARED.value),
                ),
                FieldCondition(
                    key="visibility",
                    match=MatchValue(value=VisibilityLevel.TEAM.value),
                ),
            ]
            return Filter(should=should_conditions)
        else:
            # Only owner's own docs + public
            should_conditions = [
                FieldCondition(
                    key="owner_agent_id",
                    match=MatchValue(value=agent_id),
                ),
                FieldCondition(
                    key="visibility",
                    match=MatchValue(value=VisibilityLevel.PUBLIC.value),
                ),
            ]
            return Filter(should=should_conditions)

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: Optional[Dict[str, List[float]]] = None,
        agent_id: Optional[str] = None,
        visibility: str = "private",
    ) -> List[str]:
        """Store document chunks with embeddings.

        Args:
            chunks: List of chunks to store.
            embeddings: Optional pre-computed embeddings keyed by chunk_id.
            agent_id: Owner agent ID for multi-tenancy.  If None, stored
                without owner (v2 behavior).
            visibility: Visibility level string (private/team/shared/public).

        Returns:
            List of stored chunk IDs.
        """
        client = self._get_client()

        # Generate embeddings if not provided
        if embeddings is None:
            texts = [chunk.content for chunk in chunks]
            embedding_vectors = embed_texts(texts, self.config.embedding)
            embeddings = {
                chunk.chunk_id: emb
                for chunk, emb in zip(chunks, embedding_vectors)
            }

        # Ensure collection exists (per-tenant or shared)
        if embeddings:
            sample_embedding = next(iter(embeddings.values()))
            self.ensure_collection(len(sample_embedding), agent_id)

        collection_name = self._get_collection_name(agent_id)

        # Prepare points
        points = []
        stored_ids = []

        for chunk in chunks:
            embedding = embeddings.get(chunk.chunk_id)
            if embedding is None:
                logger.warning(f"No embedding for chunk {chunk.chunk_id}, skipping")
                continue

            payload = {
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "token_count": chunk.token_count,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "created_at": datetime.utcnow().isoformat(),
                # --- v3: multi-tenancy fields ---
                "owner_agent_id": agent_id or "",
                "visibility": visibility,
                **chunk.metadata,
            }

            # Add domain if present in metadata
            if "domain" in chunk.metadata:
                payload["domain"] = chunk.metadata["domain"]

            point = PointStruct(
                id=chunk.chunk_id,
                vector=embedding,
                payload=payload,
            )
            points.append(point)
            stored_ids.append(chunk.chunk_id)

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch,
            )

        logger.info(
            f"Upserted {len(stored_ids)} chunks "
            f"(agent={agent_id}, visibility={visibility}, "
            f"collection={collection_name})"
        )
        return stored_ids

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 10,
        domain: Optional[str] = None,
        doc_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        include_shared: bool = True,
    ) -> List[RankedResult]:
        """Search for similar chunks with optional tenant filtering.

        Args:
            query: Search query.
            limit: Maximum results.
            domain: Filter by domain.
            doc_id: Filter by document ID.
            agent_id: Agent performing the search (v3 multi-tenancy).
                If None, no tenant filtering (v2 behavior).
            include_shared: Whether to include shared/team/public docs.

        Returns:
            List of ranked results.
        """
        client = self._get_client()

        # Generate query embedding
        query_embedding = embed_query(query, self.config.embedding)

        # Build filters
        filters = []
        if domain:
            filters.append(
                FieldCondition(key="domain", match=MatchValue(value=domain))
            )
        if doc_id:
            filters.append(
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
            )

        # v3: visibility/tenant filter
        visibility_filter = self._build_visibility_filter(agent_id, include_shared)
        if visibility_filter:
            # If we have must-filters AND a visibility filter, combine them
            if filters:
                # All must conditions + the visibility should-conditions
                combined = Filter(
                    must=filters,
                    should=visibility_filter.should,
                )
                query_filter = combined
            else:
                query_filter = visibility_filter
        else:
            query_filter = Filter(must=filters) if filters else None

        # Determine which collection to search
        # If namespaces are enabled, search the agent's own namespace
        collection_name = self._get_collection_name(agent_id)

        # If namespace doesn't exist yet (new agent), try legacy collection
        try:
            client.get_collection(collection_name)
        except Exception:
            collection_name = self._get_legacy_collection()

        # Search using query_points
        query_result = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=limit,
            query_filter=query_filter,
        )

        # Convert to RankedResult
        ranked_results = []
        for point in query_result.points:
            payload = point.payload
            ranked = RankedResult(
                content=payload.get("content", ""),
                title=payload.get("title", payload.get("doc_id", "")),
                domain=payload.get("domain"),
                source=payload.get("source", payload.get("doc_id", "")),
                semantic_score=point.score,
                rerank_score=0.0,
                attention_score=0.0,
                final_score=point.score,
                chunk_index=payload.get("chunk_index", 0),
                doc_id=payload.get("doc_id", ""),
                metadata=payload,
            )
            ranked_results.append(ranked)

        return ranked_results

    def search_with_rerank(
        self,
        query: str,
        config: Optional[RerankConfig] = None,
        domain: Optional[str] = None,
        agent_id: Optional[str] = None,
        include_shared: bool = True,
    ) -> List[RankedResult]:
        """Search with re-ranking using cross-encoder.

        Args:
            query: Search query.
            config: Re-ranking configuration.
            domain: Filter by domain.
            agent_id: Agent performing the search (v3).
            include_shared: Whether to include shared/team/public docs.

        Returns:
            Re-ranked results.
        """
        rerank_config = config or self.config.reranking

        # Initial search with more results
        initial_results = self.search(
            query=query,
            limit=rerank_config.top_k_initial,
            domain=domain,
            agent_id=agent_id,
            include_shared=include_shared,
        )

        if not initial_results or not rerank_config.enabled:
            return initial_results[:rerank_config.top_k_final]

        # Re-ranking will be done by the reranker module
        # For now, return top_k_final results
        return initial_results[:rerank_config.top_k_final]

    def search_with_attention(
        self,
        query: str,
        session_id: str,
        limit: int = 10,
        domain: Optional[str] = None,
        agent_id: Optional[str] = None,
        include_shared: bool = True,
    ) -> List[RankedResult]:
        """Search with attention-focused retrieval.

        Args:
            query: Search query.
            session_id: Session ID for context.
            limit: Maximum results.
            domain: Filter by domain.
            agent_id: Agent performing the search (v3).
            include_shared: Whether to include shared/team/public docs.

        Returns:
            Attention-focused results.
        """
        # Base search
        results = self.search(
            query,
            limit=limit * 2,
            domain=domain,
            agent_id=agent_id,
            include_shared=include_shared,
        )

        # Attention scoring will be done by the attention module
        # For now, return results with base scoring
        return results[:limit]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_document(
        self,
        doc_id: str,
        agent_id: Optional[str] = None,
    ) -> int:
        """Delete all chunks for a document.

        Args:
            doc_id: Document ID to delete.
            agent_id: Optional agent ID to scope deletion to a namespace.

        Returns:
            Number of chunks deleted.
        """
        client = self._get_client()
        collection_name = self._get_collection_name(agent_id)

        total_deleted = 0

        # If no namespace, also check legacy collection
        collections_to_check = [collection_name]
        if self.use_namespaces and agent_id:
            # Also clean up from legacy collection for migration safety
            collections_to_check.append(self._get_legacy_collection())

        for coll in collections_to_check:
            try:
                results = client.scroll(
                    collection_name=coll,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                        ]
                    ),
                    limit=10000,
                )

                if not results[0]:
                    continue

                point_ids = [point.id for point in results[0]]
                client.delete(
                    collection_name=coll,
                    points_selector=PointIdsList(points=point_ids),
                )
                total_deleted += len(point_ids)
                logger.info(f"Deleted {len(point_ids)} chunks from {coll}")
            except Exception:
                pass

        logger.info(
            f"Deleted {total_deleted} total chunks for document {doc_id}"
        )
        return total_deleted

    # ------------------------------------------------------------------
    # Update visibility
    # ------------------------------------------------------------------

    def update_chunk_visibility(
        self,
        doc_id: str,
        new_visibility: str,
        agent_id: Optional[str] = None,
    ) -> int:
        """Update visibility for all chunks of a document.

        Args:
            doc_id: Document ID.
            new_visibility: New visibility level.
            agent_id: Optional agent ID to scope to namespace.

        Returns:
            Number of chunks updated.
        """
        client = self._get_client()
        collection_name = self._get_collection_name(agent_id)

        # Find all chunks for the document
        results = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                ]
            ),
            limit=10000,
        )

        if not results[0]:
            return 0

        updated_points = []
        for point in results[0]:
            payload = dict(point.payload)
            payload["visibility"] = new_visibility
            updated_points.append(
                PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=payload,
                )
            )

        # Re-upsert with updated payloads
        batch_size = 100
        for i in range(0, len(updated_points), batch_size):
            batch = updated_points[i : i + batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch,
            )

        logger.info(
            f"Updated visibility to '{new_visibility}' for {len(updated_points)} "
            f"chunks of doc {doc_id}"
        )
        return len(updated_points)

    def update_chunk_ownership(
        self,
        doc_id: str,
        new_owner_agent_id: str,
        agent_id: Optional[str] = None,
    ) -> int:
        """Update owner_agent_id for all chunks of a document.

        Useful when transferring document ownership between agents.

        Args:
            doc_id: Document ID.
            new_owner_agent_id: New owner agent ID.
            agent_id: Current agent ID (for namespace lookup).

        Returns:
            Number of chunks updated.
        """
        client = self._get_client()
        collection_name = self._get_collection_name(agent_id)

        results = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                ]
            ),
            limit=10000,
        )

        if not results[0]:
            return 0

        updated_points = []
        for point in results[0]:
            payload = dict(point.payload)
            payload["owner_agent_id"] = new_owner_agent_id
            updated_points.append(
                PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=payload,
                )
            )

        batch_size = 100
        for i in range(0, len(updated_points), batch_size):
            batch = updated_points[i : i + batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch,
            )

        logger.info(
            f"Updated owner to '{new_owner_agent_id}' for {len(updated_points)} "
            f"chunks of doc {doc_id}"
        )
        return len(updated_points)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_collection_stats(
        self,
        agent_id: Optional[str] = None,
    ) -> CollectionStats:
        """Get collection statistics.

        Args:
            agent_id: Optional agent ID for namespace-specific stats.

        Returns:
            Collection statistics.
        """
        client = self._get_client()
        collection_name = self._get_collection_name(agent_id)

        try:
            collection_info = client.get_collection(collection_name)

            # Get unique domains and doc_ids
            domains = set()
            doc_ids = set()
            results = client.scroll(
                collection_name=collection_name,
                limit=10000,
            )
            for point in results[0]:
                if "domain" in point.payload:
                    domains.add(point.payload["domain"])
                if "doc_id" in point.payload:
                    doc_ids.add(point.payload["doc_id"])

            return CollectionStats(
                collection_name=collection_name,
                total_documents=len(doc_ids),
                total_chunks=collection_info.points_count,
                total_embeddings=collection_info.points_count,
                domains=list(domains),
                last_updated=datetime.utcnow(),
            )
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return CollectionStats(
                collection_name=collection_name,
                total_documents=0,
                total_chunks=0,
                total_embeddings=0,
                domains=[],
                last_updated=datetime.utcnow(),
            )

    def delete_collection(
        self,
        agent_id: Optional[str] = None,
    ) -> None:
        """Delete a collection.

        Args:
            agent_id: Optional agent ID for namespace-specific deletion.
        """
        client = self._get_client()
        collection_name = self._get_collection_name(agent_id)
        client.delete_collection(collection_name)
        # Remove from cache
        self._ensured_collections = {
            c for c in self._ensured_collections if not c.startswith(collection_name)
        }
        logger.info(f"Deleted collection: {collection_name}")
