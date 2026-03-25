"""Enhanced Qdrant vector store operations for gdrag v2.

Provides chunked storage, attention-focused search, and re-ranking integration.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from ..core.config import AppConfig, RerankConfig
from ..core.embeddings import embed_query, embed_texts
from ..models.schemas import Chunk, CollectionStats, RankedResult

logger = logging.getLogger(__name__)


class EnhancedVectorStore:
    """Enhanced Qdrant operations with chunked storage and attention support."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._client: Optional[QdrantClient] = None
        self._collection_name = config.database.qdrant_collection

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

    def ensure_collection(self, vector_size: int) -> None:
        """Ensure collection exists with correct configuration.

        Args:
            vector_size: Size of embedding vectors.
        """
        client = self._get_client()

        try:
            collection_info = client.get_collection(self._collection_name)
            current_size = collection_info.config.params.vectors.size

            if current_size != vector_size:
                logger.warning(
                    f"Collection vector size mismatch: {current_size} vs {vector_size}. "
                    f"Recreating collection."
                )
                client.delete_collection(self._collection_name)
                self._create_collection(vector_size)
        except Exception:
            logger.info(f"Creating collection: {self._collection_name}")
            self._create_collection(vector_size)

    def _create_collection(self, vector_size: int) -> None:
        """Create Qdrant collection.

        Args:
            vector_size: Size of embedding vectors.
        """
        client = self._get_client()
        client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Collection created with vector size: {vector_size}")

    def upsert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: Optional[Dict[str, List[float]]] = None,
    ) -> List[str]:
        """Store document chunks with embeddings.

        Args:
            chunks: List of chunks to store.
            embeddings: Optional pre-computed embeddings keyed by chunk_id.

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

        # Ensure collection exists
        if embeddings:
            sample_embedding = next(iter(embeddings.values()))
            self.ensure_collection(len(sample_embedding))

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
                collection_name=self._collection_name,
                points=batch,
            )

        logger.info(f"Upserted {len(stored_ids)} chunks")
        return stored_ids

    def search(
        self,
        query: str,
        limit: int = 10,
        domain: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> List[RankedResult]:
        """Search for similar chunks.

        Args:
            query: Search query.
            limit: Maximum results.
            domain: Filter by domain.
            doc_id: Filter by document ID.

        Returns:
            List of ranked results.
        """
        client = self._get_client()

        # Generate query embedding
        query_embedding = embed_query(query, self.config.embedding)

        # Build filter
        filters = []
        if domain:
            filters.append(
                FieldCondition(key="domain", match=MatchValue(value=domain))
            )
        if doc_id:
            filters.append(
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
            )

        query_filter = Filter(must=filters) if filters else None

        # Search
        results = client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=query_filter,
        )

        # Convert to RankedResult
        ranked_results = []
        for result in results:
            payload = result.payload
            ranked = RankedResult(
                content=payload.get("content", ""),
                title=payload.get("title", payload.get("doc_id", "")),
                domain=payload.get("domain"),
                source=payload.get("source", payload.get("doc_id", "")),
                semantic_score=result.score,
                rerank_score=0.0,
                attention_score=0.0,
                final_score=result.score,
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
    ) -> List[RankedResult]:
        """Search with re-ranking using cross-encoder.

        Args:
            query: Search query.
            config: Re-ranking configuration.
            domain: Filter by domain.

        Returns:
            Re-ranked results.
        """
        rerank_config = config or self.config.reranking

        # Initial search with more results
        initial_results = self.search(
            query=query,
            limit=rerank_config.top_k_initial,
            domain=domain,
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
    ) -> List[RankedResult]:
        """Search with attention-focused retrieval.

        Args:
            query: Search query.
            session_id: Session ID for context.
            limit: Maximum results.
            domain: Filter by domain.

        Returns:
            Attention-focused results.
        """
        # Base search
        results = self.search(query, limit=limit * 2, domain=domain)

        # Attention scoring will be done by the attention module
        # For now, return results with base scoring
        return results[:limit]

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks for a document.

        Args:
            doc_id: Document ID to delete.

        Returns:
            Number of chunks deleted.
        """
        client = self._get_client()

        # Find all chunks for document
        results = client.scroll(
            collection_name=self._collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                ]
            ),
            limit=10000,
        )

        if not results[0]:
            return 0

        # Delete points
        point_ids = [point.id for point in results[0]]
        client.delete(
            collection_name=self._collection_name,
            points_selector=point_ids,
        )

        logger.info(f"Deleted {len(point_ids)} chunks for document {doc_id}")
        return len(point_ids)

    def get_collection_stats(self) -> CollectionStats:
        """Get collection statistics.

        Returns:
            Collection statistics.
        """
        client = self._get_client()

        try:
            collection_info = client.get_collection(self._collection_name)

            # Get unique domains
            domains = set()
            results = client.scroll(
                collection_name=self._collection_name,
                limit=10000,
            )
            for point in results[0]:
                if "domain" in point.payload:
                    domains.add(point.payload["domain"])

            # Get unique documents
            doc_ids = set()
            for point in results[0]:
                if "doc_id" in point.payload:
                    doc_ids.add(point.payload["doc_id"])

            return CollectionStats(
                collection_name=self._collection_name,
                total_documents=len(doc_ids),
                total_chunks=collection_info.points_count,
                total_embeddings=collection_info.points_count,
                domains=list(domains),
                last_updated=datetime.utcnow(),
            )
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return CollectionStats(
                collection_name=self._collection_name,
                total_documents=0,
                total_chunks=0,
                total_embeddings=0,
                domains=[],
                last_updated=datetime.utcnow(),
            )

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        client = self._get_client()
        client.delete_collection(self._collection_name)
        logger.info(f"Deleted collection: {self._collection_name}")