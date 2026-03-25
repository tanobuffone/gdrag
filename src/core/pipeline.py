"""Query pipeline orchestration for gdrag v2.

Orchestrates the full RAG flow: embed → retrieve → rerank → compress → respond.
"""

import logging
import time
from typing import Dict, List, Optional

from ..core.attention import AttentionManager
from ..core.compressor import ContextCompressor
from ..core.config import AppConfig
from ..core.embeddings import embed_query
from ..core.graph_store import EnhancedGraphStore
from ..core.reranker import RerankerFactory
from ..core.relational_store import EnhancedRelationalStore
from ..core.session_manager import SessionManager
from ..core.vector_store import EnhancedVectorStore
from ..models.schemas import (
    EnhancedQueryRequest,
    EnhancedQueryResponse,
    QueryRecord,
    RankedResult,
    ResponseMetadata,
)

logger = logging.getLogger(__name__)


class QueryPipeline:
    """Orchestrates the complete RAG query flow.

    Coordinates embedding, retrieval, re-ranking, compression,
    and attention-focused results.
    """

    def __init__(self, config: AppConfig):
        self.config = config

        # Initialize components
        self.vector_store = EnhancedVectorStore(config)
        self.graph_store = EnhancedGraphStore(config)
        self.relational_store = EnhancedRelationalStore(config)
        self.session_manager = SessionManager(config, self.relational_store)
        self.attention_manager = AttentionManager(config.attention)
        self.compressor = ContextCompressor(config.compression)
        self.reranker = RerankerFactory.get_reranker(config.reranking)

    def execute(self, request: EnhancedQueryRequest) -> EnhancedQueryResponse:
        """Execute the full query pipeline.

        Args:
            request: Enhanced query request.

        Returns:
            Enhanced query response with results and metadata.
        """
        start_time = time.time()
        logger.info(f"Executing query: {request.query[:100]}...")

        # Step 1: Get session context if requested
        session_context = ""
        session_context_used = False
        if request.session_id and request.use_context:
            session_context = self.session_manager.get_session_context(
                request.session_id
            )
            if session_context:
                session_context_used = True
                logger.info(f"Using session context: {len(session_context)} chars")

        # Step 2: Enhance query with session context
        enhanced_query = request.query
        if session_context:
            enhanced_query = f"{request.query}\n\nContext from previous queries:\n{session_context}"

        # Step 3: Semantic search in vector store
        initial_limit = self.config.reranking.top_k_initial
        results = self.vector_store.search(
            query=enhanced_query,
            limit=initial_limit,
            domain=request.domain,
        )
        logger.info(f"Vector search returned {len(results)} results")

        # Step 4: Enrich with graph context
        graph_context = self._get_graph_context(request.query)
        if graph_context:
            # Add graph context to results
            for result in results[:3]:  # Enrich top 3
                result.metadata["graph_context"] = graph_context

        # Step 5: Enrich with relational context
        relational_results = self._get_relational_context(
            request.query, request.domain
        )
        if relational_results:
            # Add to results
            for rr in relational_results[:2]:
                ranked = RankedResult(
                    content=rr.get("content", ""),
                    title=rr.get("title", ""),
                    domain=rr.get("domain"),
                    source="postgres",
                    semantic_score=0.7,  # Default score for relational
                    rerank_score=0.0,
                    attention_score=0.0,
                    final_score=0.7,
                    chunk_index=0,
                    doc_id=rr.get("id", ""),
                    metadata=rr,
                )
                results.append(ranked)

        # Step 6: Re-rank results
        if self.config.reranking.enabled and results:
            results = self.reranker.rerank_results(
                query=request.query,
                results=results,
                top_k=self.config.reranking.top_k_final,
            )
            logger.info(f"Re-ranked to {len(results)} results")

        # Step 7: Apply attention focusing
        if request.attention_focus and results:
            results = self.attention_manager.get_focused_results(
                results=results,
                session_id=request.session_id,
                limit=request.limit,
            )
            logger.info(f"Applied attention focusing")

        # Step 8: Compress results if requested
        compressed_context = None
        tokens_saved = 0
        if request.compress_results and results:
            original_count = len(results)
            results = self.compressor.compress_results(results)
            tokens_saved = (original_count - len(results)) * 100  # Rough estimate
            logger.info(f"Compressed results: {original_count} → {len(results)}")

            # Generate compressed context summary
            if results:
                compressed_context = self.compressor.summarize_text(
                    " ".join(r.content for r in results[:3]),
                    max_tokens=self.config.compression.max_summary_tokens,
                )

        # Step 9: Limit final results
        results = results[:request.limit]

        # Step 10: Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Step 11: Build response
        response = EnhancedQueryResponse(
            query=request.query,
            results=results,
            compressed_context=compressed_context,
            session_context_used=session_context_used,
            attention_scores={
                r.doc_id: r.attention_score for r in results
            },
            metadata=ResponseMetadata(
                total_results=len(results),
                reranked=self.config.reranking.enabled,
                compressed=request.compress_results,
                session_id=request.session_id,
                embedding_provider=self.config.embedding.provider.value,
                processing_time_ms=processing_time_ms,
                tokens_saved=tokens_saved,
            ),
        )

        # Step 12: Update session if applicable
        if request.session_id:
            query_record = QueryRecord(
                query=request.query,
                results_ids=[r.doc_id for r in results],
                relevance_scores=[r.final_score for r in results],
                processing_time_ms=processing_time_ms,
            )
            self.session_manager.update_session(request.session_id, query_record)

        logger.info(
            f"Query completed in {processing_time_ms:.1f}ms, "
            f"{len(results)} results"
        )

        return response

    def _get_graph_context(self, query: str) -> Optional[str]:
        """Get context from knowledge graph.

        Args:
            query: Search query.

        Returns:
            Graph context string or None.
        """
        try:
            # Extract concepts from query
            concepts = self.graph_store.extract_concepts(query)

            if concepts:
                # Find related concepts
                related = self.graph_store.find_related_concepts(
                    concepts[:5], limit=5
                )

                if related:
                    context_parts = [f"Related concepts: {', '.join(concepts[:5])}"]
                    for rel in related[:3]:
                        context_parts.append(
                            f"{rel.source_concept} → {rel.target_concept}"
                        )
                    return ". ".join(context_parts)
        except Exception as e:
            logger.warning(f"Graph context error: {e}")

        return None

    def _get_relational_context(
        self,
        query: str,
        domain: Optional[str] = None,
    ) -> List[Dict]:
        """Get context from relational store.

        Args:
            query: Search query.
            domain: Optional domain filter.

        Returns:
            List of result dictionaries.
        """
        try:
            results = self.relational_store.full_text_search(
                query_text=query,
                domain=domain,
                limit=3,
            )
            return results
        except Exception as e:
            logger.warning(f"Relational context error: {e}")
            return []

    def execute_with_session(
        self,
        query: str,
        session_id: str,
        agent_id: str = "default",
        domain: Optional[str] = None,
        limit: int = 10,
    ) -> EnhancedQueryResponse:
        """Execute query with session management.

        Args:
            query: Query text.
            session_id: Session ID.
            agent_id: Agent identifier.
            domain: Optional domain filter.
            limit: Maximum results.

        Returns:
            Enhanced query response.
        """
        # Ensure session exists
        session = self.session_manager.get_session(session_id)
        if session is None:
            self.session_manager.create_session(agent_id, session_id)

        request = EnhancedQueryRequest(
            query=query,
            domain=domain,
            session_id=session_id,
            use_context=True,
            compress_results=False,
            attention_focus=True,
            limit=limit,
        )

        return self.execute(request)

    def execute_compressed(
        self,
        query: str,
        domain: Optional[str] = None,
        limit: int = 10,
    ) -> EnhancedQueryResponse:
        """Execute query with compression enabled.

        Args:
            query: Query text.
            domain: Optional domain filter.
            limit: Maximum results.

        Returns:
            Compressed query response.
        """
        request = EnhancedQueryRequest(
            query=query,
            domain=domain,
            session_id=None,
            use_context=False,
            compress_results=True,
            attention_focus=True,
            limit=limit,
        )

        return self.execute(request)

    def ingest_document(
        self,
        content: str,
        title: str = "",
        domain: Optional[str] = None,
        source: str = "api",
        doc_id: Optional[str] = None,
    ) -> Dict:
        """Ingest a document into the system.

        Args:
            content: Document content.
            title: Document title.
            domain: Optional domain.
            source: Source identifier.
            doc_id: Optional document ID.

        Returns:
            Ingestion statistics.
        """
        from uuid import uuid4

        from ..core.chunker import DocumentChunker
        from ..models.schemas import KnowledgeItem

        doc_id = doc_id or str(uuid4())
        logger.info(f"Ingesting document: {doc_id}")

        # Step 1: Chunk the document
        chunker = DocumentChunker(self.config.chunking)
        chunks = chunker.chunk(content, doc_id)

        # Add metadata to chunks
        for chunk in chunks:
            chunk.metadata.update({
                "title": title,
                "domain": domain,
                "source": source,
            })

        # Step 2: Store chunks in vector store
        chunk_ids = self.vector_store.upsert_chunks(chunks)

        # Step 3: Extract and store concepts in graph
        concepts = self.graph_store.extract_concepts(content)
        if concepts:
            self.graph_store.store_concepts(doc_id, concepts, domain)

        # Step 4: Store knowledge entry in relational store
        knowledge_item = KnowledgeItem(
            id=doc_id,
            title=title,
            content=content[:5000],  # Limit for storage
            domain=domain,
            source=source,
            source_type="document",
        )
        self.relational_store.store_knowledge_entry(knowledge_item)

        stats = {
            "doc_id": doc_id,
            "chunks_created": len(chunks),
            "concepts_extracted": len(concepts),
            "chunk_ids": chunk_ids,
        }

        logger.info(f"Ingested document: {stats}")
        return stats

    def get_health(self) -> Dict:
        """Get pipeline health status.

        Returns:
            Health status dictionary.
        """
        health = {
            "status": "healthy",
            "components": {},
        }

        # Check vector store
        try:
            stats = self.vector_store.get_collection_stats()
            health["components"]["vector_store"] = {
                "status": "healthy",
                "documents": stats.total_documents,
                "chunks": stats.total_chunks,
            }
        except Exception as e:
            health["components"]["vector_store"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health["status"] = "degraded"

        # Check graph store
        try:
            graph_stats = self.graph_store.get_concept_stats()
            health["components"]["graph_store"] = {
                "status": "healthy",
                "concepts": graph_stats.get("concept_count", 0),
            }
        except Exception as e:
            health["components"]["graph_store"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health["status"] = "degraded"

        # Check session manager
        try:
            session_stats = self.session_manager.get_stats()
            health["components"]["session_manager"] = {
                "status": "healthy",
                **session_stats,
            }
        except Exception as e:
            health["components"]["session_manager"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health["status"] = "degraded"

        # Check attention manager
        try:
            attention_stats = self.attention_manager.get_stats()
            health["components"]["attention_manager"] = {
                "status": "healthy",
                **attention_stats,
            }
        except Exception as e:
            health["components"]["attention_manager"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        return health