"""Enhanced MCP server for gdrag v2.

Provides MCP tools for RAG operations with session management,
attention focusing, and context compression.
"""

import logging
from typing import Any, Dict, List, Optional

from mcp.server import FastMCP

from ...core.config import load_config
from ...core.pipeline import QueryPipeline
from ...models.schemas import EnhancedQueryRequest

logger = logging.getLogger(__name__)

# Global pipeline instance
_pipeline: Optional[QueryPipeline] = None


def get_pipeline() -> QueryPipeline:
    """Get or create the query pipeline instance."""
    global _pipeline
    if _pipeline is None:
        config = load_config()
        _pipeline = QueryPipeline(config)
    return _pipeline


def create_mcp_server() -> FastMCP:
    """Create and configure the MCP server.

    Returns:
        Configured FastMCP server instance.
    """
    mcp = FastMCP("gdrag-v2")

    # ========================================================================
    # Query Tools
    # ========================================================================

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

    # ========================================================================
    # Session Tools
    # ========================================================================

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

    # ========================================================================
    # Ingestion Tools
    # ========================================================================

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

    # ========================================================================
    # Health & Stats Tools
    # ========================================================================

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

    return mcp