"""Cross-encoder re-ranking for gdrag v2.

Provides relevance scoring and re-ranking of search results using
cross-encoder models for improved accuracy.
"""

import logging
from typing import List, Optional, Tuple

from ..core.config import RerankConfig
from ..models.schemas import RankedResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-ranks search results using a cross-encoder model.

    Cross-encoders process query-document pairs together for more accurate
    relevance scoring compared to bi-encoder approaches.
    """

    def __init__(self, config: Optional[RerankConfig] = None):
        self.config = config or RerankConfig()
        self._model = None

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(f"Loading cross-encoder model: {self.config.model}")
                self._model = CrossEncoder(self.config.model)
                logger.info("Cross-encoder model loaded successfully")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for re-ranking. "
                    "Install with: pip install sentence-transformers"
                )

    def compute_scores(
        self,
        query: str,
        documents: List[str],
    ) -> List[float]:
        """Compute relevance scores for query-document pairs.

        Args:
            query: Search query.
            documents: List of document texts.

        Returns:
            List of relevance scores (higher is more relevant).
        """
        if not documents:
            return []

        self._load_model()

        # Prepare pairs
        pairs = [[query, doc] for doc in documents]

        # Compute scores
        scores = self._model.predict(pairs)

        # Normalize scores to 0-1 range
        min_score = float(min(scores))
        max_score = float(max(scores))

        if max_score > min_score:
            normalized = [
                (float(s) - min_score) / (max_score - min_score)
                for s in scores
            ]
        else:
            normalized = [0.5] * len(scores)

        return normalized

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Re-rank documents by relevance to query.

        Args:
            query: Search query.
            documents: List of document texts.
            top_k: Number of top results to return. If None, uses config.

        Returns:
            List of (original_index, score) tuples, sorted by score descending.
        """
        if not documents:
            return []

        scores = self.compute_scores(query, documents)

        # Create indexed score pairs
        indexed_scores = list(enumerate(scores))

        # Sort by score descending
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top_k
        k = top_k or self.config.top_k_final
        return indexed_scores[:k]

    def rerank_results(
        self,
        query: str,
        results: List[RankedResult],
        top_k: Optional[int] = None,
    ) -> List[RankedResult]:
        """Re-rank RankedResult objects.

        Args:
            query: Search query.
            results: List of ranked results to re-rank.
            top_k: Number of top results to return.

        Returns:
            Re-ranked results with updated scores.
        """
        if not results:
            return []

        # Extract document texts
        documents = [r.content for r in results]

        # Get re-ranking scores
        reranked_indices = self.rerank(query, documents, top_k)

        # Rebuild results with new scores
        reranked_results = []
        for original_idx, rerank_score in reranked_indices:
            original = results[original_idx]

            # Update scores
            updated = RankedResult(
                content=original.content,
                title=original.title,
                domain=original.domain,
                source=original.source,
                semantic_score=original.semantic_score,
                rerank_score=rerank_score,
                attention_score=original.attention_score,
                final_score=rerank_score,  # Use rerank score as final
                chunk_index=original.chunk_index,
                doc_id=original.doc_id,
                metadata=original.metadata,
            )
            reranked_results.append(updated)

        return reranked_results

    def batch_rerank(
        self,
        queries: List[str],
        documents: List[List[str]],
        top_k: Optional[int] = None,
    ) -> List[List[Tuple[int, float]]]:
        """Re-rank documents for multiple queries.

        Args:
            queries: List of search queries.
            documents: List of document lists, one per query.
            top_k: Number of top results per query.

        Returns:
            List of re-ranking results per query.
        """
        results = []
        for query, docs in zip(queries, documents):
            result = self.rerank(query, docs, top_k)
            results.append(result)
        return results


class SimpleReranker:
    """Simple keyword-based re-ranker as fallback.

    Uses basic text matching when cross-encoder is not available.
    """

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """Re-rank using keyword overlap.

        Args:
            query: Search query.
            documents: List of document texts.
            top_k: Number of top results.

        Returns:
            List of (original_index, score) tuples.
        """
        if not documents:
            return []

        # Tokenize query
        query_tokens = set(query.lower().split())

        # Score each document
        scores = []
        for i, doc in enumerate(documents):
            doc_tokens = set(doc.lower().split())

            # Jaccard similarity
            intersection = len(query_tokens & doc_tokens)
            union = len(query_tokens | doc_tokens)

            score = intersection / union if union > 0 else 0.0
            scores.append((i, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def rerank_results(
        self,
        query: str,
        results: List[RankedResult],
        top_k: int = 10,
    ) -> List[RankedResult]:
        """Re-rank RankedResult objects.

        Args:
            query: Search query.
            results: List of ranked results.
            top_k: Number of top results.

        Returns:
            Re-ranked results.
        """
        if not results:
            return []

        documents = [r.content for r in results]
        reranked_indices = self.rerank(query, documents, top_k)

        reranked_results = []
        for original_idx, score in reranked_indices:
            original = results[original_idx]
            updated = RankedResult(
                content=original.content,
                title=original.title,
                domain=original.domain,
                source=original.source,
                semantic_score=original.semantic_score,
                rerank_score=score,
                attention_score=original.attention_score,
                final_score=score,
                chunk_index=original.chunk_index,
                doc_id=original.doc_id,
                metadata=original.metadata,
            )
            reranked_results.append(updated)

        return reranked_results


class RerankerFactory:
    """Factory for creating reranker instances."""

    _instance = None

    @classmethod
    def get_reranker(
        cls,
        config: Optional[RerankConfig] = None,
        force_simple: bool = False,
    ):
        """Get or create a reranker instance.

        Args:
            config: Re-ranking configuration.
            force_simple: Force use of simple reranker.

        Returns:
            Reranker instance.
        """
        if force_simple:
            return SimpleReranker()

        if cls._instance is None:
            try:
                cls._instance = CrossEncoderReranker(config)
                # Test that model loads
                cls._instance._load_model()
            except Exception as e:
                logger.warning(
                    f"Failed to load cross-encoder, using simple reranker: {e}"
                )
                cls._instance = SimpleReranker()

        return cls._instance