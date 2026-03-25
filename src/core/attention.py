"""Focused attention mechanism for gdrag v2.

Provides temporal decay, relevance scoring, and diversity filtering
for context-aware retrieval.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ..core.config import AttentionConfig
from ..models.schemas import RankedResult

logger = logging.getLogger(__name__)


class AttentionItem:
    """An item being tracked for attention."""

    def __init__(
        self,
        item_id: str,
        content: str,
        initial_score: float = 1.0,
    ):
        self.item_id = item_id
        self.content = content
        self.score = initial_score
        self.access_count = 1
        self.last_accessed = datetime.utcnow()
        self.created_at = datetime.utcnow()

    def access(self, score_boost: float = 0.1) -> None:
        """Record an access to this item.

        Args:
            score_boost: Score increase on access.
        """
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
        self.score = min(1.0, self.score + score_boost)


class AttentionManager:
    """Manages focused attention for context-aware retrieval.

    Combines temporal decay, relevance scoring, and diversity
    to maintain focused context.
    """

    def __init__(self, config: Optional[AttentionConfig] = None):
        self.config = config or AttentionConfig()
        self._items: Dict[str, AttentionItem] = {}
        self._session_scores: Dict[str, Dict[str, float]] = {}

    def _apply_temporal_decay(
        self,
        score: float,
        age_hours: float,
    ) -> float:
        """Apply temporal decay to a score.

        Args:
            score: Original score.
            age_hours: Age in hours since last access.

        Returns:
            Decayed score.
        """
        decay = math.pow(self.config.decay_factor, age_hours)
        return score * decay

    def calculate_attention_score(
        self,
        result: RankedResult,
        session_id: Optional[str] = None,
    ) -> float:
        """Calculate attention score for a result.

        Args:
            result: Ranked result to score.
            session_id: Optional session ID for context.

        Returns:
            Attention score (0-1).
        """
        # Get item from tracking
        item = self._items.get(result.doc_id)

        if item is None:
            # New item, use semantic score as base
            base_score = result.semantic_score
            recency_score = 1.0  # New items get full recency
            frequency_score = 0.0
        else:
            base_score = item.score

            # Calculate recency (hours since last access)
            age_hours = (datetime.utcnow() - item.last_accessed).total_seconds() / 3600
            recency_score = self._apply_temporal_decay(1.0, age_hours)

            # Frequency score (normalized by max expected accesses)
            frequency_score = min(1.0, item.access_count / 10.0)

        # Relevance score from semantic search
        relevance_score = result.semantic_score

        # Combine scores with weights
        attention_score = (
            self.config.recency_weight * recency_score
            + self.config.relevance_weight * relevance_score
            + self.config.diversity_weight * frequency_score
        )

        # Apply session-specific boost if available
        if session_id and session_id in self._session_scores:
            session_score = self._session_scores[session_id].get(result.doc_id, 0.0)
            attention_score = 0.7 * attention_score + 0.3 * session_score

        return min(1.0, max(0.0, attention_score))

    def update_attention(
        self,
        result: RankedResult,
        session_id: Optional[str] = None,
        feedback: Optional[float] = None,
    ) -> None:
        """Update attention for a result after access.

        Args:
            result: Result that was accessed.
            session_id: Optional session ID.
            feedback: Optional user feedback (0-1).
        """
        item_id = result.doc_id

        if item_id not in self._items:
            self._items[item_id] = AttentionItem(
                item_id=item_id,
                content=result.content,
                initial_score=result.semantic_score,
            )
        else:
            self._items[item_id].access()

        # Update session scores
        if session_id:
            if session_id not in self._session_scores:
                self._session_scores[session_id] = {}

            current = self._session_scores[session_id].get(item_id, 0.5)
            if feedback is not None:
                # Use feedback to adjust score
                new_score = 0.7 * current + 0.3 * feedback
            else:
                # Boost score slightly on access
                new_score = min(1.0, current + 0.1)

            self._session_scores[session_id][item_id] = new_score

        # Maintain max items limit
        if len(self._items) > self.config.max_attention_items:
            self._cleanup_old_items()

    def _cleanup_old_items(self) -> None:
        """Remove oldest items to stay within limit."""
        # Sort by last accessed
        sorted_items = sorted(
            self._items.items(),
            key=lambda x: x[1].last_accessed,
        )

        # Remove oldest 20%
        remove_count = len(sorted_items) // 5
        for item_id, _ in sorted_items[:remove_count]:
            del self._items[item_id]

    def diversity_filter(
        self,
        results: List[RankedResult],
        max_similar: int = 2,
    ) -> List[RankedResult]:
        """Filter results for diversity.

        Ensures we don't return too many similar results.

        Args:
            results: List of ranked results.
            max_similar: Maximum similar results per document.

        Returns:
            Diversified results.
        """
        if not results:
            return []

        doc_counts: Dict[str, int] = {}
        diversified = []

        for result in results:
            doc_id = result.doc_id
            count = doc_counts.get(doc_id, 0)

            if count < max_similar:
                diversified.append(result)
                doc_counts[doc_id] = count + 1

        return diversified

    def get_focused_results(
        self,
        results: List[RankedResult],
        session_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[RankedResult]:
        """Get results with attention-focused scoring.

        Args:
            results: List of ranked results.
            session_id: Optional session ID.
            limit: Maximum results.

        Returns:
            Attention-focused results.
        """
        if not results:
            return []

        # Calculate attention scores
        scored_results = []
        for result in results:
            attention_score = self.calculate_attention_score(result, session_id)

            updated = RankedResult(
                content=result.content,
                title=result.title,
                domain=result.domain,
                source=result.source,
                semantic_score=result.semantic_score,
                rerank_score=result.rerank_score,
                attention_score=attention_score,
                final_score=(
                    0.4 * result.semantic_score
                    + 0.3 * result.rerank_score
                    + 0.3 * attention_score
                ),
                chunk_index=result.chunk_index,
                doc_id=result.doc_id,
                metadata=result.metadata,
            )
            scored_results.append(updated)

        # Sort by final score
        scored_results.sort(key=lambda r: r.final_score, reverse=True)

        # Apply diversity filter
        diversified = self.diversity_filter(scored_results)

        # Update attention for returned results
        for result in diversified[:limit]:
            self.update_attention(result, session_id)

        return diversified[:limit]

    def get_session_context(
        self,
        session_id: str,
        max_items: int = 10,
    ) -> List[Tuple[str, float]]:
        """Get most attended items for a session.

        Args:
            session_id: Session ID.
            max_items: Maximum items to return.

        Returns:
            List of (item_id, score) tuples.
        """
        if session_id not in self._session_scores:
            return []

        scores = self._session_scores[session_id]
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_scores[:max_items]

    def clear_session(self, session_id: str) -> None:
        """Clear attention data for a session.

        Args:
            session_id: Session ID to clear.
        """
        if session_id in self._session_scores:
            del self._session_scores[session_id]

    def get_stats(self) -> Dict:
        """Get attention statistics.

        Returns:
            Statistics dictionary.
        """
        return {
            "tracked_items": len(self._items),
            "active_sessions": len(self._session_scores),
            "max_items": self.config.max_attention_items,
        }