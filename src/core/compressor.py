"""Context compression for gdrag v2.

Provides summarization, extraction, and deduplication of context
to optimize token usage and relevance.
"""

import logging
import re
from typing import Dict, List, Optional, Set

from ..core.config import CompressionConfig
from ..models.schemas import RankedResult

logger = logging.getLogger(__name__)


class TextSummarizer:
    """Extractive text summarizer.

    Uses sentence scoring to extract key sentences without
    requiring external LLM calls.
    """

    def __init__(self):
        self._stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "shall", "must",
            "and", "or", "but", "not", "no", "nor", "for", "yet", "so",
            "at", "by", "to", "from", "in", "on", "of", "with", "about",
        }

    def _score_sentence(self, sentence: str, word_freq: Dict[str, int]) -> float:
        """Score a sentence based on word frequency.

        Args:
            sentence: Sentence to score.
            word_freq: Word frequency dictionary.

        Returns:
            Sentence score.
        """
        words = sentence.lower().split()
        if not words:
            return 0.0

        # Filter stop words
        content_words = [w for w in words if w not in self._stop_words]

        if not content_words:
            return 0.0

        # Calculate score as average word frequency
        score = sum(word_freq.get(w, 0) for w in content_words) / len(content_words)

        return score

    def _build_word_freq(self, text: str) -> Dict[str, int]:
        """Build word frequency dictionary.

        Args:
            text: Text to analyze.

        Returns:
            Word frequency dictionary.
        """
        words = re.findall(r"\b\w+\b", text.lower())
        freq: Dict[str, int] = {}
        for word in words:
            if word not in self._stop_words:
                freq[word] = freq.get(word, 0) + 1
        return freq

    def summarize(
        self,
        text: str,
        max_sentences: int = 3,
        max_tokens: int = 500,
    ) -> str:
        """Summarize text by extracting key sentences.

        Args:
            text: Text to summarize.
            max_sentences: Maximum sentences to include.
            max_tokens: Maximum tokens in summary.

        Returns:
            Summarized text.
        """
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= max_sentences:
            return text

        # Build word frequency
        word_freq = self._build_word_freq(text)

        # Score sentences
        scored = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, word_freq)

            # Boost first sentences
            if i < 3:
                score *= 1.5

            # Boost sentences with numbers or proper nouns
            if re.search(r"\b[A-Z][a-z]+\b", sentence):
                score *= 1.2
            if re.search(r"\d+", sentence):
                score *= 1.1

            scored.append((i, score, sentence))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Select top sentences (maintain original order)
        selected = sorted(scored[:max_sentences], key=lambda x: x[0])

        # Build summary
        summary_parts = []
        token_count = 0
        for _, _, sentence in selected:
            sentence_tokens = len(sentence.split()) * 1.3  # Rough estimate
            if token_count + sentence_tokens > max_tokens:
                break
            summary_parts.append(sentence)
            token_count += sentence_tokens

        return " ".join(summary_parts)

    def extract_key_points(
        self,
        text: str,
        num_points: int = 5,
    ) -> List[str]:
        """Extract key points from text.

        Args:
            text: Text to extract from.
            num_points: Number of key points.

        Returns:
            List of key point strings.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        word_freq = self._build_word_freq(text)

        # Score and select top sentences
        scored = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, word_freq)
            if i < 3:
                score *= 1.5
            scored.append((score, sentence))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [s for _, s in scored[:num_points]]


class ResultDeduplicator:
    """Deduplicates similar search results."""

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Similarity score (0-1).
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def deduplicate(
        self,
        results: List[RankedResult],
        max_similar: int = 1,
    ) -> List[RankedResult]:
        """Remove duplicate or highly similar results.

        Args:
            results: List of ranked results.
            max_similar: Maximum similar results to keep.

        Returns:
            Deduplicated results.
        """
        if not results:
            return []

        deduplicated: List[RankedResult] = []
        seen_contents: List[str] = []

        for result in results:
            is_duplicate = False

            for seen in seen_contents:
                similarity = self._jaccard_similarity(result.content, seen)
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(result)
                seen_contents.append(result.content)

        return deduplicated

    def merge_similar(
        self,
        results: List[RankedResult],
        threshold: float = 0.7,
    ) -> List[RankedResult]:
        """Merge similar results into combined entries.

        Args:
            results: List of ranked results.
            threshold: Similarity threshold for merging.

        Returns:
            Merged results.
        """
        if not results:
            return []

        merged: List[RankedResult] = []
        used_indices: Set[int] = set()

        for i, result in enumerate(results):
            if i in used_indices:
                continue

            # Find similar results
            similar = [result]
            for j, other in enumerate(results[i + 1 :], start=i + 1):
                if j in used_indices:
                    continue

                similarity = self._jaccard_similarity(result.content, other.content)
                if similarity >= threshold:
                    similar.append(other)
                    used_indices.add(j)

            # Merge if multiple similar found
            if len(similar) > 1:
                # Take highest scored as base
                base = max(similar, key=lambda r: r.final_score)

                # Combine content (take unique parts)
                combined_content = base.content
                for s in similar:
                    if s.content not in combined_content:
                        combined_content += f"\n\n{s.content}"

                merged_result = RankedResult(
                    content=combined_content[:2000],  # Limit size
                    title=base.title,
                    domain=base.domain,
                    source=base.source,
                    semantic_score=max(r.semantic_score for r in similar),
                    rerank_score=max(r.rerank_score for r in similar),
                    attention_score=max(r.attention_score for r in similar),
                    final_score=max(r.final_score for r in similar),
                    chunk_index=base.chunk_index,
                    doc_id=base.doc_id,
                    metadata={
                        **base.metadata,
                        "merged_from": [r.doc_id for r in similar],
                    },
                )
                merged.append(merged_result)
            else:
                merged.append(result)

        return merged


class ContextCompressor:
    """Main context compression class.

    Combines summarization, extraction, and deduplication.
    """

    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        self.summarizer = TextSummarizer()
        self.deduplicator = ResultDeduplicator()

    def compress_results(
        self,
        results: List[RankedResult],
        ratio: Optional[float] = None,
    ) -> List[RankedResult]:
        """Compress a set of results.

        Args:
            results: List of ranked results.
            ratio: Target compression ratio. If None, uses config.

        Returns:
            Compressed results.
        """
        if not results or not self.config.enabled:
            return results

        target_ratio = ratio or self.config.compression_ratio

        # Step 1: Deduplicate
        deduplicated = self.deduplicator.deduplicate(results)

        # Step 2: Merge similar
        merged = self.deduplicator.merge_similar(
            deduplicated,
            threshold=0.7,
        )

        # Step 3: Limit to target count
        target_count = max(1, int(len(results) * target_ratio))
        limited = merged[:target_count]

        # Step 4: Compress content of each result
        compressed = []
        for result in limited:
            if len(result.content) > 1000:  # Only compress long content
                summary = self.summarizer.summarize(
                    result.content,
                    max_sentences=3,
                    max_tokens=self.config.max_summary_tokens,
                )
                compressed_result = RankedResult(
                    content=summary,
                    title=result.title,
                    domain=result.domain,
                    source=result.source,
                    semantic_score=result.semantic_score,
                    rerank_score=result.rerank_score,
                    attention_score=result.attention_score,
                    final_score=result.final_score,
                    chunk_index=result.chunk_index,
                    doc_id=result.doc_id,
                    metadata={
                        **result.metadata,
                        "compressed": True,
                        "original_length": len(result.content),
                    },
                )
                compressed.append(compressed_result)
            else:
                compressed.append(result)

        logger.info(
            f"Compressed {len(results)} results to {len(compressed)} "
            f"(ratio: {len(compressed)/len(results):.2f})"
        )

        return compressed

    def summarize_text(
        self,
        text: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Summarize a text.

        Args:
            text: Text to summarize.
            max_tokens: Maximum tokens. If None, uses config.

        Returns:
            Summarized text.
        """
        max_tok = max_tokens or self.config.max_summary_tokens
        return self.summarizer.summarize(text, max_tokens=max_tok)

    def extract_key_points(
        self,
        text: str,
        num_points: int = 5,
    ) -> List[str]:
        """Extract key points from text.

        Args:
            text: Text to extract from.
            num_points: Number of key points.

        Returns:
            List of key points.
        """
        return self.summarizer.extract_key_points(text, num_points)

    def merge_similar_results(
        self,
        results: List[RankedResult],
        threshold: Optional[float] = None,
    ) -> List[RankedResult]:
        """Merge similar results.

        Args:
            results: List of ranked results.
            threshold: Similarity threshold. If None, uses 0.7.

        Returns:
            Merged results.
        """
        thresh = threshold or 0.7
        return self.deduplicator.merge_similar(results, thresh)