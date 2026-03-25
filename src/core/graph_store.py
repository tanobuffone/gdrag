"""Enhanced Memgraph graph store operations for gdrag v2.

Provides concept extraction, knowledge graph building, and graph traversal.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from ..core.config import AppConfig
from ..models.schemas import ConceptRelation

logger = logging.getLogger(__name__)


class EnhancedGraphStore:
    """Enhanced Memgraph operations with concept extraction and graph traversal."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._driver = None

    def _get_driver(self):
        """Get or create Memgraph driver."""
        if self._driver is None:
            uri = f"bolt://{self.config.database.memgraph_host}:{self.config.database.memgraph_port}"
            self._driver = GraphDatabase.driver(
                uri,
                auth=(
                    self.config.database.memgraph_user,
                    self.config.database.memgraph_password,
                ),
            )
            logger.info(f"Connected to Memgraph at {uri}")
        return self._driver

    def close(self) -> None:
        """Close the driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def _execute_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """Execute a Cypher query.

        Args:
            query: Cypher query string.
            parameters: Query parameters.

        Returns:
            List of result records as dictionaries.
        """
        driver = self._get_driver()

        try:
            with driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except ServiceUnavailable as e:
            logger.error(f"Memgraph connection error: {e}")
            raise
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise

    def extract_concepts(self, text: str, min_length: int = 3) -> List[str]:
        """Extract potential concepts from text.

        Uses simple NLP techniques to extract noun phrases and key terms.

        Args:
            text: Text to extract concepts from.
            min_length: Minimum concept length.

        Returns:
            List of extracted concepts.
        """
        # Normalize text
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Split into words
        words = text.split()

        # Simple concept extraction (noun phrases)
        concepts = set()

        # Single words (filtering common words)
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "shall", "must",
            "and", "or", "but", "not", "no", "nor", "for", "yet", "so",
            "at", "by", "to", "from", "in", "on", "of", "with", "about",
            "this", "that", "these", "those", "it", "its", "he", "she",
            "they", "them", "their", "we", "our", "you", "your", "i", "me",
        }

        for word in words:
            if len(word) >= min_length and word not in stop_words:
                concepts.add(word)

        # Two-word phrases
        for i in range(len(words) - 1):
            if (
                words[i] not in stop_words
                and words[i + 1] not in stop_words
                and len(words[i]) >= min_length
                and len(words[i + 1]) >= min_length
            ):
                phrase = f"{words[i]} {words[i + 1]}"
                concepts.add(phrase)

        return list(concepts)[:50]  # Limit to 50 concepts

    def store_concepts(
        self,
        doc_id: str,
        concepts: List[str],
        domain: Optional[str] = None,
    ) -> int:
        """Store concepts and their relationships to a document.

        Args:
            doc_id: Document ID.
            concepts: List of concepts.
            domain: Optional domain for the document.

        Returns:
            Number of concepts stored.
        """
        if not concepts:
            return 0

        stored_count = 0

        for concept in concepts:
            # Create or merge concept node
            query = """
            MERGE (c:Concept {name: $name})
            ON CREATE SET c.created_at = datetime()
            SET c.updated_at = datetime()
            RETURN c
            """
            self._execute_query(query, {"name": concept.lower()})

            # Create or merge document node
            doc_query = """
            MERGE (d:Document {id: $doc_id})
            ON CREATE SET d.created_at = datetime()
            SET d.updated_at = datetime(), d.domain = $domain
            RETURN d
            """
            self._execute_query(doc_query, {"doc_id": doc_id, "domain": domain})

            # Create relationship
            rel_query = """
            MATCH (c:Concept {name: $name})
            MATCH (d:Document {id: $doc_id})
            MERGE (c)-[r:MENTIONED_IN]->(d)
            SET r.updated_at = datetime()
            RETURN r
            """
            self._execute_query(rel_query, {"name": concept.lower(), "doc_id": doc_id})

            stored_count += 1

        logger.info(f"Stored {stored_count} concepts for document {doc_id}")
        return stored_count

    def find_related_concepts(
        self,
        concepts: List[str],
        depth: int = 2,
        limit: int = 20,
    ) -> List[ConceptRelation]:
        """Find concepts related to given concepts.

        Args:
            concepts: List of seed concepts.
            depth: Graph traversal depth.
            limit: Maximum results.

        Returns:
            List of concept relations.
        """
        if not concepts:
            return []

        # Find concepts that share documents with seed concepts
        query = """
        MATCH (seed:Concept)-[:MENTIONED_IN]->(d:Document)<-[:MENTIONED_IN]-(related:Concept)
        WHERE seed.name IN $concepts AND seed <> related
        RETURN seed.name AS source, related.name AS target, 
               count(d) AS shared_docs
        ORDER BY shared_docs DESC
        LIMIT $limit
        """
        results = self._execute_query(
            query, {"concepts": [c.lower() for c in concepts], "limit": limit}
        )

        relations = []
        for record in results:
            relation = ConceptRelation(
                source_concept=record["source"],
                target_concept=record["target"],
                relation_type="co_occurs",
                weight=record["shared_docs"],
                metadata={"shared_documents": record["shared_docs"]},
            )
            relations.append(relation)

        return relations

    def get_concept_context(self, concept: str, limit: int = 5) -> str:
        """Get textual context for a concept.

        Args:
            concept: Concept name.
            limit: Maximum documents to include.

        Returns:
            Textual context describing the concept.
        """
        # Get documents mentioning this concept
        query = """
        MATCH (c:Concept {name: $name})-[:MENTIONED_IN]->(d:Document)
        RETURN d.id AS doc_id, d.domain AS domain
        ORDER BY d.updated_at DESC
        LIMIT $limit
        """
        results = self._execute_query(query, {"name": concept.lower(), "limit": limit})

        if not results:
            return f"Concept: {concept} (no context available)"

        # Build context
        context_parts = [f"Concept: {concept}"]
        domains = set()

        for record in results:
            if record.get("domain"):
                domains.add(record["domain"])

        if domains:
            context_parts.append(f"Domains: {', '.join(domains)}")

        context_parts.append(f"Mentioned in {len(results)} documents")

        return ". ".join(context_parts)

    def build_knowledge_graph(
        self,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """Build knowledge graph from documents.

        Args:
            documents: List of documents with 'id', 'content', and optional 'domain'.

        Returns:
            Statistics about the graph built.
        """
        stats = {"concepts": 0, "documents": 0, "relations": 0}

        for doc in documents:
            doc_id = doc.get("id", "")
            content = doc.get("content", "")
            domain = doc.get("domain")

            if not doc_id or not content:
                continue

            # Extract concepts
            concepts = self.extract_concepts(content)

            # Store concepts
            stored = self.store_concepts(doc_id, concepts, domain)

            stats["concepts"] += stored
            stats["documents"] += 1
            stats["relations"] += stored  # One relation per concept-document pair

        logger.info(
            f"Built knowledge graph: {stats['concepts']} concepts, "
            f"{stats['documents']} documents, {stats['relations']} relations"
        )
        return stats

    def get_concept_stats(self, domain: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about concepts in the graph.

        Args:
            domain: Optional domain filter.

        Returns:
            Statistics dictionary.
        """
        if domain:
            query = """
            MATCH (c:Concept)-[:MENTIONED_IN]->(d:Document {domain: $domain})
            RETURN count(DISTINCT c) AS concept_count, count(DISTINCT d) AS doc_count
            """
            results = self._execute_query(query, {"domain": domain})
        else:
            query = """
            MATCH (c:Concept)
            OPTIONAL MATCH (c)-[:MENTIONED_IN]->(d:Document)
            RETURN count(DISTINCT c) AS concept_count, count(DISTINCT d) AS doc_count
            """
            results = self._execute_query(query)

        if results:
            return {
                "concept_count": results[0].get("concept_count", 0),
                "document_count": results[0].get("doc_count", 0),
                "domain": domain,
            }
        return {"concept_count": 0, "document_count": 0, "domain": domain}

    def clear_graph(self) -> None:
        """Clear all data from the graph."""
        query = "MATCH (n) DETACH DELETE n"
        self._execute_query(query)
        logger.info("Cleared graph data")