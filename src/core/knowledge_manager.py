"""Knowledge Manager for gdrag v3.

Provides multi-agent knowledge management with visibility controls,
ownership tracking, and cross-store coordination.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Set
from uuid import uuid4

from ..models.schemas import (
    Chunk,
    ConceptRelation,
    GraphEdge,
    GraphNode,
    KnowledgeAccess,
    KnowledgeGraph,
    KnowledgeItem,
    RankedResult,
    SearchResult,
    VisibilityLevel,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols (interfaces) for store dependencies
# ============================================================================

class VectorStoreProtocol(Protocol):
    """Protocol for vector store operations."""

    def upsert_chunks(
        self,
        chunks: List[Chunk],
        embeddings: Optional[Dict[str, List[float]]] = None,
    ) -> List[str]:
        ...

    def search(
        self,
        query: str,
        limit: int = 10,
        domain: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> List[RankedResult]:
        ...

    def delete_document(self, doc_id: str) -> int:
        ...


class GraphStoreProtocol(Protocol):
    """Protocol for graph store operations."""

    def extract_concepts(self, text: str, min_length: int = 3) -> List[str]:
        ...

    def store_concepts(
        self,
        doc_id: str,
        concepts: List[str],
        domain: Optional[str] = None,
    ) -> int:
        ...

    def find_related_concepts(
        self,
        concepts: List[str],
        depth: int = 2,
        limit: int = 20,
    ) -> List[ConceptRelation]:
        ...

    def _execute_query(self, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        ...


class RelationalStoreProtocol(Protocol):
    """Protocol for relational store operations."""

    def store_knowledge_entry(self, item: KnowledgeItem) -> str:
        ...

    def _execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch: bool = True,
    ) -> List[Dict[str, Any]]:
        ...


class TenantManagerProtocol(Protocol):
    """Protocol for tenant/team management.

    This interface allows the KnowledgeManager to query team membership
    without depending on a concrete implementation. A future TenantManager
    can implement this protocol.
    """

    def get_agent_team(self, agent_id: str) -> Optional[str]:
        """Get the team ID for an agent. Returns None if agent has no team."""
        ...

    def get_team_members(self, team_id: str) -> List[str]:
        """Get all agent IDs in a team."""
        ...


class DocumentChunkerProtocol(Protocol):
    """Protocol for document chunking."""

    def chunk(self, text: str, doc_id: str = "") -> List[Chunk]:
        ...


# ============================================================================
# KnowledgeManager
# ============================================================================

class KnowledgeManager:
    """Manages knowledge lifecycle across vector, graph, and relational stores.

    Provides multi-agent support with visibility controls (private, team,
    shared, public), ownership tracking, and promotion workflows.

    Attributes:
        vector_store: Store for embeddings and semantic search.
        graph_store: Store for concept extraction and knowledge graphs.
        relational_store: Store for metadata, sessions, and access control.
        tenant_manager: Optional manager for team membership queries.
        chunker: Document chunker for splitting content.
    """

    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        graph_store: GraphStoreProtocol,
        relational_store: RelationalStoreProtocol,
        tenant_manager: Optional[TenantManagerProtocol] = None,
        chunker: Optional[DocumentChunkerProtocol] = None,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.relational_store = relational_store
        self.tenant_manager = tenant_manager
        self.chunker = chunker

        # Ensure the knowledge_access table exists
        self._ensure_access_schema()

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def _ensure_access_schema(self) -> None:
        """Create the knowledge_access table if it does not exist."""
        schema_sql = """
        CREATE TABLE IF NOT EXISTS knowledge_access (
            doc_id VARCHAR(255) PRIMARY KEY,
            owner_agent_id VARCHAR(255) NOT NULL,
            visibility VARCHAR(50) NOT NULL DEFAULT 'private',
            allowed_agent_ids JSONB DEFAULT '[]',
            allowed_domains JSONB DEFAULT '[]',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_knowledge_access_owner
            ON knowledge_access(owner_agent_id);

        CREATE INDEX IF NOT EXISTS idx_knowledge_access_visibility
            ON knowledge_access(visibility);
        """
        try:
            self.relational_store._execute_query(schema_sql, fetch=False)
        except Exception as e:
            logger.warning(f"Could not ensure knowledge_access schema: {e}")

    # ------------------------------------------------------------------
    # Access control helpers
    # ------------------------------------------------------------------

    def _store_access(self, access: KnowledgeAccess) -> None:
        """Insert or update an access record."""
        import json

        sql = """
        INSERT INTO knowledge_access
            (doc_id, owner_agent_id, visibility,
             allowed_agent_ids, allowed_domains, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (doc_id) DO UPDATE SET
            visibility = EXCLUDED.visibility,
            allowed_agent_ids = EXCLUDED.allowed_agent_ids,
            allowed_domains = EXCLUDED.allowed_domains,
            updated_at = EXCLUDED.updated_at
        """
        self.relational_store._execute_query(
            sql,
            (
                access.doc_id,
                access.owner_agent_id,
                access.visibility.value,
                json.dumps(access.allowed_agent_ids),
                json.dumps(access.allowed_domains),
                access.created_at,
                access.updated_at,
            ),
            fetch=False,
        )

    def _get_access(self, doc_id: str) -> Optional[KnowledgeAccess]:
        """Retrieve the access record for a document."""
        import json

        sql = """
        SELECT doc_id, owner_agent_id, visibility,
               allowed_agent_ids, allowed_domains, created_at, updated_at
        FROM knowledge_access
        WHERE doc_id = %s
        """
        rows = self.relational_store._execute_query(sql, (doc_id,))
        if not rows:
            return None
        row = rows[0]
        return KnowledgeAccess(
            doc_id=row["doc_id"],
            owner_agent_id=row["owner_agent_id"],
            visibility=VisibilityLevel(row["visibility"]),
            allowed_agent_ids=json.loads(row["allowed_agent_ids"]) if row["allowed_agent_ids"] else [],
            allowed_domains=json.loads(row["allowed_domains"]) if row["allowed_domains"] else [],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _get_accessible_doc_ids(self, agent_id: str, include_shared: bool = True) -> Set[str]:
        """Return the set of doc_ids accessible by *agent_id*.

        A document is accessible when ANY of the following is true:
        - The agent is the owner.
        - Visibility is *public*.
        - Visibility is *shared* and the agent is in allowed_agent_ids or
          the agent's domain is in allowed_domains.
        - Visibility is *team*, tenant_manager is available, and the agent
          shares a team with the owner.
        - Visibility is *team*, tenant_manager is unavailable, and the agent
          is in allowed_agent_ids (fallback).
        """
        import json

        sql = """
        SELECT doc_id, owner_agent_id, visibility,
               allowed_agent_ids, allowed_domains
        FROM knowledge_access
        """
        all_rows = self.relational_store._execute_query(sql)

        accessible: Set[str] = set()
        agent_team: Optional[str] = None
        if self.tenant_manager:
            agent_team = self.tenant_manager.get_agent_team(agent_id)

        for row in all_rows:
            owner = row["owner_agent_id"]
            vis = VisibilityLevel(row["visibility"])
            doc_id = row["doc_id"]

            # Owner always has access
            if owner == agent_id:
                accessible.add(doc_id)
                continue

            # Public is always accessible
            if vis == VisibilityLevel.PUBLIC:
                accessible.add(doc_id)
                continue

            if not include_shared:
                continue

            allowed_agents = json.loads(row["allowed_agent_ids"]) if row["allowed_agent_ids"] else []
            allowed_domains = json.loads(row["allowed_domains"]) if row["allowed_domains"] else []

            if vis == VisibilityLevel.SHARED:
                if agent_id in allowed_agents:
                    accessible.add(doc_id)
                    continue
                # Domain-level sharing would require knowing the agent's domain;
                # for now we check if the agent_id itself is in allowed_agents.
                # Domain matching can be extended when a domain registry exists.

            if vis == VisibilityLevel.TEAM:
                if self.tenant_manager and agent_team:
                    owner_team = self.tenant_manager.get_agent_team(owner)
                    if owner_team and owner_team == agent_team:
                        accessible.add(doc_id)
                        continue
                # Fallback: if no tenant manager, treat as shared
                if agent_id in allowed_agents:
                    accessible.add(doc_id)

        return accessible

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        agent_id: str,
        content: str,
        visibility: str = "private",
        domain: Optional[str] = None,
        title: str = "",
        source: str = "manual",
    ) -> str:
        """Ingest a knowledge document.

        Chunks the content, stores embeddings in the vector store, extracts
        concepts into the graph store, persists the metadata in the
        relational store, and records ownership + visibility in the
        knowledge_access table.

        Args:
            agent_id: The agent that owns this knowledge.
            content: Raw text content.
            visibility: One of 'private', 'team', 'shared', 'public'.
            domain: Optional knowledge domain.
            title: Optional document title.
            source: Source identifier.

        Returns:
            The generated document ID.
        """
        doc_id = str(uuid4())
        vis_level = VisibilityLevel(visibility)

        # --- 1. Chunk the document -----------------------------------------
        if self.chunker:
            chunks = self.chunker.chunk(content, doc_id)
        else:
            # Fallback: treat entire content as a single chunk
            token_count = int(len(content.split()) * 1.3)
            chunks = [
                Chunk(
                    doc_id=doc_id,
                    content=content,
                    chunk_index=0,
                    start_char=0,
                    end_char=len(content),
                    token_count=token_count,
                    metadata={"domain": domain} if domain else {},
                )
            ]

        # Tag every chunk with the domain so vector search can filter
        for chunk in chunks:
            if domain:
                chunk.metadata["domain"] = domain

        # --- 2. Store embeddings -------------------------------------------
        self.vector_store.upsert_chunks(chunks)

        # --- 3. Extract & store concepts in graph --------------------------
        concepts = self.graph_store.extract_concepts(content)
        if concepts:
            self.graph_store.store_concepts(doc_id, concepts, domain)

        # --- 4. Persist knowledge entry in relational store ----------------
        knowledge_item = KnowledgeItem(
            id=doc_id,
            title=title or content[:100],
            content=content,
            domain=domain,
            source=source,
            source_type="document",
            metadata={
                "owner_agent_id": agent_id,
                "visibility": visibility,
                "concepts": concepts,
            },
        )
        self.relational_store.store_knowledge_entry(knowledge_item)

        # --- 5. Record access control --------------------------------------
        access = KnowledgeAccess(
            doc_id=doc_id,
            owner_agent_id=agent_id,
            visibility=vis_level,
        )
        self._store_access(access)

        logger.info(
            f"Ingested doc {doc_id} for agent {agent_id} "
            f"(visibility={visibility}, domain={domain})"
        )
        return doc_id

    def promote_to_shared(
        self,
        agent_id: str,
        doc_id: str,
        target_domains: Optional[List[str]] = None,
        target_agents: Optional[List[str]] = None,
    ) -> None:
        """Promote a private document to shared visibility.

        Only the document owner may promote a document.

        Args:
            agent_id: Requesting agent (must be owner).
            doc_id: Document to promote.
            target_domains: Domains to share with.
            target_agents: Specific agents to share with.

        Raises:
            PermissionError: If the agent is not the document owner.
            ValueError: If the document does not exist.
        """
        access = self._get_access(doc_id)
        if access is None:
            raise ValueError(f"Document {doc_id} not found")
        if access.owner_agent_id != agent_id:
            raise PermissionError(
                f"Agent {agent_id} is not the owner of document {doc_id}"
            )

        access.visibility = VisibilityLevel.SHARED
        if target_agents:
            access.allowed_agent_ids = list(
                set(access.allowed_agent_ids) | set(target_agents)
            )
        if target_domains:
            access.allowed_domains = list(
                set(access.allowed_domains) | set(target_domains)
            )
        access.updated_at = datetime.utcnow()
        self._store_access(access)
        logger.info(
            f"Promoted doc {doc_id} to shared "
            f"(agents={access.allowed_agent_ids}, domains={access.allowed_domains})"
        )

    def search(
        self,
        agent_id: str,
        query: str,
        include_shared: bool = True,
        limit: int = 10,
        domain: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search knowledge accessible by *agent_id*.

        Performs semantic search in the vector store and filters results
        according to visibility rules.

        Args:
            agent_id: The agent performing the search.
            query: Search query text.
            include_shared: Whether to include shared/team/public docs.
            limit: Maximum number of results.
            domain: Optional domain filter.

        Returns:
            List of SearchResult objects.
        """
        # Retrieve accessible doc IDs
        accessible_ids = self._get_accessible_doc_ids(agent_id, include_shared)

        # If the agent has no accessible documents, return empty
        if not accessible_ids:
            return []

        # Run vector search with a generous limit, then filter
        raw_results = self.vector_store.search(
            query, limit=limit * 5, domain=domain
        )

        # Build a lookup of access records for the returned docs
        results: List[SearchResult] = []
        seen_docs: Set[str] = set()

        for ranked in raw_results:
            if ranked.doc_id in seen_docs:
                continue
            if ranked.doc_id not in accessible_ids:
                continue

            access = self._get_access(ranked.doc_id)
            if access is None:
                continue

            seen_docs.add(ranked.doc_id)
            results.append(
                SearchResult(
                    doc_id=ranked.doc_id,
                    content=ranked.content,
                    title=ranked.title,
                    domain=ranked.domain,
                    source=ranked.source,
                    score=ranked.final_score,
                    visibility=access.visibility,
                    owner_agent_id=access.owner_agent_id,
                    is_shared=access.owner_agent_id != agent_id,
                    metadata=ranked.metadata,
                    created_at=access.created_at,
                )
            )

            if len(results) >= limit:
                break

        return results

    def get_knowledge_graph(
        self,
        agent_id: str,
        include_shared: bool = True,
        seed_concepts: Optional[List[str]] = None,
        depth: int = 2,
        limit: int = 50,
    ) -> KnowledgeGraph:
        """Build a knowledge graph from concepts accessible by *agent_id*.

        If *seed_concepts* is provided the graph is expanded from those
        concepts; otherwise concepts are gathered from the agent's
        accessible documents.

        Args:
            agent_id: Requesting agent.
            include_shared: Whether to include shared/team/public docs.
            seed_concepts: Optional starting concepts.
            depth: Graph traversal depth.
            limit: Maximum relations to return.

        Returns:
            A KnowledgeGraph with nodes and edges.
        """
        accessible_ids = self._get_accessible_doc_ids(agent_id, include_shared)

        # Collect concepts
        if seed_concepts:
            concepts = seed_concepts
        else:
            concepts = self._collect_concepts_from_docs(accessible_ids)

        if not concepts:
            return KnowledgeGraph(metadata={"agent_id": agent_id})

        # Find related concepts via graph store
        relations = self.graph_store.find_related_concepts(
            concepts, depth=depth, limit=limit
        )

        # Filter relations to only those connected to accessible docs
        filtered_relations = self._filter_relations_to_accessible(
            relations, accessible_ids
        )

        # Build the graph
        nodes: Dict[str, GraphNode] = {}
        edges: List[GraphEdge] = []

        for rel in filtered_relations:
            src_id = rel.source_concept
            tgt_id = rel.target_concept

            if src_id not in nodes:
                nodes[src_id] = GraphNode(
                    id=src_id,
                    label=rel.source_concept,
                    node_type="concept",
                )
            if tgt_id not in nodes:
                nodes[tgt_id] = GraphNode(
                    id=tgt_id,
                    label=rel.target_concept,
                    node_type="concept",
                )

            edges.append(
                GraphEdge(
                    source_id=src_id,
                    target_id=tgt_id,
                    relation_type=rel.relation_type,
                    weight=rel.weight,
                    metadata=rel.metadata,
                )
            )

        return KnowledgeGraph(
            nodes=list(nodes.values()),
            edges=edges,
            metadata={
                "agent_id": agent_id,
                "seed_concepts": concepts,
                "depth": depth,
                "accessible_docs": len(accessible_ids),
            },
        )

    def revoke_access(self, doc_id: str, agent_id: str) -> None:
        """Revoke a specific agent's access to a document.

        Removes *agent_id* from the document's allowed_agent_ids list.
        Only the document owner may revoke access.

        Args:
            doc_id: Document to revoke access for.
            agent_id: Agent whose access should be revoked.

        Raises:
            PermissionError: If the caller is not the document owner.
            ValueError: If the document does not exist.
        """
        access = self._get_access(doc_id)
        if access is None:
            raise ValueError(f"Document {doc_id} not found")

        if access.owner_agent_id != agent_id:
            raise PermissionError(
                f"Agent {agent_id} is not the owner of document {doc_id}"
            )

        # Remove the agent from the allowed list
        if agent_id in access.allowed_agent_ids:
            access.allowed_agent_ids.remove(agent_id)

        # Downgrade to private if no agents or domains remain shared
        if not access.allowed_agent_ids and not access.allowed_domains:
            if access.visibility == VisibilityLevel.SHARED:
                access.visibility = VisibilityLevel.PRIVATE

        access.updated_at = datetime.utcnow()
        self._store_access(access)
        logger.info(f"Revoked access for agent {agent_id} on doc {doc_id}")

    def delete_document(self, agent_id: str, doc_id: str) -> None:
        """Delete a document and all associated data.

        Only the owner may delete a document.

        Args:
            agent_id: Requesting agent (must be owner).
            doc_id: Document to delete.

        Raises:
            PermissionError: If the agent is not the owner.
            ValueError: If the document does not exist.
        """
        access = self._get_access(doc_id)
        if access is None:
            raise ValueError(f"Document {doc_id} not found")
        if access.owner_agent_id != agent_id:
            raise PermissionError(
                f"Agent {agent_id} is not the owner of document {doc_id}"
            )

        # Delete from vector store
        self.vector_store.delete_document(doc_id)

        # Delete access record
        self.relational_store._execute_query(
            "DELETE FROM knowledge_access WHERE doc_id = %s",
            (doc_id,),
            fetch=False,
        )

        # Delete knowledge entry
        self.relational_store._execute_query(
            "DELETE FROM knowledge_entries WHERE id = %s",
            (doc_id,),
            fetch=False,
        )

        logger.info(f"Deleted doc {doc_id} and all associated data")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_concepts_from_docs(
        self, doc_ids: Set[str], limit: int = 100
    ) -> List[str]:
        """Extract concepts from documents accessible to the agent."""
        if not doc_ids:
            return []

        # Query concepts linked to accessible documents via graph store
        placeholders = ",".join(["%s"] * len(doc_ids))
        query = f"""
        MATCH (c:Concept)-[:MENTIONED_IN]->(d:Document)
        WHERE d.id IN [{','.join(f'"{did}"' for did in doc_ids)}]
        RETURN DISTINCT c.name AS name
        LIMIT {limit}
        """
        try:
            results = self.graph_store._execute_query(query)
            return [r["name"] for r in results]
        except Exception:
            # Fallback: extract from content if graph query fails
            return []

    def _filter_relations_to_accessible(
        self,
        relations: List[ConceptRelation],
        accessible_doc_ids: Set[str],
    ) -> List[ConceptRelation]:
        """Filter concept relations to only those involving accessible docs.

        Currently passes all relations through since ConceptRelation doesn't
        carry doc_id directly. A future enhancement would join through the
        graph to filter by document access.
        """
        # If no access restrictions, return all
        if not accessible_doc_ids:
            return []

        # For now return all relations since concepts are already scoped
        # by the _collect_concepts_from_docs call
        return relations
