"""
RAG MCP Server
Model Context Protocol server for RAG system access
Allows any MCP-compatible agent to query the RAG system
"""

from fastmcp import FastMCP
import os
import json
import hashlib
from datetime import datetime
import qdrant_client
import psycopg2
from psycopg2.extras import RealDictCursor
from neo4j import GraphDatabase

# Initialize MCP server
mcp = FastMCP("rag-server", version="1.0.0")

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://localhost:5432/workspace")
MEMGRAPH_URL = os.getenv("MEMGRAPH_URL", "bolt://localhost:7687")

# Client helpers
def get_qdrant_client():
    return qdrant_client.QdrantClient(url=QDRANT_URL)

def get_postgres_connection():
    return psycopg2.connect(POSTGRES_URL, cursor_factory=RealDictCursor)

def get_memgraph_driver():
    return GraphDatabase.driver(MEMGRAPH_URL, auth=None)

# ─── MCP TOOLS ──────────────────────────────────────────────────────────────

@mcp.tool()
async def rag_query(query: str, domain: str = None, limit: int = 10) -> str:
    """
    Execute a complete RAG query across PostgreSQL, Qdrant, and Memgraph.
    
    Args:
        query: The search query
        domain: Optional domain filter (software, finance, academic, print3d)
        limit: Maximum number of results (default: 10, max: 100)
    
    Returns:
        JSON string with search results from all three databases
    """
    try:
        results = []
        sources = []
        
        # 1. Semantic search in Qdrant
        qdrant_results = await search_qdrant(query, limit, domain)
        results.extend(qdrant_results)
        sources.append("qdrant")
        
        # 2. Graph search in Memgraph
        memgraph_results = await search_memgraph(query, limit)
        results.extend(memgraph_results)
        sources.append("memgraph")
        
        # 3. Structured search in PostgreSQL
        postgres_results = await search_postgres(query, limit, domain)
        results.extend(postgres_results)
        sources.append("postgres")
        
        # Deduplicate
        results = deduplicate_results(results)[:limit]
        
        return json.dumps({
            "query": query,
            "results": results,
            "sources": sources,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "query": query})

@mcp.tool()
async def semantic_search(query: str, domain: str = None, limit: int = 10) -> str:
    """
    Semantic search using Qdrant vector database.
    
    Args:
        query: The search query
        domain: Optional domain filter
        limit: Maximum results (default: 10)
    
    Returns:
        JSON string with semantic search results
    """
    try:
        results = await search_qdrant(query, limit, domain)
        return json.dumps({
            "query": query,
            "results": results,
            "source": "qdrant",
            "count": len(results)
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
async def graph_search(concepts: str, limit: int = 10) -> str:
    """
    Search for related concepts in Memgraph knowledge graph.
    
    Args:
        concepts: Comma-separated list of concepts to search
        limit: Maximum results (default: 10)
    
    Returns:
        JSON string with graph relationships
    """
    try:
        concept_list = [c.strip() for c in concepts.split(",")]
        results = await search_memgraph_by_concepts(concept_list, limit)
        return json.dumps({
            "concepts": concept_list,
            "results": results,
            "source": "memgraph",
            "count": len(results)
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
async def structured_query(sql_query: str, limit: int = 100) -> str:
    """
    Execute a read-only SQL query on PostgreSQL knowledge base.
    
    Args:
        sql_query: SELECT query to execute (only SELECT allowed)
        limit: Maximum results (default: 100)
    
    Returns:
        JSON string with query results
    """
    try:
        # Security: only allow SELECT queries
        if not sql_query.strip().upper().startswith("SELECT"):
            return json.dumps({"error": "Only SELECT queries are allowed"})
        
        # Add LIMIT if not present
        if "LIMIT" not in sql_query.upper():
            sql_query += f" LIMIT {limit}"
        
        conn = get_postgres_connection()
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return json.dumps({
            "query": sql_query,
            "results": [dict(row) for row in results],
            "source": "postgres",
            "count": len(results)
        }, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
async def ingest_knowledge(content: str, domain: str, source: str, title: str = None) -> str:
    """
    Ingest new knowledge into the RAG system (all databases).
    
    Args:
        content: The content to ingest
        domain: Domain (software, finance, academic, print3d)
        source: Source identifier
        title: Optional title
    
    Returns:
        JSON string with ingestion result
    """
    try:
        doc_id = await save_knowledge(content, domain, source, title)
        return json.dumps({
            "status": "success",
            "doc_id": doc_id,
            "domain": domain,
            "source": source,
            "timestamp": datetime.now().isoformat()
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
async def get_health() -> str:
    """
    Check health status of all RAG system components.
    
    Returns:
        JSON string with health status
    """
    services = {}
    
    # Check Qdrant
    try:
        client = get_qdrant_client()
        client.get_collections()
        services["qdrant"] = "healthy"
    except Exception as e:
        services["qdrant"] = f"unhealthy: {str(e)}"
    
    # Check PostgreSQL
    try:
        conn = get_postgres_connection()
        conn.close()
        services["postgres"] = "healthy"
    except Exception as e:
        services["postgres"] = f"unhealthy: {str(e)}"
    
    # Check Memgraph
    try:
        driver = get_memgraph_driver()
        driver.close()
        services["memgraph"] = "healthy"
    except Exception as e:
        services["memgraph"] = f"unhealthy: {str(e)}"
    
    return json.dumps({
        "status": "healthy" if all("healthy" in s for s in services.values()) else "degraded",
        "services": services,
        "timestamp": datetime.now().isoformat()
    }, indent=2)

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

async def search_qdrant(query: str, limit: int, domain: str = None) -> list:
    """Search in Qdrant"""
    try:
        client = get_qdrant_client()
        query_vector = generate_embedding(query)
        
        query_filter = None
        if domain:
            query_filter = {"must": [{"key": "domain", "match": {"value": domain}}]}
        
        results = client.search(
            collection_name="workspace_knowledge",
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter
        )
        
        return [
            {
                "content": hit.payload.get("content", "")[:500],
                "title": hit.payload.get("title", ""),
                "domain": hit.payload.get("domain", ""),
                "score": hit.score,
                "db": "qdrant"
            }
            for hit in results
        ]
    except Exception as e:
        return [{"error": str(e), "db": "qdrant"}]

async def search_memgraph(query: str, limit: int) -> list:
    """Search in Memgraph"""
    try:
        driver = get_memgraph_driver()
        concepts = extract_concepts(query)
        
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Concept)-[:BELONGS_TO]->(t:Theory)
                WHERE c.name IN $concepts
                RETURN c.name as concept, c.description as description,
                       t.name as theory
                LIMIT $limit
            """, concepts=concepts, limit=limit)
            
            return [
                {
                    "content": record["description"] or "",
                    "title": record["concept"],
                    "theory": record["theory"],
                    "score": 0.8,
                    "db": "memgraph"
                }
                for record in result
            ]
    except Exception as e:
        return [{"error": str(e), "db": "memgraph"}]

async def search_memgraph_by_concepts(concepts: list, limit: int) -> list:
    """Search Memgraph by explicit concepts"""
    try:
        driver = get_memgraph_driver()
        
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Concept)-[r]->(related)
                WHERE c.name IN $concepts
                RETURN c.name as concept, type(r) as relationship,
                       labels(related)[0] as related_type,
                       related.name as related_name
                LIMIT $limit
            """, concepts=concepts, limit=limit)
            
            return [
                {
                    "concept": record["concept"],
                    "relationship": record["relationship"],
                    "related_type": record["related_type"],
                    "related_name": record["related_name"],
                    "db": "memgraph"
                }
                for record in result
            ]
    except Exception as e:
        return [{"error": str(e), "db": "memgraph"}]

async def search_postgres(query: str, limit: int, domain: str = None) -> list:
    """Search in PostgreSQL"""
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()
        
        sql = """
            SELECT id, title, content, domain, source
            FROM knowledge_entries
            WHERE content ILIKE %s OR title ILIKE %s
        """
        params = [f"%{query}%", f"%{query}%"]
        
        if domain:
            sql += " AND domain = %s"
            params.append(domain)
        
        sql += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        
        cursor.execute(sql, params)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [
            {
                "content": row["content"][:500],
                "title": row["title"],
                "domain": row["domain"],
                "source": row["source"],
                "score": 0.7,
                "db": "postgres"
            }
            for row in results
        ]
    except Exception as e:
        return [{"error": str(e), "db": "postgres"}]

async def save_knowledge(content: str, domain: str, source: str, title: str = None) -> str:
    """Save knowledge to all databases"""
    doc_id = hashlib.md5(f"{content}{datetime.now()}".encode()).hexdigest()
    
    # Save to PostgreSQL
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO knowledge_entries (id, domain, title, content, source)
            VALUES (%s, %s, %s, %s, %s)
        """, (doc_id, domain, title or "", content, source))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"PostgreSQL error: {e}")
    
    # Save to Qdrant
    try:
        client = get_qdrant_client()
        embedding = generate_embedding(content)
        client.upsert(
            collection_name="workspace_knowledge",
            points=[{
                "id": doc_id,
                "vector": embedding,
                "payload": {
                    "content": content,
                    "title": title or "",
                    "domain": domain,
                    "source": source
                }
            }]
        )
    except Exception as e:
        print(f"Qdrant error: {e}")
    
    # Save to Memgraph
    try:
        driver = get_memgraph_driver()
        with driver.session() as session:
            session.run("""
                MERGE (c:Concept {name: $name})
                SET c.description = $desc, c.domain = $domain
            """, name=title or doc_id, desc=content[:500], domain=domain)
        driver.close()
    except Exception as e:
        print(f"Memgraph error: {e}")
    
    return doc_id

def generate_embedding(text: str) -> list:
    """Generate embedding (placeholder)"""
    import random
    return [random.random() for _ in range(1536)]

def extract_concepts(query: str) -> list:
    """Extract concepts from query"""
    words = query.lower().split()
    return [w for w in words if len(w) > 3][:5]

def deduplicate_results(results: list) -> list:
    """Deduplicate results"""
    seen = set()
    unique = []
    for r in results:
        key = hashlib.md5(r.get("content", "")[:100].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique

# ─── RUN SERVER ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()