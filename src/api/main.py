"""
RAG API - Sistema de Retrieval-Augmented Generation
Accesible para Cline y otros agentes IA
Version 2.0 - Enhanced with session management, attention, and compression
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import json
import hashlib
from datetime import datetime
import qdrant_client
from qdrant_client.models import Distance, VectorParams
import psycopg2
from psycopg2.extras import RealDictCursor
from neo4j import GraphDatabase

# Import v2 components
from .v2.router import api_v2_router
from .v2.middleware import RequestLoggingMiddleware, SessionMiddleware

app = FastAPI(
    title="RAG API",
    description="Sistema de Retrieval-Augmented Generation para agentes IA - Version 2.0",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# v2 Middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SessionMiddleware)

# Include v2 router
app.include_router(api_v2_router)

# Security
security = HTTPBearer()

# ─── CONFIGURATION ──────────────────────────────────────────────────────────

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://localhost:5432/workspace")
MEMGRAPH_URL = os.getenv("MEMGRAPH_URL", "bolt://localhost:7687")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")

# Agent permissions
AGENT_PERMISSIONS = {
    "cline": ["read", "write", "ingest", "admin"],
    "claude": ["read"],
    "gpt": ["read"],
    "copilot": ["read"],
    "custom-agent": ["read", "write"],
}

# ─── MODELS ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="Query de búsqueda")
    domain: Optional[str] = Field(None, description="Dominio: software, finance, academic, print3d")
    limit: int = Field(10, description="Número máximo de resultados", ge=1, le=100)

class SearchRequest(BaseModel):
    query: str = Field(..., description="Query de búsqueda")
    limit: int = Field(10, description="Número máximo de resultados", ge=1, le=100)
    domain: Optional[str] = Field(None, description="Filtrar por dominio")

class IngestRequest(BaseModel):
    content: str = Field(..., description="Contenido a ingestar")
    domain: str = Field(..., description="Dominio del contenido")
    source: str = Field(..., description="Fuente del contenido")
    title: Optional[str] = Field(None, description="Título del documento")
    metadata: Optional[dict] = Field(None, description="Metadatos adicionales")

class QueryResponse(BaseModel):
    query: str
    results: List[dict]
    sources: List[str]
    timestamp: str
    agent_id: str

class HealthResponse(BaseModel):
    status: str
    services: dict
    timestamp: str

# ─── CLIENTS ─────────────────────────────────────────────────────────────────

def get_qdrant_client():
    """Get Qdrant client"""
    return qdrant_client.QdrantClient(url=QDRANT_URL)

def get_postgres_connection():
    """Get PostgreSQL connection"""
    return psycopg2.connect(POSTGRES_URL, cursor_factory=RealDictCursor)

def get_memgraph_driver():
    """Get Memgraph driver"""
    return GraphDatabase.driver(MEMGRAPH_URL, auth=None)

# ─── AUTHENTICATION ──────────────────────────────────────────────────────────

def validate_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Validate API token and return agent info"""
    token = credentials.credentials
    
    # Simple token validation (in production, use JWT)
    for agent_id, permissions in AGENT_PERMISSIONS.items():
        expected_token = hashlib.sha256(f"{agent_id}:{JWT_SECRET}".encode()).hexdigest()[:32]
        if token == expected_token:
            return {"agent_id": agent_id, "permissions": permissions}
    
    raise HTTPException(status_code=401, detail="Invalid API token")

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(agent_info: dict = Depends(validate_token)):
        if permission not in agent_info["permissions"]:
            raise HTTPException(status_code=403, detail=f"Permission '{permission}' required")
        return agent_info
    return decorator

# ─── ENDPOINTS ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services = {}
    
    # Check Qdrant
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
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
    
    return HealthResponse(
        status="healthy" if all("healthy" in s for s in services.values()) else "degraded",
        services=services,
        timestamp=datetime.now().isoformat()
    )

@app.post("/api/v1/query", response_model=QueryResponse)
async def rag_query(
    request: QueryRequest,
    agent_info: dict = Depends(require_permission("read"))
):
    """Execute RAG query across all databases"""
    try:
        results = []
        sources = []
        
        # 1. Semantic search in Qdrant
        qdrant_results = await search_qdrant(request.query, request.limit, request.domain)
        results.extend(qdrant_results)
        sources.append("qdrant")
        
        # 2. Graph search in Memgraph
        memgraph_results = await search_memgraph(request.query, request.limit)
        results.extend(memgraph_results)
        sources.append("memgraph")
        
        # 3. Structured search in PostgreSQL
        postgres_results = await search_postgres(request.query, request.limit, request.domain)
        results.extend(postgres_results)
        sources.append("postgres")
        
        # Deduplicate and rank
        results = deduplicate_results(results)
        results = results[:request.limit]
        
        return QueryResponse(
            query=request.query,
            results=results,
            sources=sources,
            timestamp=datetime.now().isoformat(),
            agent_id=agent_info["agent_id"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/search/semantic")
async def semantic_search(
    request: SearchRequest,
    agent_info: dict = Depends(require_permission("read"))
):
    """Semantic search in Qdrant only"""
    try:
        results = await search_qdrant(request.query, request.limit, request.domain)
        return {"results": results, "source": "qdrant"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/search/graph")
async def graph_search(
    request: SearchRequest,
    agent_info: dict = Depends(require_permission("read"))
):
    """Graph search in Memgraph only"""
    try:
        results = await search_memgraph(request.query, request.limit)
        return {"results": results, "source": "memgraph"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/search/structured")
async def structured_search(
    request: SearchRequest,
    agent_info: dict = Depends(require_permission("read"))
):
    """Structured search in PostgreSQL only"""
    try:
        results = await search_postgres(request.query, request.limit, request.domain)
        return {"results": results, "source": "postgres"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/ingest")
async def ingest_knowledge(
    request: IngestRequest,
    agent_info: dict = Depends(require_permission("write"))
):
    """Ingest new knowledge into the RAG system"""
    try:
        doc_id = await save_knowledge(
            content=request.content,
            domain=request.domain,
            source=request.source,
            title=request.title,
            metadata=request.metadata,
            agent_id=agent_info["agent_id"]
        )
        return {
            "status": "ingested",
            "doc_id": doc_id,
            "agent_id": agent_info["agent_id"],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agents")
async def list_agents(agent_info: dict = Depends(require_permission("admin"))):
    """List registered agents and their permissions"""
    return {"agents": AGENT_PERMISSIONS}

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

async def search_qdrant(query: str, limit: int, domain: Optional[str] = None) -> list:
    """Search in Qdrant"""
    try:
        client = get_qdrant_client()
        
        # Generate embedding (placeholder - use actual embedding model)
        query_vector = generate_embedding(query)
        
        # Build filter
        query_filter = None
        if domain:
            query_filter = {"must": [{"key": "domain", "match": {"value": domain}}]}
        
        # Search
        results = client.search(
            collection_name="workspace_knowledge",
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter
        )
        
        return [
            {
                "content": hit.payload.get("content", ""),
                "title": hit.payload.get("title", ""),
                "domain": hit.payload.get("domain", ""),
                "source": hit.payload.get("source", ""),
                "score": hit.score,
                "db": "qdrant"
            }
            for hit in results
        ]
    except Exception as e:
        print(f"Qdrant search error: {e}")
        return []

async def search_memgraph(query: str, limit: int) -> list:
    """Search in Memgraph"""
    try:
        driver = get_memgraph_driver()
        
        with driver.session() as session:
            # Extract concepts from query
            concepts = extract_concepts(query)
            
            # Search related concepts
            result = session.run("""
                MATCH (c:Concept)-[:BELONGS_TO]->(t:Theory)
                WHERE c.name IN $concepts
                RETURN c.name as concept, c.description as description, 
                       t.name as theory, t.framework as framework
                LIMIT $limit
            """, concepts=concepts, limit=limit)
            
            return [
                {
                    "content": record["description"] or "",
                    "title": record["concept"],
                    "domain": "academic",
                    "source": "memgraph",
                    "theory": record["theory"],
                    "score": 0.8,
                    "db": "memgraph"
                }
                for record in result
            ]
    except Exception as e:
        print(f"Memgraph search error: {e}")
        return []

async def search_postgres(query: str, limit: int, domain: Optional[str] = None) -> list:
    """Search in PostgreSQL"""
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()
        
        # Build query
        sql = """
            SELECT id, title, content, domain, source, created_at
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
                "content": row["content"],
                "title": row["title"],
                "domain": row["domain"],
                "source": row["source"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "score": 0.7,
                "db": "postgres"
            }
            for row in results
        ]
    except Exception as e:
        print(f"PostgreSQL search error: {e}")
        return []

async def save_knowledge(
    content: str,
    domain: str,
    source: str,
    title: Optional[str] = None,
    metadata: Optional[dict] = None,
    agent_id: str = "unknown"
) -> str:
    """Save knowledge to all databases"""
    doc_id = hashlib.md5(f"{content}{datetime.now()}".encode()).hexdigest()
    
    # 1. Save to PostgreSQL
    try:
        conn = get_postgres_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO knowledge_entries (id, domain, title, content, source, metadata, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (doc_id, domain, title or "", content, source, 
              json.dumps(metadata or {}), agent_id))
        
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"PostgreSQL save error: {e}")
    
    # 2. Save to Qdrant
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
                    "source": source,
                    "created_by": agent_id,
                    "created_at": datetime.now().isoformat()
                }
            }]
        )
    except Exception as e:
        print(f"Qdrant save error: {e}")
    
    # 3. Save to Memgraph
    try:
        driver = get_memgraph_driver()
        
        with driver.session() as session:
            # Create concept node
            session.run("""
                MERGE (c:Concept {name: $name})
                SET c.description = $description, c.domain = $domain,
                    c.source = $source, c.created_at = datetime()
            """, name=title or doc_id, description=content[:500], 
               domain=domain, source=source)
        
        driver.close()
    except Exception as e:
        print(f"Memgraph save error: {e}")
    
    return doc_id

def generate_embedding(text: str) -> list:
    """Generate embedding for text (placeholder)"""
    # In production, use sentence-transformers or OpenAI embeddings
    import random
    return [random.random() for _ in range(1536)]

def extract_concepts(query: str) -> list:
    """Extract concepts from query (placeholder)"""
    # In production, use NLP to extract concepts
    words = query.lower().split()
    return [w for w in words if len(w) > 3][:5]

def deduplicate_results(results: list) -> list:
    """Deduplicate results by content similarity"""
    seen = set()
    unique_results = []
    
    for result in results:
        content_hash = hashlib.md5(result.get("content", "")[:100].encode()).hexdigest()
        if content_hash not in seen:
            seen.add(content_hash)
            unique_results.append(result)
    
    return unique_results

# ─── STARTUP ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("🚀 RAG API starting...")
    print(f"   Qdrant: {QDRANT_URL}")
    print(f"   PostgreSQL: {POSTGRES_URL}")
    print(f"   Memgraph: {MEMGRAPH_URL}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)