"""FastAPI dependency injection for gdrag v2.

Provides reusable dependencies for config, pipeline, and agent identity.
"""

from functools import lru_cache
from typing import Optional

from fastapi import Header, HTTPException, Request

from ...core.config import AppConfig, load_config
from ...core.pipeline import QueryPipeline


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return the cached application configuration (loaded once per process)."""
    return load_config()


def get_pipeline(request: Request) -> QueryPipeline:
    """Return the QueryPipeline singleton stored in app state.

    The pipeline is initialised during the FastAPI lifespan startup event.
    Raises 503 if the pipeline is not yet ready.
    """
    pipeline: Optional[QueryPipeline] = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialised. Check startup logs.",
        )
    return pipeline


def get_agent_id(x_agent_id: Optional[str] = Header(default=None)) -> str:
    """Extract the agent identity from the X-Agent-ID request header.

    Falls back to 'anonymous' when the header is absent.
    """
    return x_agent_id or "anonymous"
