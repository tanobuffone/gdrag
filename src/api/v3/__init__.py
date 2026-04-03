"""gdrag v3 API module."""

from .tasks import tasks_router
from .knowledge import knowledge_router

__all__ = ["tasks_router", "knowledge_router"]
