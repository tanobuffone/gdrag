"""Custom middleware for gdrag v2 API.

Provides logging, rate limiting, and request tracking.
"""

import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs all API requests with timing information."""

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Process request and log details.

        Args:
            request: Incoming request.
            call_next: Next middleware/endpoint.

        Returns:
            Response from downstream.
        """
        start_time = time.time()

        # Extract request info
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"

        # Get agent ID if present
        agent_id = request.headers.get("X-Agent-ID", "unknown")

        logger.info(
            f"Request: {method} {path} "
            f"from {client_ip} "
            f"agent={agent_id}"
        )

        # Process request
        response = await call_next(request)

        # Calculate timing
        process_time = (time.time() - start_time) * 1000

        # Add timing header
        response.headers["X-Process-Time-Ms"] = str(round(process_time, 2))

        logger.info(
            f"Response: {method} {path} "
            f"status={response.status_code} "
            f"time={process_time:.1f}ms"
        )

        return response


class SessionMiddleware(BaseHTTPMiddleware):
    """Tracks active sessions and adds session context."""

    def __init__(self, app):
        super().__init__(app)
        self._active_sessions = set()

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """Process request with session tracking.

        Args:
            request: Incoming request.
            call_next: Next middleware/endpoint.

        Returns:
            Response from downstream.
        """
        # Extract session ID from header or query
        session_id = request.headers.get(
            "X-Session-ID",
            request.query_params.get("session_id"),
        )

        if session_id:
            self._active_sessions.add(session_id)

            # Clean up old sessions periodically
            if len(self._active_sessions) > 1000:
                # Keep only recent sessions (simple cleanup)
                self._active_sessions = set(list(self._active_sessions)[-500:])

        response = await call_next(request)

        # Add session tracking header
        if session_id:
            response.headers["X-Session-Tracked"] = "true"

        return response

    def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        return len(self._active_sessions)