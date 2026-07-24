"""
FastAPI middleware for request-level observability.

Adds two pieces of instrumentation to every request:

1. **Request timing** — records ``mra_graph_execution_seconds`` and
   ``mra_queries_total`` Prometheus metrics.
2. **Trace context** — injects ``X-Trace-ID`` response header for
   correlation with LangSmith traces and structured logs.

Usage
-----
Mount onto an existing FastAPI app *without* modifying the app module::

    from observability.middleware import add_observability_middleware
    from research_agent_api_v2 import app

    add_observability_middleware(app)
"""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from observability.prometheus_metrics import (
    record_graph_execution,
    record_query,
)

logger = logging.getLogger(__name__)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Records request duration, status, and trace-ID header."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Generate or forward trace ID
        trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
        start = time.time()

        try:
            response = await call_next(request)
            duration = time.time() - start

            # Only record metrics for the /query endpoint
            if request.url.path == "/query":
                status = "success" if response.status_code < 400 else "error"
                record_query(status=status)
                record_graph_execution(duration)

                logger.info(
                    "Request completed",
                    extra={
                        "trace_id": trace_id,
                        "node": "api",
                    },
                )

            response.headers["X-Trace-ID"] = trace_id
            return response

        except Exception:
            duration = time.time() - start
            if request.url.path == "/query":
                record_query(status="error")
                record_graph_execution(duration)
            raise


def add_observability_middleware(app) -> None:  # type: ignore[type-arg]
    """Mount ObservabilityMiddleware onto a FastAPI/Starlette app.

    Parameters
    ----------
    app : FastAPI
        The application instance.
    """
    app.add_middleware(ObservabilityMiddleware)
    logger.info("Observability middleware mounted on FastAPI app")
