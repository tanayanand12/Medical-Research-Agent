"""
Observability package for Medical Research Agent (Phase 5).

Provides LangSmith tracing, Prometheus metrics, structured JSON logging,
FastAPI middleware, and an instrumented graph wrapper.

Quick start::

    from observability import init_observability

    # Call once at application startup
    init_observability()
"""

from observability.langsmith_tracer import (
    init_langsmith,
    is_tracing_enabled,
    get_client,
    get_project,
    trace_node,
    trace_llm_call,
)
from observability.prometheus_metrics import (
    init_prometheus,
    is_metrics_enabled,
    record_llm_call,
    record_graph_execution,
    record_tool_execution,
    record_node_execution,
    record_query,
    record_fallback,
    record_error,
)
from observability.structured_logging import configure_structured_logging
from observability.middleware import add_observability_middleware
from observability.traced_graph import get_traced_graph, build_traced_graph

import logging

logger = logging.getLogger(__name__)


def init_observability(
    prometheus_port: int | None = None,
    langsmith_api_key: str | None = None,
    langsmith_project: str | None = None,
    log_level: str | None = None,
) -> dict:
    """One-call initialisation of all observability subsystems.

    Returns a status dict summarising what was enabled::

        {
            "langsmith": True,
            "prometheus": True,
            "structured_logging": True,
        }
    """
    configure_structured_logging(level=log_level)
    ls_ok = init_langsmith(api_key=langsmith_api_key, project=langsmith_project)
    prom_ok = init_prometheus(port=prometheus_port)

    status = {
        "langsmith": ls_ok,
        "prometheus": prom_ok,
        "structured_logging": True,
    }

    logger.info("Observability initialised: %s", status)
    return status


__all__ = [
    # Top-level init
    "init_observability",
    # LangSmith
    "init_langsmith",
    "is_tracing_enabled",
    "get_client",
    "get_project",
    "trace_node",
    "trace_llm_call",
    # Prometheus
    "init_prometheus",
    "is_metrics_enabled",
    "record_llm_call",
    "record_graph_execution",
    "record_tool_execution",
    "record_node_execution",
    "record_query",
    "record_fallback",
    "record_error",
    # Logging
    "configure_structured_logging",
    # Middleware
    "add_observability_middleware",
    # Traced graph
    "get_traced_graph",
    "build_traced_graph",
]
