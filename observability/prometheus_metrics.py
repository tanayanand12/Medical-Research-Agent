"""
Prometheus metrics for Medical Research Agent.

Defines all application-level metrics and provides helper functions that
the rest of the codebase can call without importing ``prometheus_client``
directly.

Metrics
-------
- ``mra_llm_calls_total``              Counter   — per-model LLM call count
- ``mra_llm_tokens_total``             Counter   — cumulative token count (direction label)
- ``mra_llm_cost_usd_total``           Counter   — cumulative cost in USD
- ``mra_llm_latency_seconds``          Histogram — per-call LLM latency
- ``mra_graph_execution_seconds``       Histogram — end-to-end graph execution time
- ``mra_tool_execution_seconds``        Histogram — per-tool retrieval latency
- ``mra_node_execution_seconds``        Histogram — per-node execution latency
- ``mra_queries_total``                 Counter   — total queries received
- ``mra_fallbacks_total``               Counter   — fallback regenerations triggered
- ``mra_errors_total``                  Counter   — recoverable errors

All metrics use the ``mra_`` prefix (Medical Research Agent).

Usage
-----
Call ``init_prometheus(port)`` at startup to expose ``/metrics`` on the given
port (default 9000).  Then use the ``record_*`` helpers from any module::

    from observability.prometheus_metrics import record_llm_call

    record_llm_call(model="gpt-4o", tokens_in=150, tokens_out=300,
                    cost_usd=0.006, latency_sec=1.2)

If ``prometheus_client`` is not installed, all helpers become silent no-ops.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — degrade gracefully if prometheus_client is missing
# ---------------------------------------------------------------------------
_prom_available = False

try:
    from prometheus_client import (  # type: ignore[import-untyped]
        Counter,
        Histogram,
        start_http_server,
    )
    _prom_available = True
except ImportError:
    logger.info(
        "prometheus_client not installed — metrics disabled. "
        "Install with: pip install prometheus-client"
    )

# ---------------------------------------------------------------------------
# Metric definitions (created only when prometheus_client is available)
# ---------------------------------------------------------------------------

if _prom_available:
    LLM_CALLS = Counter(
        "mra_llm_calls_total",
        "Total LLM calls",
        ["model"],
    )

    LLM_TOKENS = Counter(
        "mra_llm_tokens_total",
        "Cumulative token count",
        ["model", "direction"],  # direction = "input" | "output"
    )

    LLM_COST = Counter(
        "mra_llm_cost_usd_total",
        "Cumulative LLM cost in USD",
        ["model"],
    )

    LLM_LATENCY = Histogram(
        "mra_llm_latency_seconds",
        "LLM call latency in seconds",
        ["model"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0),
    )

    GRAPH_EXECUTION = Histogram(
        "mra_graph_execution_seconds",
        "End-to-end graph execution time in seconds",
        buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 30.0, 60.0),
    )

    TOOL_EXECUTION = Histogram(
        "mra_tool_execution_seconds",
        "Per-tool retrieval latency in seconds",
        ["tool_name"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0),
    )

    NODE_EXECUTION = Histogram(
        "mra_node_execution_seconds",
        "Per-node execution latency in seconds",
        ["node_name"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0),
    )

    QUERIES_TOTAL = Counter(
        "mra_queries_total",
        "Total queries received",
        ["status"],  # "success" | "error"
    )

    FALLBACKS_TOTAL = Counter(
        "mra_fallbacks_total",
        "Fallback regenerations triggered",
    )

    ERRORS_TOTAL = Counter(
        "mra_errors_total",
        "Recoverable errors (per node)",
        ["node_name"],
    )

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

_metrics_server_started = False


def init_prometheus(port: Optional[int] = None) -> bool:
    """Start the Prometheus HTTP metrics server.

    Parameters
    ----------
    port : int, optional
        Port to serve ``/metrics`` on.  Falls back to ``PROMETHEUS_PORT``
        env var, then 9000.

    Returns
    -------
    bool
        True if the server started successfully.
    """
    global _metrics_server_started

    if not _prom_available:
        logger.info("Prometheus metrics disabled (prometheus_client not installed)")
        return False

    if _metrics_server_started:
        logger.debug("Prometheus metrics server already running")
        return True

    port = port or int(os.getenv("PROMETHEUS_PORT", "9000"))

    try:
        start_http_server(port)
        _metrics_server_started = True
        logger.info("Prometheus metrics server started on port %d", port)
        return True
    except OSError as exc:
        logger.warning(
            "Could not start Prometheus metrics server on port %d: %s",
            port, exc,
        )
        return False


def is_metrics_enabled() -> bool:
    """Return True if Prometheus metrics are available."""
    return _prom_available


# ---------------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------------

def record_llm_call(
    model: str,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    latency_sec: float = 0.0,
) -> None:
    """Record a single LLM call across all relevant metrics."""
    if not _prom_available:
        return
    LLM_CALLS.labels(model=model).inc()
    LLM_TOKENS.labels(model=model, direction="input").inc(tokens_in)
    LLM_TOKENS.labels(model=model, direction="output").inc(tokens_out)
    LLM_COST.labels(model=model).inc(cost_usd)
    LLM_LATENCY.labels(model=model).observe(latency_sec)


def record_graph_execution(duration_sec: float) -> None:
    """Record a complete graph execution duration."""
    if not _prom_available:
        return
    GRAPH_EXECUTION.observe(duration_sec)


def record_tool_execution(tool_name: str, duration_sec: float) -> None:
    """Record a single MCP tool execution duration."""
    if not _prom_available:
        return
    TOOL_EXECUTION.labels(tool_name=tool_name).observe(duration_sec)


def record_node_execution(node_name: str, duration_sec: float) -> None:
    """Record a single graph node execution duration."""
    if not _prom_available:
        return
    NODE_EXECUTION.labels(node_name=node_name).observe(duration_sec)


def record_query(status: str = "success") -> None:
    """Increment the query counter.  ``status`` is ``"success"`` or ``"error"``."""
    if not _prom_available:
        return
    QUERIES_TOTAL.labels(status=status).inc()


def record_fallback() -> None:
    """Increment the fallback counter."""
    if not _prom_available:
        return
    FALLBACKS_TOTAL.inc()


def record_error(node_name: str) -> None:
    """Increment the per-node error counter."""
    if not _prom_available:
        return
    ERRORS_TOTAL.labels(node_name=node_name).inc()
