"""
Traced graph wrapper — instruments the LangGraph without modifying graph.py.

Imports the original ``build_graph`` and wraps each node function with:
- ``@trace_node``  (LangSmith tracing)
- Prometheus ``mra_node_execution_seconds`` + ``mra_tool_execution_seconds``

Usage
-----
Replace ``from graph import get_graph`` with::

    from observability.traced_graph import get_traced_graph

The returned compiled graph behaves identically to the original but
emits observability signals on every invocation.
"""

import functools
import logging
import time
from typing import Any, Callable, Dict, Optional

from observability.langsmith_tracer import trace_node
from observability.prometheus_metrics import (
    record_fallback,
    record_node_execution,
    record_tool_execution,
    record_error,
)

logger = logging.getLogger(__name__)


def _instrument_node(name: str, fn: Callable) -> Callable:
    """Wrap a node function with LangSmith tracing + Prometheus metrics.

    The wrapper:
    1. Records ``mra_node_execution_seconds`` for every node.
    2. After ``parallel_retrieve``, also records per-tool durations.
    3. After ``fallback_regen``, records ``mra_fallbacks_total`` if triggered.
    4. On error, records ``mra_errors_total``.
    5. Delegates to ``@trace_node`` for LangSmith.
    """

    @trace_node(name)
    @functools.wraps(fn)
    def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        try:
            result = fn(state)

            # Post-node metric hooks
            if name == "parallel_retrieve":
                _record_tool_metrics(result)
            elif name == "fallback_regen":
                if result.get("fallback_triggered"):
                    record_fallback()

            return result

        except Exception:
            record_error(name)
            raise

        finally:
            record_node_execution(name, time.time() - start)

    return wrapper


def _record_tool_metrics(state: Dict[str, Any]) -> None:
    """Extract per-tool timing from retrieval results and record to Prometheus."""
    retrieval_time = state.get("retrieval_time_sec", {})
    for tool_name, duration in retrieval_time.items():
        record_tool_execution(tool_name, duration)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

_traced_graph_instance: Optional[Any] = None


def build_traced_graph() -> Any:
    """Build a LangGraph StateGraph with tracing + metrics on every node.

    Imports the original node functions, wraps them, and assembles an
    identical graph topology.  Does **not** modify ``graph.py`` or any
    node source file.
    """
    from langgraph.graph import StateGraph, END  # type: ignore[import-untyped]
    from agent_state import AgentState
    from edges import after_classify_intent, after_evaluate_coherence

    # Import original (un-instrumented) node functions
    from nodes.classify_intent import classify_intent
    from nodes.discover_skills import discover_skills
    from nodes.parallel_retrieve import parallel_retrieve
    from nodes.synthesise import synthesise
    from nodes.score_confidence import score_confidence
    from nodes.evaluate_coherence import evaluate_coherence
    from nodes.fallback_regen import fallback_regen
    from nodes.format_response import format_response

    # Wrap each node
    nodes = {
        "classify_intent": _instrument_node("classify_intent", classify_intent),
        "discover_skills": _instrument_node("discover_skills", discover_skills),
        "parallel_retrieve": _instrument_node("parallel_retrieve", parallel_retrieve),
        "synthesise": _instrument_node("synthesise", synthesise),
        "score_confidence": _instrument_node("score_confidence", score_confidence),
        "evaluate_coherence": _instrument_node("evaluate_coherence", evaluate_coherence),
        "fallback_regen": _instrument_node("fallback_regen", fallback_regen),
        "format_response": _instrument_node("format_response", format_response),
    }

    # Build graph with same topology as graph.py
    graph = StateGraph(AgentState)

    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        after_classify_intent,
        {
            "discover_skills": "discover_skills",
            "format_response": "format_response",
        },
    )

    graph.add_edge("discover_skills", "parallel_retrieve")
    graph.add_edge("parallel_retrieve", "synthesise")
    graph.add_edge("synthesise", "score_confidence")
    graph.add_edge("score_confidence", "evaluate_coherence")

    graph.add_conditional_edges(
        "evaluate_coherence",
        after_evaluate_coherence,
        {
            "fallback_regen": "fallback_regen",
            "format_response": "format_response",
        },
    )

    graph.add_edge("fallback_regen", "format_response")
    graph.add_edge("format_response", END)

    compiled = graph.compile()
    logger.info(
        "Traced LangGraph compiled — all 8 nodes instrumented with "
        "LangSmith tracing + Prometheus metrics"
    )
    return compiled


def get_traced_graph() -> Any:
    """Get or create the global traced graph instance (lazy singleton)."""
    global _traced_graph_instance
    if _traced_graph_instance is None:
        _traced_graph_instance = build_traced_graph()
    return _traced_graph_instance
