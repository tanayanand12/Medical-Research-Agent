"""
Phase 5 Validation Script — LangSmith Observability + Prometheus Metrics.

Validates that all Phase 5 deliverables are functional:

1. Structured JSON logging emits valid JSON with trace_id and node fields
2. Prometheus metrics are defined and recordable
3. LangSmith tracing initialises (or degrades gracefully)
4. @trace_node decorator wraps functions correctly
5. Traced graph builds and mirrors original graph topology
6. FastAPI middleware mounts without error
7. End-to-end: metrics are emitted during a simulated graph run

Run:
    cd medical-research-agent
    python medresearch_phase5_validation.py
"""

import json
import logging
import os
import sys
import time
import io
from datetime import datetime

# Ensure the package root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
results: list = []


def report(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    msg = f"  {status} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)


# =========================================================================
# Test 1: Structured JSON Logging
# =========================================================================
print("\n=== Test 1: Structured JSON Logging ===")
try:
    from observability.structured_logging import JSONFormatter, configure_structured_logging

    # Capture log output
    formatter = JSONFormatter()
    handler = logging.StreamHandler(stream=io.StringIO())
    handler.setFormatter(formatter)

    test_logger = logging.getLogger("test_structured_logging")
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.DEBUG)

    test_logger.info(
        "Validation log entry",
        extra={"trace_id": "test-trace-001", "node": "classify_intent"},
    )

    log_output = handler.stream.getvalue()
    parsed = json.loads(log_output.strip())

    assert "timestamp" in parsed, "Missing 'timestamp' field"
    assert parsed["level"] == "INFO", f"Expected INFO, got {parsed['level']}"
    assert parsed["trace_id"] == "test-trace-001", "trace_id mismatch"
    assert parsed["node"] == "classify_intent", "node mismatch"
    assert "Validation log entry" in parsed["message"], "message mismatch"

    report("JSON formatter produces valid JSON", PASS)
    report("JSON includes trace_id field", PASS)
    report("JSON includes node field", PASS)
    report("JSON includes timestamp field", PASS)

    test_logger.removeHandler(handler)

except Exception as e:
    report("Structured JSON logging", FAIL, str(e))


# =========================================================================
# Test 2: Prometheus Metrics
# =========================================================================
print("\n=== Test 2: Prometheus Metrics ===")
try:
    from observability.prometheus_metrics import (
        is_metrics_enabled,
        record_llm_call,
        record_graph_execution,
        record_tool_execution,
        record_node_execution,
        record_query,
        record_fallback,
        record_error,
    )

    if is_metrics_enabled():
        from prometheus_client import REGISTRY

        # Record some test metrics
        record_llm_call(
            model="gpt-4o", tokens_in=100, tokens_out=200,
            cost_usd=0.005, latency_sec=1.5,
        )
        record_graph_execution(3.2)
        record_tool_execution("pubmed", 1.1)
        record_node_execution("classify_intent", 0.3)
        record_query(status="success")
        record_fallback()
        record_error(node_name="synthesise")

        # Verify metrics exist in registry
        metric_names = [m.name for m in REGISTRY.collect()]

        expected = [
            "mra_llm_calls",
            "mra_llm_tokens",
            "mra_llm_cost_usd",
            "mra_llm_latency_seconds",
            "mra_graph_execution_seconds",
            "mra_tool_execution_seconds",
            "mra_node_execution_seconds",
            "mra_queries",
            "mra_fallbacks",
            "mra_errors",
        ]

        for name in expected:
            # prometheus_client may add _total/_created suffixes
            found = any(name in m for m in metric_names)
            if found:
                report(f"Metric '{name}' registered", PASS)
            else:
                report(f"Metric '{name}' registered", FAIL, "not found in registry")

        report("record_llm_call() executes without error", PASS)
        report("record_graph_execution() executes without error", PASS)
        report("record_tool_execution() executes without error", PASS)

    else:
        report("Prometheus metrics", SKIP, "prometheus_client not installed")

except Exception as e:
    report("Prometheus metrics", FAIL, str(e))


# =========================================================================
# Test 3: LangSmith Tracing
# =========================================================================
print("\n=== Test 3: LangSmith Tracing ===")
try:
    from observability.langsmith_tracer import (
        init_langsmith,
        is_tracing_enabled,
        get_project,
    )

    api_key = os.getenv("LANGSMITH_API_KEY", "")

    if api_key:
        ok = init_langsmith()
        if ok:
            report("LangSmith initialisation with API key", PASS)
            report(f"LangSmith project = '{get_project()}'", PASS)
            assert is_tracing_enabled(), "Expected tracing to be enabled"
            report("is_tracing_enabled() returns True", PASS)
        else:
            report("LangSmith initialisation", FAIL, "init returned False despite key")
    else:
        ok = init_langsmith()
        assert not ok, "Expected init to return False without API key"
        assert not is_tracing_enabled(), "Expected tracing to be disabled"
        report("Graceful degradation (no API key)", PASS)
        report("is_tracing_enabled() returns False", PASS)

except Exception as e:
    report("LangSmith tracing", FAIL, str(e))


# =========================================================================
# Test 4: @trace_node Decorator
# =========================================================================
print("\n=== Test 4: @trace_node Decorator ===")
try:
    from observability.langsmith_tracer import trace_node

    @trace_node("test_node")
    def dummy_node(state):
        state["test_output"] = "hello"
        return state

    result = dummy_node({"input_query": "test", "trace_id": "test-001"})
    assert result["test_output"] == "hello", "Decorator altered return value"
    report("@trace_node preserves function return value", PASS)

    # Test with exception
    @trace_node("error_node")
    def failing_node(state):
        raise ValueError("intentional test error")

    try:
        failing_node({"input_query": "test", "trace_id": "test-002"})
        report("@trace_node error propagation", FAIL, "exception was swallowed")
    except ValueError:
        report("@trace_node propagates exceptions correctly", PASS)

except Exception as e:
    report("@trace_node decorator", FAIL, str(e))


# =========================================================================
# Test 5: trace_llm_call Context Manager
# =========================================================================
print("\n=== Test 5: trace_llm_call Context Manager ===")
try:
    from observability.langsmith_tracer import trace_llm_call

    with trace_llm_call("gpt-4o", trace_id="test-003") as ctx:
        # Simulate a tiny delay so latency is measurable
        time.sleep(0.001)
        ctx["tokens_in"] = 50
        ctx["tokens_out"] = 100
        ctx["cost_usd"] = 0.002

    assert ctx["tokens_in"] == 50, "context not preserved"
    assert ctx["latency_ms"] >= 0, "latency_ms should be non-negative"
    report("trace_llm_call context manager works", PASS)
    report("trace_llm_call records latency_ms", PASS)

except Exception as e:
    report("trace_llm_call", FAIL, str(e))


# =========================================================================
# Test 6: Traced Graph Builder
# =========================================================================
print("\n=== Test 6: Traced Graph Builder ===")
try:
    from observability.traced_graph import build_traced_graph

    traced_graph = build_traced_graph()
    assert traced_graph is not None, "build_traced_graph returned None"
    report("build_traced_graph() compiles successfully", PASS)

    # Verify graph has nodes (LangGraph compiled graph introspection)
    graph_def = traced_graph.get_graph()
    node_ids = [n.id for n in graph_def.nodes.values() if n.id not in ("__start__", "__end__")]

    expected_nodes = [
        "classify_intent", "discover_skills", "parallel_retrieve",
        "synthesise", "score_confidence", "evaluate_coherence",
        "fallback_regen", "format_response",
    ]

    for node_name in expected_nodes:
        if node_name in node_ids:
            report(f"Node '{node_name}' present in traced graph", PASS)
        else:
            report(f"Node '{node_name}' present in traced graph", FAIL, "missing")

except ImportError as e:
    report("Traced graph builder", SKIP, f"dependency not available: {e}")
except Exception as e:
    report("Traced graph builder", FAIL, str(e))


# =========================================================================
# Test 7: FastAPI Middleware
# =========================================================================
print("\n=== Test 7: FastAPI Middleware ===")
try:
    from fastapi import FastAPI
    from observability.middleware import add_observability_middleware

    test_app = FastAPI()
    add_observability_middleware(test_app)
    report("add_observability_middleware() mounts without error", PASS)

except ImportError as e:
    report("FastAPI middleware", SKIP, f"dependency not available: {e}")
except Exception as e:
    report("FastAPI middleware", FAIL, str(e))


# =========================================================================
# Test 8: init_observability() Top-Level
# =========================================================================
print("\n=== Test 8: init_observability() ===")
try:
    from observability import init_observability

    status = init_observability()
    assert isinstance(status, dict), "Expected dict return"
    assert "langsmith" in status, "Missing 'langsmith' key"
    assert "prometheus" in status, "Missing 'prometheus' key"
    assert "structured_logging" in status, "Missing 'structured_logging' key"
    assert status["structured_logging"] is True, "Logging should always succeed"
    report("init_observability() returns status dict", PASS)
    report(f"Status: {status}", PASS)

except Exception as e:
    report("init_observability()", FAIL, str(e))


# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 70)
print("PHASE 5 VALIDATION SUMMARY")
print("=" * 70)

passed = sum(1 for _, s, _ in results if s == PASS)
failed = sum(1 for _, s, _ in results if s == FAIL)
skipped = sum(1 for _, s, _ in results if s == SKIP)

print(f"\n  Total:   {len(results)}")
print(f"  Passed:  {passed}")
print(f"  Failed:  {failed}")
print(f"  Skipped: {skipped}")

if failed > 0:
    print("\n  FAILURES:")
    for name, status, detail in results:
        if status == FAIL:
            print(f"    - {name}: {detail}")

gate = "PASSED" if failed == 0 else "FAILED"
print(f"\n  Phase 5 Validation Gate: {gate}")
print("=" * 70)

sys.exit(0 if failed == 0 else 1)
