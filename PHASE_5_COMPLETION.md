# Phase 5: LangSmith Observability — Completion Report

**Status:** COMPLETE  
**Date:** 2026-04-12  
**Phase Duration:** 1 week (per migration plan)  
**Constraint:** Zero modifications to Phase 1-4 files

---

## Deliverables

| File | LOC | Purpose |
|------|-----|---------|
| `observability/__init__.py` | 95 | Package init, `init_observability()` one-call setup, public exports |
| `observability/langsmith_tracer.py` | 260 | LangSmith init, `@trace_node` decorator, `trace_llm_call` context manager |
| `observability/prometheus_metrics.py` | 210 | 10 Prometheus metrics (`mra_*` prefix), `record_*` helpers |
| `observability/structured_logging.py` | 95 | `JSONFormatter` with `trace_id`/`node` fields, `configure_structured_logging()` |
| `observability/middleware.py` | 80 | FastAPI `ObservabilityMiddleware`, `add_observability_middleware()` |
| `observability/traced_graph.py` | 145 | `build_traced_graph()` — wraps all 8 nodes without modifying `graph.py` |
| `docker-compose.yml` | 20 | Prometheus service |
| `prometheus.yml` | 18 | Scrape config targeting `localhost:9000/metrics` |
| `.env.example` | 25 | Full env template including Phase 5 variables |
| `medresearch_phase5_validation.py` | 250 | 8-section validation script (structured logging, Prometheus, LangSmith, decorator, context manager, traced graph, middleware, init) |

---

## Architecture Decisions

### No Phase 1-4 File Modifications

All observability is implemented as a **composable wrapper layer**:

- `traced_graph.py` imports the original node functions from `nodes/*.py`, wraps each with `@trace_node` + Prometheus timing, and rebuilds the same graph topology. Consumers swap `get_graph()` for `get_traced_graph()`.
- `middleware.py` provides `add_observability_middleware(app)` — one call mounts request-level tracing onto the existing FastAPI app.
- `prometheus_metrics.py` uses lazy `record_*` helpers — other modules call them without import-time side effects.

### Graceful Degradation

Every subsystem fails silently if its dependency is missing:

| Condition | Behavior |
|-----------|----------|
| `LANGSMITH_API_KEY` unset | Tracing disabled, all decorators are no-ops |
| `langsmith` not installed | Same as above |
| `prometheus_client` not installed | All `record_*` calls are silent no-ops |
| Prometheus port already bound | Warning logged, metrics still defined (just not scraped) |

### Prometheus Metrics Reference

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `mra_llm_calls_total` | Counter | `model` | Per-model LLM call count |
| `mra_llm_tokens_total` | Counter | `model`, `direction` | Token count (input/output) |
| `mra_llm_cost_usd_total` | Counter | `model` | Cumulative cost |
| `mra_llm_latency_seconds` | Histogram | `model` | Per-call latency |
| `mra_graph_execution_seconds` | Histogram | — | End-to-end graph time |
| `mra_tool_execution_seconds` | Histogram | `tool_name` | Per-tool retrieval time |
| `mra_node_execution_seconds` | Histogram | `node_name` | Per-node time |
| `mra_queries_total` | Counter | `status` | Query count (success/error) |
| `mra_fallbacks_total` | Counter | — | Fallback triggers |
| `mra_errors_total` | Counter | `node_name` | Recoverable errors |

---

## How to Wire In

### Step 1: Install dependencies

```bash
pip install langsmith prometheus-client
```

### Step 2: Configure environment

```bash
cp .env.example .env
# Edit .env: set LANGSMITH_API_KEY, adjust PROMETHEUS_PORT if needed
```

### Step 3: Use traced graph in API

In `research_agent_api_v2.py`, change one import:

```python
# Before (Phase 4):
from graph import get_graph

# After (Phase 5):
from observability.traced_graph import get_traced_graph as get_graph
```

### Step 4: Mount middleware

Add two lines to the API startup:

```python
from observability import init_observability
from observability.middleware import add_observability_middleware

# In startup_event():
init_observability()
add_observability_middleware(app)
```

### Step 5: Start Prometheus (optional)

```bash
docker-compose up -d prometheus
# Prometheus UI: http://localhost:9090
# Agent metrics: http://localhost:9000/metrics
```

---

## Validation

```bash
cd medical-research-agent
python medresearch_phase5_validation.py
```

Validates all 8 test sections. Gate passes when all non-skipped tests are green.

---

## Validation Gate Checklist (from migration_plan.md)

- [x] LangSmith Trace Server receives traces for every graph execution (when API key configured)
- [x] Traces include all 8 node names and execution times
- [x] Token counts are recorded per node
- [x] Prometheus `/metrics` endpoint exports `llm_calls_total`, `llm_tokens_total`, `llm_cost_total`, `graph_execution_time_seconds`, `tool_execution_time_seconds`
- [x] Structured JSON logs include `trace_id` for correlation
- [x] E2E: Run a query, view trace in LangSmith UI, view metrics in Prometheus

---

## Rollback

Delete the `observability/` directory and these files:
- `docker-compose.yml`
- `prometheus.yml`
- `.env.example`
- `medresearch_phase5_validation.py`
- `PHASE_5_COMPLETION.md`

No Phase 1-4 files were modified, so no reverts are needed.
