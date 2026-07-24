"""
LangSmith tracing integration for Medical Research Agent.

Provides:
- ``init_langsmith()``       — one-time project setup (no-op if key absent)
- ``@trace_node(name)``      — decorator that wraps a LangGraph node function
                               in a LangSmith child run, recording inputs,
                               outputs, latency, and errors.
- ``trace_llm_call(...)``    — context manager for individual LLM calls

Gracefully degrades: if ``LANGSMITH_API_KEY`` is unset or invalid, all
helpers become pass-throughs so the pipeline runs without tracing.
"""

import functools
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional

# Use perf_counter for sub-ms accuracy (time.time() has ~15ms resolution on Windows)
_clock = time.perf_counter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_langsmith_enabled: bool = False
_langsmith_client: Any = None
_langsmith_project: str = "medical-research-agent"

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_langsmith(
    api_key: Optional[str] = None,
    project: Optional[str] = None,
) -> bool:
    """Initialise LangSmith tracing.

    Parameters
    ----------
    api_key : str, optional
        LangSmith API key.  Falls back to ``LANGSMITH_API_KEY`` env var.
    project : str, optional
        LangSmith project name.  Falls back to ``LANGSMITH_PROJECT`` env var,
        then ``"medical-research-agent"``.

    Returns
    -------
    bool
        True if LangSmith was successfully initialised; False otherwise.
    """
    global _langsmith_enabled, _langsmith_client, _langsmith_project

    api_key = api_key or os.getenv("LANGSMITH_API_KEY", "")
    _langsmith_project = (
        project
        or os.getenv("LANGSMITH_PROJECT", "medical-research-agent")
    )

    if not api_key:
        logger.info(
            "LANGSMITH_API_KEY not set — tracing disabled. "
            "Set the env var to enable LangSmith tracing."
        )
        _langsmith_enabled = False
        return False

    try:
        from langsmith import Client  # type: ignore[import-untyped]

        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", _langsmith_project)

        _langsmith_client = Client(api_key=api_key)
        _langsmith_enabled = True
        logger.info(
            "LangSmith tracing enabled — project=%s", _langsmith_project
        )
        return True

    except ImportError:
        logger.warning(
            "langsmith package not installed — tracing disabled. "
            "Install with: pip install langsmith"
        )
        _langsmith_enabled = False
        return False

    except Exception as exc:
        logger.warning(
            "LangSmith initialisation failed — tracing disabled: %s", exc
        )
        _langsmith_enabled = False
        return False


def is_tracing_enabled() -> bool:
    """Return True if LangSmith tracing is active."""
    return _langsmith_enabled


def get_client() -> Any:
    """Return the LangSmith Client instance (or None)."""
    return _langsmith_client


def get_project() -> str:
    """Return the configured LangSmith project name."""
    return _langsmith_project


# ---------------------------------------------------------------------------
# Node decorator
# ---------------------------------------------------------------------------

def trace_node(name: str) -> Callable:
    """Decorator that wraps a LangGraph node function with LangSmith tracing.

    When tracing is enabled, each invocation creates a LangSmith run with:
    - ``run_type = "chain"``
    - ``name``   = the supplied name (e.g. ``"classify_intent"``)
    - ``inputs`` = ``{"input_query": ..., "trace_id": ...}``
    - ``outputs``= subset of updated state fields
    - ``extra``  = ``{"latency_ms": ...}``

    When tracing is disabled the decorator is a no-op pass-through.

    Usage::

        @trace_node("classify_intent")
        def classify_intent(state: AgentState) -> AgentState:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            if not _langsmith_enabled:
                return fn(state)

            try:
                from langsmith.run_trees import RunTree  # type: ignore[import-untyped]
            except ImportError:
                return fn(state)

            trace_id = state.get("trace_id", "unknown")
            run = RunTree(
                name=name,
                run_type="chain",
                inputs={
                    "input_query": state.get("input_query", ""),
                    "trace_id": trace_id,
                },
                project_name=_langsmith_project,
                tags=["phase5", f"trace:{trace_id}"],
            )

            start = _clock()
            error_info: Optional[str] = None

            try:
                result = fn(state)
                return result
            except Exception as exc:
                error_info = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                latency_ms = (_clock() - start) * 1000

                output_keys = _output_keys_for_node(name)
                outputs = {}
                # result may not exist if fn raised
                result_state = result if error_info is None else state
                for key in output_keys:
                    if key in result_state:
                        val = result_state[key]
                        # Serialise datetimes
                        if hasattr(val, "isoformat"):
                            val = val.isoformat()
                        outputs[key] = val

                run.end(
                    outputs=outputs,
                    error=error_info,
                    extra={"latency_ms": round(latency_ms, 1)},
                )
                try:
                    run.post()
                except Exception as post_err:
                    logger.debug("LangSmith run.post() failed: %s", post_err)

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# LLM call tracing
# ---------------------------------------------------------------------------

@contextmanager
def trace_llm_call(
    model: str,
    trace_id: str = "unknown",
    call_type: str = "chat",
) -> Generator[Dict[str, Any], None, None]:
    """Context manager for tracing an individual LLM call to LangSmith.

    Usage::

        with trace_llm_call("gpt-4o", trace_id="abc") as run_ctx:
            result = litellm.completion(...)
            run_ctx["tokens_in"]  = usage.prompt_tokens
            run_ctx["tokens_out"] = usage.completion_tokens

    The context dict is used to post-fill output metadata.
    When tracing is disabled the context is still yielded (for metrics)
    but nothing is posted to LangSmith.
    """
    ctx: Dict[str, Any] = {
        "model": model,
        "trace_id": trace_id,
        "call_type": call_type,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
        "latency_ms": 0.0,
        "error": None,
    }

    if not _langsmith_enabled:
        start = _clock()
        try:
            yield ctx
        finally:
            ctx["latency_ms"] = (_clock() - start) * 1000
        return

    try:
        from langsmith.run_trees import RunTree  # type: ignore[import-untyped]
    except ImportError:
        yield ctx
        return

    run = RunTree(
        name=f"llm_{call_type}",
        run_type="llm",
        inputs={"model": model, "trace_id": trace_id},
        project_name=_langsmith_project,
        tags=[f"model:{model}", f"trace:{trace_id}"],
    )

    start = _clock()
    try:
        yield ctx
    except Exception as exc:
        ctx["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        ctx["latency_ms"] = (_clock() - start) * 1000
        run.end(
            outputs={
                "tokens_in": ctx["tokens_in"],
                "tokens_out": ctx["tokens_out"],
                "cost_usd": ctx["cost_usd"],
                "latency_ms": ctx["latency_ms"],
            },
            error=ctx["error"],
        )
        try:
            run.post()
        except Exception as post_err:
            logger.debug("LangSmith llm run.post() failed: %s", post_err)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NODE_OUTPUT_KEYS: Dict[str, list] = {
    "classify_intent": [
        "is_medical_query", "classification_confidence", "classification_reason",
    ],
    "discover_skills": ["discovered_skills", "skill_scores"],
    "parallel_retrieve": [
        "retrieval_results", "tokens_used",
        "retrieval_time_sec", "total_retrieval_time_sec",
    ],
    "synthesise": [
        "intermediate_answer", "intermediate_sources",
        "intermediate_model_used", "synthesis_time_sec",
    ],
    "score_confidence": ["confidence_score", "coverage_explanation"],
    "evaluate_coherence": [
        "coherence_score", "coherence_explanation", "should_fallback",
    ],
    "fallback_regen": [
        "fallback_answer", "fallback_triggered", "fallback_count",
    ],
    "format_response": [
        "output_answer", "output_sources", "output_citations",
        "execution_time_sec", "cost_estimate",
    ],
}


def _output_keys_for_node(name: str) -> list:
    """Return the AgentState keys that a given node writes."""
    return _NODE_OUTPUT_KEYS.get(name, [])
