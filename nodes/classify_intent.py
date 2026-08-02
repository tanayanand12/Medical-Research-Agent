"""
Node 1: classify_intent — Medical domain filtering.

Determines if query is within medical/clinical domain.
If not medical, triggers early exit gate.
"""

from agent_state import AgentState
from evaluation_core import RuntimeDeadlineExceeded
from query_classifier import QueryClassifier
from llm_client import LLMCallResult, LLMClient
from runtime_verification import build_attempt_event
import logging
import time

logger = logging.getLogger(__name__)


def _record_classifier_call(
    state: AgentState, trace_id: str, call_result: object
) -> None:
    if not isinstance(call_result, LLMCallResult):
        return
    event = build_attempt_event(
        trace_id=str(trace_id),
        attempt_id=f"{trace_id}:classifier:1",
        parent_attempt_id=None,
        stage="intent_classification",
        component="query_classifier",
        status=str(call_result.status or "success"),
        repair_status="initial",
        model=call_result.model,
        model_revision=call_result.model_revision,
        tokens_in=call_result.tokens_in,
        tokens_out=call_result.tokens_out,
        cost_usd=call_result.cost_usd,
        latency_sec=call_result.latency_sec,
        finish_reason=call_result.finish_reason,
        deadline_exhausted=(
            call_result.error_type == "RuntimeDeadlineExceeded"
        ),
        error_type=call_result.error_type,
        provider_metadata=call_result.provider_metadata,
    )
    attempts = list(state.get("attempt_telemetry") or [])
    if any(
        item.get("event_id") == event["event_id"] for item in attempts
    ):
        return
    attempts.append(event)
    usage = dict(state.get("token_usage") or {})
    state["token_usage"] = {
        "input": int(usage.get("input") or 0) + call_result.tokens_in,
        "output": int(usage.get("output") or 0) + call_result.tokens_out,
        "total": int(usage.get("total") or 0)
        + call_result.tokens_in
        + call_result.tokens_out,
    }
    state["cost_estimate"] = float(
        state.get("cost_estimate") or 0.0
    ) + call_result.cost_usd
    state["attempt_telemetry"] = attempts


def classify_intent(state: AgentState) -> AgentState:
    """
    Classify whether query is medical domain.

    Reuses legacy query_classifier.py logic, but now routes LLM calls
    through LLMClient (not hardcoded OpenAI).

    Args:
        state: Current agent state

    Returns:
        Updated state with classification results
    """
    trace_id = state.get("trace_id", "unknown")
    context = state.setdefault("context", {})
    if context.get("_runtime_deadline_at_monotonic") is None:
        context["_runtime_deadline_at_monotonic"] = (
            time.monotonic()
            + float(context.get("runtime_verification_deadline_sec", 60.0))
        )

    try:
        # Instantiate classifier with LLMClient
        llm_client = LLMClient()
        classifier = QueryClassifier(llm_client=llm_client)

        # Classify query
        deadline_at = state.get("context", {}).get(
            "_runtime_deadline_at_monotonic"
        )
        llm_kwargs = {
            "_telemetry_stage": "intent_classification",
            "_telemetry_attempt_id": f"{trace_id}:classifier:1",
            "_telemetry_repair_status": "initial",
        }
        if deadline_at is not None:
            remaining = float(deadline_at) - time.monotonic()
            if remaining <= 0:
                raise RuntimeDeadlineExceeded(
                    "runtime deadline expired before classification"
                )
            llm_kwargs.update(
                {
                    "timeout": max(0.1, remaining),
                    "client_max_attempts": 1,
                    "deadline_at": float(deadline_at),
                }
            )
        is_medical, confidence, reason = classifier.classify_with_reason(
            state["input_query"],
            llm_kwargs=llm_kwargs,
        )

        # Update state
        state["is_medical_query"] = is_medical
        state["classification_confidence"] = confidence
        state["classification_reason"] = reason

        logger.info(
            "[%s] classify_intent is_medical=%s confidence=%.2f",
            trace_id,
            is_medical,
            confidence,
        )
        _record_classifier_call(
            state, str(trace_id), classifier.last_call_result
        )

    except RuntimeDeadlineExceeded:
        if "classifier" in locals():
            _record_classifier_call(
                state, str(trace_id), classifier.last_call_result
            )
        logger.warning(
            "[%s] classify_intent deadline exhausted", trace_id
        )
        state["is_medical_query"] = True
        state["classification_confidence"] = 0.0
        state["classification_reason"] = "runtime_deadline_exhausted"
        state["error_occurred"] = True
        state["is_partial_response"] = True
        state["error_messages"].append(
            "Classification error_type=RuntimeDeadlineExceeded"
        )
    except Exception as e:
        logger.error(
            "[%s] classify_intent failed error_type=%s",
            trace_id,
            type(e).__name__,
        )
        # Fallback: assume medical if classifier fails (fail-safe)
        state["is_medical_query"] = True
        state["classification_confidence"] = 0.5
        state["classification_reason"] = (
            "Classifier failed; conservatively treating the query as medical."
        )
        state["error_occurred"] = True
        state["error_messages"].append(
            f"Classification error_type={type(e).__name__}"
        )

    return state
