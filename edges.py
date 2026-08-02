"""
Edge functions for conditional routing in LangGraph StateGraph.

Defines decision logic for which node executes after certain nodes.
"""

from agent_state import AgentState
from runtime_verification import evidence_limited_answer
import logging

logger = logging.getLogger(__name__)

_RECOGNIZED_VERIFICATION_STATUSES = {
    "accept",
    "retry_retrieval",
    "retry_synthesis",
    "evidence_limited",
}


def after_classify_intent(state: AgentState) -> str:
    """
    Router after classify_intent node.

    If query is not medical, skip all retrieval and synthesis stages;
    go directly to format_response with a rejection message (early exit).

    If query is medical, proceed to discover_skills normally.

    Args:
        state: Current agent state

    Returns:
        Next node name: "discover_skills" or "format_response"
    """
    trace_id = state.get("trace_id", "unknown")

    if not state["is_medical_query"]:
        logger.info(
            f"[{trace_id}] Non-medical query detected. "
            f"Confidence: {state['classification_confidence']:.2f}. "
            f"Reason: {state['classification_reason']} "
            f"Routing directly to format_response."
        )
        return "format_response"

    logger.info(
        f"[{trace_id}] Medical query confirmed. "
        f"Confidence: {state['classification_confidence']:.2f}. "
        f"Proceeding to skill discovery."
    )
    return "discover_skills"


def after_evaluate_coherence(state: AgentState) -> str:
    """
    Router after evaluate_coherence node.

    If coherence is low (< threshold) AND we haven't already tried fallback,
    trigger fallback regeneration.

    Otherwise, proceed directly to format_response.

    Args:
        state: Current agent state

    Returns:
        Next node name: "fallback_regen" or "format_response"
    """
    trace_id = state.get("trace_id", "unknown")
    coherence = state["coherence_score"]
    fallback_count = state["fallback_count"]
    max_fallbacks = max(
        0,
        min(
            1,
            int(state.get("context", {}).get("max_synthesis_repairs", 1)),
        ),
    )
    coherence_threshold = 0.6
    verification_decision = state.get("verification_decision")
    decision_is_valid = (
        isinstance(verification_decision, dict)
        and verification_decision.get("status")
        in _RECOGNIZED_VERIFICATION_STATUSES
        and verification_decision.get("valid") is True
    )
    if decision_is_valid:
        should_repair = (
            verification_decision.get("status") == "retry_synthesis"
        )
        routing_source = "runtime_verifier"
    else:
        # Conservative compatibility policy: absent, empty, malformed, or
        # unavailable verifier state falls back to the legacy coherence gate.
        should_repair = bool(state.get("should_fallback")) or (
            coherence < coherence_threshold
        )
        routing_source = "legacy_coherence"

    if should_repair and fallback_count < max_fallbacks:
        logger.info(
            f"[{trace_id}] Coherence score ({coherence:.2f}) < threshold ({coherence_threshold}). "
            f"Fallback count: {fallback_count}. "
            f"Triggering fallback regeneration. "
            f"Reason: {state['coherence_explanation']}. "
            f"Routing source: {routing_source}"
        )
        state["should_fallback"] = True
        state["fallback_reason"] = (
            f"coherence_score={coherence:.2f} < threshold={coherence_threshold}"
        )
        return "fallback_regen"

    if should_repair and fallback_count >= max_fallbacks:
        logger.warning(
            f"[{trace_id}] Coherence score ({coherence:.2f}) is low, "
            f"but fallback already attempted {fallback_count} time(s). "
            f"Skipping further fallback and proceeding to format_response."
        )
        state["should_fallback"] = False
        state["evidence_limited"] = True
        state["is_partial_response"] = True
        state["intermediate_answer"] = evidence_limited_answer(
            "Runtime verification could not complete synthesis repair "
            "within the repair budget."
        )

    logger.info(
        "[%s] Proceeding directly to format_response "
        "(coherence=%.2f, routing_source=%s).",
        trace_id,
        coherence,
        routing_source,
    )
    return "format_response"
