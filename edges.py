"""
Edge functions for conditional routing in LangGraph StateGraph.

Defines decision logic for which node executes after certain nodes.
"""

from agent_state import AgentState
import logging

logger = logging.getLogger(__name__)


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
    coherence_threshold = 0.6

    if coherence < coherence_threshold and fallback_count < 1:
        logger.info(
            f"[{trace_id}] Coherence score ({coherence:.2f}) < threshold ({coherence_threshold}). "
            f"Fallback count: {fallback_count}. "
            f"Triggering fallback regeneration. "
            f"Reason: {state['coherence_explanation']}"
        )
        state["should_fallback"] = True
        state["fallback_reason"] = (
            f"coherence_score={coherence:.2f} < threshold={coherence_threshold}"
        )
        return "fallback_regen"

    if coherence < coherence_threshold and fallback_count >= 1:
        logger.warning(
            f"[{trace_id}] Coherence score ({coherence:.2f}) is low, "
            f"but fallback already attempted {fallback_count} time(s). "
            f"Skipping further fallback and proceeding to format_response."
        )
        state["should_fallback"] = False

    logger.info(
        f"[{trace_id}] Coherence score ({coherence:.2f}) >= threshold. "
        f"Proceeding directly to format_response."
    )
    return "format_response"
