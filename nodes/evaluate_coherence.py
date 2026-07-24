"""
Node 6: evaluate_coherence — Coherence scoring for fallback decision.

Uses LLM to judge whether the synthesized answer is clinically coherent,
logically consistent, and grounded in retrieved documents.

Reuses fallback.py logic, but routes through LLMClient.
"""

from agent_state import AgentState
from fallback import FallbackMechanism
from llm_client import LLMClient
import logging

logger = logging.getLogger(__name__)


def evaluate_coherence(state: AgentState) -> AgentState:
    """
    Evaluate the coherence of the synthesized answer.

    Uses FallbackMechanism to score answer coherence. If score is low,
    the router will trigger fallback_regen node.

    Args:
        state: Current agent state

    Returns:
        Updated state with coherence_score and should_fallback decision
    """
    trace_id = state.get("trace_id", "unknown")
    coherence_threshold = 0.6

    try:
        llm_client = LLMClient()
        fallback_mechanism = FallbackMechanism(llm_client=llm_client)

        # Evaluate coherence
        coherence_score, explanation = fallback_mechanism.evaluate_coherence(
            query=state["input_query"],
            answer=state["intermediate_answer"],
            sources=state["intermediate_sources"],
        )

        state["coherence_score"] = coherence_score
        state["coherence_explanation"] = explanation

        # Determine if fallback should trigger
        fallback_count = state.get("fallback_count", 0)
        should_fallback = coherence_score < coherence_threshold and fallback_count < 1

        state["should_fallback"] = should_fallback
        if should_fallback:
            state["fallback_reason"] = (
                f"Coherence score {coherence_score:.2f} < threshold {coherence_threshold}"
            )

        logger.info(
            f"[{trace_id}] evaluate_coherence: Score={coherence_score:.2f}. "
            f"Should fallback: {should_fallback}. Explanation: {explanation}"
        )

    except Exception as e:
        logger.error(
            f"[{trace_id}] evaluate_coherence: Failed to evaluate coherence: {str(e)}",
            exc_info=True,
        )
        # If coherence evaluation fails, don't trigger fallback (conservative)
        state["coherence_score"] = 0.7
        state["coherence_explanation"] = f"Evaluation error, defaulting to moderate coherence: {str(e)}"
        state["should_fallback"] = False
        state["error_occurred"] = True
        state["error_messages"].append(f"Coherence evaluation error: {str(e)}")

    return state
