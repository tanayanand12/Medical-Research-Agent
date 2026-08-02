"""
Node 5: score_confidence — Coverage-based confidence calculation.

Computes confidence score based on how many tools successfully returned results.
"""

from agent_state import AgentState
import logging

logger = logging.getLogger(__name__)


def score_confidence(state: AgentState) -> AgentState:
    """
    Calculate confidence score based on retrieval coverage.

    Confidence = (# tools with results) / (# tools selected)

    If no tools were selected or none returned results, confidence = 0.0

    Args:
        state: Current agent state

    Returns:
        Updated state with confidence_score and coverage_explanation
    """
    trace_id = state.get("trace_id", "unknown")

    try:
        # Count tools with successful results
        successful_tools = []
        failed_tools = []

        for tool_name in state["discovered_skills"]:
            result = state["retrieval_results"].get(tool_name, {})
            if result.get("error") is None and result.get("results"):
                successful_tools.append(tool_name)
            else:
                failed_tools.append(tool_name)

        # Calculate confidence
        total_tools = len(state["discovered_skills"])
        if total_tools == 0:
            confidence = 0.0
            explanation = "No tools were selected for retrieval."
        else:
            confidence = len(successful_tools) / total_tools
            explanation = (
                f"{len(successful_tools)}/{total_tools} tools returned results. "
                f"Successful: {successful_tools}. Failed: {failed_tools}."
            )

        state["confidence_score"] = confidence
        state["coverage_explanation"] = explanation

        logger.info(
            f"[{trace_id}] score_confidence: Confidence={confidence:.2f}. {explanation}"
        )

    except Exception as e:
        logger.error(
            "[%s] score_confidence failed error_type=%s",
            trace_id,
            type(e).__name__,
        )
        state["confidence_score"] = 0.0
        state["coverage_explanation"] = (
            f"Confidence calculation error_type={type(e).__name__}"
        )
        state["error_occurred"] = True
        state["error_messages"].append(
            f"Confidence scoring error_type={type(e).__name__}"
        )

    return state
