"""
Node 1: classify_intent — Medical domain filtering.

Determines if query is within medical/clinical domain.
If not medical, triggers early exit gate.
"""

from agent_state import AgentState
from query_classifier import QueryClassifier
from llm_client import LLMClient
import logging

logger = logging.getLogger(__name__)


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

    try:
        # Instantiate classifier with LLMClient
        llm_client = LLMClient()
        classifier = QueryClassifier(llm_client=llm_client)

        # Classify query
        is_medical, confidence, reason = classifier.classify_with_reason(
            state["input_query"]
        )

        # Update state
        state["is_medical_query"] = is_medical
        state["classification_confidence"] = confidence
        state["classification_reason"] = reason

        logger.info(
            f"[{trace_id}] classify_intent: is_medical={is_medical}, "
            f"confidence={confidence:.2f}, reason={reason}"
        )

    except Exception as e:
        logger.error(
            f"[{trace_id}] classify_intent: Classification failed: {str(e)}",
            exc_info=True,
        )
        # Fallback: assume medical if classifier fails (fail-safe)
        state["is_medical_query"] = True
        state["classification_confidence"] = 0.5
        state["classification_reason"] = f"Classifier error, defaulting to medical: {str(e)}"
        state["error_occurred"] = True
        state["error_messages"].append(f"Classification error: {str(e)}")

    return state
