"""
Node 7: fallback_regen — Fallback answer regeneration.

Triggered when coherence score is low. Regenerates answer with more conservative settings.
Only attempts fallback once per query.
"""

from agent_state import AgentState
from llm_client import LLMClient
import logging
import time

logger = logging.getLogger(__name__)


def fallback_regen(state: AgentState) -> AgentState:
    """
    Regenerate answer if coherence was low.

    Uses lower temperature (0.3) for more conservative answer.
    Does not attempt fallback again if already failed.

    Args:
        state: Current agent state

    Returns:
        Updated state with fallback_answer and fallback metadata
    """
    trace_id = state.get("trace_id", "unknown")

    # Check if fallback should actually run
    if not state.get("should_fallback", False):
        logger.debug(f"[{trace_id}] fallback_regen: Skipping (should_fallback=False)")
        state["fallback_triggered"] = False
        state["fallback_answer"] = None
        return state

    if state.get("fallback_count", 0) >= 1:
        logger.warning(
            f"[{trace_id}] fallback_regen: Fallback already attempted. Skipping."
        )
        state["fallback_triggered"] = False
        state["fallback_answer"] = None
        return state

    try:
        llm_client = LLMClient()
        model_id = state.get("context", {}).get("model_id")

        # Conservative fallback prompt
        system_prompt = """You are a conservative medical expert.
Your task is to provide a safe, evidence-based answer to a medical question.

Guidelines:
1. Prioritize safety and accuracy over comprehensiveness
2. State clearly when evidence is limited
3. Recommend consulting specialist if needed
4. Do NOT speculate or extrapolate beyond evidence
5. Be brief and direct

Provide a concise, evidence-based answer."""

        user_message = f"""Question: {state["input_query"]}

Previous answer was scored as incoherent. Please provide a more conservative, carefully reasoned answer.
Focus on what is well-established and safe to say."""

        start_time = time.time()

        # Call LLM with lower temperature for more deterministic output
        fallback_answer = llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model=model_id,
            temperature=0.3,  # Lower than synthesis (0.7)
            max_tokens=800,
        )

        elapsed_time = time.time() - start_time

        # Update state
        state["fallback_answer"] = fallback_answer
        state["fallback_triggered"] = True
        state["fallback_count"] += 1
        state["intermediate_answer"] = fallback_answer  # Use fallback as final answer

        logger.info(
            f"[{trace_id}] fallback_regen: Generated fallback answer. "
            f"Time: {elapsed_time:.2f}s. Length: {len(fallback_answer)} chars."
        )

    except Exception as e:
        logger.error(
            f"[{trace_id}] fallback_regen: Failed to regenerate: {str(e)}",
            exc_info=True,
        )
        # Keep original answer if fallback fails
        state["fallback_answer"] = None
        state["fallback_triggered"] = False
        state["error_occurred"] = True
        state["error_messages"].append(f"Fallback regeneration error: {str(e)}")

    return state
