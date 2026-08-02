"""
Node 6: evaluate_coherence — Coherence scoring for fallback decision.

Builds a qrel-free synthesis trace and applies the shared runtime verifier.
The legacy field names are retained for API/state compatibility.
"""

from agent_state import AgentState
from evaluation_core import build_orchestrator_evaluation_trace
from fallback import FallbackMechanism  # compatibility import for existing patch points
from runtime_verification import (
    build_runtime_verifier,
    calculate_combined_confidence,
    evidence_limited_answer,
    evidence_limited_decision,
    record_conditional_verifier_telemetry,
)
import logging

logger = logging.getLogger(__name__)


def evaluate_coherence(state: AgentState) -> AgentState:
    """
    Verify the synthesized answer and decide bounded synthesis repair.

    Args:
        state: Current agent state

    Returns:
        Updated state with coherence_score and should_fallback decision
    """
    trace_id = state.get("trace_id", "unknown")
    try:
        repair_count = state.get("fallback_count", 0)
        max_repairs = max(
            0,
            min(
                1,
                int(state.get("context", {}).get("max_synthesis_repairs", 1)),
            ),
        )
        has_evidence = any(
            result.get("results")
            for result in state.get("retrieval_results", {}).values()
        )
        trace = build_orchestrator_evaluation_trace(
            state,
            attempt_id=f"{trace_id}:orchestrator:{repair_count + 1}",
        )
        decision = build_runtime_verifier(state.get("context", {})).verify(
            trace,
            retries_remaining=(
                max_repairs - repair_count if has_evidence else 0
            ),
        )
        record_conditional_verifier_telemetry(state, trace, decision)
        if decision.status == "retry_retrieval":
            decision.status = "evidence_limited"
            decision.target_stage = "none"
            decision.recommended_retry_changes = {}
            decision.structured_feedback.append(
                {
                    "check": "retrieval_retry_unavailable",
                    "message": (
                        "Final synthesis verification found inadequate evidence "
                        "after the bounded retrieval stage completed."
                    ),
                }
            )

        components = dict(decision.component_scores)
        confidence, explanation = calculate_combined_confidence(components)

        state["evaluation_traces"] = list(
            state.get("evaluation_traces", [])
        ) + [trace.to_dict()]
        state["verification_history"] = list(
            state.get("verification_history", [])
        ) + [decision.to_dict()]
        state["verification_decision"] = decision.to_dict()
        state["confidence_components"] = components
        state["runtime_quality_score"] = confidence
        state["runtime_quality_explanation"] = explanation
        state["coherence_score"] = confidence
        state["coherence_explanation"] = (
            "Runtime verifier status="
            f"{decision.status}; failed_checks={decision.failed_checks}. "
            f"{explanation}"
        )
        state["coherence_eval_model_used"] = decision.verifier_model
        state["should_fallback"] = decision.status == "retry_synthesis"
        state["evidence_limited"] = decision.status == "evidence_limited"
        if state["evidence_limited"]:
            state["intermediate_answer"] = evidence_limited_answer(
                "The retrieved evidence or answer grounding did not pass runtime verification."
            )
            state["is_partial_response"] = True
        if state["should_fallback"]:
            state["fallback_reason"] = ", ".join(decision.failed_checks)
        if not decision.valid:
            state["error_occurred"] = True
            state["is_partial_response"] = True
            state["error_messages"].append(
                f"Runtime verification error: {decision.error}"
            )

        logger.info(
            "[%s] evaluate_coherence: status=%s score=%.2f repair=%s checks=%s",
            trace_id,
            decision.status,
            confidence,
            state["should_fallback"],
            decision.failed_checks,
        )

    except Exception as e:
        logger.error(
            "[%s] evaluate_coherence failed error_type=%s",
            trace_id,
            type(e).__name__,
        )
        # Never convert verifier failure into a passing score.
        state["coherence_score"] = 0.0
        state["coherence_explanation"] = (
            f"Runtime verification failed error_type={type(e).__name__}"
        )
        state["should_fallback"] = False
        state["evidence_limited"] = True
        state["runtime_quality_score"] = 0.0
        state["runtime_quality_explanation"] = (
            "Runtime verification failed; quality score forced to 0."
        )
        state["confidence_components"] = {
            "retrieval_coverage": 0.0,
            "evidence_sufficiency": 0.0,
            "claim_grounding": 0.0,
            "citation_support": 0.0,
            "query_coverage": 0.0,
            "verifier_confidence": 0.0,
        }
        decision = evidence_limited_decision(
            target_agent="orchestrator",
            failed_check="verifier_failure",
            message="Final synthesis verification failed.",
            valid=False,
            error=f"verifier_failed:{type(e).__name__}",
        )
        state["verification_decision"] = decision.to_dict()
        state["verification_history"] = list(
            state.get("verification_history", [])
        ) + [decision.to_dict()]
        state["intermediate_answer"] = evidence_limited_answer(
            "Runtime verification failed before the answer could be accepted."
        )
        state["error_occurred"] = True
        state["is_partial_response"] = True
        state["error_messages"].append(
            f"Coherence evaluation error_type={type(e).__name__}"
        )

    return state
