"""
Node 7: fallback_regen — Fallback answer regeneration.

Triggered when coherence score is low. Regenerates answer with more conservative settings.
Only attempts fallback once per query.
"""

from agent_state import AgentState
from evaluation_core import build_orchestrator_evaluation_trace
from llm_client import LLMCallResult, LLMClient
from runtime_verification import (
    build_evidence_context,
    build_attempt_event,
    build_runtime_verifier,
    calculate_combined_confidence,
    call_llm_with_metadata,
    evidence_limited_answer,
    evidence_limited_decision,
    record_conditional_verifier_telemetry,
    record_llm_call_results,
)
import json
import logging
import time

logger = logging.getLogger(__name__)


def _finalize_failed_repair(
    state: AgentState,
    *,
    failed_check: str,
    message: str,
    valid: bool,
    error: str | None = None,
    call_result: LLMCallResult | None = None,
    telemetry_recorded: bool = False,
) -> AgentState:
    """Record a terminal, schema-complete repair attempt on every exit."""
    trace_id = state.get("trace_id", "unknown")
    state["fallback_count"] = min(1, state.get("fallback_count", 0) + 1)
    state["fallback_triggered"] = False
    state["fallback_answer"] = None
    state["intermediate_answer"] = evidence_limited_answer(message)
    state["evidence_limited"] = True
    state["is_partial_response"] = True
    previous_feedback = list(
        (state.get("verification_decision") or {}).get(
            "structured_feedback", []
        )
    )
    if error:
        state["error_occurred"] = True
        state["error_messages"].append(f"Synthesis repair error: {error}")

    previous_traces = list(state.get("evaluation_traces", []))
    parent_attempt_id = (
        previous_traces[-1].get("attempt_id") if previous_traces else None
    )
    terminal_attempt_id = (
        f"{trace_id}:orchestrator:{state['fallback_count'] + 1}"
    )
    deadline_exhausted = bool(
        "deadline" in failed_check
        or (
            call_result is not None
            and call_result.error_type == "RuntimeDeadlineExceeded"
        )
    )
    terminal_event = build_attempt_event(
        trace_id=str(trace_id),
        attempt_id=terminal_attempt_id,
        parent_attempt_id=parent_attempt_id,
        stage="top_level_synthesis_repair",
        component="orchestrator",
        status=(
            "deadline_exhausted"
            if deadline_exhausted
            else "error"
            if error
            else "skipped"
        ),
        repair_status="synthesis_repair",
        model=(
            call_result.model
            if call_result is not None
            else str(state.get("context", {}).get("model_id") or "")
        ),
        model_revision=(
            call_result.model_revision if call_result is not None else ""
        ),
        tokens_in=(call_result.tokens_in if call_result is not None else 0),
        tokens_out=(
            call_result.tokens_out if call_result is not None else 0
        ),
        cost_usd=(
            call_result.cost_usd if call_result is not None else 0.0
        ),
        latency_sec=(
            call_result.latency_sec if call_result is not None else 0.0
        ),
        finish_reason=(
            call_result.finish_reason
            if call_result is not None
            else "not_completed"
        ),
        deadline_exhausted=deadline_exhausted,
        error_type=(
            "RuntimeDeadlineExceeded"
            if deadline_exhausted
            else (
                call_result.error_type
                if call_result is not None and call_result.error_type
                else str(error or failed_check)
            )
        ),
        provider_metadata=(
            call_result.provider_metadata
            if call_result is not None
            else None
        ),
    )
    attempts = list(state.get("attempt_telemetry") or [])
    if not any(
        item.get("event_id") == terminal_event["event_id"]
        for item in attempts
    ):
        attempts.append(terminal_event)
        if call_result is not None and not telemetry_recorded:
            usage = dict(state.get("token_usage") or {})
            state["token_usage"] = {
                "input": int(usage.get("input") or 0)
                + call_result.tokens_in,
                "output": int(usage.get("output") or 0)
                + call_result.tokens_out,
                "total": int(usage.get("total") or 0)
                + call_result.tokens_in
                + call_result.tokens_out,
            }
            state["cost_estimate"] = float(
                state.get("cost_estimate") or 0.0
            ) + call_result.cost_usd
    state["attempt_telemetry"] = attempts
    decision = evidence_limited_decision(
        target_agent="orchestrator",
        failed_check=failed_check,
        message=message,
        valid=valid,
        error=error,
    )
    terminal_trace = build_orchestrator_evaluation_trace(
        state,
        answer=state["intermediate_answer"],
        attempt_id=terminal_attempt_id,
        parent_attempt_id=parent_attempt_id,
    )
    terminal_trace.verification_decisions.append(decision)
    state["evaluation_traces"] = previous_traces + [terminal_trace.to_dict()]
    state["verification_history"] = list(
        state.get("verification_history", [])
    ) + [decision.to_dict()]
    state["verification_decision"] = decision.to_dict()
    state["confidence_components"] = dict(decision.component_scores)
    state["runtime_quality_score"] = 0.0
    state["runtime_quality_explanation"] = message
    state["coherence_score"] = 0.0
    state["coherence_explanation"] = message
    state["should_fallback"] = False
    state["repair_history"] = list(state.get("repair_history", [])) + [
        {
            "attempt": state["fallback_count"],
            "target_stage": "synthesis",
            "target_agent": "orchestrator",
            "parent_attempt_id": parent_attempt_id,
            "feedback": previous_feedback,
            "evidence_document_ids": [
                item.get("document_id")
                for item in state.get("synthesis_context", [])
                if item.get("document_id")
            ],
            "decision": decision.to_dict(),
        }
    ]
    return state


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
        model_id = state.get("context", {}).get("model_id")
        evidence_text, sources_used, included_context = build_evidence_context(state)
        if not included_context:
            return _finalize_failed_repair(
                state,
                failed_check="empty_repair_evidence",
                message="Synthesis repair was skipped because no evidence remained.",
                valid=True,
            )

        deadline_at = state.get("context", {}).get(
            "_runtime_deadline_at_monotonic"
        )
        if deadline_at is not None and time.monotonic() >= float(deadline_at):
            return _finalize_failed_repair(
                state,
                failed_check="repair_deadline_exceeded",
                message="The configured runtime deadline expired before synthesis repair.",
                valid=True,
            )

        llm_client = LLMClient()
        history_reader = getattr(
            llm_client, "thread_call_history", None
        )
        history_start = (
            len(history_reader()) if callable(history_reader) else 0
        )
        decision = dict(state.get("verification_decision") or {})
        feedback = decision.get("structured_feedback") or []
        previous_answer = state.get("intermediate_answer", "")

        # Conservative fallback prompt
        system_prompt = """You are a conservative medical expert.
Repair an answer using only the supplied evidence.

Guidelines:
1. Prioritize safety and accuracy over comprehensiveness
2. State clearly when evidence is limited
3. Do NOT speculate or extrapolate beyond evidence
4. Every factual claim must cite a matching evidence marker such as [1]
5. Remove or qualify assertions identified by verifier feedback
6. Be brief and direct

Provide a concise, evidence-based answer."""

        user_message = f"""Question: {state["input_query"]}

Previous answer:
{previous_answer}

Structured verifier feedback:
{json.dumps(feedback, ensure_ascii=False, indent=2)}

Available Evidence (the repair must use only this evidence):
{evidence_text}

Return the repaired answer with resolvable evidence markers."""

        deadline_kwargs = {}
        if deadline_at is not None:
            deadline_kwargs = {
                "timeout": max(
                    0.1, float(deadline_at) - time.monotonic()
                ),
                "client_max_attempts": 1,
                "deadline_at": float(deadline_at),
            }

        # Call LLM with lower temperature for more deterministic output
        call_result = call_llm_with_metadata(
            llm_client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model=model_id,
            temperature=0.3,  # Lower than synthesis (0.7)
            max_tokens=800,
            _telemetry_stage="top_level_synthesis_repair",
            _telemetry_attempt_id=f"{trace_id}:orchestrator:2",
            _telemetry_repair_status="synthesis_repair",
            **deadline_kwargs,
        )
        fallback_answer = call_result.text
        elapsed_time = call_result.latency_sec
        provider_calls = (
            [
                call
                for call in list(history_reader())[history_start:]
                if isinstance(call, LLMCallResult)
                and str(
                    call.provider_metadata.get("telemetry_stage")
                    or ""
                )
                == "top_level_synthesis_repair"
            ]
            if callable(history_reader)
            else []
        )
        if not provider_calls:
            provider_calls = [call_result]

        # Update state
        state["fallback_answer"] = fallback_answer
        state["fallback_triggered"] = True
        state["fallback_count"] = state.get("fallback_count", 0) + 1
        state["intermediate_answer"] = fallback_answer
        state["intermediate_sources"] = sources_used
        state["intermediate_model_used"] = (
            f"{call_result.model}@{call_result.model_revision}"
            if call_result.model_revision
            else call_result.model
        )
        state["synthesis_context"] = included_context
        state["synthesis_time_sec"] = (
            state.get("synthesis_time_sec", 0.0) + elapsed_time
        )
        state["synthesis_tokens_in"] = (
            int(state.get("synthesis_tokens_in", 0))
            + sum(call.tokens_in for call in provider_calls)
        )
        state["synthesis_tokens_out"] = (
            int(state.get("synthesis_tokens_out", 0))
            + sum(call.tokens_out for call in provider_calls)
        )
        state["last_synthesis_cost_usd"] = call_result.cost_usd
        repair_attempt_id = (
            f"{trace_id}:orchestrator:{state['fallback_count'] + 1}"
        )
        repair_parent_id = (
            state.get("evaluation_traces", [{}])[-1].get("attempt_id")
            if state.get("evaluation_traces")
            else None
        )
        record_llm_call_results(
            state,
            provider_calls,
            trace_id=str(trace_id),
            base_attempt_id=repair_attempt_id,
            parent_attempt_id=repair_parent_id,
            stage="top_level_synthesis_repair",
            component="orchestrator",
            repair_status="synthesis_repair",
        )

        previous_traces = state.get("evaluation_traces", [])
        parent_attempt_id = (
            previous_traces[-1].get("attempt_id") if previous_traces else None
        )
        repaired_trace = build_orchestrator_evaluation_trace(
            state,
            answer=fallback_answer,
            attempt_id=f"{trace_id}:orchestrator:{state['fallback_count'] + 1}",
            parent_attempt_id=parent_attempt_id,
        )
        repair_tokens_in = sum(
            call.tokens_in for call in provider_calls
        )
        repair_tokens_out = sum(
            call.tokens_out for call in provider_calls
        )
        repair_cost = sum(
            call.cost_usd for call in provider_calls
        )
        repair_latency = sum(
            call.latency_sec for call in provider_calls
        )
        repaired_trace.token_usage = {
            "input": repair_tokens_in,
            "output": repair_tokens_out,
            "total": repair_tokens_in + repair_tokens_out,
        }
        repaired_trace.cost_breakdown_usd = {"repair": repair_cost}
        repaired_trace.stage_latency_sec = {
            "synthesis": repair_latency,
            "total": repair_latency,
        }
        repaired_trace.exact_model = call_result.model
        repaired_trace.model_revision = call_result.model_revision
        repaired_decision = build_runtime_verifier(
            state.get("context", {})
        ).verify(
            repaired_trace, retries_remaining=0
        )
        record_conditional_verifier_telemetry(
            state, repaired_trace, repaired_decision
        )
        state["evaluation_traces"] = list(previous_traces) + [
            repaired_trace.to_dict()
        ]
        state["verification_history"] = list(
            state.get("verification_history", [])
        ) + [repaired_decision.to_dict()]
        state["verification_decision"] = repaired_decision.to_dict()
        state["confidence_components"] = dict(
            repaired_decision.component_scores
        )
        confidence, explanation = calculate_combined_confidence(
            state["confidence_components"]
        )
        state["runtime_quality_score"] = confidence
        state["runtime_quality_explanation"] = explanation
        repair_event = {
            "attempt": state["fallback_count"],
            "target_stage": "synthesis",
            "target_agent": "orchestrator",
            "parent_attempt_id": parent_attempt_id,
            "feedback": feedback,
            "evidence_document_ids": [
                item["document_id"] for item in included_context
            ],
            "decision": repaired_decision.to_dict(),
        }
        state["repair_history"] = list(state.get("repair_history", [])) + [
            repair_event
        ]
        if repaired_decision.status != "accept":
            state["intermediate_answer"] = evidence_limited_answer(
                "The bounded synthesis repair did not pass runtime verification."
            )
            state["evidence_limited"] = True
            state["is_partial_response"] = True
        else:
            state["evidence_limited"] = False

        logger.info(
            f"[{trace_id}] fallback_regen: Generated evidence-grounded repair. "
            f"Time: {elapsed_time:.2f}s. Verification: {repaired_decision.status}."
        )

    except Exception as e:
        logger.error(
            "[%s] fallback_regen failed error_type=%s",
            trace_id,
            type(e).__name__,
        )
        failed_result = None
        provider_calls = []
        if "llm_client" in locals():
            history_reader = getattr(
                llm_client, "thread_call_history", None
            )
            history = (
                list(history_reader() or [])
                if callable(history_reader)
                else []
            )
            slice_start = (
                history_start
                if "history_start" in locals()
                else len(history)
            )
            provider_calls = [
                call
                for call in history[slice_start:]
                if isinstance(call, LLMCallResult)
                and str(
                    call.provider_metadata.get("telemetry_stage")
                    or ""
                )
                == "top_level_synthesis_repair"
            ]
            if provider_calls:
                candidate = provider_calls[-1]
                metadata = dict(candidate.provider_metadata or {})
                if (
                    isinstance(candidate, LLMCallResult)
                    and metadata.get("telemetry_stage")
                    == "top_level_synthesis_repair"
                ):
                    failed_result = candidate
        if provider_calls:
            previous_traces = list(
                state.get("evaluation_traces") or []
            )
            parent_attempt_id = (
                previous_traces[-1].get("attempt_id")
                if previous_traces
                else None
            )
            repair_attempt_id = (
                f"{trace_id}:orchestrator:"
                f"{min(1, state.get('fallback_count', 0) + 1) + 1}"
            )
            record_llm_call_results(
                state,
                provider_calls,
                trace_id=str(trace_id),
                base_attempt_id=repair_attempt_id,
                parent_attempt_id=parent_attempt_id,
                stage="top_level_synthesis_repair",
                component="orchestrator",
                repair_status="synthesis_repair",
            )
            state["synthesis_tokens_in"] = int(
                state.get("synthesis_tokens_in") or 0
            ) + sum(call.tokens_in for call in provider_calls)
            state["synthesis_tokens_out"] = int(
                state.get("synthesis_tokens_out") or 0
            ) + sum(call.tokens_out for call in provider_calls)
        return _finalize_failed_repair(
            state,
            failed_check="repair_failure",
            message="The bounded synthesis repair failed.",
            valid=False,
            error=f"repair_failed:{type(e).__name__}",
            call_result=failed_result,
            telemetry_recorded=bool(provider_calls),
        )

    return state
