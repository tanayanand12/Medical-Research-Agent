"""
Node 8: format_response — AMA citation formatting + disclaimers.

Formats final answer with:
1. AMA-style citations for all referenced papers
2. Clinical disclaimer
3. [FALLBACK] tag if fallback was triggered
4. Early-exit response if query was non-medical
"""

from agent_state import AgentState
from citation_formatter import format_citations_to_ama
from datetime import datetime
from evaluation_core import (
    EVALUATION_TRACE_SCHEMA_VERSION,
    EvaluationTrace,
    VerificationDecision,
    build_orchestrator_evaluation_trace,
    validate_evaluation_trace,
)
from runtime_verification import evidence_limited_answer, evidence_limited_decision
import logging
import math
import time

logger = logging.getLogger(__name__)


CLINICAL_DISCLAIMER = """[DISCLAIMER: This response is AI-generated and not a substitute for professional medical advice.
Always consult with a qualified healthcare provider before making any medical decisions.
This information is for educational purposes only.]"""

FALLBACK_TAG = "\n\n[NOTE: This response was regenerated using fallback mechanism due to coherence concerns.]"

NON_MEDICAL_RESPONSE = """I cannot provide information on this topic as it falls outside the scope of medical research.
Please consult with appropriate professionals or resources for this question."""


def _complete_verification_payload(raw: object) -> bool:
    """Reject default-filled or malformed terminal verifier payloads."""
    if not isinstance(raw, dict):
        return False
    required = {
        "status",
        "component_scores",
        "failed_checks",
        "structured_feedback",
        "target_stage",
        "target_agent",
        "recommended_retry_changes",
        "verifier_confidence",
        "valid",
        "verifier_model",
        "prompt_version",
        "raw_decision",
    }
    if not required.issubset(raw):
        return False
    confidence = raw.get("verifier_confidence")
    component_scores = raw.get("component_scores")
    required_components = {
        "retrieval_coverage",
        "evidence_sufficiency",
        "claim_grounding",
        "citation_support",
        "query_coverage",
        "verifier_confidence",
    }
    scores_valid = bool(
        isinstance(component_scores, dict)
        and required_components.issubset(component_scores)
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0
            for value in component_scores.values()
        )
    )
    return bool(
        raw.get("status")
        in {"accept", "retry_retrieval", "retry_synthesis", "evidence_limited"}
        and scores_valid
        and isinstance(raw.get("failed_checks"), list)
        and isinstance(raw.get("structured_feedback"), list)
        and isinstance(raw.get("recommended_retry_changes"), dict)
        and isinstance(raw.get("raw_decision"), dict)
        and isinstance(raw.get("target_stage"), str)
        and raw.get("target_stage")
        and isinstance(raw.get("target_agent"), str)
        and raw.get("target_agent")
        and isinstance(raw.get("verifier_model"), str)
        and raw.get("verifier_model")
        and isinstance(raw.get("prompt_version"), str)
        and raw.get("prompt_version")
        and isinstance(raw.get("valid"), bool)
        and not isinstance(confidence, bool)
        and isinstance(confidence, (int, float))
        and math.isfinite(float(confidence))
        and 0.0 <= float(confidence) <= 1.0
        and (
            (
                raw.get("status") == "accept"
                and raw.get("valid") is True
                and not raw.get("failed_checks")
                and not raw.get("error")
                and raw.get("target_stage") == "none"
                and raw.get("target_agent") == "orchestrator"
                and not raw.get("recommended_retry_changes")
            )
            or raw.get("status") != "accept"
        )
    )


def _acceptance_bound_to_latest_trace(
    state: AgentState, decision: VerificationDecision
) -> bool:
    """Require terminal acceptance to match the latest valid synthesis trace."""
    traces = list(state.get("evaluation_traces") or [])
    if not traces or not isinstance(traces[-1], dict):
        return False
    try:
        trace = EvaluationTrace.from_dict(traces[-1])
    except (TypeError, ValueError):
        return False
    if validate_evaluation_trace(trace):
        return False
    if (
        trace.trace_id != str(state.get("trace_id") or "")
        or not trace.synthesis_manifest_present
        or not trace.final_context_spans
        or trace.answer != str(state.get("intermediate_answer") or "")
        or not trace.verification_decisions
    ):
        return False
    bound_decision = trace.verification_decisions[-1]
    return bool(
        bound_decision.status == "accept"
        and bound_decision.valid
        and not bound_decision.failed_checks
        and not bound_decision.error
        and bound_decision.target_stage == "none"
        and bound_decision.target_agent == "orchestrator"
        and bound_decision.status == decision.status
        and bound_decision.valid == decision.valid
        and bound_decision.verifier_model == decision.verifier_model
        and bound_decision.prompt_version == decision.prompt_version
        and bound_decision.verifier_confidence == decision.verifier_confidence
        and bound_decision.target_stage == decision.target_stage
        and bound_decision.target_agent == decision.target_agent
    )


def _terminal_decision(state: AgentState) -> VerificationDecision:
    if state.get("is_medical_query") is False:
        return VerificationDecision(
            status="accept",
            component_scores={},
            failed_checks=[],
            structured_feedback=[],
            target_stage="none",
            target_agent="orchestrator",
            recommended_retry_changes={},
            verifier_confidence=1.0,
            valid=True,
            verifier_model="deterministic_query_classifier",
            prompt_version="runtime-terminal-v1",
            raw_decision={"terminal_reason": "non_medical_early_exit"},
        )

    raw = state.get("verification_decision")
    if _complete_verification_payload(raw):
        try:
            decision = VerificationDecision.from_dict(raw)
            if (
                decision.status == "accept"
                and decision.valid
                and _acceptance_bound_to_latest_trace(state, decision)
                and not state.get("evidence_limited")
                and not state.get("error_occurred")
            ):
                decision.raw_decision = dict(decision.raw_decision)
                decision.raw_decision.setdefault(
                    "terminal_reason", "accepted"
                )
                return decision
            if decision.status == "evidence_limited":
                decision.raw_decision = dict(decision.raw_decision)
                decision.raw_decision.setdefault(
                    "terminal_reason", "evidence_limited"
                )
                return decision
            if decision.status in {
                "accept",
                "retry_retrieval",
                "retry_synthesis",
            }:
                return evidence_limited_decision(
                    target_agent="orchestrator",
                    failed_check="terminal_retry_unresolved",
                    message=(
                        "A nonterminal or invalid verification decision remained "
                        "when the response was delivered."
                    ),
                    valid=bool(decision.valid),
                    error=decision.error,
                )
        except (TypeError, ValueError):
            pass
    if state.get("evidence_limited") or state.get("error_occurred"):
        return evidence_limited_decision(
            target_agent="orchestrator",
            failed_check="terminal_evidence_limited",
            message="The delivered response is partial or evidence limited.",
            valid=False,
            error=(
                "upstream_runtime_error"
                if state.get("error_messages")
                else None
            ),
        )
    return evidence_limited_decision(
        target_agent="orchestrator",
        failed_check="terminal_verification_missing_or_malformed",
        message=(
            "The medical response did not have a complete, valid terminal "
            "verification decision."
        ),
        valid=False,
        error="terminal verification unavailable",
    )


def _append_terminal_trace(state: AgentState) -> None:
    """Record exactly the answer delivered by the API as the final attempt."""
    trace_id = str(state.get("trace_id") or "unknown")
    traces = list(state.get("evaluation_traces") or [])
    parent_attempt_id = traces[-1].get("attempt_id") if traces else None
    try:
        terminal_trace = build_orchestrator_evaluation_trace(
            state,
            answer=str(state.get("output_answer") or ""),
            attempt_id=f"{trace_id}:orchestrator:terminal",
            parent_attempt_id=parent_attempt_id,
        )
    except Exception as exc:
        logger.error(
            "[%s] Terminal trace adapter failed: %s",
            trace_id,
            type(exc).__name__,
        )
        trace_error = f"terminal_trace_adapter_failure:{type(exc).__name__}"
        terminal_trace = EvaluationTrace(
            schema_version=EVALUATION_TRACE_SCHEMA_VERSION,
            trace_id=trace_id,
            attempt_id=f"{trace_id}:orchestrator:terminal",
            parent_attempt_id=parent_attempt_id,
            agent_name="orchestrator",
            domain="multi_source",
            original_query=str(state.get("input_query") or "unknown"),
            expanded_queries=[],
            retrieval_configuration={},
            retrieved_documents=[],
            reranked_documents=[],
            final_context_document_ids=[],
            final_context_spans=[],
            answer=str(state.get("output_answer") or ""),
            atomic_claims=[],
            citations=[],
            stage_latency_sec={},
            token_usage={},
            cost_breakdown_usd={},
            exact_model=str(
                state.get("intermediate_model_used") or "unknown"
            ),
            model_revision="",
            prompt_version="runtime-terminal-v1",
            config_version="runtime-v1",
            errors=[trace_error],
            partial_response=True,
            trace_role="terminal_delivery",
            synthesis_manifest_present=False,
            answer_origin="evidence_limited",
        )
        terminal_trace.verification_decisions.append(
            evidence_limited_decision(
                target_agent="orchestrator",
                failed_check="terminal_trace_adapter_failure",
                message=(
                    "The response was delivered, but its full terminal trace "
                    "could not be adapted."
                ),
                valid=False,
                error=trace_error,
            )
        )
    terminal_trace.prompt_version = "runtime-terminal-v1"
    terminal_trace.trace_role = "terminal_delivery"
    terminal_trace.partial_response = bool(
        terminal_trace.partial_response or state.get("is_partial_response")
    )
    if state.get("error_messages") and "upstream_runtime_error" not in (
        terminal_trace.errors
    ):
        terminal_trace.errors = list(terminal_trace.errors) + [
            "upstream_runtime_error"
        ]
    terminal_trace.token_usage = {
        str(key): int(value)
        for key, value in dict(state.get("token_usage") or {}).items()
        if isinstance(value, (int, float))
    }
    terminal_trace.cost_breakdown_usd = {
        "request_total": float(state.get("cost_estimate") or 0.0)
    }
    terminal_trace.stage_latency_sec["request_total"] = float(
        state.get("execution_time_sec") or 0.0
    )
    if not terminal_trace.verification_decisions:
        terminal_trace.verification_decisions.append(_terminal_decision(state))
    state["evaluation_traces"] = traces + [terminal_trace.to_dict()]


def format_response(state: AgentState) -> AgentState:
    """
    Format final response for user delivery.

    Handles three cases:
    1. Non-medical query: Return rejection message
    2. Medical query with results: Format with citations and disclaimer
    3. Medical query with partial/no results: Return best-effort answer with warning

    Args:
        state: Current agent state

    Returns:
        Updated state with output_answer, output_citations, output_sources, etc.
    """
    trace_id = state.get("trace_id", "unknown")
    timestamp_end = datetime.utcnow()

    try:
        # Case 1: Non-medical query
        if not state.get("is_medical_query", False):
            state["verification_decision"] = _terminal_decision(state).to_dict()
            state["answer_origin"] = "non_medical_early_exit"
            state["output_answer"] = NON_MEDICAL_RESPONSE
            state["output_sources"] = []
            state["output_citations"] = []
            state["output_disclaimer"] = CLINICAL_DISCLAIMER
            state["timestamp_end"] = timestamp_end
            state["execution_time_sec"] = (
                timestamp_end - state["timestamp_start"]
            ).total_seconds()
            logger.info(
                f"[{trace_id}] format_response: Non-medical query. "
                f"Returning rejection response."
            )
            _append_terminal_trace(state)
            return state

        # Case 2 & 3: Medical query. Coherence is never terminal proof of
        # grounding; only a complete, valid runtime-verifier acceptance is.
        terminal_decision = _terminal_decision(state)
        state["verification_decision"] = terminal_decision.to_dict()
        if terminal_decision.status != "accept":
            state["evidence_limited"] = True
            state["is_partial_response"] = True
            state["fallback_triggered"] = False
            state["answer_origin"] = "evidence_limited"
            state["intermediate_answer"] = evidence_limited_answer(
                "Runtime verification could not establish a grounded answer."
            )

        # Build answer
        final_answer = state["intermediate_answer"]

        # Add fallback tag if triggered
        if state.get("fallback_triggered", False):
            final_answer += FALLBACK_TAG

        # Prepend disclaimer
        final_answer = f"{CLINICAL_DISCLAIMER}\n\n{final_answer}"

        # Add partial response warning if applicable
        if state.get("is_partial_response", False):
            warning = (
                "\n\n[NOTE: Some data sources could not be retrieved. "
                "This answer may be incomplete.]"
            )
            final_answer += warning

        # Extract citations from retrieval results
        citations = []
        sources = state.get("intermediate_sources", [])

        synthesis_context = state.get("synthesis_context")
        if synthesis_context is not None:
            citation_records = [
                item.get("citation_metadata", {})
                for item in synthesis_context
                if item.get("citation_metadata")
            ]
        else:
            # A missing manifest is invalid for generated medical output.
            # Never imply grounding by substituting retrieved documents that
            # were not recorded as prompt context.
            citation_records = []

        for result in citation_records:
            # Format as AMA citation
            try:
                ama_citation = format_citations_to_ama(
                    {
                        "title": result.get("title"),
                        "authors": result.get("authors", []),
                        "year": result.get("year"),
                        "journal": result.get("journal", ""),
                        "volume": result.get("volume"),
                        "issue": result.get("issue"),
                        "pages": result.get("pages"),
                        "doi": result.get("doi"),
                        "pmid": result.get("pmid"),
                    }
                )
                if ama_citation:
                    citations.append(ama_citation)
            except Exception as cite_error:
                logger.warning(
                    "[%s] citation formatting failed error_type=%s",
                    trace_id,
                    type(cite_error).__name__,
                )

        # Build citations section
        citations_section = ""
        if citations:
            citations_section = "\n\nReferences:\n" + "\n".join(
                [f"{i+1}. {c}" for i, c in enumerate(citations)]
            )
            final_answer += citations_section

        # Update state
        state["output_answer"] = final_answer
        state["output_sources"] = sources
        state["output_citations"] = citations
        state["output_disclaimer"] = CLINICAL_DISCLAIMER
        state["timestamp_end"] = timestamp_end

        # Calculate total execution time
        execution_time = (timestamp_end - state["timestamp_start"]).total_seconds()
        state["execution_time_sec"] = execution_time

        logger.info(
            f"[{trace_id}] format_response: Formatted answer with {len(citations)} citations. "
            f"Total time: {execution_time:.2f}s. Fallback: {state.get('fallback_triggered', False)}"
        )

    except Exception as e:
        logger.error(
            "[%s] format_response failed error_type=%s",
            trace_id,
            type(e).__name__,
        )
        terminal_decision = evidence_limited_decision(
            target_agent="orchestrator",
            failed_check="response_formatting_failure",
            message=(
                "The response could not be safely formatted for delivery."
            ),
            valid=False,
            error=f"format_response_failed:{type(e).__name__}",
        )
        state["verification_decision"] = terminal_decision.to_dict()
        state["evidence_limited"] = True
        state["is_partial_response"] = True
        state["fallback_triggered"] = False
        state["answer_origin"] = "evidence_limited"
        state["intermediate_answer"] = evidence_limited_answer(
            "The verified response could not be safely formatted."
        )
        state["output_answer"] = (
            f"{CLINICAL_DISCLAIMER}\n\n"
            f"{state['intermediate_answer']}\n\n"
            f"[NOTE: Response formatting encountered an error.]"
        )
        state["output_sources"] = []
        state["output_citations"] = []
        state["output_disclaimer"] = CLINICAL_DISCLAIMER
        state["timestamp_end"] = timestamp_end
        timestamp_start = state.get("timestamp_start")
        state["execution_time_sec"] = (
            (timestamp_end - timestamp_start).total_seconds()
            if isinstance(timestamp_start, datetime)
            else 0.0
        )
        state["error_occurred"] = True
        state.setdefault("error_messages", []).append(
            f"Response formatting error_type={type(e).__name__}"
        )

    _append_terminal_trace(state)
    return state
