"""
Node 4: synthesise — LLM answer generation with context.

Aggregates retrieved results into a coherent answer using the LLM.
Reuses aggregator.py logic, but routes through LLMClient.
"""

from agent_state import AgentState
from aggregator import Aggregator  # compatibility import for existing patch points
from llm_client import LLMCallResult, LLMClient
from runtime_verification import (
    build_evidence_context,
    build_attempt_event,
    call_llm_with_metadata,
    evidence_limited_answer,
    record_llm_call_results,
)
import logging
import time

logger = logging.getLogger(__name__)


def synthesise(state: AgentState) -> AgentState:
    """
    Generate answer by synthesizing retrieved documents.

    Reuses legacy aggregator.py; now routes LLM calls through LLMClient.

    Args:
        state: Current agent state

    Returns:
        Updated state with intermediate_answer and synthesis metadata
    """
    trace_id = state.get("trace_id", "unknown")
    start_time = time.time()
    model_id = state.get("context", {}).get("model_id")

    try:
        # Extract model from context; default to LLMClient default
        context_text, sources_used, included_context = build_evidence_context(state)
        state["synthesis_context"] = included_context
        if not included_context:
            state["intermediate_answer"] = evidence_limited_answer(
                "No usable documents were retrieved."
            )
            state["intermediate_sources"] = []
            state["intermediate_model_used"] = model_id or "not_called"
            state["synthesis_time_sec"] = 0.0
            state["synthesis_tokens_in"] = 0
            state["synthesis_tokens_out"] = 0
            state["evidence_limited"] = True
            state["is_partial_response"] = True
            logger.warning(
                "[%s] synthesise: no usable evidence; returning evidence-limited response",
                trace_id,
            )
            return state

        deadline_at = state.get("context", {}).get(
            "_runtime_deadline_at_monotonic"
        )
        if deadline_at is not None and time.monotonic() >= float(deadline_at):
            state["intermediate_answer"] = evidence_limited_answer(
                "The configured runtime deadline expired before synthesis."
            )
            state["intermediate_sources"] = sources_used
            state["intermediate_model_used"] = model_id or "not_called"
            state["synthesis_time_sec"] = 0.0
            state["evidence_limited"] = True
            state["is_partial_response"] = True
            state["attempt_telemetry"] = list(
                state.get("attempt_telemetry") or []
            ) + [
                build_attempt_event(
                    trace_id=str(trace_id),
                    attempt_id=f"{trace_id}:orchestrator:1",
                    parent_attempt_id=None,
                    stage="top_level_synthesis",
                    component="orchestrator",
                    status="deadline_exhausted",
                    repair_status="initial",
                    model=str(model_id or ""),
                    latency_sec=0.0,
                    finish_reason="not_started",
                    deadline_exhausted=True,
                    error_type="RuntimeDeadlineExceeded",
                )
            ]
            return state

        llm_client = LLMClient()
        history_reader = getattr(
            llm_client, "thread_call_history", None
        )
        history_start = (
            len(history_reader()) if callable(history_reader) else 0
        )
        state["evidence_limited"] = False

        system_prompt = """You are an evidence-based medical research assistant.
Your task is to answer medical/clinical questions using the provided scientific evidence.

Guidelines:
1. Answer directly and clearly
2. Ground your answer in the provided evidence
3. Mention specific findings from the literature
4. If evidence is limited or conflicting, state this clearly
5. Do NOT make up information or cite sources not provided
6. Cite each factual claim with the matching evidence marker, such as [1]
7. Use professional medical terminology appropriately

You will synthesize multiple sources into a coherent, clinically-informed answer."""

        user_message = f"""Question: {state["input_query"]}

Available Evidence:
{context_text}

Please provide a comprehensive, evidence-based answer to the question."""

        # Call LLM through LLMClient
        deadline_kwargs = {}
        if deadline_at is not None:
            deadline_kwargs = {
                "timeout": max(
                    0.1, float(deadline_at) - time.monotonic()
                ),
                "client_max_attempts": 1,
                "deadline_at": float(deadline_at),
            }

        call_result = call_llm_with_metadata(
            llm_client,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model=model_id,
            temperature=0.7,
            max_tokens=1000,
            _telemetry_stage="top_level_synthesis",
            _telemetry_attempt_id=f"{trace_id}:orchestrator:1",
            _telemetry_repair_status="initial",
            **deadline_kwargs,
        )
        answer = call_result.text
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
                == "top_level_synthesis"
            ]
            if callable(history_reader)
            else []
        )
        if not provider_calls:
            provider_calls = [call_result]

        # Update state
        state["intermediate_answer"] = answer
        state["intermediate_sources"] = sources_used
        state["intermediate_model_used"] = (
            f"{call_result.model}@{call_result.model_revision}"
            if call_result.model_revision
            else call_result.model
        )
        state["synthesis_time_sec"] = elapsed_time
        state["synthesis_tokens_in"] = (
            int(state.get("synthesis_tokens_in", 0))
            + sum(call.tokens_in for call in provider_calls)
        )
        state["synthesis_tokens_out"] = (
            int(state.get("synthesis_tokens_out", 0))
            + sum(call.tokens_out for call in provider_calls)
        )
        state["last_synthesis_cost_usd"] = call_result.cost_usd
        record_llm_call_results(
            state,
            provider_calls,
            trace_id=str(trace_id),
            base_attempt_id=f"{trace_id}:orchestrator:1",
            parent_attempt_id=None,
            stage="top_level_synthesis",
            component="orchestrator",
            repair_status="initial",
        )

        logger.info(
            f"[{trace_id}] synthesise: Generated answer using {len(sources_used)} sources "
            f"({', '.join(sources_used)}). Time: {elapsed_time:.2f}s. "
            f"Answer length: {len(answer)} chars."
        )

    except Exception as e:
        logger.error(
            "[%s] synthesise failed error_type=%s",
            trace_id,
            type(e).__name__,
        )
        state["intermediate_answer"] = (
            "I was unable to synthesize an answer because the generation "
            "stage failed. Please review the available evidence."
        )
        state["intermediate_sources"] = []
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
            history_slice = history[slice_start:]
            provider_calls = [
                call
                for call in history_slice
                if isinstance(call, LLMCallResult)
                and str(
                    call.provider_metadata.get("telemetry_stage")
                    or ""
                )
                == "top_level_synthesis"
            ]
            if provider_calls:
                candidate = provider_calls[-1]
                metadata = dict(candidate.provider_metadata or {})
                if (
                    isinstance(candidate, LLMCallResult)
                    and metadata.get("telemetry_stage")
                    == "top_level_synthesis"
                ):
                    failed_result = candidate
        state["intermediate_model_used"] = (
            (
                f"{failed_result.model}@{failed_result.model_revision}"
                if failed_result.model_revision
                else failed_result.model
            )
            if failed_result is not None
            else "error"
        )
        state["synthesis_time_sec"] = (
            failed_result.latency_sec if failed_result is not None else 0.0
        )
        state["error_occurred"] = True
        state["error_messages"].append(
            f"Synthesis error_type={type(e).__name__}"
        )
        error_type = (
            failed_result.error_type
            if failed_result is not None and failed_result.error_type
            else type(e).__name__
        )
        if provider_calls:
            record_llm_call_results(
                state,
                provider_calls,
                trace_id=str(trace_id),
                base_attempt_id=f"{trace_id}:orchestrator:1",
                parent_attempt_id=None,
                stage="top_level_synthesis",
                component="orchestrator",
                repair_status="initial",
            )
            state["synthesis_tokens_in"] = int(
                state.get("synthesis_tokens_in") or 0
            ) + sum(call.tokens_in for call in provider_calls)
            state["synthesis_tokens_out"] = int(
                state.get("synthesis_tokens_out") or 0
            ) + sum(call.tokens_out for call in provider_calls)
            return state
        event = build_attempt_event(
                trace_id=str(trace_id),
                attempt_id=f"{trace_id}:orchestrator:1",
                parent_attempt_id=None,
                stage="top_level_synthesis",
                component="orchestrator",
                status=(
                    "deadline_exhausted"
                    if error_type == "RuntimeDeadlineExceeded"
                    else "error"
                ),
                repair_status="initial",
                model=(
                    failed_result.model
                    if failed_result is not None
                    else str(model_id or "")
                ),
                model_revision=(
                    failed_result.model_revision
                    if failed_result is not None
                    else ""
                ),
                tokens_in=(
                    failed_result.tokens_in
                    if failed_result is not None
                    else 0
                ),
                tokens_out=(
                    failed_result.tokens_out
                    if failed_result is not None
                    else 0
                ),
                cost_usd=(
                    failed_result.cost_usd
                    if failed_result is not None
                    else 0.0
                ),
                latency_sec=(
                    failed_result.latency_sec
                    if failed_result is not None
                    else max(0.0, time.time() - start_time)
                ),
                finish_reason=(
                    failed_result.finish_reason
                    if failed_result is not None
                    else "error"
                ),
                deadline_exhausted=(
                    error_type == "RuntimeDeadlineExceeded"
                ),
                error_type=error_type,
                provider_metadata=(
                    failed_result.provider_metadata
                    if failed_result is not None
                    else None
                ),
            )
        attempts = list(state.get("attempt_telemetry") or [])
        if not any(
            item.get("event_id") == event["event_id"]
            for item in attempts
        ):
            attempts.append(event)
            if failed_result is not None:
                usage = dict(state.get("token_usage") or {})
                state["token_usage"] = {
                    "input": int(usage.get("input") or 0)
                    + failed_result.tokens_in,
                    "output": int(usage.get("output") or 0)
                    + failed_result.tokens_out,
                    "total": int(usage.get("total") or 0)
                    + failed_result.tokens_in
                    + failed_result.tokens_out,
                }
                state["cost_estimate"] = float(
                    state.get("cost_estimate") or 0.0
                ) + failed_result.cost_usd
        state["attempt_telemetry"] = attempts

    return state
