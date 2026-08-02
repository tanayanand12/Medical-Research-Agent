"""Per-call and per-attempt telemetry helpers for runtime verification."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List

from evaluation_core import (
    ATTEMPT_EVENT_SCHEMA_VERSION,
    EvaluationTrace,
    VerificationDecision,
)
from llm_client import LLMCallResult


_SAFE_PROVIDER_METADATA_KEYS = {
    "provider",
    "provider_attempt",
    "input_count",
    "request_timeout_sec",
    "response_id",
}


def build_attempt_event(
    *,
    trace_id: str,
    attempt_id: str,
    parent_attempt_id: Any,
    stage: str,
    component: str,
    status: str,
    repair_status: str,
    model: str = "",
    model_revision: str = "",
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    latency_sec: float = 0.0,
    finish_reason: str = "",
    prompt_version: str = "",
    deadline_exhausted: bool = False,
    error_type: str = "",
    provider_metadata: Any = None,
    event_id: str = "",
) -> Dict[str, Any]:
    """Build the canonical, qrel-free telemetry event used by all stages."""
    safe_provider_metadata = {
        str(key): value
        for key, value in dict(provider_metadata or {}).items()
        if key in _SAFE_PROVIDER_METADATA_KEYS
        and isinstance(value, (str, int, float, bool, type(None)))
    }
    usage = {
        "input": max(0, int(tokens_in)),
        "output": max(0, int(tokens_out)),
        "total": max(0, int(tokens_in)) + max(0, int(tokens_out)),
    }
    normalized_cost = max(0.0, float(cost_usd))
    normalized_latency = max(0.0, float(latency_sec))
    normalized_event_id = event_id or f"{attempt_id}:{stage}"
    return {
        "schema_version": ATTEMPT_EVENT_SCHEMA_VERSION,
        "event_id": normalized_event_id,
        "trace_id": str(trace_id),
        "attempt_id": str(attempt_id),
        "parent_attempt_id": (
            str(parent_attempt_id) if parent_attempt_id else None
        ),
        "stage": str(stage),
        "component": str(component),
        "status": str(status),
        "repair_status": str(repair_status),
        "model": str(model or ""),
        "model_revision": str(model_revision or ""),
        "prompt_version": str(prompt_version or ""),
        "token_usage": usage,
        "cost_usd": normalized_cost,
        "latency_sec": normalized_latency,
        "finish_reason": str(finish_reason or ""),
        "deadline_exhausted": bool(deadline_exhausted),
        "error_type": str(error_type or ""),
        "provider_metadata": safe_provider_metadata,
        # Backward-compatible aliases used by existing API consumers.
        "token_total": usage["total"],
        "cost_total_usd": normalized_cost,
        "cost_breakdown_usd": {"total": normalized_cost},
    }


def call_llm_with_metadata(client: Any, **kwargs: Any) -> LLMCallResult:
    """Use structured calls when available and adapt legacy/mock clients."""
    if hasattr(client, "chat_with_metadata"):
        result = client.chat_with_metadata(**kwargs)
        if isinstance(result, LLMCallResult):
            return result
        raise TypeError("chat_with_metadata must return LLMCallResult")
    started_at = time.monotonic()
    text = client.chat(**kwargs)
    requested_model = str(kwargs.get("model") or getattr(client, "default_model", ""))
    if "@" in requested_model:
        model, revision = requested_model.rsplit("@", 1)
    else:
        model, revision = requested_model, ""
    return LLMCallResult(
        text=str(text),
        model=model or "unknown",
        model_revision=revision,
        tokens_in=0,
        tokens_out=0,
        cost_usd=0.0,
        latency_sec=max(0.0, time.monotonic() - started_at),
        finish_reason="",
        provider_metadata={"provider": "legacy_or_mock"},
    )


def record_llm_call_results(
    state: Dict[str, Any],
    call_results: Iterable[LLMCallResult],
    *,
    trace_id: str,
    base_attempt_id: str,
    parent_attempt_id: Any,
    stage: str,
    component: str,
    repair_status: str,
    prompt_version: str = "",
) -> List[Dict[str, Any]]:
    """Record every physical provider call and update request aggregates."""
    calls = [
        call
        for call in call_results
        if isinstance(call, LLMCallResult)
    ]
    attempts = list(state.get("attempt_telemetry") or [])
    seen = {
        str(event.get("event_id"))
        for event in attempts
        if isinstance(event, dict) and event.get("event_id")
    }
    usage = dict(state.get("token_usage") or {})
    tokens_in = int(usage.get("input") or 0)
    tokens_out = int(usage.get("output") or 0)
    total_cost = float(state.get("cost_estimate") or 0.0)
    compute_time = float(
        state.get("attempt_compute_time_sec") or 0.0
    )
    added: List[Dict[str, Any]] = []
    multiple = len(calls) > 1

    for index, call in enumerate(calls, 1):
        provider_attempt = int(
            call.provider_metadata.get("provider_attempt") or index
        )
        attempt_id = (
            f"{base_attempt_id}:provider:{provider_attempt}"
            if multiple
            else base_attempt_id
        )
        event_id = (
            f"{base_attempt_id}:{stage}:provider:{provider_attempt}"
            if multiple
            else f"{base_attempt_id}:{stage}"
        )
        deadline_exhausted = (
            call.error_type == "RuntimeDeadlineExceeded"
        )
        event = build_attempt_event(
            trace_id=trace_id,
            attempt_id=attempt_id,
            parent_attempt_id=parent_attempt_id,
            stage=stage,
            component=component,
            status=(
                "deadline_exhausted"
                if deadline_exhausted
                else str(call.status or "success")
            ),
            repair_status=repair_status,
            model=call.model,
            model_revision=call.model_revision,
            tokens_in=call.tokens_in,
            tokens_out=call.tokens_out,
            cost_usd=call.cost_usd,
            latency_sec=call.latency_sec,
            finish_reason=call.finish_reason,
            prompt_version=prompt_version,
            deadline_exhausted=deadline_exhausted,
            error_type=call.error_type,
            provider_metadata=call.provider_metadata,
            event_id=event_id,
        )
        if event_id in seen:
            continue
        seen.add(event_id)
        attempts.append(event)
        added.append(event)
        tokens_in += int(call.tokens_in or 0)
        tokens_out += int(call.tokens_out or 0)
        total_cost += float(call.cost_usd or 0.0)
        compute_time += float(call.latency_sec or 0.0)

    state["attempt_telemetry"] = attempts
    state["token_usage"] = {
        "input": tokens_in,
        "output": tokens_out,
        "total": tokens_in + tokens_out,
    }
    state["cost_estimate"] = total_cost
    state["attempt_compute_time_sec"] = compute_time
    return added


def aggregate_attempt_telemetry(
    traces: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-attempt usage without treating wall latency as additive."""
    attempts: List[Dict[str, Any]] = []
    seen_event_ids = set()
    token_total = 0
    cost_total = 0.0
    attempt_compute_time = 0.0
    for trace in traces:
        canonical_events = list(trace.get("attempt_events") or [])
        if canonical_events:
            for event in canonical_events:
                event_id = str(
                    event.get("event_id")
                    or f"{event.get('attempt_id')}:{event.get('stage')}"
                )
                if event_id in seen_event_ids:
                    continue
                seen_event_ids.add(event_id)
                usage = dict(event.get("token_usage") or {})
                event_tokens = int(
                    usage.get("total")
                    if isinstance(usage.get("total"), (int, float))
                    else int(usage.get("input") or 0)
                    + int(usage.get("output") or 0)
                )
                event_cost = float(
                    event.get("cost_usd")
                    if isinstance(event.get("cost_usd"), (int, float))
                    else event.get("cost_total_usd") or 0.0
                )
                event_latency = float(event.get("latency_sec") or 0.0)
                token_total += event_tokens
                cost_total += event_cost
                attempt_compute_time += event_latency
                attempts.append(dict(event))
            continue

        usage = dict(trace.get("token_usage") or {})
        attempt_tokens = int(
            usage.get("total")
            if isinstance(usage.get("total"), (int, float))
            else int(usage.get("input") or 0) + int(usage.get("output") or 0)
        )
        costs = {
            str(key): float(value)
            for key, value in dict(trace.get("cost_breakdown_usd") or {}).items()
            if isinstance(value, (int, float))
        }
        attempt_cost = sum(costs.values())
        latencies = dict(trace.get("stage_latency_sec") or {})
        attempt_latency = float(latencies.get("total") or 0.0)
        token_total += attempt_tokens
        cost_total += attempt_cost
        attempt_compute_time += attempt_latency
        event = build_attempt_event(
            trace_id=str(trace.get("trace_id") or ""),
            attempt_id=str(trace.get("attempt_id") or ""),
            parent_attempt_id=trace.get("parent_attempt_id"),
            stage=str(trace.get("trace_role") or "agent_attempt"),
            component=str(trace.get("agent_name") or ""),
            status="success" if not trace.get("errors") else "error",
            repair_status=(
                "synthesis_repair"
                if trace.get("parent_attempt_id")
                else "initial"
            ),
            model=str(trace.get("exact_model") or ""),
            model_revision=str(trace.get("model_revision") or ""),
            tokens_in=int(usage.get("input") or 0),
            tokens_out=int(usage.get("output") or 0),
            cost_usd=attempt_cost,
            latency_sec=attempt_latency,
        )
        event["cost_breakdown_usd"] = costs
        event["token_usage"] = usage
        event["token_total"] = attempt_tokens
        event["cost_total_usd"] = attempt_cost
        attempts.append(event)
    return {
        "tokens_used": token_total,
        "cost_usd": cost_total,
        "attempt_compute_time_sec": attempt_compute_time,
        "attempt_telemetry": attempts,
    }


def record_conditional_verifier_telemetry(
    state: Dict[str, Any],
    trace: EvaluationTrace,
    decision: VerificationDecision,
) -> None:
    """Add top-level conditional-judge calls to aggregate request telemetry."""
    results = list(
        decision.raw_decision.get("conditional_claim_verification") or []
    )
    if not results:
        return

    aggregate_usage = dict(state.get("token_usage") or {})
    attempts = list(state.get("attempt_telemetry") or [])
    total_cost = 0.0
    total_in = 0
    total_out = 0
    total_tokens = 0
    for index, result in enumerate(results, 1):
        usage = dict(result.get("token_usage") or {})
        tokens_in = int(usage.get("input") or 0)
        tokens_out = int(usage.get("output") or 0)
        token_total = int(
            usage.get("total")
            if isinstance(usage.get("total"), (int, float))
            else tokens_in + tokens_out
        )
        cost = float(result.get("cost_usd") or 0.0)
        error_type = str(result.get("error_type") or "")
        deadline_exhausted = error_type in {
            "TimeoutError",
            "RuntimeDeadlineExceeded",
        }
        event = build_attempt_event(
            trace_id=trace.trace_id,
            attempt_id=f"{trace.attempt_id}:verification:{index}",
            parent_attempt_id=trace.attempt_id,
            stage="conditional_claim_verification",
            component=trace.agent_name,
            status=(
                "success"
                if result.get("valid") is True
                else "deadline_exhausted"
                if deadline_exhausted
                else "error"
            ),
            repair_status="verification",
            model=str(result.get("model") or ""),
            model_revision=str(result.get("model_revision") or ""),
            prompt_version=str(result.get("prompt_version") or ""),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            latency_sec=float(result.get("latency_sec") or 0.0),
            finish_reason=str(result.get("finish_reason") or ""),
            deadline_exhausted=deadline_exhausted,
            error_type=error_type,
            provider_metadata=result.get("provider_metadata"),
        )
        event["valid"] = result.get("valid") is True
        if not any(
            item.get("event_id") == event["event_id"] for item in attempts
        ):
            attempts.append(event)
            total_in += tokens_in
            total_out += tokens_out
            total_tokens += token_total
            total_cost += cost

    state["token_usage"] = {
        "input": int(aggregate_usage.get("input") or 0) + total_in,
        "output": int(aggregate_usage.get("output") or 0) + total_out,
        "total": int(aggregate_usage.get("total") or 0) + total_tokens,
    }
    state["cost_estimate"] = float(state.get("cost_estimate") or 0.0) + total_cost
    state["attempt_telemetry"] = attempts
