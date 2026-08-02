"""Bounded retry context construction for runtime verification."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from evaluation_core import EvaluationTrace, VerificationDecision

_MAX_RETRY_TOP_K = 20
_MAX_QUERY_ADDITIONS = 8
_MAX_QUERY_ADDITION_LENGTH = 80
_SUPPORTED_RETRIEVAL_METHODS = {"hybrid_with_sparse_fallback"}
_SYNTHESIS_CONSTRAINTS = {
    "preserve_evidence",
    "require_resolvable_citations",
    "remove_or_qualify_unsupported_assertions",
}
_SUPPORTED_KEYS = {
    "query_additions",
    "top_k",
    "retrieval_method",
    "preserve_original_query",
    *_SYNTHESIS_CONSTRAINTS,
}


def build_retry_request(
    *,
    original_query: str,
    context: Dict[str, Any],
    decision: VerificationDecision,
    trace: EvaluationTrace,
    attempt_number: int,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Apply structured feedback without mutating the caller's context."""
    changes = dict(decision.recommended_retry_changes)
    applied_changes: Dict[str, Any] = {}
    rejected_changes: Dict[str, str] = {}
    raw_additions = changes.get("query_additions", [])
    additions = []
    if isinstance(raw_additions, (list, tuple)):
        for value in raw_additions[:_MAX_QUERY_ADDITIONS]:
            addition = " ".join(str(value).split())
            if (
                addition
                and len(addition) <= _MAX_QUERY_ADDITION_LENGTH
                and addition not in additions
            ):
                additions.append(addition)
            elif addition:
                rejected_changes["query_additions"] = "invalid_or_oversized_value"
        if len(raw_additions) > _MAX_QUERY_ADDITIONS:
            rejected_changes["query_additions"] = "too_many_values"
    elif raw_additions:
        rejected_changes["query_additions"] = "invalid_type"
    retry_query = original_query
    if additions:
        retry_query = f"{original_query} {' '.join(additions)}"

    retry_context = dict(context)
    retry_context["original_query"] = original_query
    retry_context["attempt_id"] = (
        f"{trace.trace_id}:{trace.agent_name}:{attempt_number + 1}"
    )
    retry_context["parent_attempt_id"] = trace.attempt_id
    retry_context["verification_feedback"] = list(decision.structured_feedback)
    retry_context["retry_count"] = attempt_number

    if additions:
        applied_changes["query_additions"] = additions
    if "top_k" in changes:
        try:
            top_k = max(1, min(_MAX_RETRY_TOP_K, int(changes["top_k"])))
            retry_context["top_k"] = top_k
            applied_changes["top_k"] = top_k
        except (TypeError, ValueError):
            rejected_changes["top_k"] = "invalid_value"
    if "retrieval_method" in changes:
        retrieval_method = str(changes["retrieval_method"])
        if retrieval_method in _SUPPORTED_RETRIEVAL_METHODS:
            retry_context["retrieval_method"] = retrieval_method
            applied_changes["retrieval_method"] = retrieval_method
        else:
            rejected_changes["retrieval_method"] = "unsupported_value"
    if changes.get("preserve_original_query") is True:
        applied_changes["preserve_original_query"] = True
    for key in _SYNTHESIS_CONSTRAINTS:
        if changes.get(key) is True and decision.target_stage == "synthesis":
            retry_context.setdefault("repair_constraints", {})[key] = True
            applied_changes[key] = True
    for key in changes:
        if key not in _SUPPORTED_KEYS:
            rejected_changes[key] = "unsupported_key"

    actual_configuration: Dict[str, Any] = {
        "query": retry_query,
        "original_query_preserved": retry_query.startswith(original_query),
    }
    actual_configuration.update({
        key: retry_context[key]
        for key in ("top_k", "retrieval_method")
        if key in retry_context
    })
    if retry_context.get("repair_constraints"):
        actual_configuration["repair_constraints"] = dict(
            retry_context["repair_constraints"]
        )

    repair_event = {
        "attempt": attempt_number,
        "target_stage": decision.target_stage,
        "target_agent": decision.target_agent,
        "parent_attempt_id": trace.attempt_id,
        "feedback": list(decision.structured_feedback),
        "recommended_changes": changes,
        "applied_changes": applied_changes,
        "rejected_changes": rejected_changes,
        "actual_configuration": actual_configuration,
    }
    history = list(context.get("repair_history") or [])
    history.append(repair_event)
    retry_context["repair_history"] = history
    return retry_query, retry_context, repair_event
