"""Evidence-preserving synthesis repair shared by heterogeneous agents."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any, Dict

from evaluation_core import (
    EvaluationTrace,
    VerificationDecision,
    build_agent_evaluation_trace,
)
from llm_client import LLMClient
from runtime_verification.telemetry import (
    build_attempt_event,
    call_llm_with_metadata,
)


def repair_agent_synthesis(
    *,
    agent_output: Any,
    trace: EvaluationTrace,
    decision: VerificationDecision,
    original_query: str,
    context: Dict[str, Any],
) -> Any:
    """Repair only the agent answer while freezing its verified evidence."""
    spans = list(trace.final_context_spans)
    if not spans:
        raise ValueError("agent synthesis repair requires frozen evidence")

    evidence = "\n\n".join(
        f"[{index}] Document ID: {span.document_id}\nEvidence: {span.text}"
        for index, span in enumerate(spans, 1)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Repair the medical answer using only the supplied frozen evidence. "
                "Every factual claim must cite a matching marker such as [1]. "
                "Do not add evidence, rerun retrieval, or speculate."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {original_query}\n\n"
                f"Previous answer:\n{trace.answer}\n\n"
                "Structured verifier feedback:\n"
                f"{json.dumps(decision.structured_feedback, ensure_ascii=False)}\n\n"
                f"Frozen evidence:\n{evidence}\n\n"
                "Return only the repaired evidence-grounded answer."
            ),
        },
    ]
    model = str(context.get("model_id") or trace.exact_model or "") or None
    deadline_at = context.get("_runtime_deadline_at_monotonic")
    deadline_kwargs: Dict[str, Any] = {}
    if deadline_at is not None:
        remaining = float(deadline_at) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "runtime deadline expired before agent synthesis repair"
            )
        deadline_kwargs = {
            "timeout": remaining,
            "client_max_attempts": 1,
            "deadline_at": float(deadline_at),
        }

    call_result = call_llm_with_metadata(
        LLMClient(),
        messages=messages,
        model=model,
        temperature=0.2,
        max_tokens=800,
        **deadline_kwargs,
    )
    repaired_answer = call_result.text
    repair_latency = call_result.latency_sec

    documents = [
        {
            "document_id": document.document_id,
            "text": document.text,
            "score": document.raw_score,
            "original_rank": document.original_rank,
            "retrieval_method": document.retrieval_method,
            "score_type": document.score_type,
            "metadata": dict(document.metadata),
        }
        for document in trace.retrieved_documents
    ]
    document_by_id = {
        document["document_id"]: document for document in documents
    }
    reranked = [
        {
            **document_by_id[span.document_id],
            "text": span.text,
        }
        for span in spans
        if span.document_id in document_by_id
    ]
    trace_context = dict(context)
    trace_context.update(trace.retrieval_configuration)
    trace_context["retrieval_only"] = False
    trace_context["repair_history"] = list(context.get("repair_history") or [])
    repair_attempt_id = str(
        trace_context.get("attempt_id") or f"{trace.attempt_id}:repair"
    )
    trace_id = str(trace_context.get("trace_id") or trace.trace_id)
    repaired_trace = build_agent_evaluation_trace(
        agent_name=trace.agent_name,
        domain=trace.domain,
        original_query=original_query,
        state={
            "expanded_query": (
                trace.expanded_queries[-1]
                if trace.expanded_queries
                else original_query
            ),
            "retrieval_results": documents,
            "reranked_results": reranked,
            "synthesis_context": [
                {
                    "document_id": span.document_id,
                    "text": span.text,
                    "start_char": span.start_char,
                    "original_length": span.original_length,
                    "truncated": span.truncated,
                    "citation_marker": index,
                }
                for index, span in enumerate(spans, 1)
            ],
            "answer": repaired_answer,
            "citations": list(getattr(agent_output, "citations", []) or []),
            "model_used": (
                f"{call_result.model}@{call_result.model_revision}"
                if call_result.model_revision
                else call_result.model
            ),
            "execution_time_sec": repair_latency,
            "stage_latency_sec": {"synthesis": repair_latency},
            "token_usage": {
                "input": call_result.tokens_in,
                "output": call_result.tokens_out,
                "total": call_result.tokens_in + call_result.tokens_out,
            },
            "cost_breakdown_usd": {"repair": call_result.cost_usd},
            "attempt_events": [
                build_attempt_event(
                    trace_id=trace_id,
                    attempt_id=repair_attempt_id,
                    parent_attempt_id=trace.attempt_id,
                    stage="agent_synthesis_repair",
                    component=trace.agent_name,
                    status=str(call_result.status or "success"),
                    repair_status="synthesis_repair",
                    model=call_result.model,
                    model_revision=call_result.model_revision,
                    tokens_in=call_result.tokens_in,
                    tokens_out=call_result.tokens_out,
                    cost_usd=call_result.cost_usd,
                    latency_sec=call_result.latency_sec,
                    finish_reason=call_result.finish_reason,
                    deadline_exhausted=(
                        call_result.error_type
                        == "RuntimeDeadlineExceeded"
                    ),
                    error_type=call_result.error_type,
                    provider_metadata=call_result.provider_metadata,
                )
            ],
            "error": None,
        },
        context=trace_context,
    )
    return replace(
        agent_output,
        answer=repaired_answer,
        execution_time_sec=(
            float(getattr(agent_output, "execution_time_sec", 0.0))
            + repair_latency
        ),
        error=None,
        evaluation_trace=repaired_trace,
    )
