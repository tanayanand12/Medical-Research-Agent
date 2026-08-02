"""Versioned, qrel-free schemas shared by runtime and offline evaluation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


EVALUATION_TRACE_SCHEMA_VERSION = "1.0.0"
ATTEMPT_EVENT_SCHEMA_VERSION = "1.0.0"
RUNTIME_VERIFIER_PROMPT_VERSION = "runtime-verifier-v1"
RUNTIME_SYNTHESIS_PROMPT_VERSION = "runtime-synthesis-v1"


@dataclass
class RetrievedDocument:
    document_id: str
    source: str
    provenance: str
    retrieval_method: str
    original_rank: int
    raw_score: Optional[float]
    score_type: str
    content_hash: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankedDocument:
    document_id: str
    rank: int
    reranker_score: Optional[float]
    reranker_model: str
    reranker_revision: str = ""


@dataclass
class ContextSpan:
    document_id: str
    start_char: int
    end_char: int
    text: str
    content_hash: str
    truncated: bool = False
    original_length: int = 0


@dataclass
class AtomicClaim:
    claim_id: str
    text: str
    cited_document_ids: List[str] = field(default_factory=list)


@dataclass
class CitationResolution:
    citation: str
    document_id: Optional[str]
    resolved: bool
    reason: str = ""


@dataclass
class VerificationDecision:
    status: str
    component_scores: Dict[str, float]
    failed_checks: List[str]
    structured_feedback: List[Dict[str, Any]]
    target_stage: str
    target_agent: str
    recommended_retry_changes: Dict[str, Any]
    verifier_confidence: float
    valid: bool = True
    error: Optional[str] = None
    verifier_model: str = "deterministic"
    verifier_model_revision: str = ""
    prompt_version: str = RUNTIME_VERIFIER_PROMPT_VERSION
    raw_decision: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "VerificationDecision":
        return cls(**value)


@dataclass
class EvaluationTrace:
    schema_version: str
    trace_id: str
    attempt_id: str
    parent_attempt_id: Optional[str]
    agent_name: str
    domain: str
    original_query: str
    expanded_queries: List[str]
    retrieval_configuration: Dict[str, Any]
    retrieved_documents: List[RetrievedDocument]
    reranked_documents: List[RerankedDocument]
    final_context_document_ids: List[str]
    final_context_spans: List[ContextSpan]
    answer: str
    atomic_claims: List[AtomicClaim]
    citations: List[CitationResolution]
    stage_latency_sec: Dict[str, float]
    token_usage: Dict[str, int]
    cost_breakdown_usd: Dict[str, float]
    exact_model: str
    model_revision: str
    prompt_version: str
    config_version: str
    errors: List[str]
    partial_response: bool
    verification_decisions: List[VerificationDecision] = field(default_factory=list)
    retry_feedback: List[Dict[str, Any]] = field(default_factory=list)
    repair_history: List[Dict[str, Any]] = field(default_factory=list)
    trace_role: str = "attempt"
    synthesis_manifest_present: bool = False
    answer_origin: str = "generated"
    attempt_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "EvaluationTrace":
        data = dict(value)
        data["retrieved_documents"] = [
            item if isinstance(item, RetrievedDocument) else RetrievedDocument(**item)
            for item in data.get("retrieved_documents", [])
        ]
        data["reranked_documents"] = [
            item if isinstance(item, RerankedDocument) else RerankedDocument(**item)
            for item in data.get("reranked_documents", [])
        ]
        data["final_context_spans"] = [
            item if isinstance(item, ContextSpan) else ContextSpan(**item)
            for item in data.get("final_context_spans", [])
        ]
        data["atomic_claims"] = [
            item if isinstance(item, AtomicClaim) else AtomicClaim(**item)
            for item in data.get("atomic_claims", [])
        ]
        data["citations"] = [
            item
            if isinstance(item, CitationResolution)
            else CitationResolution(**item)
            for item in data.get("citations", [])
        ]
        data["verification_decisions"] = [
            item
            if isinstance(item, VerificationDecision)
            else VerificationDecision.from_dict(item)
            for item in data.get("verification_decisions", [])
        ]
        return cls(**data)


def validate_evaluation_trace(trace: EvaluationTrace) -> List[str]:
    """Return schema errors without raising or consulting benchmark references."""
    errors: List[str] = []
    if trace.schema_version != EVALUATION_TRACE_SCHEMA_VERSION:
        errors.append("unsupported_schema_version")
    for name in (
        "trace_id",
        "attempt_id",
        "agent_name",
        "domain",
        "original_query",
        "prompt_version",
        "config_version",
    ):
        if not getattr(trace, name, ""):
            errors.append(f"missing_{name}")

    retrieval_only = bool(
        trace.retrieval_configuration.get("retrieval_only")
    )
    if retrieval_only and trace.answer.strip():
        errors.append("retrieval_only_trace_contains_generated_answer")
    if trace.answer.strip() and trace.answer_origin == "generated":
        if not trace.synthesis_manifest_present:
            errors.append("missing_synthesis_manifest")
        elif not trace.final_context_spans:
            errors.append("empty_synthesis_manifest")

    document_ids = [doc.document_id for doc in trace.retrieved_documents]
    if any(not value for value in document_ids):
        errors.append("missing_document_id")
    if len(document_ids) != len(set(document_ids)):
        errors.append("duplicate_document_id")
    for document in trace.retrieved_documents:
        expected_hash = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
        if document.content_hash != expected_hash:
            errors.append("retrieved_document_hash_mismatch")

    known_ids = set(document_ids)
    documents_by_id = {
        document.document_id: document for document in trace.retrieved_documents
    }
    for doc in trace.reranked_documents:
        if doc.document_id not in known_ids:
            errors.append(f"reranked_document_not_retrieved:{doc.document_id}")
    for document_id in trace.final_context_document_ids:
        if document_id not in known_ids:
            errors.append(f"context_document_not_retrieved:{document_id}")
    for span in trace.final_context_spans:
        source = documents_by_id.get(span.document_id)
        if source is None:
            errors.append(f"context_span_document_not_retrieved:{span.document_id}")
            continue
        expected_hash = hashlib.sha256(span.text.encode("utf-8")).hexdigest()
        if span.content_hash != expected_hash:
            errors.append("context_span_hash_mismatch")
        if (
            span.start_char < 0
            or span.end_char < span.start_char
            or span.end_char > len(source.text)
        ):
            errors.append("context_span_offsets_invalid")
        elif source.text[span.start_char : span.end_char] != span.text:
            errors.append("context_span_source_mismatch")
        if span.original_length != len(source.text):
            errors.append("context_span_original_length_mismatch")
    if trace.final_context_document_ids != [
        span.document_id for span in trace.final_context_spans
    ]:
        errors.append("context_manifest_order_mismatch")
    final_context_ids = set(trace.final_context_document_ids)
    for claim in trace.atomic_claims:
        for document_id in claim.cited_document_ids:
            if document_id not in known_ids:
                errors.append(
                    f"claim_citation_not_retrieved:{claim.claim_id}:{document_id}"
                )
            elif document_id not in final_context_ids:
                errors.append(
                    "claim_citation_not_in_final_context:"
                    f"{claim.claim_id}:{document_id}"
                )
    for citation in trace.citations:
        if (
            citation.resolved
            and citation.document_id
            and citation.document_id not in final_context_ids
        ):
            errors.append(
                f"resolved_citation_not_in_final_context:{citation.citation}"
            )
    seen_event_ids = set()
    for event in trace.attempt_events:
        if not isinstance(event, dict):
            errors.append("attempt_event_not_object")
            continue
        if event.get("schema_version") != ATTEMPT_EVENT_SCHEMA_VERSION:
            errors.append("unsupported_attempt_event_schema_version")
        for key in (
            "event_id",
            "trace_id",
            "attempt_id",
            "stage",
            "component",
            "status",
            "repair_status",
        ):
            if not isinstance(event.get(key), str) or not event.get(key):
                errors.append(f"invalid_attempt_event_{key}")
        event_id = str(event.get("event_id") or "")
        if event_id in seen_event_ids:
            errors.append(f"duplicate_attempt_event_id:{event_id}")
        seen_event_ids.add(event_id)
        if event.get("trace_id") != trace.trace_id:
            errors.append("attempt_event_trace_id_mismatch")
        if event.get("status") not in {
            "success",
            "error",
            "deadline_exhausted",
            "skipped",
        }:
            errors.append("invalid_attempt_event_status")
        usage = event.get("token_usage")
        if not isinstance(usage, dict):
            errors.append("invalid_attempt_event_token_usage")
        else:
            token_values = [
                usage.get("input"),
                usage.get("output"),
                usage.get("total"),
            ]
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in token_values
            ):
                errors.append("invalid_attempt_event_token_usage")
            elif usage["total"] != usage["input"] + usage["output"]:
                errors.append("attempt_event_token_total_mismatch")
        for key in ("cost_usd", "latency_sec"):
            value = event.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                errors.append(f"invalid_attempt_event_{key}")
        if not isinstance(event.get("provider_metadata"), dict):
            errors.append("invalid_attempt_event_provider_metadata")
        for key in (
            "model",
            "model_revision",
            "finish_reason",
            "prompt_version",
            "error_type",
        ):
            if not isinstance(event.get(key), str):
                errors.append(f"invalid_attempt_event_{key}")
        if not isinstance(event.get("deadline_exhausted"), bool):
            errors.append("invalid_attempt_event_deadline_exhausted")
    return errors
