"""Shared, versioned evaluation data contracts with no benchmark dependencies."""

from .deadline import (
    RuntimeDeadlineExceeded,
    ensure_deadline,
    remaining_seconds,
    sleep_with_deadline,
)
from .privacy import (
    redact_sensitive_values,
    redact_trace_for_response,
    redact_traces_for_response,
    safe_error_type,
    sanitize_sensitive_text,
    stable_query_fingerprint,
)
from .schemas import (
    ATTEMPT_EVENT_SCHEMA_VERSION,
    EVALUATION_TRACE_SCHEMA_VERSION,
    RUNTIME_SYNTHESIS_PROMPT_VERSION,
    RUNTIME_VERIFIER_PROMPT_VERSION,
    AtomicClaim,
    CitationResolution,
    ContextSpan,
    EvaluationTrace,
    RerankedDocument,
    RetrievedDocument,
    VerificationDecision,
    validate_evaluation_trace,
)
from .trace_adapter import (
    build_agent_evaluation_trace,
    build_orchestrator_evaluation_trace,
    content_hash,
    document_text,
    stable_document_id,
)

__all__ = [
    "ATTEMPT_EVENT_SCHEMA_VERSION",
    "EVALUATION_TRACE_SCHEMA_VERSION",
    "RUNTIME_SYNTHESIS_PROMPT_VERSION",
    "RUNTIME_VERIFIER_PROMPT_VERSION",
    "RuntimeDeadlineExceeded",
    "AtomicClaim",
    "CitationResolution",
    "ContextSpan",
    "EvaluationTrace",
    "RerankedDocument",
    "RetrievedDocument",
    "VerificationDecision",
    "build_agent_evaluation_trace",
    "build_orchestrator_evaluation_trace",
    "content_hash",
    "document_text",
    "ensure_deadline",
    "redact_sensitive_values",
    "redact_trace_for_response",
    "redact_traces_for_response",
    "remaining_seconds",
    "safe_error_type",
    "sanitize_sensitive_text",
    "stable_query_fingerprint",
    "stable_document_id",
    "sleep_with_deadline",
    "validate_evaluation_trace",
]
