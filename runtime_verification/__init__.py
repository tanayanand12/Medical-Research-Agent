"""Bounded, qrel-free runtime verification and repair primitives."""

from .claim_verifier import ConditionalClaimVerifier, HIGH_RISK_CLAIM_PROMPT_VERSION
from .confidence import CONFIDENCE_WEIGHTS, calculate_combined_confidence
from .evidence import build_evidence_context, evidence_limited_answer
from .factory import build_runtime_verifier
from .repair import repair_agent_synthesis
from .retry_policy import build_retry_request
from .telemetry import (
    ATTEMPT_EVENT_SCHEMA_VERSION,
    aggregate_attempt_telemetry,
    build_attempt_event,
    call_llm_with_metadata,
    record_conditional_verifier_telemetry,
    record_llm_call_results,
)
from .verifier import RuntimeVerifier, VerifierConfig, evidence_limited_decision

__all__ = [
    "CONFIDENCE_WEIGHTS",
    "ATTEMPT_EVENT_SCHEMA_VERSION",
    "ConditionalClaimVerifier",
    "HIGH_RISK_CLAIM_PROMPT_VERSION",
    "RuntimeVerifier",
    "VerifierConfig",
    "build_evidence_context",
    "build_retry_request",
    "build_runtime_verifier",
    "aggregate_attempt_telemetry",
    "build_attempt_event",
    "call_llm_with_metadata",
    "record_conditional_verifier_telemetry",
    "record_llm_call_results",
    "calculate_combined_confidence",
    "evidence_limited_decision",
    "evidence_limited_answer",
    "repair_agent_synthesis",
]
