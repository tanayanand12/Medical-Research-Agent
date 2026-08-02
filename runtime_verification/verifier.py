"""Qrel-free runtime verification with deterministic checks first."""

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from evaluation_core import (
    EvaluationTrace,
    VerificationDecision,
    validate_evaluation_trace,
)
from runtime_verification.entities import (
    entity_attribution_status,
    extract_salient_entities,
    normalize_biomedical_text,
)


SimilarityScorer = Callable[[str, Sequence[str]], float]
ClaimVerifier = Callable[[str, Sequence[str]], Dict[str, Any]]


def _empty_components() -> Dict[str, float]:
    return {
        "retrieval_coverage": 0.0,
        "evidence_sufficiency": 0.0,
        "claim_grounding": 0.0,
        "citation_support": 0.0,
        "query_coverage": 0.0,
        "retrieval_query_coverage": 0.0,
        "evidence_query_coverage": 0.0,
        "answer_query_coverage": 0.0,
        "verifier_confidence": 0.0,
    }


def evidence_limited_decision(
    *,
    target_agent: str,
    failed_check: str,
    message: str,
    valid: bool,
    error: Optional[str] = None,
) -> VerificationDecision:
    """Build a complete terminal decision for bounded-control failures."""
    return VerificationDecision(
        status="evidence_limited",
        component_scores=_empty_components(),
        failed_checks=[failed_check],
        structured_feedback=[{"check": failed_check, "message": message}],
        target_stage="none",
        target_agent=target_agent,
        recommended_retry_changes={},
        verifier_confidence=0.0,
        valid=valid,
        error=error,
        raw_decision={"terminal_reason": failed_check},
    )

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?")
_INTERNAL_CAMEL_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b")
_NEGATION_RE = re.compile(r"\b(?:no|not|never|without|neither|nor|didn't|doesn't)\b", re.I)
_HIGH_RISK_RE = re.compile(
    r"\b(?:mortality|death|contraindicat|dose|dosage|pregnan|adverse|harm|risk)\w*\b",
    re.I,
)
_INCREASE_RE = re.compile(
    r"\b(?:increase|increased|increases|higher|raise|raised|worsen|worsened)\b",
    re.I,
)
_DECREASE_RE = re.compile(
    r"\b(?:decrease|decreased|decreases|lower|lowered|reduce|reduced|reduces)\b",
    re.I,
)
_FAILURE_PHRASES = (
    "i was unable",
    "unable to synthesize",
    "unable to synthesise",
    "as an ai",
    "error synthesising",
    "error synthesizing",
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "with",
}
_CONCEPT_PATTERNS = {
    "population": (
        r"\badults?\b",
        r"\bchildren\b",
        r"\bpediatric\b",
        r"\bpaediatric\b",
        r"\bpregnan\w*\b",
        r"\bneonates?\b",
        r"\belderly\b",
        r"\bolder adults?\b",
        r"\bwomen\b",
        r"\bmen\b",
    ),
    "timeframe": (
        r"\b(?:19|20)\d{2}\b",
        r"\bsince\b",
        r"\brecent\w*\b",
        r"\blatest\b",
        r"\bwithin\s+\d+\s+(?:days?|weeks?|months?|years?)\b",
    ),
    "study_design": (
        r"\brandomi[sz]ed(?: controlled)? trials?\b",
        r"\brcts?\b",
        r"\bcohort(?: study| studies)?\b",
        r"\bcase control\b",
        r"\bmeta analysis\b",
        r"\bsystematic review\b",
        r"\bobservational stud(?:y|ies)\b",
    ),
}


@dataclass(frozen=True)
class VerifierConfig:
    min_documents: int = 1
    min_query_coverage: float = 0.35
    top_k_increment: int = 3
    max_top_k: int = 20
    require_citations_for_factual_claims: bool = True
    min_claim_lexical_support: float = 0.45
    min_supported_claim_confidence: float = 0.70


class RuntimeVerifier:
    """Shared verifier used for every agent and final synthesis."""

    def __init__(
        self,
        config: Optional[VerifierConfig] = None,
        *,
        similarity_scorer: Optional[SimilarityScorer] = None,
        claim_verifier: Optional[ClaimVerifier] = None,
    ) -> None:
        self.config = config or VerifierConfig()
        self.similarity_scorer = similarity_scorer
        self.claim_verifier = claim_verifier

    def verify(
        self,
        trace: EvaluationTrace,
        *,
        retries_remaining: int = 1,
        retrieval_retries_remaining: Optional[int] = None,
        synthesis_repairs_remaining: Optional[int] = None,
    ) -> VerificationDecision:
        schema_errors = validate_evaluation_trace(trace)
        if schema_errors:
            decision = VerificationDecision(
                status="evidence_limited",
                component_scores=_empty_components(),
                failed_checks=["invalid_evaluation_trace"],
                structured_feedback=[
                    {
                        "check": "invalid_evaluation_trace",
                        "message": "Evaluation trace failed schema validation.",
                        "errors": schema_errors,
                    }
                ],
                target_stage="none",
                target_agent=trace.agent_name,
                recommended_retry_changes={},
                verifier_confidence=0.0,
                valid=False,
                error="; ".join(schema_errors),
                raw_decision={"schema_errors": schema_errors},
            )
            trace.verification_decisions.append(decision)
            return decision

        try:
            check_result = self._deterministic_checks(trace)
            failed_checks = check_result["failed_checks"]
            feedback = check_result["feedback"]
            components = check_result["components"]
            raw_decision = check_result["raw_decision"]
            verifier_valid = True
            verifier_model = "deterministic"
            verifier_revision = ""
            prompt_version = "runtime-verifier-v1"
            semantic_confidences: List[float] = []

            if self.similarity_scorer is not None:
                similarity = float(
                    self.similarity_scorer(
                        trace.original_query,
                        [item.text for item in trace.final_context_spans],
                    )
                )
                components["evidence_similarity"] = max(
                    0.0, min(1.0, similarity)
                )
                raw_decision["similarity_semantics"] = (
                    "relevance triage only; not entailment or faithfulness"
                )

            if self.claim_verifier is not None:
                semantic = self._conditional_claim_checks(
                    trace,
                    failed_checks,
                    feedback,
                    required_claims=dict(
                        check_result["conditional_claim_reasons"]
                    ),
                )
                raw_decision["conditional_claim_verification"] = semantic["results"]
                verifier_valid = bool(semantic["valid"])
                verifier_model = str(semantic.get("model") or verifier_model)
                verifier_revision = str(semantic.get("model_revision") or "")
                prompt_version = str(semantic.get("prompt_version") or prompt_version)
                semantic_confidences = list(semantic.get("confidences") or [])
                _apply_verifier_telemetry(trace, semantic)
                claim_count = len(trace.atomic_claims)
                if claim_count:
                    deterministic_supported = len(
                        check_result[
                            "deterministically_supported_claim_ids"
                        ]
                    )
                    semantic_supported = len(
                        semantic.get("supported_claim_ids") or []
                    )
                    components["claim_grounding"] = (
                        deterministic_supported + semantic_supported
                    ) / claim_count

            retrieval_failures = {
                "empty_retrieval",
                "insufficient_retrieval",
                "malformed_document",
                "duplicate_document",
                "missing_query_components",
                "retrieval_rank_anomaly",
                "reranking_anomaly",
            }
            synthesis_failures = set(failed_checks) - retrieval_failures

            retrieval_budget = (
                retries_remaining
                if retrieval_retries_remaining is None
                else retrieval_retries_remaining
            )
            synthesis_budget = (
                retries_remaining
                if synthesis_repairs_remaining is None
                else synthesis_repairs_remaining
            )
            has_retrieval_failure = bool(
                retrieval_failures.intersection(failed_checks)
            )
            if not verifier_valid:
                status = "evidence_limited"
                target_stage = "none"
            elif failed_checks and has_retrieval_failure:
                if retrieval_budget > 0:
                    status = "retry_retrieval"
                    target_stage = "retrieval"
                else:
                    status = "evidence_limited"
                    target_stage = "none"
            elif failed_checks:
                if synthesis_budget > 0:
                    status = "retry_synthesis"
                    target_stage = "synthesis"
                else:
                    status = "evidence_limited"
                    target_stage = "none"
            elif (
                failed_checks
                and retrieval_retries_remaining is None
                and synthesis_repairs_remaining is None
                and retries_remaining > 0
            ):
                if retrieval_failures.intersection(failed_checks):
                    status = "retry_retrieval"
                    target_stage = "retrieval"
                else:
                    status = "retry_synthesis"
                    target_stage = "synthesis"
            elif failed_checks:
                status = "evidence_limited"
                target_stage = "none"
            else:
                status = "accept"
                target_stage = "none"

            retry_changes = self._recommended_changes(
                trace,
                failed_checks=failed_checks,
                retrieval_failure=bool(
                    retrieval_failures.intersection(failed_checks)
                ),
            )
            verifier_confidence = (
                0.0
                if not verifier_valid
                else (
                    min(semantic_confidences)
                    if semantic_confidences
                    else (1.0 if failed_checks or components else 0.5)
                )
            )
            components["verifier_confidence"] = verifier_confidence

            decision = VerificationDecision(
                status=status,
                component_scores=components,
                failed_checks=failed_checks,
                structured_feedback=feedback,
                target_stage=target_stage,
                target_agent=trace.agent_name,
                recommended_retry_changes=retry_changes,
                verifier_confidence=verifier_confidence,
                valid=verifier_valid,
                error=(
                    str(semantic.get("error"))
                    if self.claim_verifier is not None and semantic.get("error")
                    else None
                ),
                verifier_model=verifier_model,
                verifier_model_revision=verifier_revision,
                prompt_version=prompt_version,
                raw_decision=raw_decision,
            )
        except Exception as exc:
            decision = VerificationDecision(
                status="evidence_limited",
                component_scores=_empty_components(),
                failed_checks=["verifier_failure"],
                structured_feedback=[
                    {
                        "check": "verifier_failure",
                        "message": "Runtime verification failed; output cannot be accepted.",
                    }
                ],
                target_stage="none",
                target_agent=trace.agent_name,
                recommended_retry_changes={},
                verifier_confidence=0.0,
                valid=False,
                error=f"verifier_failed:{type(exc).__name__}",
                raw_decision={"exception_type": type(exc).__name__},
            )

        trace.verification_decisions.append(decision)
        return decision

    def _deterministic_checks(self, trace: EvaluationTrace) -> Dict[str, Any]:
        failed: List[str] = []
        feedback: List[Dict[str, Any]] = []
        documents = trace.retrieved_documents
        retrieval_only = bool(trace.retrieval_configuration.get("retrieval_only"))
        context_by_id = {
            span.document_id: span.text for span in trace.final_context_spans
        }
        directly_supported_high_risk_claim_ids: Set[str] = set()
        high_risk_results: List[Dict[str, Any]] = []
        conditional_claim_reasons: Dict[str, str] = {}
        deterministically_supported_claim_ids: Set[str] = set()

        def add(check: str, message: str, **details: Any) -> None:
            if check not in failed:
                failed.append(check)
            item = {"check": check, "message": message}
            item.update(details)
            feedback.append(item)

        if not documents:
            add(
                "empty_retrieval",
                "No evidence documents were retrieved.",
                suggested_action="broaden source-specific query and increase top_k",
            )
        elif len(documents) < self.config.min_documents:
            add(
                "insufficient_retrieval",
                f"Only {len(documents)} evidence documents were retrieved.",
            )

        ids: Set[str] = set()
        hashes: Set[str] = set()
        retrieval_ranks: Set[int] = set()
        for document in documents:
            if not document.document_id or not document.text.strip():
                add(
                    "malformed_document",
                    "A retrieved document has no stable ID or usable content.",
                    document_id=document.document_id,
                )
            if document.document_id in ids or document.content_hash in hashes:
                add(
                    "duplicate_document",
                    "Duplicate evidence documents were detected.",
                    document_id=document.document_id,
                )
            ids.add(document.document_id)
            hashes.add(document.content_hash)
            if document.original_rank <= 0 or document.original_rank in retrieval_ranks:
                add(
                    "retrieval_rank_anomaly",
                    "Retrieval ranks are missing or duplicated.",
                    document_id=document.document_id,
                )
            retrieval_ranks.add(document.original_rank)

        rerank_ids: Set[str] = set()
        rerank_ranks: Set[int] = set()
        for document in trace.reranked_documents:
            if (
                document.document_id not in ids
                or document.document_id in rerank_ids
                or document.rank <= 0
                or document.rank in rerank_ranks
            ):
                add(
                    "reranking_anomaly",
                    "Reranked documents do not form a valid ordering of retrieved evidence.",
                    document_id=document.document_id,
                )
            rerank_ids.add(document.document_id)
            rerank_ranks.add(document.rank)

        query_tokens = _tokens(trace.original_query)
        retrieval_tokens = _tokens(" ".join(document.text for document in documents))
        final_context_text = " ".join(
            span.text for span in trace.final_context_spans
        )
        evaluation_evidence_text = (
            " ".join(document.text for document in documents)
            if retrieval_only
            else final_context_text
        )
        evidence_tokens = _tokens(evaluation_evidence_text)
        answer_tokens = _tokens(trace.answer)
        retrieval_covered = query_tokens.intersection(retrieval_tokens)
        evidence_covered = query_tokens.intersection(evidence_tokens)
        answer_covered = query_tokens.intersection(answer_tokens)
        retrieval_query_coverage = (
            len(retrieval_covered) / len(query_tokens) if query_tokens else 1.0
        )
        evidence_query_coverage = (
            len(evidence_covered) / len(query_tokens) if query_tokens else 1.0
        )
        answer_query_coverage = (
            1.0
            if retrieval_only
            else (len(answer_covered) / len(query_tokens) if query_tokens else 1.0)
        )
        missing_terms = sorted(query_tokens - evidence_covered)
        missing_answer_terms = sorted(query_tokens - answer_covered)
        evidence_missing_concepts = _missing_query_concepts(
            trace.original_query, evaluation_evidence_text
        )
        answer_missing_concepts = (
            {} if retrieval_only else _missing_query_concepts(trace.original_query, trace.answer)
        )
        if (
            evidence_query_coverage < self.config.min_query_coverage
            or evidence_missing_concepts
        ):
            add(
                "missing_query_components",
                (
                    "Retrieved evidence does not cover enough query concepts."
                    if retrieval_only
                    else "Final synthesis context does not cover enough query concepts."
                ),
                missing_terms=missing_terms,
                missing_concepts=sorted(evidence_missing_concepts),
                missing_concept_values=evidence_missing_concepts,
                evidence_scope=(
                    "retrieved_documents"
                    if retrieval_only
                    else "final_context_spans"
                ),
            )
        if not retrieval_only and (
            answer_query_coverage < self.config.min_query_coverage
            or answer_missing_concepts
        ):
            add(
                "missing_answer_query_components",
                "The answer does not address enough important query concepts.",
                missing_terms=missing_answer_terms,
                missing_concepts=sorted(answer_missing_concepts),
                missing_concept_values=answer_missing_concepts,
            )

        unresolved = (
            []
            if retrieval_only
            else [item for item in trace.citations if not item.resolved]
        )
        if unresolved:
            add(
                "unresolved_citation",
                "One or more citation identifiers do not resolve to final context.",
                citations=[item.citation for item in unresolved],
            )

        supported_claims = 0
        for claim in ([] if retrieval_only else trace.atomic_claims):
            cited_texts = [
                context_by_id[document_id]
                for document_id in claim.cited_document_ids
                if document_id in context_by_id
            ]
            if (
                self.config.require_citations_for_factual_claims
                and _looks_factual(claim.text)
                and not cited_texts
            ):
                add(
                    "uncited_factual_claim",
                    "A factual claim has no resolvable evidence citation.",
                    claim_id=claim.claim_id,
                    claim=claim.text,
                )
                continue

            attribution_status = entity_attribution_status(
                claim.text, cited_texts
            )
            if attribution_status == "mismatch":
                add(
                    "entity_mismatch",
                    "A cited claim attributes evidence to a different entity.",
                    claim_id=claim.claim_id,
                    claim=claim.text,
                )
                continue
            mismatch = self._claim_mismatch(claim.text, cited_texts)
            if mismatch:
                add(
                    mismatch,
                    "A cited claim conflicts with or exceeds its cited evidence.",
                    claim_id=claim.claim_id,
                    claim=claim.text,
                )
                continue
            if attribution_status == "uncertain":
                if self.claim_verifier is None:
                    add(
                        "entity_attribution_verification_required",
                        (
                            "Entity attribution is ambiguous and requires "
                            "cited-context-only semantic verification."
                        ),
                        claim_id=claim.claim_id,
                        claim=claim.text,
                    )
                    continue
                conditional_claim_reasons[
                    claim.claim_id
                ] = "ambiguous_entity_attribution"
            if _HIGH_RISK_RE.search(claim.text):
                if _direct_high_risk_support(claim.text, cited_texts):
                    directly_supported_high_risk_claim_ids.add(claim.claim_id)
                    conditional_claim_reasons.pop(claim.claim_id, None)
                    high_risk_results.append(
                        {
                            "claim_id": claim.claim_id,
                            "decision": "supported_by_direct_match",
                            "evidence_document_ids": list(claim.cited_document_ids),
                            "method": "deterministic_exact_support",
                        }
                    )
                elif self.claim_verifier is None:
                    add(
                        "high_risk_claim_unverified",
                        (
                            "A high-risk claim was not an unambiguous direct match "
                            "to its cited evidence and no conditional verifier was available."
                        ),
                        claim_id=claim.claim_id,
                        claim=claim.text,
                    )
                    high_risk_results.append(
                        {
                            "claim_id": claim.claim_id,
                            "decision": "uncertain",
                            "valid": False,
                            "error": "conditional verifier not configured",
                        }
                    )
                    continue
                else:
                    conditional_claim_reasons.setdefault(
                        claim.claim_id, "high_risk_claim"
                    )
            if cited_texts and claim.claim_id not in conditional_claim_reasons:
                supported_claims += 1
                deterministically_supported_claim_ids.add(claim.claim_id)

        answer_lower = trace.answer.lower().strip()
        if not retrieval_only and (
            not answer_lower
            or any(value in answer_lower for value in _FAILURE_PHRASES)
        ):
            add(
                "answer_failure",
                "The generated answer is empty, a refusal, or an obvious failure.",
            )
        elif not retrieval_only and not trace.atomic_claims:
            add(
                "claim_extraction_empty",
                "The answer contains no verifiable atomic claim.",
            )
        if not retrieval_only and (
            answer_lower.endswith("...")
            or answer_lower.endswith("[truncated]")
        ):
            add("answer_truncation", "The generated answer appears truncated.")

        claim_count = 0 if retrieval_only else len(trace.atomic_claims)
        resolved_count = sum(item.resolved for item in trace.citations)
        components = {
            "retrieval_coverage": min(
                1.0, len(documents) / max(1, self.config.min_documents)
            ),
            "evidence_sufficiency": (
                min(
                    1.0,
                    (
                        evidence_query_coverage
                        + min(1.0, len(trace.final_context_spans) / 3.0)
                    )
                    / 2.0,
                )
                if trace.final_context_spans
                else 0.0
            ),
            "claim_grounding": (
                supported_claims / claim_count
                if claim_count
                else (1.0 if retrieval_only else 0.0)
            ),
            "citation_support": (
                resolved_count / len(trace.citations)
                if trace.citations
                else (1.0 if retrieval_only or not claim_count else 0.0)
            ),
            "query_coverage": (
                evidence_query_coverage
                if retrieval_only
                else min(evidence_query_coverage, answer_query_coverage)
            ),
            "retrieval_query_coverage": retrieval_query_coverage,
            "evidence_query_coverage": evidence_query_coverage,
            "answer_query_coverage": answer_query_coverage,
        }
        return {
            "failed_checks": failed,
            "feedback": feedback,
            "components": components,
            "raw_decision": {
                "deterministic": True,
                "query_missing_terms": missing_terms,
                "answer_missing_terms": missing_answer_terms,
                "coverage_evidence_scope": (
                    "retrieved_documents"
                    if retrieval_only
                    else "final_context_spans"
                ),
                "high_risk_claims": high_risk_results,
            },
            "directly_supported_high_risk_claim_ids": sorted(
                directly_supported_high_risk_claim_ids
            ),
            "conditional_claim_reasons": conditional_claim_reasons,
            "deterministically_supported_claim_ids": sorted(
                deterministically_supported_claim_ids
            ),
        }

    def _claim_mismatch(
        self, claim: str, cited_texts: Sequence[str]
    ) -> Optional[str]:
        if not cited_texts:
            return None
        evidence = " ".join(cited_texts)
        evidence_lower = evidence.lower()

        claim_numbers = set(_NUMBER_RE.findall(claim))
        evidence_numbers = set(_NUMBER_RE.findall(evidence))
        if claim_numbers - evidence_numbers:
            return "unsupported_numeric_assertion"

        claim_camel_entities = set(_INTERNAL_CAMEL_ENTITY_RE.findall(claim))
        evidence_camel_entities = set(_INTERNAL_CAMEL_ENTITY_RE.findall(evidence))
        if (
            claim_camel_entities
            and evidence_camel_entities
            and claim_camel_entities.isdisjoint(evidence_camel_entities)
        ):
            return "entity_mismatch"

        claim_negated = bool(_NEGATION_RE.search(claim))
        evidence_negated = bool(_NEGATION_RE.search(evidence))
        lexical_overlap = _overlap(claim, evidence)
        if claim_negated != evidence_negated and lexical_overlap >= 0.35:
            return "negation_mismatch"
        claim_increases = bool(_INCREASE_RE.search(claim))
        claim_decreases = bool(_DECREASE_RE.search(claim))
        evidence_increases = bool(_INCREASE_RE.search(evidence))
        evidence_decreases = bool(_DECREASE_RE.search(evidence))
        if (claim_increases and evidence_decreases) or (
            claim_decreases and evidence_increases
        ):
            return "directional_mismatch"
        if lexical_overlap < self.config.min_claim_lexical_support:
            return "unsupported_claim"

        return None

    def _conditional_claim_checks(
        self,
        trace: EvaluationTrace,
        failed_checks: List[str],
        feedback: List[Dict[str, Any]],
        *,
        required_claims: Dict[str, str],
    ) -> Dict[str, Any]:
        context_by_id = {
            span.document_id: span.text for span in trace.final_context_spans
        }
        raw_results: List[Dict[str, Any]] = []
        valid = True
        error: Optional[str] = None
        model = ""
        model_revision = ""
        prompt_version = ""
        confidences: List[float] = []
        tokens_in = 0
        tokens_out = 0
        token_total = 0
        cost_usd = 0.0
        latency_sec = 0.0
        supported_claim_ids: List[str] = []
        for claim in trace.atomic_claims:
            verification_reason = required_claims.get(claim.claim_id)
            if not verification_reason:
                continue
            evidence = [
                context_by_id[document_id]
                for document_id in claim.cited_document_ids
                if document_id in context_by_id
            ]
            try:
                raw_result = self.claim_verifier(claim.text, evidence)  # type: ignore[misc]
                result = dict(raw_result) if isinstance(raw_result, dict) else {}
            except Exception as exc:
                result = {
                    "decision": "uncertain",
                    "valid": False,
                    "error": f"claim_verifier_failed:{type(exc).__name__}",
                    "exception_type": type(exc).__name__,
                }
            result.setdefault("claim_id", claim.claim_id)
            result.setdefault("verification_reason", verification_reason)
            raw_results.append(result)
            decision = str(result.get("decision") or "").lower()
            result_valid = result.get("valid") is True
            if decision not in {"supported", "unsupported", "uncertain"}:
                result_valid = False
                result["valid"] = False
                result["error"] = "malformed conditional claim verifier output"
                decision = "uncertain"
                result["decision"] = decision
            confidence = result.get("confidence")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or not 0.0 <= float(confidence) <= 1.0
            ):
                result_valid = False
                result["valid"] = False
                result["error"] = (
                    "malformed conditional claim verifier confidence"
                )
            rationale = result.get("rationale")
            if (
                decision in {"supported", "unsupported"}
                and (
                    not isinstance(rationale, str)
                    or not rationale.strip()
                )
            ):
                result_valid = False
                result["valid"] = False
                result["error"] = (
                    "malformed conditional claim verifier rationale"
                )
            if (
                result_valid
                and decision == "supported"
                and float(confidence)
                < self.config.min_supported_claim_confidence
            ):
                decision = "uncertain"
                result["decision"] = decision
                result["downgrade_reason"] = (
                    "supported_confidence_below_minimum"
                )
            if result.get("model"):
                model = str(result["model"])
            if result.get("model_revision"):
                model_revision = str(result["model_revision"])
            if result.get("prompt_version"):
                prompt_version = str(result["prompt_version"])
            if result_valid:
                confidences.append(float(confidence))
            usage = dict(result.get("token_usage") or {})
            result_tokens_in = int(usage.get("input") or 0)
            result_tokens_out = int(usage.get("output") or 0)
            result_token_total = int(
                usage.get("total")
                if isinstance(usage.get("total"), (int, float))
                else result_tokens_in + result_tokens_out
            )
            tokens_in += result_tokens_in
            tokens_out += result_tokens_out
            token_total += result_token_total
            cost_usd += float(result.get("cost_usd") or 0.0)
            latency_sec += float(result.get("latency_sec") or 0.0)
            if not result_valid:
                valid = False
                error = str(
                    result.get("error") or "conditional claim verifier unavailable"
                )
                unavailable_check = (
                    "high_risk_verifier_unavailable"
                    if verification_reason == "high_risk_claim"
                    else "entity_attribution_verifier_unavailable"
                )
                if unavailable_check not in failed_checks:
                    failed_checks.append(unavailable_check)
                feedback.append(
                    {
                        "check": unavailable_check,
                        "message": (
                            "Conditional verification of an ambiguous claim was "
                            "unavailable or invalid; the claim was not accepted."
                        ),
                        "claim_id": claim.claim_id,
                        "error": error,
                    }
                )
                continue
            if decision == "unsupported":
                if "conditional_claim_unsupported" not in failed_checks:
                    failed_checks.append("conditional_claim_unsupported")
                feedback.append(
                    {
                        "check": "conditional_claim_unsupported",
                        "message": "Independent verification did not support a high-risk claim.",
                        "claim_id": claim.claim_id,
                    }
                )
            elif decision == "uncertain":
                if "conditional_claim_uncertain" not in failed_checks:
                    failed_checks.append("conditional_claim_uncertain")
                feedback.append(
                    {
                        "check": "conditional_claim_uncertain",
                        "message": (
                            "Conditional verification could not establish support "
                            "for a high-risk claim."
                        ),
                        "claim_id": claim.claim_id,
                    }
                )
            else:
                supported_claim_ids.append(claim.claim_id)
        return {
            "results": raw_results,
            "valid": valid,
            "error": error,
            "model": model,
            "model_revision": model_revision,
            "prompt_version": prompt_version,
            "confidences": confidences,
            "token_usage": {
                "input": tokens_in,
                "output": tokens_out,
                "total": token_total,
            },
            "cost_usd": cost_usd,
            "latency_sec": latency_sec,
            "supported_claim_ids": supported_claim_ids,
        }

    def _recommended_changes(
        self,
        trace: EvaluationTrace,
        *,
        failed_checks: Sequence[str],
        retrieval_failure: bool,
    ) -> Dict[str, Any]:
        if not failed_checks:
            return {}
        if retrieval_failure:
            current_top_k = int(
                trace.retrieval_configuration.get("top_k")
                or self.config.min_documents
            )
            missing = _tokens(trace.original_query) - _tokens(
                " ".join(item.text for item in trace.final_context_spans)
            )
            return {
                "top_k": min(
                    self.config.max_top_k,
                    current_top_k + self.config.top_k_increment,
                ),
                "query_additions": sorted(missing),
                "retrieval_method": "hybrid_with_sparse_fallback",
                "preserve_original_query": True,
            }
        return {
            "preserve_evidence": True,
            "require_resolvable_citations": True,
            "remove_or_qualify_unsupported_assertions": True,
        }


def _tokens(text: str) -> Set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if token.lower() not in _STOPWORDS and len(token) > 1
    }


def _apply_verifier_telemetry(
    trace: EvaluationTrace, semantic: Dict[str, Any]
) -> None:
    """Attach conditional-judge usage to its attempt without hiding prior work."""
    usage = dict(semantic.get("token_usage") or {})
    verifier_in = int(usage.get("input") or 0)
    verifier_out = int(usage.get("output") or 0)
    verifier_total = int(
        usage.get("total")
        if isinstance(usage.get("total"), (int, float))
        else verifier_in + verifier_out
    )
    current = dict(trace.token_usage or {})
    current_in = int(current.get("input") or 0)
    current_out = int(current.get("output") or 0)
    current_total = int(
        current.get("total")
        if isinstance(current.get("total"), (int, float))
        else current_in + current_out
    )
    trace.token_usage = {
        **current,
        "input": current_in + verifier_in,
        "output": current_out + verifier_out,
        "total": current_total + verifier_total,
    }

    verifier_cost = float(semantic.get("cost_usd") or 0.0)
    trace.cost_breakdown_usd = dict(trace.cost_breakdown_usd or {})
    trace.cost_breakdown_usd["verification"] = (
        float(trace.cost_breakdown_usd.get("verification") or 0.0)
        + verifier_cost
    )

    verifier_latency = float(semantic.get("latency_sec") or 0.0)
    trace.stage_latency_sec = dict(trace.stage_latency_sec or {})
    trace.stage_latency_sec["verification"] = (
        float(trace.stage_latency_sec.get("verification") or 0.0)
        + verifier_latency
    )
    trace.stage_latency_sec["total"] = (
        float(trace.stage_latency_sec.get("total") or 0.0)
        + verifier_latency
    )

    from runtime_verification.telemetry import build_attempt_event

    for index, result in enumerate(semantic.get("results") or [], 1):
        result_usage = dict(result.get("token_usage") or {})
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
                if bool(result.get("valid", False))
                else "deadline_exhausted"
                if deadline_exhausted
                else "error"
            ),
            repair_status="verification",
            model=str(result.get("model") or ""),
            model_revision=str(result.get("model_revision") or ""),
            prompt_version=str(result.get("prompt_version") or ""),
            tokens_in=int(result_usage.get("input") or 0),
            tokens_out=int(result_usage.get("output") or 0),
            cost_usd=float(result.get("cost_usd") or 0.0),
            latency_sec=float(result.get("latency_sec") or 0.0),
            finish_reason=str(result.get("finish_reason") or ""),
            deadline_exhausted=deadline_exhausted,
            error_type=error_type,
            provider_metadata=result.get("provider_metadata"),
        )
        if not any(
            item.get("event_id") == event["event_id"]
            for item in trace.attempt_events
        ):
            trace.attempt_events.append(event)


def _overlap(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens:
        return 1.0
    return len(left_tokens.intersection(right_tokens)) / len(left_tokens)


def _looks_factual(text: str) -> bool:
    words = _tokens(text)
    return len(words) >= 3 and not text.strip().endswith("?")


def _missing_query_concepts(query: str, text: str) -> Dict[str, List[str]]:
    """Return query concept categories absent from the supplied text."""
    normalized_query = normalize_biomedical_text(query)
    normalized_text = normalize_biomedical_text(text)
    missing: Dict[str, List[str]] = {}
    for category, patterns in _CONCEPT_PATTERNS.items():
        query_values = [
            match.group(0)
            for pattern in patterns
            for match in re.finditer(pattern, normalized_query, re.I)
        ]
        absent = [
            value
            for value in query_values
            if normalize_biomedical_text(value) not in normalized_text
        ]
        if absent:
            missing[category] = sorted(set(absent))

    query_entities = extract_salient_entities(query)
    text_entities = extract_salient_entities(text)
    for category in ("drug", "intervention", "disease"):
        expected = query_entities.get(category, set())
        if expected and expected.isdisjoint(text_entities.get(category, set())):
            missing.setdefault("intervention", []).extend(sorted(expected))
    return missing


def _direct_high_risk_support(claim: str, cited_texts: Sequence[str]) -> bool:
    """Accept only unambiguous near-verbatim support when no judge is needed."""
    if not cited_texts:
        return False
    evidence = " ".join(cited_texts)
    if _overlap(claim, evidence) < 0.9:
        return False
    claim_numbers = set(_NUMBER_RE.findall(claim))
    evidence_numbers = set(_NUMBER_RE.findall(evidence))
    if claim_numbers - evidence_numbers:
        return False
    if entity_attribution_status(claim, cited_texts) in {"mismatch", "uncertain"}:
        return False
    if bool(_NEGATION_RE.search(claim)) != bool(_NEGATION_RE.search(evidence)):
        return False
    claim_direction = (
        bool(_INCREASE_RE.search(claim)),
        bool(_DECREASE_RE.search(claim)),
    )
    evidence_direction = (
        bool(_INCREASE_RE.search(evidence)),
        bool(_DECREASE_RE.search(evidence)),
    )
    return claim_direction == evidence_direction
