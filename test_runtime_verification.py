import pytest
import json

from evaluation_core import (
    build_agent_evaluation_trace,
    content_hash,
    stable_document_id,
)
from llm_client import LLMCallResult
from runtime_verification import (
    ConditionalClaimVerifier,
    RuntimeVerifier,
    VerifierConfig,
    build_runtime_verifier,
    calculate_combined_confidence,
    record_conditional_verifier_telemetry,
)


def _trace(
    *,
    query="Do GLP-1 drugs reduce cardiovascular events in adults?",
    evidence="GLP-1 therapy reduced cardiovascular events by 20% in adults.",
    answer="GLP-1 therapy reduced cardiovascular events by 20% in adults [1].",
    citations=None,
    top_k=5,
):
    documents = []
    if evidence is not None:
        documents = [
            {
                "doc_id": "doc-1",
                "text": evidence,
                "score": 0.9,
                "original_rank": 1,
                "metadata": {
                    "title": "Outcome trial",
                    "source": "PubMed",
                    "source_type": "peer-reviewed publication",
                    "authority": "journal",
                    "publication_type": "randomized trial",
                    "publication_date": "2024-01-01",
                    "pmid": "123",
                },
            }
        ]
    state = {
        "expanded_query": query,
        "retrieval_results": documents,
        "reranked_results": documents,
        "answer": answer,
        "citations": citations if citations is not None else ["Outcome trial"],
        "model_used": "test-model@abc123",
        "retrieval_time_sec": 0.2,
        "execution_time_sec": 0.4,
        "error": None,
    }
    if documents:
        state["synthesis_context"] = [
            {
                "document_id": stable_document_id(
                    documents[0], "pubmed", 1
                ),
                "text": evidence,
                "start_char": 0,
                "original_length": len(evidence),
                "truncated": False,
                "citation_marker": 1,
            }
        ]
    else:
        state["synthesis_context"] = []
        state["answer_origin"] = (
            "evidence_limited" if str(answer or "").strip() else "retrieval_only"
        )
        state["retrieval_only"] = not bool(str(answer or "").strip())
    context = {
        "trace_id": "trace-1",
        "attempt_id": "attempt-1",
        "top_k": top_k,
    }
    if state.get("retrieval_only"):
        context["retrieval_only"] = True
    return build_agent_evaluation_trace(
        agent_name="search_pubmed",
        domain="pubmed",
        original_query=query,
        state=state,
        context=context,
    )


def test_strong_output_is_accepted_without_retry():
    decision = RuntimeVerifier().verify(_trace(), retries_remaining=1)

    assert decision.status == "accept"
    assert decision.target_stage == "none"
    assert decision.failed_checks == []
    assert decision.valid is True
    assert set(decision.component_scores) >= {
        "retrieval_coverage",
        "evidence_sufficiency",
        "claim_grounding",
        "citation_support",
        "query_coverage",
        "verifier_confidence",
    }


def test_weak_retrieval_generates_actionable_feedback():
    trace = _trace(
        query=(
            "In adults with diabetes, what randomized trials since 2020 "
            "report cardiovascular mortality?"
        ),
        evidence=None,
        answer="No evidence was found.",
        citations=[],
        top_k=5,
    )

    decision = RuntimeVerifier().verify(trace, retries_remaining=1)

    assert decision.status == "retry_retrieval"
    assert decision.target_stage == "retrieval"
    assert decision.target_agent == "search_pubmed"
    assert "empty_retrieval" in decision.failed_checks
    assert decision.structured_feedback
    assert decision.recommended_retry_changes["top_k"] > 5
    assert decision.recommended_retry_changes["query_additions"]


@pytest.mark.parametrize(
    ("evidence", "answer", "expected_check"),
    [
        (
            "The intervention reduced events by 20% in adults.",
            "The intervention reduced events by 50% in adults [1].",
            "unsupported_numeric_assertion",
        ),
        (
            "Treatment increased mortality in the study population.",
            "Treatment did not increase mortality in the study population [1].",
            "negation_mismatch",
        ),
        (
            "DrugBeta reduced admissions in the trial.",
            "DrugAlpha reduced admissions in the trial [1].",
            "entity_mismatch",
        ),
    ],
)
def test_similarity_does_not_override_deterministic_claim_mismatches(
    evidence, answer, expected_check
):
    verifier = RuntimeVerifier(similarity_scorer=lambda _query, _texts: 0.99)

    decision = verifier.verify(
        _trace(evidence=evidence, answer=answer), retries_remaining=1
    )

    assert decision.component_scores["evidence_similarity"] == 0.99
    assert expected_check in decision.failed_checks
    assert decision.status != "accept"


def test_verifier_failure_never_silently_passes(monkeypatch):
    verifier = RuntimeVerifier()

    def fail(_trace):
        raise RuntimeError("verifier unavailable")

    monkeypatch.setattr(verifier, "_deterministic_checks", fail)
    decision = verifier.verify(_trace(), retries_remaining=1)

    assert decision.status == "evidence_limited"
    assert decision.valid is False
    assert decision.error == "verifier_failed:RuntimeError"
    assert decision.verifier_confidence == 0.0


def test_empty_retrieval_after_retry_exhaustion_is_evidence_limited():
    decision = RuntimeVerifier().verify(
        _trace(evidence=None, answer="", citations=[]),
        retries_remaining=0,
    )

    assert decision.status == "evidence_limited"
    assert decision.target_stage == "none"
    assert decision.valid is True


@pytest.mark.parametrize(
    ("evidence", "answer", "expected_check"),
    [
        (
            "Aspirin reduced mortality in the observed cohort.",
            "Aspirin increased mortality in the observed cohort [1].",
            "directional_mismatch",
        ),
        (
            "Metformin lowered glucose in adults with diabetes.",
            "Yoga improved sleep quality in adults [1].",
            "unsupported_claim",
        ),
    ],
)
def test_citation_presence_alone_does_not_prove_claim_support(
    evidence, answer, expected_check
):
    decision = RuntimeVerifier().verify(
        _trace(evidence=evidence, answer=answer), retries_remaining=0
    )

    assert decision.status == "evidence_limited"
    assert expected_check in decision.failed_checks
    assert decision.component_scores["claim_grounding"] == 0.0


def test_invalid_trace_schema_cannot_be_accepted():
    trace = _trace()
    trace.schema_version = "999"

    decision = RuntimeVerifier().verify(trace, retries_remaining=1)

    assert decision.status == "evidence_limited"
    assert decision.valid is False
    assert "invalid_evaluation_trace" in decision.failed_checks


def test_short_unverifiable_answer_is_not_silently_accepted():
    decision = RuntimeVerifier().verify(
        _trace(answer="Effective."), retries_remaining=0
    )

    assert decision.status == "evidence_limited"
    assert "claim_extraction_empty" in decision.failed_checks


def test_directly_supported_high_risk_claim_does_not_require_optional_judge():
    trace = _trace(
        query="Does aspirin reduce mortality in adults?",
        evidence="Aspirin reduced mortality in adults.",
        answer="Aspirin reduced mortality in adults [1].",
    )

    decision = RuntimeVerifier().verify(trace, retries_remaining=0)

    assert decision.status == "accept"
    assert "high_risk_claim_unverified" not in decision.failed_checks
    assert decision.raw_decision["high_risk_claims"][0]["decision"] == (
        "supported_by_direct_match"
    )


def test_conditional_high_risk_verifier_uses_only_cited_final_context():
    calls = []

    def claim_verifier(claim, evidence):
        calls.append((claim, list(evidence)))
        return {
            "decision": "supported",
            "confidence": 0.91,
            "rationale": "The cited evidence supports the claim.",
            "valid": True,
            "model": "judge-model",
            "model_revision": "judge-revision",
            "prompt_version": "high-risk-claim-v2",
            "raw_decision": {"label": "entailed"},
            "token_usage": {"input": 4, "output": 3, "total": 7},
            "cost_usd": 0.002,
            "latency_sec": 0.04,
        }

    trace = _trace(
        query="What dose lowers mortality in adults?",
        evidence="A 5 mg dose was associated with lower mortality in adults.",
        answer="A 5 mg dose lowered mortality in adults [1].",
    )
    verifier = RuntimeVerifier(claim_verifier=claim_verifier)

    decision = verifier.verify(trace, retries_remaining=1)

    assert decision.status == "accept"
    assert calls == [
        (
                "A 5 mg dose lowered mortality in adults.",
            ["A 5 mg dose was associated with lower mortality in adults."],
        )
    ]
    assert decision.verifier_model == "judge-model"
    assert decision.verifier_model_revision == "judge-revision"
    assert decision.prompt_version == "high-risk-claim-v2"
    assert trace.token_usage == {"input": 4, "output": 3, "total": 7}
    assert trace.cost_breakdown_usd["verification"] == pytest.approx(0.002)
    assert trace.stage_latency_sec["verification"] == pytest.approx(0.04)


@pytest.mark.parametrize(
    "result",
    [
        {
            "decision": "uncertain",
            "valid": False,
            "error": "claim verifier timeout",
            "model": "judge-model",
            "model_revision": "r1",
            "prompt_version": "high-risk-claim-v2",
        },
        {"unexpected": "malformed"},
    ],
)
def test_unavailable_or_malformed_high_risk_verifier_fails_conservatively(result):
    verifier = RuntimeVerifier(
        claim_verifier=lambda _claim, _evidence: dict(result)
    )
    trace = _trace(
        query="Does aspirin increase pregnancy risk?",
        evidence="Evidence about aspirin exposure in pregnancy was inconclusive.",
        answer="Aspirin increases pregnancy risk [1].",
    )

    decision = verifier.verify(trace, retries_remaining=1)

    assert decision.status == "evidence_limited"
    assert decision.valid is False
    assert "high_risk_verifier_unavailable" in decision.failed_checks
    assert decision.verifier_confidence == 0.0


@pytest.mark.parametrize(
    ("evidence", "answer"),
    [
        (
            "Ibuprofen reduced platelet aggregation in adults.",
            "Aspirin reduced platelet aggregation in adults [1].",
        ),
        (
            "Insulin lowered glycated hemoglobin in adults with diabetes.",
            "Metformin lowered glycated hemoglobin in adults with diabetes [1].",
        ),
        (
            "The intervention prevented stroke in high-risk adults.",
            "The intervention prevented myocardial infarction in high-risk adults [1].",
        ),
        (
            "The EMA authorized the medicine after review.",
            "The FDA authorized the medicine after review [1].",
        ),
        (
            "Liraglutide reduced body weight in adults with obesity.",
            "Semaglutide reduced body weight in adults with obesity [1].",
        ),
        (
            "The intervention reduced symptoms in children.",
            "The intervention reduced symptoms in adults [1].",
        ),
        (
            "Apixaban reduced embolic events in adults.",
            "Warfarin reduced embolic events in adults [1].",
        ),
    ],
)
def test_realistic_biomedical_entity_substitutions_are_rejected(evidence, answer):
    decision = RuntimeVerifier().verify(
        _trace(evidence=evidence, answer=answer), retries_remaining=0
    )

    assert decision.status == "evidence_limited"
    assert "entity_mismatch" in decision.failed_checks


def test_every_claimed_salient_entity_must_appear_in_cited_evidence():
    decision = RuntimeVerifier().verify(
        _trace(
            evidence="Aspirin reduced platelet aggregation in adults.",
            answer=(
                "Aspirin and ibuprofen reduced platelet aggregation "
                "in adults [1]."
            ),
        ),
        retries_remaining=0,
    )

    assert decision.status == "evidence_limited"
    assert "entity_mismatch" in decision.failed_checks


@pytest.mark.parametrize(
    ("evidence", "answer"),
    [
        (
            "Acetylsalicylic acid reduced platelet aggregation in adults.",
            "Aspirin reduced platelet aggregation in adults [1].",
        ),
        (
            "A heart attack was prevented in high-risk adults.",
            "Myocardial infarction was prevented in high-risk adults [1].",
        ),
        (
            "The US Food and Drug Administration authorized the medicine.",
            "The FDA authorized the medicine [1].",
        ),
        (
            "Aspirin reduced platelet aggregation in adults.",
            "ASA reduced platelet aggregation in adults [1].",
        ),
        (
            "Myocardial infarction was prevented in high-risk adults.",
            "MI was prevented in high-risk adults [1].",
        ),
        (
            "The intervention improved outcomes in elderly participants.",
            "The intervention improved outcomes in older adults [1].",
        ),
        (
            "Warfarin reduced embolic events in adults.",
            "Coumadin reduced embolic events in adults [1].",
        ),
    ],
)
def test_supported_biomedical_aliases_do_not_create_entity_mismatches(
    evidence, answer
):
    decision = RuntimeVerifier().verify(
        _trace(evidence=evidence, answer=answer), retries_remaining=0
    )

    assert "entity_mismatch" not in decision.failed_checks
    assert "entity_attribution_uncertain" not in decision.failed_checks


def test_unresolved_salient_entity_attribution_is_conservative():
    decision = RuntimeVerifier().verify(
        _trace(
            evidence="The intervention reduced platelet aggregation in adults.",
            answer="Aspirin reduced platelet aggregation in adults [1].",
        ),
        retries_remaining=0,
    )

    assert decision.status == "evidence_limited"
    assert (
        "entity_attribution_verification_required"
        in decision.failed_checks
    )


def test_unknown_lowercase_drug_substitution_requires_cited_semantic_verification():
    calls = []

    def judge(claim, evidence):
        calls.append((claim, list(evidence)))
        return {
            "decision": "unsupported",
            "confidence": 0.95,
            "rationale": "The cited evidence names givinostat, not vamorolone.",
            "valid": True,
            "model": "mock-judge",
            "model_revision": "r1",
            "prompt_version": "mock-v1",
        }

    trace = _trace(
        query="Did vamorolone improve motor function in boys?",
        evidence="Givinostat improved motor function in boys in the cited trial.",
        answer="Vamorolone improved motor function in boys in the cited trial [1].",
    )
    decision = RuntimeVerifier(claim_verifier=judge).verify(
        trace, retries_remaining=0
    )

    assert decision.status == "evidence_limited"
    assert "conditional_claim_unsupported" in decision.failed_checks
    assert calls == [
        (
            "Vamorolone improved motor function in boys in the cited trial.",
            ["Givinostat improved motor function in boys in the cited trial."],
        )
    ]
    assert (
        decision.raw_decision["conditional_claim_verification"][0][
            "verification_reason"
        ]
        == "ambiguous_entity_attribution"
    )


def test_unknown_entity_ambiguity_fails_closed_without_claim_verifier():
    trace = _trace(
        query="Did ozanimod reduce relapses in adults?",
        evidence="Etrasimod reduced relapses in adults in the cited study.",
        answer="Ozanimod reduced relapses in adults in the cited study [1].",
    )

    decision = RuntimeVerifier().verify(trace, retries_remaining=0)

    assert decision.status == "evidence_limited"
    assert (
        "entity_attribution_verification_required"
        in decision.failed_checks
    )


def test_unknown_entity_exact_overlap_still_requires_semantic_judge():
    calls = []

    def claim_verifier(claim, evidence):
        calls.append((claim, list(evidence)))
        return {
            "decision": "supported",
            "confidence": 0.91,
            "valid": True,
            "model": "judge-model",
            "model_revision": "r1",
            "prompt_version": "high-risk-claim-v2",
            "rationale": "The cited sentence attributes the outcome to ozanimod.",
        }

    trace = _trace(
        query="Did ozanimod reduce relapses in adults?",
        evidence="Ozanimod reduced relapses in adults in the cited study.",
        answer="Ozanimod reduced relapses in adults in the cited study [1].",
    )

    decision = RuntimeVerifier(claim_verifier=claim_verifier).verify(
        trace, retries_remaining=0
    )

    assert decision.status == "accept"
    assert calls == [
        (
            "Ozanimod reduced relapses in adults in the cited study.",
            ["Ozanimod reduced relapses in adults in the cited study."],
        )
    ]


@pytest.mark.parametrize(
    ("entity", "evidence"),
    [
        (
            "edoxaban",
            "Edoxaban improved outcomes in the cited trial.",
        ),
        (
            "AZD1234",
            "AZD1234 improved outcomes in the cited trial.",
        ),
        (
            "JNJ-1234",
            "JNJ-1234 improved outcomes in the cited trial.",
        ),
        (
            "mk-3475",
            "mk-3475 improved outcomes in the cited trial.",
        ),
    ],
)
def test_inferred_drugs_and_biomedical_codes_require_semantic_verification(
    entity, evidence
):
    calls = []

    def claim_verifier(claim, cited_evidence):
        calls.append((claim, list(cited_evidence)))
        return {
            "decision": "supported",
            "confidence": 0.92,
            "rationale": "The cited sentence directly attributes the outcome.",
            "valid": True,
            "model": "judge-model",
            "prompt_version": "high-risk-claim-v2",
        }

    trace = _trace(
        query=f"Did {entity} improve outcomes?",
        evidence=evidence,
        answer=f"{entity} improved outcomes in the cited trial [1].",
    )
    decision = RuntimeVerifier(claim_verifier=claim_verifier).verify(
        trace, retries_remaining=0
    )

    assert decision.status == "accept"
    assert len(calls) == 1
    assert (
        decision.raw_decision["conditional_claim_verification"][0][
            "verification_reason"
        ]
        == "ambiguous_entity_attribution"
    )


def test_unknown_entity_lexical_presence_cannot_prove_attribution():
    calls = []

    def claim_verifier(claim, evidence):
        calls.append((claim, list(evidence)))
        return {
            "decision": "unsupported",
            "confidence": 0.96,
            "valid": True,
            "model": "judge-model",
            "model_revision": "r1",
            "prompt_version": "high-risk-claim-v2",
            "rationale": "The outcome is attributed to givinostat, not vamorolone.",
        }

    trace = _trace(
        query="Did vamorolone improve motor function?",
        evidence=(
            "Vamorolone was the comparator; givinostat improved motor "
            "function in the cited trial."
        ),
        answer="Vamorolone improved motor function in the cited trial [1].",
    )

    decision = RuntimeVerifier(claim_verifier=claim_verifier).verify(
        trace, retries_remaining=0
    )

    assert decision.status == "evidence_limited"
    assert "conditional_claim_unsupported" in decision.failed_checks
    assert len(calls) == 1


def test_excluded_document_cannot_inflate_final_context_coverage():
    query = (
        "In pregnant adults, what randomized trials since 2024 report "
        "aspirin outcomes?"
    )
    included = {
        "document_id": "included",
        "text": "Aspirin outcomes were reported in adults.",
        "score": 0.9,
        "original_rank": 1,
    }
    excluded = {
        "document_id": "excluded",
        "text": "A randomized trial in pregnant adults was published in 2024.",
        "score": 0.8,
        "original_rank": 2,
    }
    trace = build_agent_evaluation_trace(
        agent_name="search_pubmed",
        domain="pubmed",
        original_query=query,
        state={
            "expanded_query": query,
            "retrieval_results": [included, excluded],
            "reranked_results": [included, excluded],
            "synthesis_context": [
                {
                    "document_id": "included",
                    "text": included["text"],
                    "start_char": 0,
                    "original_length": len(included["text"]),
                    "citation_marker": 1,
                }
            ],
            "answer": "Aspirin outcomes were reported in adults [1].",
            "citations": [],
            "model_used": "test-model",
            "error": None,
        },
        context={"trace_id": "coverage", "attempt_id": "coverage:1", "top_k": 2},
    )

    decision = RuntimeVerifier().verify(trace, retries_remaining=0)

    assert decision.component_scores["retrieval_query_coverage"] > (
        decision.component_scores["evidence_query_coverage"]
    )
    assert "missing_query_components" in decision.failed_checks
    feedback = next(
        item
        for item in decision.structured_feedback
        if item["check"] == "missing_query_components"
    )
    assert set(feedback["missing_concepts"]) >= {"population", "timeframe", "study_design"}


def test_truncated_retrieved_text_outside_manifest_cannot_support_claim():
    full_evidence = (
        "The intervention reduced symptoms. "
        "The excluded tail reported severe liver injury."
    )
    trace = _trace(
        query="What did the intervention do?",
        evidence=full_evidence,
        answer="The intervention caused severe liver injury [1].",
    )
    included_text = "The intervention reduced symptoms."
    span = trace.final_context_spans[0]
    span.text = included_text
    span.end_char = len(included_text)
    span.content_hash = content_hash(included_text)
    span.truncated = True

    decision = RuntimeVerifier().verify(trace, retries_remaining=0)

    assert decision.status == "evidence_limited"
    assert "unsupported_claim" in decision.failed_checks


def test_answer_coverage_is_separate_from_final_evidence_coverage():
    query = "What randomized trials since 2024 studied aspirin in pregnant adults?"
    evidence = (
        "A 2024 randomized trial studied aspirin in pregnant adults and reported "
        "maternal outcomes."
    )
    decision = RuntimeVerifier().verify(
        _trace(
            query=query,
            evidence=evidence,
            answer="Aspirin was studied in adults [1].",
        ),
        retries_remaining=0,
    )

    assert decision.component_scores["evidence_query_coverage"] > (
        decision.component_scores["answer_query_coverage"]
    )
    assert "missing_answer_query_components" in decision.failed_checks


def test_retrieval_only_trace_does_not_require_answer_coverage():
    trace = _trace(answer="", citations=[])
    trace.retrieval_configuration["retrieval_only"] = True
    trace.atomic_claims = []
    trace.citations = []

    decision = RuntimeVerifier().verify(trace, retries_remaining=0)

    assert "missing_answer_query_components" not in decision.failed_checks
    assert decision.component_scores["answer_query_coverage"] == 1.0


def test_structured_claim_verifier_enforces_deadline_and_cited_evidence_only():
    captured = {}

    class _JudgeLLM:
        default_model = "judge-default"

        def chat_with_metadata(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return LLMCallResult(
                text=(
                    '{"decision":"supported","confidence":0.88,'
                    '"rationale":"The cited sentence directly supports the claim."}'
                ),
                model="provider-judge",
                model_revision="provider-r7",
                tokens_in=9,
                tokens_out=4,
                cost_usd=0.003,
                latency_sec=0.02,
                finish_reason="stop",
                provider_metadata={"provider": "mock"},
            )

    verifier = ConditionalClaimVerifier(
        llm_client=_JudgeLLM(),
        model="judge-model@revision-2",
        timeout_sec=3.0,
    )
    result = verifier(
        "Aspirin reduced mortality.",
        ["Aspirin reduced mortality in the cited cohort."],
    )

    prompt = "\n".join(item["content"] for item in captured["messages"])
    assert "Aspirin reduced mortality in the cited cohort." in prompt
    assert "benchmark" not in prompt.lower()
    assert captured["kwargs"]["client_max_attempts"] == 1
    assert 0 < captured["kwargs"]["timeout"] <= 3.0
    assert result["decision"] == "supported"
    assert result["valid"] is True
    assert result["model"] == "provider-judge"
    assert result["model_revision"] == "provider-r7"
    assert result["token_usage"] == {"input": 9, "output": 4, "total": 13}
    assert result["cost_usd"] == pytest.approx(0.003)
    assert result["prompt_version"]


def test_structured_claim_verifier_returns_uncertain_on_timeout():
    class _TimeoutLLM:
        default_model = "judge-model"

        def chat(self, *_args, **_kwargs):
            raise TimeoutError("provider deadline exceeded")

    result = ConditionalClaimVerifier(
        llm_client=_TimeoutLLM(), timeout_sec=0.5
    )("Aspirin reduced mortality.", ["Cited evidence."])

    assert result["decision"] == "uncertain"
    assert result["valid"] is False
    assert result["error_type"] == "TimeoutError"


@pytest.mark.parametrize(
    "confidence",
    [True, False, None, "0.9", float("nan"), float("inf"), -0.1, 1.1],
)
def test_claim_verifier_rejects_malformed_confidence(confidence):
    class _JudgeLLM:
        default_model = "judge-model"

        def chat(self, *_args, **_kwargs):
            return json.dumps(
                {
                    "decision": "supported",
                    "confidence": confidence,
                    "rationale": "Cited evidence directly supports the claim.",
                }
            )

    result = ConditionalClaimVerifier(llm_client=_JudgeLLM())(
        "Aspirin reduced mortality.",
        ["Aspirin reduced mortality in the cited cohort."],
    )

    assert result["decision"] == "uncertain"
    assert result["valid"] is False
    assert result["confidence"] == 0.0
    assert result["error_type"] == "ValueError"
    assert "raw_decision" in result


def test_supported_claim_below_minimum_confidence_becomes_uncertain():
    class _JudgeLLM:
        default_model = "judge-model"

        def chat(self, *_args, **_kwargs):
            return json.dumps(
                {
                    "decision": "supported",
                    "confidence": 0.0,
                    "rationale": "Weak support.",
                }
            )

    result = ConditionalClaimVerifier(
        llm_client=_JudgeLLM(), minimum_supported_confidence=0.75
    )(
        "Aspirin reduced mortality.",
        ["Aspirin reduced mortality in the cited cohort."],
    )

    assert result["decision"] == "uncertain"
    assert result["valid"] is True
    assert result["confidence"] == 0.0
    assert result["downgrade_reason"] == "supported_confidence_below_minimum"


@pytest.mark.parametrize("decision", ["supported", "unsupported"])
def test_claim_verifier_requires_rationale_for_definitive_decisions(decision):
    class _JudgeLLM:
        default_model = "judge-model"

        def chat(self, *_args, **_kwargs):
            return json.dumps(
                {"decision": decision, "confidence": 0.9, "rationale": ""}
            )

    result = ConditionalClaimVerifier(llm_client=_JudgeLLM())(
        "Aspirin reduced mortality.",
        ["Aspirin reduced mortality in the cited cohort."],
    )

    assert result["decision"] == "uncertain"
    assert result["valid"] is False
    assert result["error_type"] == "ValueError"


@pytest.mark.parametrize("rationale", [True, 1, [], {}])
def test_claim_verifier_rejects_non_string_rationale(rationale):
    class _JudgeLLM:
        default_model = "judge-model"

        def chat(self, *_args, **_kwargs):
            return json.dumps(
                {
                    "decision": "supported",
                    "confidence": 0.9,
                    "rationale": rationale,
                }
            )

    result = ConditionalClaimVerifier(llm_client=_JudgeLLM())(
        "Aspirin reduced mortality.",
        ["Aspirin reduced mortality in the cited cohort."],
    )

    assert result["decision"] == "uncertain"
    assert result["valid"] is False
    assert result["error_type"] == "ValueError"


@pytest.mark.parametrize(
    "result",
    [
        {
            "decision": "supported",
            "confidence": 0.95,
            "rationale": "Looks supported.",
            "valid": "false",
        },
        {
            "decision": "supported",
            "confidence": 0.2,
            "rationale": "Weak support.",
            "valid": True,
        },
        {
            "decision": "supported",
            "confidence": 0.95,
            "valid": True,
        },
    ],
)
def test_runtime_verifier_revalidates_injected_claim_verifier_output(result):
    trace = _trace(
        query="What dose lowers mortality in adults?",
        evidence="A 5 mg dose was associated with lower mortality in adults.",
        answer="A 5 mg dose lowered mortality in adults [1].",
    )

    decision = RuntimeVerifier(
        claim_verifier=lambda _claim, _evidence: dict(result)
    ).verify(trace, retries_remaining=0)

    assert decision.status == "evidence_limited"
    assert not decision.valid or "conditional_claim_uncertain" in (
        decision.failed_checks
    )


def test_claim_verifier_accepts_valid_supported_response():
    class _JudgeLLM:
        default_model = "judge-model"

        def chat(self, *_args, **_kwargs):
            return json.dumps(
                {
                    "decision": "supported",
                    "confidence": 0.81,
                    "rationale": "The cited evidence directly supports the claim.",
                }
            )

    result = ConditionalClaimVerifier(
        llm_client=_JudgeLLM(), minimum_supported_confidence=0.75
    )(
        "Aspirin reduced mortality.",
        ["Aspirin reduced mortality in the cited cohort."],
    )

    assert result["decision"] == "supported"
    assert result["valid"] is True
    assert result["confidence"] == pytest.approx(0.81)


def test_conditional_verifier_telemetry_updates_request_aggregates():
    trace = _trace(
        query="What dose lowers mortality in adults?",
        evidence="A 5 mg dose was associated with lower mortality in adults.",
        answer="A 5 mg dose lowered mortality in adults [1].",
    )
    decision = RuntimeVerifier(
        claim_verifier=lambda _claim, _evidence: {
            "decision": "supported",
            "confidence": 0.9,
            "rationale": "The cited evidence supports the claim.",
            "valid": True,
            "model": "judge-model",
            "model_revision": "r3",
            "prompt_version": "high-risk-claim-v2",
            "token_usage": {"input": 4, "output": 3, "total": 7},
            "cost_usd": 0.002,
            "latency_sec": 0.04,
        }
    ).verify(trace)
    state = {
        "token_usage": {"input": 6, "output": 4, "total": 10},
        "cost_estimate": 0.1,
        "attempt_telemetry": [],
    }

    record_conditional_verifier_telemetry(state, trace, decision)
    record_conditional_verifier_telemetry(state, trace, decision)

    assert state["token_usage"] == {"input": 10, "output": 7, "total": 17}
    assert state["cost_estimate"] == pytest.approx(0.102)
    assert len(state["attempt_telemetry"]) == 1
    assert state["attempt_telemetry"][0]["model"] == "judge-model"
    assert state["attempt_telemetry"][0]["parent_attempt_id"] == trace.attempt_id


def test_invalid_string_valid_flag_records_error_telemetry():
    trace = _trace(
        query="What dose lowers mortality in adults?",
        evidence="A 5 mg dose was associated with lower mortality in adults.",
        answer="A 5 mg dose lowered mortality in adults [1].",
    )
    decision = RuntimeVerifier(
        claim_verifier=lambda _claim, _evidence: {
            "decision": "supported",
            "confidence": 0.9,
            "rationale": "Looks supported.",
            "valid": "false",
        }
    ).verify(trace, retries_remaining=0)
    state = {"attempt_telemetry": []}

    record_conditional_verifier_telemetry(state, trace, decision)

    assert state["attempt_telemetry"][0]["status"] == "error"


def test_unsupported_claim_below_minimum_confidence_is_invalid():
    class _JudgeLLM:
        default_model = "judge-model"

        def chat(self, *_args, **_kwargs):
            return json.dumps(
                {
                    "decision": "unsupported",
                    "confidence": 0.2,
                    "rationale": "Weak contradiction.",
                }
            )

    result = ConditionalClaimVerifier(
        llm_client=_JudgeLLM(), minimum_supported_confidence=0.75
    )(
        "Aspirin increased mortality.",
        ["Aspirin reduced mortality in the cited cohort."],
    )

    assert result["decision"] == "uncertain"
    assert result["valid"] is False
    assert result["error_type"] == "ValueError"


def test_claim_verifier_requires_rationale_for_uncertain():
    class _JudgeLLM:
        default_model = "judge-model"

        def chat(self, *_args, **_kwargs):
            return json.dumps(
                {"decision": "uncertain", "confidence": 0.4, "rationale": ""}
            )

    result = ConditionalClaimVerifier(llm_client=_JudgeLLM())(
        "Aspirin reduced mortality.",
        ["Aspirin reduced mortality in the cited cohort."],
    )

    assert result["decision"] == "uncertain"
    assert result["valid"] is False
    assert result["error_type"] == "ValueError"


def test_short_unknown_entity_substitution_requires_semantic_verification():
    calls = []

    def claim_verifier(claim, cited_evidence):
        calls.append((claim, list(cited_evidence)))
        return {
            "decision": "unsupported",
            "confidence": 0.93,
            "rationale": "The cited evidence names Dax, not Q17.",
            "valid": True,
            "model": "judge-model",
            "prompt_version": "high-risk-claim-v2",
        }

    trace = _trace(
        query="Did Q17 improve outcomes in adults?",
        evidence="Dax improved outcomes in adults in the cited trial.",
        answer="Q17 improved outcomes in adults in the cited trial [1].",
    )
    decision = RuntimeVerifier(claim_verifier=claim_verifier).verify(
        trace, retries_remaining=0
    )

    assert decision.status == "evidence_limited"
    assert len(calls) == 1


@pytest.mark.parametrize("entity", ["Dax", "Q17"])
def test_short_unknown_entities_without_judge_do_not_accept(entity):
    trace = _trace(
        query=f"Did {entity} improve outcomes?",
        evidence=f"{entity} improved outcomes in adults in the cited trial.",
        answer=f"{entity} improved outcomes in adults in the cited trial [1].",
    )
    decision = RuntimeVerifier().verify(trace, retries_remaining=0)

    assert decision.status != "accept"


def test_runtime_verifier_factory_wires_conditional_judge():
    class _JudgeLLM:
        default_model = "judge-model"

        def chat(self, *_args, **_kwargs):
            return '{"decision":"uncertain","confidence":0.4,"rationale":"ambiguous"}'

    verifier = build_runtime_verifier(
        {"claim_verifier_model": "judge-model"},
        llm_client=_JudgeLLM(),
    )

    assert isinstance(verifier.claim_verifier, ConditionalClaimVerifier)


def test_combined_confidence_formula_is_explicit_and_bounded():
    components = {
        "retrieval_coverage": 1.0,
        "evidence_sufficiency": 0.8,
        "claim_grounding": 0.6,
        "citation_support": 1.0,
        "query_coverage": 0.8,
        "verifier_confidence": 0.9,
    }

    score, explanation = calculate_combined_confidence(components)

    assert score == pytest.approx(0.81)
    assert 0.0 <= score <= 1.0
    assert "not a clinically calibrated probability" in explanation
