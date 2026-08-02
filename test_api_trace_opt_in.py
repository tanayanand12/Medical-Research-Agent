import json
import logging
import time

from fastapi.testclient import TestClient
import pytest

import research_agent_api_v2 as api_module
from evaluation_core import (
    VerificationDecision,
    build_orchestrator_evaluation_trace,
)
from nodes.format_response import NON_MEDICAL_RESPONSE, format_response
from unicode_safe_logging import SensitiveDataFilter


SECRET = "sk-test-super-secret"
EVIDENCE = "Private full evidence text that must not leave the API."


class _CompiledGraphStub:
    def invoke(self, state):
        state.update(
            {
                "output_answer": "Grounded answer.",
                "output_sources": ["search_pubmed"],
                "output_citations": ["Citation."],
                "confidence_score": 0.8,
                "execution_time_sec": 0.05,
                "cost_estimate": 0.01,
                "fallback_triggered": False,
                "is_partial_response": False,
                "error_occurred": False,
                "verification_decision": {
                    "status": "accept",
                    "valid": True,
                    "raw_decision": {
                        "provider_payload": EVIDENCE,
                        "authorization": f"Bearer {SECRET}",
                    },
                },
                "confidence_components": {"claim_grounding": 1.0},
                "runtime_quality_score": 0.8,
                "runtime_quality_explanation": "test",
                "evidence_limited": False,
                "attempt_telemetry": [
                    {
                        "attempt_id": "attempt-1",
                        "token_total": 3,
                        "model": f"model-{SECRET}",
                        "provider_metadata": {
                            "authorization": f"Bearer {SECRET}"
                        },
                    }
                ],
                "token_usage": {"input": 2, "output": 1, "total": 3},
                "evaluation_traces": [
                    {
                        "trace_id": state["trace_id"],
                        "attempt_id": "attempt-1",
                        "answer": "Grounded answer.",
                        "original_query": state["input_query"],
                        "retrieval_configuration": {
                            "top_k": 5,
                            "authorization": f"Bearer {SECRET}",
                        },
                        "retrieved_documents": [
                            {
                                "document_id": "doc-1",
                                "text": EVIDENCE,
                                "content_hash": "hash",
                                "metadata": {"api_key": SECRET, "title": "Study"},
                            }
                        ],
                        "final_context_spans": [
                            {
                                "document_id": "doc-1",
                                "text": EVIDENCE,
                                "content_hash": "span-hash",
                            }
                        ],
                        "errors": [f"provider rejected key {SECRET}"],
                        "repair_history": [
                            {
                                "actual_configuration": {
                                    "query": state["input_query"],
                                }
                            }
                        ],
                    }
                ],
            }
        )
        return state


def _complete_decision(status, *, valid=True):
    return {
        "status": status,
        "component_scores": {
            "retrieval_coverage": 0.9,
            "evidence_sufficiency": 0.9,
            "claim_grounding": 0.9,
            "citation_support": 0.9,
            "query_coverage": 0.9,
            "verifier_confidence": 0.9,
        },
        "failed_checks": [] if status == "accept" else ["test_failure"],
        "structured_feedback": [],
        "target_stage": "none",
        "target_agent": "orchestrator",
        "recommended_retry_changes": {},
        "verifier_confidence": 0.9,
        "valid": valid,
        "error": None if valid else "verifier unavailable",
        "verifier_model": "deterministic",
        "verifier_model_revision": "",
        "prompt_version": "runtime-verifier-v1",
        "raw_decision": {},
    }


def _decision_with(status, **changes):
    decision = _complete_decision(status)
    decision.update(changes)
    return decision


class _TerminalVerificationGraphStub:
    def __init__(self, decision, *, is_medical=True, bind_decision=False):
        self.decision = decision
        self.is_medical = is_medical
        self.bind_decision = bind_decision

    def invoke(self, state):
        document = {
            "document_id": "terminal-doc",
            "text": "Aspirin reduced platelet aggregation in the cited study.",
            "score": 0.9,
        }
        answer = (
            "Aspirin reduced platelet aggregation in the cited study [1]."
            if self.bind_decision
            else "Unsupported answer must not be delivered."
        )
        state.update(
            {
                "is_medical_query": self.is_medical,
                "intermediate_answer": answer,
                "intermediate_sources": ["search_pubmed"],
                "retrieval_results": {
                    "search_pubmed": {"results": [document]}
                },
                "synthesis_context": [
                    {
                        "document_id": "terminal-doc",
                        "text": document["text"],
                        "citation_marker": 1,
                    }
                ],
                "verification_decision": self.decision,
                "confidence_score": 0.8,
                "confidence_components": {},
                "runtime_quality_score": 0.8,
                "runtime_quality_explanation": "test",
                "fallback_triggered": False,
                "is_partial_response": False,
                "error_occurred": False,
                "error_messages": [],
                "evidence_limited": False,
                "cost_estimate": 0.0,
                "token_usage": {"input": 0, "output": 0, "total": 0},
                "attempt_telemetry": [],
                "evaluation_traces": [],
            }
        )
        if self.bind_decision and isinstance(self.decision, dict):
            trace = build_orchestrator_evaluation_trace(state)
            trace.verification_decisions.append(
                VerificationDecision.from_dict(self.decision)
            )
            state["evaluation_traces"] = [trace.to_dict()]
        return format_response(state)


def _client(monkeypatch):
    monkeypatch.setattr(api_module, "get_graph", lambda: _CompiledGraphStub())
    return TestClient(api_module.app)


def test_endpoint_omits_response_traces_by_default(monkeypatch):
    with _client(monkeypatch) as client:
        response = client.post(
            "/query",
            json={"question": "What is the evidence for aspirin?"},
        )

    assert response.status_code == 200
    body = response.json()
    assert "evaluation_traces" not in body
    serialized = json.dumps(body)
    assert SECRET not in serialized
    assert EVIDENCE not in serialized
    assert body["trace_policy"] == {
        "internal_capture": True,
        "response_included": False,
        "response_redacted": False,
        "observability_persistence": "not_enabled_by_endpoint",
    }


def test_endpoint_returns_only_redacted_traces_when_opted_in(monkeypatch):
    question = "Patient ZQ-771 asks about private aspirin exposure."
    with _client(monkeypatch) as client:
        response = client.post(
            "/query",
            json={
                "question": question,
                "include_evaluation_traces": True,
            },
        )

    assert response.status_code == 200
    body = response.json()
    serialized = json.dumps(body)
    trace = body["evaluation_traces"][0]
    assert trace["answer"] == body["answer"]
    assert trace["original_query"] == "[REDACTED]"
    assert trace["retrieved_documents"][0]["text"] == "[REDACTED]"
    assert trace["final_context_spans"][0]["text"] == "[REDACTED]"
    assert trace["response_redaction"]["full_evidence_removed"] is True
    assert SECRET not in serialized
    assert EVIDENCE not in serialized
    assert question not in serialized
    assert body["trace_policy"]["response_redacted"] is True


def test_endpoint_logs_query_fingerprint_not_query_or_secrets(monkeypatch, caplog):
    question = "Patient Jane Doe asks about a private treatment."
    caplog.set_level(logging.INFO)

    with _client(monkeypatch) as client:
        response = client.post(
            "/query",
            json={"question": question, "model_id": f"model-{SECRET}"},
        )

    assert response.status_code == 200
    logs = caplog.text
    assert question not in logs
    assert SECRET not in logs
    assert "query_sha256=" in logs


def test_sensitive_log_filter_scrubs_message_arguments_and_exceptions():
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="Authorization=%s",
        args=(f"Bearer {SECRET}",),
        exc_info=None,
    )

    assert SensitiveDataFilter().filter(record) is True
    assert SECRET not in record.getMessage()
    assert "Bearer" not in record.getMessage()


def test_sensitive_log_filter_scrubs_full_basic_authorization_value():
    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="Authorization: Basic %s",
        args=("dXNlcjp2ZXJ5LXNlbnNpdGl2ZS1wYXNzd29yZA==",),
        exc_info=None,
    )

    assert SensitiveDataFilter().filter(record) is True
    assert "dXNlcjp2ZXJ5LXNlbnNpdGl2ZS1wYXNzd29yZA" not in record.getMessage()
    assert "Basic" not in record.getMessage()


@pytest.mark.parametrize(
    ("decision", "expected_limited"),
    [
        (None, True),
        ({}, True),
        ({"status": "malformed", "valid": True}, True),
        ({"status": "accept", "valid": True}, True),
        (_complete_decision("accept", valid=False), True),
        (_decision_with("accept", target_stage="retrieval"), True),
        (
            _decision_with(
                "accept",
                recommended_retry_changes={"top_k": 10},
            ),
            True,
        ),
        (_complete_decision("retry_retrieval"), True),
        (_complete_decision("retry_synthesis"), True),
        (_complete_decision("accept"), True),
        (_complete_decision("evidence_limited"), True),
    ],
)
def test_endpoint_terminal_verification_is_fail_closed(
    monkeypatch, decision, expected_limited
):
    monkeypatch.setattr(
        api_module,
        "get_graph",
        lambda: _TerminalVerificationGraphStub(decision),
    )

    with TestClient(api_module.app) as client:
        response = client.post(
            "/query",
            json={
                "question": "What did the cited evidence show?",
                "include_evaluation_traces": True,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["evidence_limited"] is expected_limited
    expected_status = "evidence_limited" if expected_limited else "accept"
    assert body["verification"]["status"] == expected_status
    if expected_limited:
        assert "insufficient evidence" in body["answer"].lower()
        assert "Unsupported answer must not be delivered." not in body["answer"]
    else:
        assert "Unsupported answer must not be delivered." in body["answer"]
    assert body["evaluation_traces"][-1]["answer"] == body["answer"]
    assert (
        body["evaluation_traces"][-1]["verification_decisions"][-1]["status"]
        == expected_status
    )


def test_endpoint_accepts_only_decision_bound_to_latest_valid_trace(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "get_graph",
        lambda: _TerminalVerificationGraphStub(
            _complete_decision("accept"),
            bind_decision=True,
        ),
    )

    with TestClient(api_module.app) as client:
        response = client.post(
            "/query",
            json={"question": "What did the cited evidence show?"},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["evidence_limited"] is False
    assert body["verification"]["status"] == "accept"
    assert "Aspirin reduced platelet aggregation" in body["answer"]


def test_endpoint_rejects_accept_decision_not_bound_to_latest_trace(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "get_graph",
        lambda: _TerminalVerificationGraphStub(
            _complete_decision("accept"),
            bind_decision=False,
        ),
    )

    with TestClient(api_module.app) as client:
        response = client.post(
            "/query",
            json={"question": "What did the cited evidence show?"},
        )

    body = response.json()
    assert response.status_code == 200
    assert body["evidence_limited"] is True
    assert body["verification"]["status"] == "evidence_limited"
    assert "Unsupported answer must not be delivered." not in body["answer"]


def test_endpoint_starts_absolute_deadline_before_graph_entry(monkeypatch):
    observed = {}

    class _DeadlineGraph(_CompiledGraphStub):
        def invoke(self, state):
            observed["deadline_at"] = state["context"].get(
                "_runtime_deadline_at_monotonic"
            )
            return super().invoke(state)

    monkeypatch.setattr(api_module, "get_graph", lambda: _DeadlineGraph())
    started = time.monotonic()
    with TestClient(api_module.app) as client:
        response = client.post(
            "/query",
            json={
                "question": "What is the evidence for aspirin?",
                "runtime_verification_deadline_sec": 5,
            },
        )

    assert response.status_code == 200
    assert started < observed["deadline_at"] <= time.monotonic() + 5


def test_endpoint_non_medical_early_exit_does_not_require_runtime_verifier(
    monkeypatch,
):
    monkeypatch.setattr(
        api_module,
        "get_graph",
        lambda: _TerminalVerificationGraphStub(None, is_medical=False),
    )

    with TestClient(api_module.app) as client:
        response = client.post(
            "/query",
            json={
                "question": "Who won a fictional sports match?",
                "include_evaluation_traces": True,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == NON_MEDICAL_RESPONSE
    assert body["evidence_limited"] is False
    assert body["verification"]["status"] == "accept"
    assert body["verification"]["verifier_model"] == (
        "deterministic_query_classifier"
    )
    assert body["evaluation_traces"][-1]["answer"] == body["answer"]
