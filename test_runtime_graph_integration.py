import importlib
from datetime import datetime

from agents.base import AgentOutput
from evaluation_core import build_agent_evaluation_trace, stable_document_id
from research_agent_api_v2 import QueryRequest, QueryResponse


QUERY = "Do GLP-1 drugs reduce cardiovascular events in adults?"
EVIDENCE = "GLP-1 therapy reduced cardiovascular events by 20% in adults."


def _output(with_evidence, attempt_id):
    sources = (
        [
            {
                "doc_id": "doc-1",
                "text": EVIDENCE,
                "score": 0.9,
                "original_rank": 1,
                "metadata": {
                    "title": "Outcome trial",
                    "source": "PubMed",
                    "pmid": "123",
                    "year": 2024,
                },
            }
        ]
        if with_evidence
        else []
    )
    answer = (
        "GLP-1 therapy reduced cardiovascular events by 20% in adults [1]."
        if with_evidence
        else "No evidence was found."
    )
    state = {
        "expanded_query": QUERY,
        "retrieval_results": sources,
        "reranked_results": sources,
        "answer": answer,
        "citations": ["Outcome trial"] if sources else [],
        "model_used": "test-model@revision-1",
        "retrieval_time_sec": 0.1,
        "execution_time_sec": 0.2,
        "error": None,
    }
    if sources:
        state["synthesis_context"] = [
            {
                "document_id": stable_document_id(
                    sources[0], "pubmed", 1
                ),
                "text": EVIDENCE,
                "start_char": 0,
                "original_length": len(EVIDENCE),
                "truncated": False,
                "citation_marker": 1,
            }
        ]
    else:
        state["synthesis_context"] = []
        state["answer_origin"] = "evidence_limited"
    trace = build_agent_evaluation_trace(
        agent_name="search_pubmed",
        domain="pubmed",
        original_query=QUERY,
        state=state,
        context={
            "trace_id": "trace-integration",
            "attempt_id": attempt_id,
            "top_k": 5,
        },
    )
    return AgentOutput(
        answer=answer,
        citations=state["citations"],
        confidence=0.8 if sources else 0.0,
        sources=sources,
        model_used="test-model@revision-1",
        domain="pubmed",
        execution_time_sec=0.2,
        evaluation_trace=trace,
    )


class _WeakThenStrongAgent:
    def __init__(self):
        self.calls = []

    def invoke(self, query, context):
        self.calls.append((query, dict(context)))
        return _output(len(self.calls) > 1, context["attempt_id"])


def _initial_state():
    return {
        "input_query": QUERY,
        "context": {
            "model_id": "test-model",
            "top_k": 5,
            "max_agent_retries": 1,
            "max_agent_synthesis_repairs": 1,
            "max_synthesis_repairs": 1,
            "runtime_verification_deadline_sec": 5,
        },
        "trace_id": "trace-integration",
        "timestamp_start": datetime.utcnow(),
        "is_medical_query": True,
        "classification_confidence": 0.0,
        "classification_reason": "",
        "discovered_skills": [],
        "skill_scores": {},
        "retrieval_results": {},
        "tokens_used": {},
        "retrieval_time_sec": {},
        "total_retrieval_time_sec": 0.0,
        "intermediate_answer": "",
        "intermediate_sources": [],
        "intermediate_model_used": "",
        "synthesis_tokens_in": 0,
        "synthesis_tokens_out": 0,
        "synthesis_time_sec": 0.0,
        "synthesis_context": [],
        "confidence_score": 0.0,
        "confidence_components": {},
        "runtime_quality_score": 0.0,
        "runtime_quality_explanation": "",
        "coverage_explanation": "",
        "coherence_score": 0.0,
        "coherence_explanation": "",
        "should_fallback": False,
        "coherence_eval_model_used": "",
        "fallback_count": 0,
        "fallback_answer": None,
        "fallback_triggered": False,
        "fallback_reason": "",
        "evaluation_traces": [],
        "verification_history": [],
        "verification_decision": {},
        "repair_history": [],
        "evidence_limited": False,
        "attempt_telemetry": [],
        "token_usage": {"input": 0, "output": 0, "total": 0},
        "runtime_executor_metrics": {},
        "output_answer": "",
        "output_sources": [],
        "output_citations": [],
        "output_disclaimer": "",
        "timestamp_end": datetime.utcnow(),
        "execution_time_sec": 0.0,
        "cost_estimate": 0.0,
        "error_occurred": False,
        "error_messages": [],
        "is_partial_response": False,
    }


def test_mocked_full_graph_runs_selective_retry_and_grounded_repair(monkeypatch):
    graph_module = importlib.import_module("graph")
    parallel_module = importlib.import_module("nodes.parallel_retrieve")
    synthesis_module = importlib.import_module("nodes.synthesise")
    fallback_module = importlib.import_module("nodes.fallback_regen")
    agent = _WeakThenStrongAgent()

    monkeypatch.setattr(parallel_module, "_get_agent_graph", lambda _name: agent)
    monkeypatch.setattr(
        graph_module,
        "classify_intent",
        lambda state: {
            **state,
            "is_medical_query": True,
            "classification_confidence": 1.0,
            "classification_reason": "test",
        },
    )
    monkeypatch.setattr(
        graph_module,
        "discover_skills",
        lambda state: {
            **state,
            "discovered_skills": ["search_pubmed"],
            "skill_scores": {"search_pubmed": 1.0},
        },
    )

    class _InitialSynthesisLLM:
        def chat(self, **_kwargs):
            return "GLP-1 therapy reduced events by 20% [9]."

    class _RepairLLM:
        def chat(self, **_kwargs):
            return (
                "GLP-1 therapy reduced cardiovascular events by 20% "
                "in adults [1]."
            )

    monkeypatch.setattr(synthesis_module, "LLMClient", lambda: _InitialSynthesisLLM())
    monkeypatch.setattr(fallback_module, "LLMClient", lambda: _RepairLLM())
    monkeypatch.setattr(graph_module, "parallel_retrieve", parallel_module.parallel_retrieve)
    monkeypatch.setattr(graph_module, "synthesise", synthesis_module.synthesise)
    monkeypatch.setattr(graph_module, "fallback_regen", fallback_module.fallback_regen)

    result = graph_module.build_graph().invoke(_initial_state())

    assert len(agent.calls) == 2
    assert result["fallback_count"] == 1
    assert result["fallback_triggered"] is True
    assert result["verification_decision"]["status"] == "accept"
    assert result["evidence_limited"] is False
    assert "GLP-1 therapy reduced cardiovascular events by 20%" in result[
        "output_answer"
    ]
    assert len(result["evaluation_traces"]) == 5
    assert result["evaluation_traces"][-1]["answer"] == result["output_answer"]
    assert result["evaluation_traces"][-1]["attempt_id"].endswith(":terminal")
    assert {event["target_stage"] for event in result["repair_history"]} == {
        "retrieval",
        "synthesis",
    }
    assert set(result["confidence_components"]) >= {
        "retrieval_coverage",
        "evidence_sufficiency",
        "claim_grounding",
        "citation_support",
        "query_coverage",
        "verifier_confidence",
    }


def test_query_response_adds_optional_verification_metadata():
    response = QueryResponse(
        answer="answer",
        sources=[],
        citations=[],
        confidence=0.5,
        trace_id="trace",
        execution_time_sec=0.1,
        cost_estimate=0.0,
        fallback_triggered=False,
        is_partial_response=False,
        error_occurred=False,
    )

    assert response.verification is None
    assert response.evaluation_traces is None
    assert response.confidence_components is None
    assert response.runtime_quality_score is None
    assert response.evidence_limited is False


def test_runtime_retry_budgets_and_deadline_are_request_configurable():
    request = QueryRequest(
        question=QUERY,
        max_agent_retries=0,
        max_agent_synthesis_repairs=0,
        max_synthesis_repairs=0,
        runtime_verification_deadline_sec=12.5,
    )

    assert request.max_agent_retries == 0
    assert request.max_agent_synthesis_repairs == 0
    assert request.max_synthesis_repairs == 0
    assert request.runtime_verification_deadline_sec == 12.5
    assert request.include_evaluation_traces is False
