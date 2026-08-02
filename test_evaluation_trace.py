import json
import threading
from dataclasses import fields
import importlib

import pytest

from agents.base import AgentOutput, SubAgentGraph
from agents.clinical_trials_agent.graph import ClinicalTrialsAgentGraph
from agents.fda_agent.graph import FDAAgentGraph
from agents.local_agent.graph import LocalAgentGraph
from agents.pubmed_agent.graph import PubMedAgentGraph
from evaluation_core import (
    EVALUATION_TRACE_SCHEMA_VERSION,
    build_agent_evaluation_trace,
    build_orchestrator_evaluation_trace,
    content_hash,
    stable_document_id,
    validate_evaluation_trace,
)
from llm_client import LLMCallResult
from runtime_verification import ATTEMPT_EVENT_SCHEMA_VERSION


class _StaticGraph:
    def __init__(self, result):
        self.result = result

    def invoke(self, _state):
        return dict(self.result)


class _DeterministicCompiledAgent(SubAgentGraph):
    domain = "compiled_test"

    def _expand_query_node(self, state):
        return {"expanded_query": f"{state['input_query']} expanded"}

    def _retrieve_node(self, _state):
        return {
            "retrieval_results": [
                {
                    "document_id": "compiled-doc",
                    "text": "Aspirin reduced platelet aggregation in adults.",
                    "score": 0.9,
                    "original_rank": 1,
                }
            ],
            "retrieval_time_sec": 0.01,
            "error": None,
        }

    def _rerank_node(self, state):
        return {"reranked_results": list(state["retrieval_results"])}

    def _synthesise_node(self, state):
        document = state["reranked_results"][0]
        return {
            "answer": "Aspirin reduced platelet aggregation in adults [1].",
            "citations": ["Compiled source"],
            "confidence": 0.9,
            "model_used": "mock-model@mock-revision",
            "synthesis_context": [
                {
                    "document_id": document["document_id"],
                    "text": document["text"],
                    "start_char": 0,
                    "original_length": len(document["text"]),
                    "citation_marker": 1,
                }
            ],
            "execution_time_sec": 0.02,
        }


class _StructuredTelemetryAgent(SubAgentGraph):
    domain = "telemetry_test"

    def _expand_query_node(self, state):
        return {"expanded_query": state["input_query"]}

    def _retrieve_node(self, _state):
        return {
            "retrieval_results": [
                {
                    "document_id": "telemetry-doc",
                    "text": "The exact evidence supports the answer.",
                    "score": 0.9,
                    "original_rank": 1,
                }
            ],
            "retrieval_time_sec": 0.02,
            "error": None,
        }

    def _rerank_node(self, state):
        return {"reranked_results": list(state["retrieval_results"])}


def _graph_result(domain):
    result = {
        "expanded_query": f"expanded {domain} query",
        "retrieval_results": [
            {
                "doc_id": 7,
                "text": "Semaglutide reduced cardiovascular events by 20% in adults.",
                "score": 0.81,
                "metadata": {
                    "title": "Cardiovascular outcomes",
                    "source": domain,
                    "pmid": "12345" if domain == "pubmed" else None,
                    "nct_id": "NCT0001" if domain == "clinical_trials" else None,
                    "record_id": "FDA-1" if domain == "fda" else None,
                    "year": 2024,
                    "study_type": "randomized trial",
                },
            }
        ],
        "reranked_results": [
            {
                "doc_id": 7,
                "text": "Semaglutide reduced cardiovascular events by 20% in adults.",
                "score": 0.93,
                "original_rank": 1,
                "metadata": {
                    "title": "Cardiovascular outcomes",
                    "source": domain,
                    "pmid": "12345" if domain == "pubmed" else None,
                    "nct_id": "NCT0001" if domain == "clinical_trials" else None,
                    "record_id": "FDA-1" if domain == "fda" else None,
                    "year": 2024,
                    "study_type": "randomized trial",
                },
            }
        ],
        "answer": "Semaglutide reduced cardiovascular events by 20% [1].",
        "citations": ["1. Cardiovascular outcomes."],
        "confidence": 0.8,
        "model_used": "test-model@revision-1",
        "retrieval_time_sec": 0.12,
        "execution_time_sec": 0.25,
        "error": None,
    }
    document = result["reranked_results"][0]
    result["synthesis_context"] = [
        {
            "document_id": stable_document_id(document, domain, 1),
            "text": document["text"],
            "start_char": 0,
            "original_length": len(document["text"]),
            "truncated": False,
            "citation_marker": 1,
        }
    ]
    return result


@pytest.mark.parametrize(
    ("agent_class", "invoke_method", "domain"),
    [
        (LocalAgentGraph, SubAgentGraph.invoke, "local"),
        (PubMedAgentGraph, PubMedAgentGraph.invoke, "pubmed"),
        (FDAAgentGraph, FDAAgentGraph.invoke, "fda"),
        (
            ClinicalTrialsAgentGraph,
            ClinicalTrialsAgentGraph.invoke,
            "clinical_trials",
        ),
    ],
)
def test_all_agents_produce_schema_valid_sidecar_traces(
    agent_class, invoke_method, domain
):
    agent = agent_class.__new__(agent_class)
    agent._invoke_lock = threading.RLock()
    agent._compiled_graph = _StaticGraph(_graph_result(domain))

    output = invoke_method(
        agent,
        "Do GLP-1 drugs reduce cardiovascular events?",
        {
            "trace_id": "trace-1",
            "attempt_id": f"{domain}-attempt-1",
            "top_k": 5,
            "model_id": "test-model",
        },
    )

    assert isinstance(output, AgentOutput)
    assert output.evaluation_trace is not None
    assert output.evaluation_trace.schema_version == EVALUATION_TRACE_SCHEMA_VERSION
    assert output.evaluation_trace.domain == domain
    assert output.evaluation_trace.expanded_queries == [f"expanded {domain} query"]
    assert output.evaluation_trace.retrieved_documents
    assert output.evaluation_trace.reranked_documents
    assert output.evaluation_trace.atomic_claims
    assert output.evaluation_trace.atomic_claims[0].cited_document_ids
    assert validate_evaluation_trace(output.evaluation_trace) == []
    json.dumps(output.evaluation_trace.to_dict())


def test_existing_agent_output_construction_remains_compatible():
    original_fields = {
        "answer",
        "citations",
        "confidence",
        "sources",
        "model_used",
        "domain",
        "execution_time_sec",
        "error",
    }

    output = AgentOutput(answer="answer", citations=[], confidence=0.5)

    assert original_fields.issubset({item.name for item in fields(AgentOutput)})
    assert output.answer == "answer"
    assert output.evaluation_trace is None


def test_mocked_compiled_agent_graph_emits_valid_trace():
    agent = _DeterministicCompiledAgent()

    output = agent.invoke(
        "Does aspirin affect platelets?",
        {
            "trace_id": "compiled-trace",
            "attempt_id": "compiled-attempt-1",
            "top_k": 1,
        },
    )

    assert output.evaluation_trace is not None
    assert output.evaluation_trace.attempt_id == "compiled-attempt-1"
    assert output.evaluation_trace.answer == output.answer
    assert validate_evaluation_trace(output.evaluation_trace) == []


def test_real_agent_graph_records_structured_generation_telemetry():
    class _StructuredLLM:
        default_model = "configured-model"

        def chat_with_metadata(self, **_kwargs):
            return LLMCallResult(
                text="The exact evidence supports the answer [1].",
                model="actual-model",
                model_revision="revision-7",
                tokens_in=17,
                tokens_out=9,
                cost_usd=0.004,
                latency_sec=0.12,
                finish_reason="stop",
                provider_metadata={"provider": "mock"},
            )

    agent = _StructuredTelemetryAgent()
    agent._llm = _StructuredLLM()
    output = agent.invoke(
        "What does the evidence show?",
        {
            "trace_id": "telemetry-trace",
            "attempt_id": "telemetry-attempt-1",
        },
    )
    trace = output.evaluation_trace

    assert trace is not None
    assert trace.exact_model == "actual-model"
    assert trace.model_revision == "revision-7"
    assert trace.token_usage == {"input": 17, "output": 9, "total": 26}
    assert trace.cost_breakdown_usd == {"generation": pytest.approx(0.004)}
    assert trace.stage_latency_sec["generation"] == pytest.approx(0.12)
    assert len(trace.attempt_events) == 1
    event = trace.attempt_events[0]
    assert event["schema_version"] == ATTEMPT_EVENT_SCHEMA_VERSION
    assert event["stage"] == "agent_synthesis"
    assert event["attempt_id"] == "telemetry-attempt-1"
    assert event["repair_status"] == "initial"
    assert event["model"] == "actual-model"
    assert event["model_revision"] == "revision-7"
    assert event["token_usage"] == {"input": 17, "output": 9, "total": 26}
    assert event["cost_usd"] == pytest.approx(0.004)
    assert event["latency_sec"] == pytest.approx(0.12)
    assert event["finish_reason"] == "stop"

    event["token_usage"]["total"] += 1
    assert "attempt_event_token_total_mismatch" in (
        validate_evaluation_trace(trace)
    )


def test_failed_agent_generation_uses_recorded_provider_metadata():
    failed_call = LLMCallResult(
        text="",
        model="failed-model",
        model_revision="failed-revision",
        tokens_in=11,
        tokens_out=0,
        cost_usd=0.003,
        latency_sec=0.25,
        finish_reason="error",
        provider_metadata={"provider": "mock"},
        status="error",
        error_type="ProviderError",
    )

    class _FailedLLM:
        default_model = "configured-model"

        @staticmethod
        def thread_call_history():
            return [failed_call]

    agent = _StructuredTelemetryAgent.__new__(_StructuredTelemetryAgent)
    agent._llm = _FailedLLM()
    telemetry = agent._failure_telemetry(
        {
            "context": {
                "trace_id": "trace-failed-call",
                "attempt_id": "trace-failed-call:agent:1",
            }
        },
        RuntimeError("provider failed"),
        stage="agent_synthesis",
        latency_sec=9.0,
    )

    assert telemetry["model_used"] == "failed-model@failed-revision"
    assert telemetry["token_usage"] == {
        "input": 11,
        "output": 0,
        "total": 11,
    }
    event = telemetry["attempt_events"][0]
    assert event["model"] == "failed-model"
    assert event["latency_sec"] == pytest.approx(0.25)
    assert event["cost_usd"] == pytest.approx(0.003)
    assert event["error_type"] == "ProviderError"


def test_failed_generation_does_not_reuse_prior_invocation_telemetry():
    prior_call = LLMCallResult(
        text="prior answer",
        model="prior-model",
        model_revision="old",
        tokens_in=99,
        tokens_out=50,
        cost_usd=1.0,
        latency_sec=2.0,
        finish_reason="stop",
        provider_metadata={"provider": "mock"},
    )

    class _LLM:
        default_model = "configured-model"

        @staticmethod
        def thread_call_history():
            return [prior_call]

    agent = _StructuredTelemetryAgent.__new__(
        _StructuredTelemetryAgent
    )
    agent._llm = _LLM()
    telemetry = agent._failure_telemetry(
        {
            "context": {
                "trace_id": "trace-current",
                "attempt_id": "trace-current:agent:1",
                "_llm_history_start": 1,
            }
        },
        RuntimeError("failed before provider call"),
        stage="agent_synthesis",
        latency_sec=0.1,
    )

    assert telemetry["model_used"] == "configured-model"
    assert telemetry["token_usage"]["total"] == 0
    assert telemetry["cost_breakdown_usd"]["generation"] == 0.0


@pytest.mark.parametrize(
    "agent_class",
    [LocalAgentGraph, FDAAgentGraph, ClinicalTrialsAgentGraph],
)
def test_real_agent_synthesis_nodes_emit_exact_context_manifest(agent_class):
    class _LLM:
        default_model = "mock-model"

        def chat(self, *_args, **_kwargs):
            return "The cited source supports the answer [1]."

    agent = agent_class.__new__(agent_class)
    agent._llm = _LLM()
    document = {
        "text": "Exact source text supplied to the synthesis prompt.",
        "score": 0.9,
        "doc_id": 1,
        "metadata": {"title": "Manifest source", "year": 2024},
    }
    result = agent._synthesise_node(
        {
            "input_query": "What does the source show?",
            "context": {},
            "retrieval_results": [document],
            "reranked_results": [document],
        }
    )

    assert result["synthesis_context"] == [
        {
            "document_id": stable_document_id(document, agent.domain, 1),
            "text": document["text"],
            "start_char": 0,
            "original_length": len(document["text"]),
            "truncated": False,
            "citation_marker": 1,
        }
    ]


def test_fallback_document_id_is_stable_across_rank_changes():
    document = {
        "text": "Stable evidence text.",
        "metadata": {"title": "Stable title", "source": "PubMed"},
    }

    assert stable_document_id(document, "pubmed", 1) == stable_document_id(
        document, "pubmed", 9
    )


def test_explicit_stable_document_id_is_not_rewrapped():
    document = {"document_id": "pubmed:pmid:123", "text": "evidence"}

    assert stable_document_id(document, "multi_source", 1) == "pubmed:pmid:123"


def test_chunks_from_same_source_record_have_distinct_stable_ids():
    first = {
        "text": "First chunk.",
        "metadata": {"pmid": "123", "source": "PubMed"},
    }
    second = {
        "text": "Second chunk.",
        "metadata": {"pmid": "123", "source": "PubMed"},
    }

    assert stable_document_id(first, "pubmed", 1) != stable_document_id(
        second, "pubmed", 2
    )


def test_multi_citation_markers_resolve_to_individual_documents():
    documents = [
        {
            "doc_id": index,
            "text": f"Evidence {index}.",
            "score": 1.0,
            "metadata": {"source": "PubMed"},
        }
        for index in range(1, 4)
    ]
    synthesis_context = [
        {
            "document_id": stable_document_id(
                document, "search_pubmed", rank
            ),
            "text": document["text"],
            "original_length": len(document["text"]),
            "citation_index": rank,
        }
        for rank, document in enumerate(documents, 1)
    ]
    trace = build_orchestrator_evaluation_trace(
        {
            "trace_id": "trace-citations",
            "input_query": "question",
            "context": {"model_id": "test-model"},
            "discovered_skills": ["search_pubmed"],
            "retrieval_results": {"search_pubmed": {"results": documents}},
            "intermediate_answer": "The evidence is mixed [1, 2-3].",
            "synthesis_context": synthesis_context,
            "intermediate_model_used": "test-model",
            "total_retrieval_time_sec": 0.1,
            "synthesis_time_sec": 0.1,
            "synthesis_tokens_in": 0,
            "synthesis_tokens_out": 0,
            "cost_estimate": 0.0,
        }
    )

    assert len(trace.atomic_claims[0].cited_document_ids) == 3
    assert all(item.resolved for item in trace.citations)


def test_orchestrator_attempt_cost_excludes_prior_retrieval_cost():
    trace = build_orchestrator_evaluation_trace(
        {
            "trace_id": "trace-cost",
            "input_query": "question",
            "context": {"model_id": "test-model"},
            "discovered_skills": [],
            "retrieval_results": {},
            "intermediate_answer": "Evidence-limited response.",
            "intermediate_model_used": "test-model",
            "synthesis_tokens_in": 4,
            "synthesis_tokens_out": 2,
            "synthesis_time_sec": 0.1,
            "cost_estimate": 0.06,
            "last_synthesis_cost_usd": 0.01,
        }
    )

    assert trace.cost_breakdown_usd == {"synthesis": 0.01}


def test_trace_adapter_failure_does_not_break_agent_output(monkeypatch):
    evaluation_core = importlib.import_module("evaluation_core")
    monkeypatch.setattr(
        evaluation_core,
        "build_agent_evaluation_trace",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("bad metadata")),
    )
    agent = LocalAgentGraph.__new__(LocalAgentGraph)
    agent._invoke_lock = threading.RLock()
    agent._compiled_graph = _StaticGraph(_graph_result("local"))

    output = SubAgentGraph.invoke(agent, "question", {"trace_id": "trace"})

    assert output.answer
    assert output.evaluation_trace is None


def test_agent_trace_uses_exact_prompt_marker_mapping_and_truncation():
    documents = [
        {
            "doc_id": 1,
            "text": "A" * 900,
            "score": 0.9,
            "metadata": {"pmid": "100", "source": "PubMed"},
        },
        {
            "doc_id": 2,
            "text": "B" * 900,
            "score": 0.8,
            "metadata": {"pmid": "100", "source": "PubMed"},
        },
        {
            "doc_id": 3,
            "text": "Second study found benefit in adults.",
            "score": 0.7,
            "metadata": {"pmid": "200", "source": "PubMed"},
        },
    ]
    manifest = [
        {
            "document_id": stable_document_id(document, "pubmed", index),
            "text": document["text"][:800],
            "original_length": len(document["text"]),
            "truncated": len(document["text"]) > 800,
            "citation_marker": 1 if index < 3 else 2,
        }
        for index, document in enumerate(documents, 1)
    ]
    trace = build_agent_evaluation_trace(
        agent_name="search_pubmed",
        domain="pubmed",
        original_query="benefit in adults",
        state={
            "expanded_query": "benefit in adults",
            "retrieval_results": documents,
            "reranked_results": documents,
            "synthesis_context": manifest,
            "answer": "Second study found benefit in adults [2].",
            "citations": [
                "[1] First. PMID: 100",
                "[2] Second. PMID: 200",
            ],
            "model_used": "test-model",
        },
        context={"trace_id": "trace-manifest"},
    )

    assert trace.atomic_claims[0].cited_document_ids == [
        manifest[2]["document_id"]
    ]
    assert trace.final_context_spans[0].truncated is True
    assert len(trace.final_context_spans[0].text) == 800


def test_explicit_empty_synthesis_manifest_does_not_restore_raw_evidence():
    trace = build_orchestrator_evaluation_trace(
        {
            "trace_id": "trace-empty-context",
            "input_query": "question",
            "context": {"model_id": "test-model"},
            "discovered_skills": ["search_pubmed"],
            "retrieval_results": {
                "search_pubmed": {
                    "results": [{"abstract": "Raw quarantined evidence."}]
                }
            },
            "synthesis_context": [],
            "intermediate_answer": "No supported answer.",
            "intermediate_model_used": "test-model",
        }
    )

    assert trace.retrieved_documents
    assert trace.final_context_document_ids == []
    assert trace.final_context_spans == []


def test_generated_answer_without_synthesis_manifest_is_invalid():
    document = {
        "text": "Pirfenidone reduced exacerbations in the cited study.",
        "metadata": {"pmid": "manifest-1"},
    }
    trace = build_agent_evaluation_trace(
        agent_name="search_pubmed",
        domain="pubmed",
        original_query="Did pirfenidone reduce exacerbations?",
        state={
            "retrieval_results": [document],
            "reranked_results": [document],
            "answer": "Pirfenidone reduced exacerbations [1].",
            "model_used": "test-model",
        },
        context={"trace_id": "trace-missing-manifest"},
    )

    assert trace.final_context_document_ids == []
    assert trace.final_context_spans == []
    assert "missing_synthesis_manifest" in validate_evaluation_trace(trace)


def test_empty_manifest_cannot_support_a_generated_claim():
    document = {
        "text": "Quarantined evidence says pirfenidone reduced exacerbations.",
        "metadata": {"pmid": "manifest-2"},
    }
    trace = build_agent_evaluation_trace(
        agent_name="search_pubmed",
        domain="pubmed",
        original_query="Did pirfenidone reduce exacerbations?",
        state={
            "retrieval_results": [document],
            "reranked_results": [document],
            "synthesis_context": [],
            "answer": "Pirfenidone reduced exacerbations [1].",
            "model_used": "test-model",
        },
        context={"trace_id": "trace-empty-generated-manifest"},
    )

    assert trace.final_context_spans == []
    assert "empty_synthesis_manifest" in validate_evaluation_trace(trace)


def test_phantom_manifest_entry_cannot_rebind_numeric_citation():
    document = {
        "document_id": "real-document",
        "text": "The real document discusses aspirin.",
        "score": 0.9,
    }
    trace = build_agent_evaluation_trace(
        agent_name="search_pubmed",
        domain="pubmed",
        original_query="What supports the claim?",
        state={
            "retrieval_results": [document],
            "reranked_results": [document],
            "synthesis_context": [
                {
                    "document_id": "phantom-document",
                    "text": "Phantom evidence supports the claim.",
                    "citation_marker": 1,
                },
                {
                    "document_id": "real-document",
                    "text": document["text"],
                    "citation_marker": 2,
                },
            ],
            "answer": "Phantom evidence supports the claim [1].",
            "model_used": "test-model",
        },
        context={"trace_id": "trace-phantom-manifest"},
    )

    assert trace.final_context_document_ids == [
        "phantom-document",
        "real-document",
    ]
    assert trace.atomic_claims[0].cited_document_ids == ["phantom-document"]
    assert any(
        error == "context_document_not_retrieved:phantom-document"
        for error in validate_evaluation_trace(trace)
    )


def test_retrieval_only_trace_without_generated_answer_may_omit_manifest():
    document = {
        "text": "Retrieved evidence only.",
        "metadata": {"pmid": "manifest-3"},
    }
    trace = build_agent_evaluation_trace(
        agent_name="search_pubmed",
        domain="pubmed",
        original_query="Retrieve evidence.",
        state={
            "retrieval_results": [document],
            "reranked_results": [document],
            "answer": "",
            "model_used": "",
        },
        context={
            "trace_id": "trace-retrieval-only-no-manifest",
            "retrieval_only": True,
        },
    )

    errors = validate_evaluation_trace(trace)
    assert "missing_synthesis_manifest" not in errors
    assert "empty_synthesis_manifest" not in errors


def test_trace_validation_rejects_context_not_derived_from_source():
    trace = build_agent_evaluation_trace(
        agent_name="search_pubmed",
        domain="pubmed",
        original_query="question",
        state=_graph_result("pubmed"),
        context={"trace_id": "trace-tampered"},
    )
    trace.final_context_spans[0].text = "Tampered context."
    trace.final_context_spans[0].end_char = len("Tampered context.")
    trace.final_context_spans[0].content_hash = content_hash("Tampered context.")

    assert "context_span_source_mismatch" in validate_evaluation_trace(trace)


def test_trace_validation_recomputes_retrieved_document_hash():
    trace = build_agent_evaluation_trace(
        agent_name="search_pubmed",
        domain="pubmed",
        original_query="question",
        state=_graph_result("pubmed"),
        context={"trace_id": "trace-document-tamper"},
    )
    trace.retrieved_documents[0].text = "Modified document text."

    assert "retrieved_document_hash_mismatch" in validate_evaluation_trace(trace)


def test_trace_validation_rejects_modified_span_hash_and_invalid_offsets():
    trace = build_agent_evaluation_trace(
        agent_name="search_pubmed",
        domain="pubmed",
        original_query="question",
        state=_graph_result("pubmed"),
        context={"trace_id": "trace-span-tamper"},
    )
    trace.final_context_spans[0].content_hash = "invalid"
    trace.final_context_spans[0].end_char = len(trace.retrieved_documents[0].text) + 1

    errors = validate_evaluation_trace(trace)

    assert "context_span_hash_mismatch" in errors
    assert "context_span_offsets_invalid" in errors


def test_claim_citation_to_retrieved_but_excluded_document_is_invalid():
    documents = [
        {"document_id": "included", "text": "Included evidence.", "score": 0.9},
        {"document_id": "excluded", "text": "Excluded evidence.", "score": 0.8},
    ]
    trace = build_agent_evaluation_trace(
        agent_name="search_pubmed",
        domain="pubmed",
        original_query="question",
        state={
            "retrieval_results": documents,
            "reranked_results": documents,
            "synthesis_context": [
                {
                    "document_id": "included",
                    "text": "Included evidence.",
                    "start_char": 0,
                    "original_length": len("Included evidence."),
                    "citation_marker": 1,
                }
            ],
                "answer": "A factual claim exists [1].",
            "model_used": "test-model",
        },
        context={"trace_id": "trace-excluded-citation"},
    )
    trace.atomic_claims[0].cited_document_ids = ["excluded"]

    assert (
        f"claim_citation_not_in_final_context:{trace.atomic_claims[0].claim_id}:excluded"
        in validate_evaluation_trace(trace)
    )


def test_orchestrator_trace_propagates_partial_and_terminal_errors():
    state = {
        "trace_id": "trace-errors",
        "input_query": "question",
        "context": {"model_id": "test-model"},
        "discovered_skills": [],
        "retrieval_results": {},
        "synthesis_context": [],
        "intermediate_answer": "Evidence is limited.",
        "intermediate_model_used": "test-model",
        "error_occurred": True,
        "error_messages": ["retrieval timeout", "verifier unavailable"],
        "is_partial_response": True,
    }

    trace = build_orchestrator_evaluation_trace(state)

    assert trace.partial_response is True
    assert trace.errors == ["retrieval timeout", "verifier unavailable"]


def test_orchestrator_trace_records_actual_truncated_context_span():
    full_text = "evidence " * 400
    state = {
        "trace_id": "trace-context",
        "input_query": "question",
        "context": {"model_id": "test-model"},
        "discovered_skills": ["search_pubmed"],
        "retrieval_results": {
            "search_pubmed": {
                "results": [
                    {
                        "document_id": "search_pubmed:doc:1",
                        "title": "Long source",
                        "abstract": full_text,
                    }
                ]
            }
        },
        "synthesis_context": [
            {
                "document_id": "search_pubmed:doc:1",
                "text": full_text[:100],
                "original_length": len(full_text),
                "truncated": True,
            }
        ],
        "intermediate_answer": "An answer [1].",
        "intermediate_model_used": "test-model",
        "total_retrieval_time_sec": 0.1,
        "synthesis_time_sec": 0.1,
        "synthesis_tokens_in": 0,
        "synthesis_tokens_out": 0,
        "cost_estimate": 0.0,
    }

    trace = build_orchestrator_evaluation_trace(state)

    assert trace.final_context_spans[0].text == full_text[:100]
    assert trace.final_context_spans[0].truncated is True
    assert trace.final_context_spans[0].original_length == len(full_text)


def test_orchestrator_trace_preserves_exact_manifest_start_offset():
    full_text = "Repeated evidence. Padding. Repeated evidence."
    excerpt = "Repeated evidence."
    start_char = full_text.rindex(excerpt)
    state = {
        "trace_id": "trace-context-offset",
        "input_query": "question",
        "context": {"model_id": "test-model"},
        "discovered_skills": ["search_pubmed"],
        "retrieval_results": {
            "search_pubmed": {
                "results": [
                    {
                        "document_id": "search_pubmed:doc:offset",
                        "abstract": full_text,
                    }
                ]
            }
        },
        "synthesis_context": [
            {
                "document_id": "search_pubmed:doc:offset",
                "text": excerpt,
                "start_char": start_char,
                "original_length": len(full_text),
                "truncated": True,
            }
        ],
        "intermediate_answer": "An answer [1].",
        "intermediate_model_used": "test-model",
    }

    trace = build_orchestrator_evaluation_trace(state)

    assert trace.final_context_spans[0].start_char == start_char
    assert trace.final_context_spans[0].end_char == (
        start_char + len(excerpt)
    )
    assert validate_evaluation_trace(trace) == []


def test_real_agent_graph_records_query_expansion_and_synthesis_telemetry():
    class _DualCallLLM:
        default_model = "configured-model"

        def __init__(self):
            self._history = []

        def chat_with_metadata(self, **_kwargs):
            stage = str(_kwargs.get("_telemetry_stage") or "agent_llm_call")
            tokens = 5 if stage == "agent_query_expansion" else 17
            result = LLMCallResult(
                text=(
                    "expanded query"
                    if stage == "agent_query_expansion"
                    else "The exact evidence supports the answer [1]."
                ),
                model=f"{stage}-model",
                model_revision=f"{stage}-revision",
                tokens_in=tokens,
                tokens_out=3,
                cost_usd=0.001 if stage == "agent_query_expansion" else 0.004,
                latency_sec=0.05 if stage == "agent_query_expansion" else 0.12,
                finish_reason="stop",
                provider_metadata={
                    "provider": "mock",
                    "telemetry_stage": stage,
                },
            )
            self._history.append(result)
            return result

        def chat(self, **kwargs):
            return self.chat_with_metadata(**kwargs).text

        def thread_call_history(self):
            return list(self._history)

    class _ExpansionTelemetryAgent(_StructuredTelemetryAgent):
        def _expand_query_node(self, state):
            from agents.base import llm_telemetry_kwargs

            self.llm.chat(
                messages=[{"role": "user", "content": state["input_query"]}],
                **llm_telemetry_kwargs(state, "agent_query_expansion"),
            )
            return {"expanded_query": f"{state['input_query']} expanded"}

    agent = _ExpansionTelemetryAgent()
    agent._llm = _DualCallLLM()
    output = agent.invoke(
        "What does the evidence show?",
        {
            "trace_id": "telemetry-trace",
            "attempt_id": "telemetry-attempt-1",
        },
    )
    trace = output.evaluation_trace

    assert trace is not None
    assert trace.token_usage == {"input": 22, "output": 6, "total": 28}
    assert len(trace.attempt_events) == 2
    stages = {event["stage"] for event in trace.attempt_events}
    assert stages == {"agent_query_expansion", "agent_synthesis"}
    assert validate_evaluation_trace(trace) == []


def test_pubmed_manifest_uses_unresolved_marker_not_rank():
    from agents.pubmed_agent.graph import PubMedAgentGraph

    agent = PubMedAgentGraph.__new__(PubMedAgentGraph)
    manifest = agent._build_synthesis_context(
        [
            {
                "text": "Evidence without matching PMID.",
                "metadata": {"pmid": "99999999"},
            }
        ],
        citations=["1. Unrelated citation PMID: 11111111."],
    )

    assert manifest[0]["citation_marker"] == "?"
