import asyncio
import importlib
import threading
import time
from datetime import datetime

import pytest

from agents.base import AgentOutput
from edges import after_evaluate_coherence
from evaluation_core import (
    VerificationDecision,
    build_agent_evaluation_trace,
    build_orchestrator_evaluation_trace,
    stable_document_id,
    validate_evaluation_trace,
)
from llm_client import LLMCallResult
from nodes.evaluate_coherence import evaluate_coherence
from nodes.fallback_regen import fallback_regen
from nodes.format_response import format_response
from nodes.parallel_retrieve import (
    _agent_output_to_mcp_result,
    _apply_invocation_telemetry,
    invoke_agent_with_timeout,
    invoke_tool_with_timeout,
    parallel_retrieve,
)
from nodes.synthesise import synthesise
from runtime_verification import (
    ATTEMPT_EVENT_SCHEMA_VERSION,
    RuntimeVerifier,
    build_attempt_event,
    build_evidence_context,
    build_retry_request,
    repair_agent_synthesis,
)
from runtime_verification.executor import BoundedExecutor, ExecutorSaturatedError


QUERY = "Do GLP-1 drugs reduce cardiovascular events in adults?"
EVIDENCE = "GLP-1 therapy reduced cardiovascular events by 20% in adults."


def test_repeated_provider_stage_calls_receive_unique_event_ids():
    trace = _agent_output().evaluation_trace
    assert trace is not None
    trace.attempt_events = []
    calls = [
        LLMCallResult(
            text="",
            model="model",
            model_revision="r1",
            tokens_in=index,
            tokens_out=1,
            cost_usd=0.001,
            latency_sec=0.01,
            finish_reason="stop",
            provider_metadata={
                "telemetry_stage": "agent_query_extraction",
                "telemetry_attempt_id": trace.attempt_id,
            },
        )
        for index in (2, 3)
    ]

    _apply_invocation_telemetry(
        trace,
        {
            "calls": 2,
            "tokens_in": 5,
            "tokens_out": 2,
            "cost_usd": 0.002,
            "latency_sec": 0.02,
            "call_results": calls,
        },
    )

    assert len(trace.attempt_events) == 2
    assert len({event["event_id"] for event in trace.attempt_events}) == 2


def test_invocation_telemetry_replaces_lossy_agent_summary_event():
    trace = _agent_output().evaluation_trace
    assert trace is not None
    trace.attempt_events = [
        build_attempt_event(
            trace_id=trace.trace_id,
            attempt_id=trace.attempt_id,
            parent_attempt_id=None,
            stage="agent_synthesis",
            component="pubmed",
            status="success",
            repair_status="initial",
        )
    ]
    calls = [
        LLMCallResult(
            text="",
            model="model",
            model_revision="r1",
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            latency_sec=0.01,
            finish_reason="error",
            provider_metadata={
                "telemetry_stage": "agent_synthesis",
                "telemetry_attempt_id": trace.attempt_id,
                "provider_attempt": 1,
            },
            status="error",
            error_type="RuntimeError",
        ),
        LLMCallResult(
            text="answer",
            model="model",
            model_revision="r1",
            tokens_in=3,
            tokens_out=2,
            cost_usd=0.001,
            latency_sec=0.02,
            finish_reason="stop",
            provider_metadata={
                "telemetry_stage": "agent_synthesis",
                "telemetry_attempt_id": trace.attempt_id,
                "provider_attempt": 2,
            },
        ),
    ]

    _apply_invocation_telemetry(
        trace,
        {
            "calls": 2,
            "tokens_in": 3,
            "tokens_out": 2,
            "cost_usd": 0.001,
            "latency_sec": 0.03,
            "call_results": calls,
        },
    )

    events = [
        event
        for event in trace.attempt_events
        if event["stage"] == "agent_synthesis"
    ]
    assert [event["status"] for event in events] == [
        "error",
        "success",
    ]


def _agent_output(
    *,
    tool_name="search_pubmed",
    evidence=EVIDENCE,
    answer="GLP-1 therapy reduced cardiovascular events by 20% in adults [1].",
    attempt_id="attempt-1",
    parent_attempt_id=None,
    tokens=12,
    cost=0.03,
):
    sources = []
    if evidence is not None:
        sources = [
            {
                "doc_id": "doc-1",
                "text": evidence,
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
    state = {
        "expanded_query": QUERY,
        "retrieval_results": sources,
        "reranked_results": sources,
        "answer": answer,
        "citations": ["Outcome trial"] if sources else [],
        "model_used": "test-model@revision-1",
        "retrieval_time_sec": 0.2,
        "execution_time_sec": 0.4,
        "token_usage": {"total": tokens},
        "cost_breakdown_usd": {"generation": cost},
        "error": None,
    }
    if sources:
        state["synthesis_context"] = [
            {
                "document_id": stable_document_id(
                    sources[0], "pubmed", 1
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
    trace = build_agent_evaluation_trace(
        agent_name=tool_name,
        domain="pubmed",
        original_query=QUERY,
        state=state,
        context={
            "trace_id": "trace-1",
            "attempt_id": attempt_id,
            "parent_attempt_id": parent_attempt_id,
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
        execution_time_sec=0.4,
        evaluation_trace=trace,
    )


class _SequenceAgent:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def invoke(self, query, context):
        self.calls.append((query, dict(context)))
        index = min(len(self.calls) - 1, len(self.outputs) - 1)
        output = self.outputs[index]
        if output.evaluation_trace is not None and len(self.calls) > 1:
            output.evaluation_trace.attempt_id = context["attempt_id"]
            output.evaluation_trace.parent_attempt_id = context["parent_attempt_id"]
            output.evaluation_trace.original_query = context["original_query"]
            output.evaluation_trace.retry_feedback = list(
                context["verification_feedback"]
            )
        return output


def test_successful_retry_replaces_result_and_preserves_attempt_history():
    agent = _SequenceAgent(
        [
            _agent_output(evidence=None, answer="No evidence was found."),
            _agent_output(attempt_id="attempt-2", parent_attempt_id="attempt-1"),
        ]
    )

    result = asyncio.run(
        invoke_agent_with_timeout(
            tool_name="search_pubmed",
            agent_graph=agent,
            query=QUERY,
            context={"top_k": 5},
            timeout_sec=2.0,
            trace_id="trace-1",
            verifier=RuntimeVerifier(),
            max_retries=1,
            deadline_at=time.monotonic() + 5,
        )
    )

    assert len(agent.calls) == 2
    retry_query, retry_context = agent.calls[1]
    assert retry_query != QUERY
    assert retry_context["original_query"] == QUERY
    assert retry_context["top_k"] > 5
    assert retry_context["verification_feedback"]
    assert result["agent_answer"].startswith("GLP-1 therapy")
    assert result["verification_decision"]["status"] == "accept"
    assert len(result["evaluation_traces"]) == 2
    assert result["evaluation_traces"][1]["parent_attempt_id"]
    assert result["repair_history"][0]["target_stage"] == "retrieval"


def test_only_failed_agent_retries(monkeypatch):
    weak_then_strong = _SequenceAgent(
        [
            _agent_output(
                tool_name="search_pubmed",
                evidence=None,
                answer="No evidence was found.",
            ),
            _agent_output(tool_name="search_pubmed", attempt_id="attempt-2"),
        ]
    )
    always_strong = _SequenceAgent(
        [_agent_output(tool_name="search_fda", attempt_id="fda-attempt-1")]
    )
    agents = {
        "search_pubmed": weak_then_strong,
        "search_fda": always_strong,
    }
    parallel_module = importlib.import_module("nodes.parallel_retrieve")
    monkeypatch.setattr(parallel_module, "_get_agent_graph", lambda name: agents[name])
    state = {
        "trace_id": "trace-1",
        "input_query": QUERY,
        "context": {
            "top_k": 5,
            "runtime_verification_deadline_sec": 5,
            "max_agent_retries": 1,
        },
        "discovered_skills": ["search_pubmed", "search_fda"],
        "retrieval_results": {},
        "tokens_used": {},
        "retrieval_time_sec": {},
        "error_messages": [],
        "error_occurred": False,
        "is_partial_response": False,
    }

    result = parallel_retrieve(state)

    assert len(weak_then_strong.calls) == 2
    assert len(always_strong.calls) == 1
    assert len(result["evaluation_traces"]) == 3


def test_retry_exhaustion_terminates_safely():
    agent = _SequenceAgent(
        [_agent_output(evidence=None, answer="No evidence was found.")]
    )

    result = asyncio.run(
        invoke_agent_with_timeout(
            tool_name="search_pubmed",
            agent_graph=agent,
            query=QUERY,
            context={"top_k": 5},
            timeout_sec=2.0,
            trace_id="trace-1",
            verifier=RuntimeVerifier(),
            max_retries=1,
            deadline_at=time.monotonic() + 5,
        )
    )

    assert len(agent.calls) == 2
    assert result["verification_decision"]["status"] == "evidence_limited"
    assert result["evidence_limited"] is True


def test_model_cost_latency_and_trace_metadata_survive_retry():
    agent = _SequenceAgent(
        [
            _agent_output(evidence=None, answer="No evidence was found."),
            _agent_output(tokens=44, cost=0.07, attempt_id="attempt-2"),
        ]
    )

    result = asyncio.run(
        invoke_agent_with_timeout(
            tool_name="search_pubmed",
            agent_graph=agent,
            query=QUERY,
            context={"top_k": 5},
            timeout_sec=2.0,
            trace_id="trace-1",
            verifier=RuntimeVerifier(),
            max_retries=1,
            deadline_at=time.monotonic() + 5,
        )
    )
    final_trace = result["evaluation_traces"][-1]

    assert final_trace["exact_model"] == "test-model"
    assert final_trace["model_revision"] == "revision-1"
    assert final_trace["token_usage"]["total"] == 44
    assert final_trace["cost_breakdown_usd"]["generation"] == 0.07
    assert final_trace["stage_latency_sec"]["total"] == 0.4
    assert result["tokens_used"] == 56
    assert result["cost"] == pytest.approx(0.10)
    assert [
        item["stage"] for item in result["attempt_telemetry"]
    ] == [
        "agent_generation_legacy",
        "retrieval",
        "agent_generation_legacy",
        "retrieval",
    ]
    assert [
        item["token_total"] for item in result["attempt_telemetry"]
    ] == [12, 0, 44, 0]
    assert result["attempt_compute_time_sec"] == pytest.approx(0.8)


def _top_level_state(answer, *, with_results=True):
    results = (
        [
            {
                "title": "Outcome trial",
                "abstract": EVIDENCE,
                "score": 0.9,
                "pmid": "123",
                "year": 2024,
            }
        ]
        if with_results
        else []
    )
    synthesis_context = []
    if results:
        synthesis_context = [
            {
                "document_id": stable_document_id(
                    results[0], "search_pubmed", 1
                ),
                "text": EVIDENCE,
                "original_length": len(EVIDENCE),
                "truncated": False,
                "citation_index": 1,
            }
        ]
    return {
        "trace_id": "trace-1",
        "input_query": QUERY,
        "context": {"model_id": "test-model", "max_synthesis_repairs": 1},
        "discovered_skills": ["search_pubmed"],
        "retrieval_results": {
            "search_pubmed": {
                "results": results,
                "error": None,
                "tokens_used": 0,
                "retrieval_time_sec": 0.2,
            }
        },
        "tokens_used": {"search_pubmed": 0},
        "retrieval_time_sec": {"search_pubmed": 0.2},
        "total_retrieval_time_sec": 0.2,
        "intermediate_answer": answer,
        "synthesis_context": synthesis_context,
        "intermediate_sources": ["search_pubmed"] if results else [],
        "intermediate_model_used": "test-model",
        "synthesis_tokens_in": 0,
        "synthesis_tokens_out": 0,
        "synthesis_time_sec": 0.1,
        "coherence_score": 0.0,
        "coherence_explanation": "",
        "should_fallback": False,
        "fallback_count": 0,
        "fallback_answer": None,
        "fallback_triggered": False,
        "fallback_reason": "",
        "evaluation_traces": [],
        "verification_history": [],
        "repair_history": [],
        "confidence_components": {},
        "confidence_score": 0.0,
        "coverage_explanation": "",
        "error_occurred": False,
        "error_messages": [],
        "is_partial_response": False,
        "evidence_limited": False,
        "cost_estimate": 0.0,
    }


def test_invalid_citations_trigger_synthesis_repair():
    state = _top_level_state(
        "GLP-1 therapy reduced cardiovascular events by 20% in adults [9]."
    )

    result = evaluate_coherence(state)

    assert result["should_fallback"] is True
    assert result["verification_decision"]["status"] == "retry_synthesis"
    assert "unresolved_citation" in result["verification_decision"]["failed_checks"]


def test_final_repair_prompt_contains_original_evidence(monkeypatch):
    captured = {}

    class _FakeLLM:
        def chat(self, messages, **_kwargs):
            captured["messages"] = messages
            return (
                "GLP-1 therapy reduced cardiovascular events by 20% "
                "in adults [1]."
            )

    fallback_module = importlib.import_module("nodes.fallback_regen")
    monkeypatch.setattr(fallback_module, "LLMClient", lambda: _FakeLLM())
    state = _top_level_state(
        "GLP-1 therapy reduced cardiovascular events by 20% in adults [9]."
    )
    state = evaluate_coherence(state)

    result = fallback_regen(state)

    prompt = "\n".join(message["content"] for message in captured["messages"])
    assert EVIDENCE in prompt
    assert "unresolved_citation" in prompt
    assert result["fallback_triggered"] is True
    assert result["fallback_count"] == 1
    assert result["verification_decision"]["status"] == "accept"


def test_top_level_synthesis_and_repair_accumulate_structured_telemetry(
    monkeypatch,
):
    class _InitialLLM:
        def chat_with_metadata(self, **_kwargs):
            return LLMCallResult(
                text=(
                    "GLP-1 therapy reduced cardiovascular events by 20% "
                    "in adults [9]."
                ),
                model="initial-model",
                model_revision="initial-r1",
                tokens_in=10,
                tokens_out=5,
                cost_usd=0.01,
                latency_sec=0.04,
                finish_reason="stop",
                provider_metadata={"provider": "mock"},
            )

    class _RepairLLM:
        def chat_with_metadata(self, **_kwargs):
            return LLMCallResult(
                text=(
                    "GLP-1 therapy reduced cardiovascular events by 20% "
                    "in adults [1]."
                ),
                model="repair-model",
                model_revision="repair-r2",
                tokens_in=6,
                tokens_out=4,
                cost_usd=0.02,
                latency_sec=0.03,
                finish_reason="stop",
                provider_metadata={"provider": "mock"},
            )

    synthesis_module = importlib.import_module("nodes.synthesise")
    fallback_module = importlib.import_module("nodes.fallback_regen")
    monkeypatch.setattr(synthesis_module, "LLMClient", lambda: _InitialLLM())
    monkeypatch.setattr(fallback_module, "LLMClient", lambda: _RepairLLM())
    state = _top_level_state("")

    state = synthesise(state)
    state = evaluate_coherence(state)
    result = fallback_regen(state)
    repair_trace = result["evaluation_traces"][-1]

    assert result["synthesis_tokens_in"] == 16
    assert result["synthesis_tokens_out"] == 9
    assert result["cost_estimate"] == pytest.approx(0.03)
    assert result["intermediate_model_used"] == "repair-model@repair-r2"
    assert repair_trace["exact_model"] == "repair-model"
    assert repair_trace["model_revision"] == "repair-r2"
    assert repair_trace["token_usage"]["total"] == 10
    assert repair_trace["cost_breakdown_usd"]["repair"] == pytest.approx(0.02)
    assert len(result["attempt_telemetry"]) == 2
    assert {
        event["schema_version"] for event in result["attempt_telemetry"]
    } == {ATTEMPT_EVENT_SCHEMA_VERSION}
    assert [event["stage"] for event in result["attempt_telemetry"]] == [
        "top_level_synthesis",
        "top_level_synthesis_repair",
    ]
    assert [event["repair_status"] for event in result["attempt_telemetry"]] == [
        "initial",
        "synthesis_repair",
    ]
    assert sum(
        event["token_usage"]["total"]
        for event in result["attempt_telemetry"]
    ) == 25
    assert sum(
        event["cost_usd"] for event in result["attempt_telemetry"]
    ) == pytest.approx(0.03)


def test_top_level_synthesis_records_failed_provider_retry(
    monkeypatch,
):
    calls = {"count": 0}

    class _Message:
        content = (
            "GLP-1 therapy reduced cardiovascular events by 20% "
            "in adults [1]."
        )

    class _Choice:
        message = _Message()
        finish_reason = "stop"

    class _Usage:
        prompt_tokens = 3
        completion_tokens = 2

    class _Response:
        choices = [_Choice()]
        usage = _Usage()
        model = "test-model@r1"
        id = "response-1"

    def flaky_completion(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("503 temporarily unavailable")
        return _Response()

    monkeypatch.setattr("llm_client.litellm.completion", flaky_completion)
    monkeypatch.setattr("llm_client.time.sleep", lambda _delay: None)

    result = synthesise(_top_level_state(""))

    events = [
        event
        for event in result["attempt_telemetry"]
        if event["stage"] == "top_level_synthesis"
    ]
    assert [event["status"] for event in events] == [
        "error",
        "success",
    ]
    assert [
        event["provider_metadata"]["provider_attempt"]
        for event in events
    ] == [1, 2]


def test_top_level_repair_records_failed_provider_retry(monkeypatch):
    calls = {"count": 0}

    class _Message:
        content = (
            "GLP-1 therapy reduced cardiovascular events by 20% "
            "in adults [1]."
        )

    class _Choice:
        message = _Message()
        finish_reason = "stop"

    class _Usage:
        prompt_tokens = 3
        completion_tokens = 2

    class _Response:
        choices = [_Choice()]
        usage = _Usage()
        model = "test-model@r2"
        id = "repair-response-1"

    def flaky_completion(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("503 temporarily unavailable")
        return _Response()

    monkeypatch.setattr("llm_client.litellm.completion", flaky_completion)
    monkeypatch.setattr("llm_client.time.sleep", lambda _delay: None)
    state = _top_level_state(
        "GLP-1 therapy reduced cardiovascular events [9]."
    )
    state = evaluate_coherence(state)

    result = fallback_regen(state)

    events = [
        event
        for event in result["attempt_telemetry"]
        if event["stage"] == "top_level_synthesis_repair"
    ]
    assert [event["status"] for event in events] == [
        "error",
        "success",
    ]
    assert [
        event["provider_metadata"]["provider_attempt"]
        for event in events
    ] == [1, 2]


def test_empty_retrieval_returns_evidence_limited_without_llm(monkeypatch):
    class _ForbiddenLLM:
        def chat(self, *_args, **_kwargs):
            raise AssertionError("LLM must not be called without evidence")

    synthesis_module = importlib.import_module("nodes.synthesise")
    monkeypatch.setattr(synthesis_module, "LLMClient", lambda: _ForbiddenLLM())
    state = _top_level_state("", with_results=False)

    result = synthesise(state)

    assert result["evidence_limited"] is True
    assert "insufficient evidence" in result["intermediate_answer"].lower()
    assert result["intermediate_sources"] == []


def test_synthesis_and_trace_use_the_same_document_text_field():
    state = _top_level_state("Canonical full text supports the answer [1].")
    state["retrieval_results"]["search_pubmed"]["results"] = [
        {
            "document_id": "doc-1",
            "text": "Canonical full text supports the answer.",
            "abstract": "A different abstract must not be selected.",
            "title": "Study",
        }
    ]

    _, _, manifest = build_evidence_context(state)
    state["synthesis_context"] = manifest
    trace = build_orchestrator_evaluation_trace(state)

    assert manifest[0]["text"] == "Canonical full text supports the answer."
    assert validate_evaluation_trace(trace) == []


def test_unrouteable_final_retrieval_failure_replaces_rejected_answer():
    state = _top_level_state("Aspirin reduced headache severity [1].")
    state["retrieval_results"]["search_pubmed"]["results"][0][
        "abstract"
    ] = "Aspirin reduced headache severity in adults."
    rejected_answer = state["intermediate_answer"]

    result = evaluate_coherence(state)

    assert result["verification_decision"]["status"] == "evidence_limited"
    assert result["evidence_limited"] is True
    assert result["intermediate_answer"] != rejected_answer
    assert "insufficient evidence" in result["intermediate_answer"].lower()


def test_runtime_quality_does_not_overwrite_legacy_coverage_confidence():
    state = _top_level_state(
        "GLP-1 therapy reduced cardiovascular events by 20% in adults [1]."
    )
    state["confidence_score"] = 1.0
    state["coverage_explanation"] = "1/1 tools returned results."

    result = evaluate_coherence(state)

    assert result["confidence_score"] == 1.0
    assert result["coverage_explanation"] == "1/1 tools returned results."
    assert 0.0 <= result["runtime_quality_score"] <= 1.0
    assert result["runtime_quality_explanation"]


def test_failed_repair_exit_records_terminal_decision_and_attempt(monkeypatch):
    class _FailingLLM:
        def chat(self, *_args, **_kwargs):
            raise RuntimeError("provider unavailable")

    fallback_module = importlib.import_module("nodes.fallback_regen")
    monkeypatch.setattr(fallback_module, "LLMClient", lambda: _FailingLLM())
    state = _top_level_state(
        "GLP-1 therapy reduced cardiovascular events by 20% in adults [9]."
    )
    state = evaluate_coherence(state)

    result = fallback_regen(state)

    decision = result["verification_decision"]
    assert decision["status"] == "evidence_limited"
    assert decision["valid"] is False
    assert decision["target_stage"] == "none"
    assert "repair_failure" in decision["failed_checks"]
    assert result["fallback_triggered"] is False
    assert len(result["evaluation_traces"]) == 2
    assert result["repair_history"][-1]["target_stage"] == "synthesis"


def test_invalid_verifier_output_is_quarantined_from_synthesis():
    state = _top_level_state(
        "GLP-1 therapy reduced cardiovascular events by 20% in adults [1]."
    )
    result = state["retrieval_results"]["search_pubmed"]
    result["verification_decision"] = {
        "status": "evidence_limited",
        "valid": False,
    }
    result["evidence_limited"] = True

    evidence_text, sources, included = build_evidence_context(state)

    assert evidence_text == ""
    assert sources == []
    assert included == []


def test_mcp_fallback_uses_shared_trace_verifier_and_retry():
    class _Tool:
        def __init__(self):
            self.calls = []

        async def invoke_async(self, query, context):
            self.calls.append((query, dict(context)))
            if len(self.calls) == 1:
                return {"results": [], "error": None}
            return {
                "results": [
                    {
                        "abstract": EVIDENCE,
                        "pmid": "123",
                        "score": 0.9,
                    }
                ],
                "error": None,
            }

    tool = _Tool()
    result = asyncio.run(
        invoke_tool_with_timeout(
            tool_name="search_pubmed_deep",
            tool=tool,
            query=QUERY,
            context={"top_k": 2},
            timeout_sec=1.0,
            trace_id="trace-mcp",
            verifier=RuntimeVerifier(),
            max_retries=1,
            deadline_at=time.monotonic() + 5,
        )
    )

    assert len(tool.calls) == 2
    assert result["verification_decision"]["status"] == "accept"
    assert len(result["evaluation_traces"]) == 2
    assert result["evaluation_traces"][1]["parent_attempt_id"]


def test_mcp_tool_error_payload_cannot_expose_raw_medical_query():
    class _Tool:
        async def invoke_async(self, query, context):
            return {
                "results": [],
                "error": f"provider failed while processing {QUERY}",
            }

    result = asyncio.run(
        invoke_tool_with_timeout(
            tool_name="search_pubmed_deep",
            tool=_Tool(),
            query=QUERY,
            context={"top_k": 2},
            timeout_sec=1.0,
            trace_id="trace-private-error",
            verifier=RuntimeVerifier(),
            max_retries=0,
            deadline_at=time.monotonic() + 5,
        )
    )

    assert result["error"] == "tool_retrieval_error"
    assert all(
        QUERY not in error
        for trace in result["evaluation_traces"]
        for error in trace["errors"]
    )


def test_agent_timeout_is_not_retried_while_worker_may_still_run():
    class _SlowAgent:
        def __init__(self):
            self.calls = 0

        def invoke(self, _query, _context):
            self.calls += 1
            time.sleep(0.3)
            return _agent_output()

    agent = _SlowAgent()
    started_at = time.monotonic()
    result = asyncio.run(
        invoke_agent_with_timeout(
            tool_name="search_pubmed",
            agent_graph=agent,
            query=QUERY,
            context={"top_k": 5},
            timeout_sec=0.01,
            trace_id="trace-timeout",
            verifier=RuntimeVerifier(),
            max_retries=1,
            deadline_at=time.monotonic() + 1,
        )
    )
    elapsed = time.monotonic() - started_at

    assert agent.calls == 1
    assert elapsed < 0.2
    assert result["verification_decision"]["status"] == "evidence_limited"
    assert result["verification_decision"]["target_stage"] == "none"
    assert len(result["evaluation_traces"]) == 1


def test_repeated_timeouts_cannot_create_unbounded_workers():
    parallel_module = importlib.import_module("nodes.parallel_retrieve")
    executor = BoundedExecutor(
        max_workers=2,
        max_queue=0,
        thread_name_prefix="runtime-verifier-stress",
    )
    release = threading.Event()

    async def exercise():
        outcomes = []
        for _ in range(10):
            try:
                await parallel_module._run_blocking_with_timeout(
                    lambda: release.wait(1.0),
                    timeout=0.01,
                    executor=executor,
                )
            except (asyncio.TimeoutError, ExecutorSaturatedError) as exc:
                outcomes.append(type(exc).__name__)
        return outcomes

    outcomes = asyncio.run(exercise())
    snapshot = executor.metrics()
    worker_threads = [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("runtime-verifier-stress")
    ]

    assert len(outcomes) == 10
    assert len(worker_threads) <= 2
    assert snapshot["max_workers"] == 2
    assert snapshot["timed_out"] <= 2
    assert snapshot["rejected"] >= 8
    assert snapshot["in_flight"] <= 2

    release.set()
    executor.shutdown(wait=True, cancel_futures=True)
    assert executor.metrics()["shutdown"] is True


def test_synthesis_passes_remaining_deadline_to_llm(monkeypatch):
    captured = {}

    class _DeadlineAwareLLM:
        def chat(self, _messages=None, **kwargs):
            captured.update(kwargs)
            return (
                "GLP-1 therapy reduced cardiovascular events by 20% "
                "in adults [1]."
            )

    synthesis_module = importlib.import_module("nodes.synthesise")
    monkeypatch.setattr(
        synthesis_module, "LLMClient", lambda: _DeadlineAwareLLM()
    )
    state = _top_level_state("")
    state["context"]["_runtime_deadline_at_monotonic"] = time.monotonic() + 5

    synthesise(state)

    assert 0 < captured["timeout"] <= 5
    assert captured["client_max_attempts"] == 1


def test_agent_synthesis_retry_reuses_frozen_evidence(monkeypatch):
    captured = {}

    class _RepairLLM:
        def chat(self, messages, **_kwargs):
            captured["prompt"] = "\n".join(
                message["content"] for message in messages
            )
            return (
                "GLP-1 therapy reduced cardiovascular events by 20% "
                "in adults [1]."
            )

    repair_module = importlib.import_module("runtime_verification.repair")
    monkeypatch.setattr(repair_module, "LLMClient", lambda: _RepairLLM())
    agent = _SequenceAgent(
        [
            _agent_output(
                answer=(
                    "GLP-1 therapy reduced cardiovascular events by 20% "
                    "in adults [9]."
                )
            )
        ]
    )

    result = asyncio.run(
        invoke_agent_with_timeout(
            tool_name="search_pubmed",
            agent_graph=agent,
            query=QUERY,
            context={"top_k": 5, "model_id": "test-model"},
            timeout_sec=1.0,
            trace_id="trace-agent-repair",
            verifier=RuntimeVerifier(),
            max_retries=1,
            deadline_at=time.monotonic() + 5,
        )
    )

    assert len(agent.calls) == 1
    assert EVIDENCE in captured["prompt"]
    assert result["verification_decision"]["status"] == "accept"
    assert len(result["evaluation_traces"]) == 2
    assert result["evaluation_traces"][1]["parent_attempt_id"]


def test_expired_agent_repair_deadline_does_not_start_provider_call(
    monkeypatch,
):
    class _ForbiddenLLM:
        def chat(self, **_kwargs):
            raise AssertionError("expired repair must not call the provider")

    repair_module = importlib.import_module("runtime_verification.repair")
    monkeypatch.setattr(repair_module, "LLMClient", lambda: _ForbiddenLLM())
    output = _agent_output()
    decision = VerificationDecision(
        status="retry_synthesis",
        component_scores={},
        failed_checks=["unsupported_claim"],
        structured_feedback=[],
        target_stage="synthesis",
        target_agent="search_pubmed",
        recommended_retry_changes={"preserve_evidence": True},
        verifier_confidence=1.0,
    )

    with pytest.raises(TimeoutError, match="deadline expired"):
        repair_agent_synthesis(
            agent_output=output,
            trace=output.evaluation_trace,
            decision=decision,
            original_query=QUERY,
            context={
                "model_id": "test-model",
                "_runtime_deadline_at_monotonic": time.monotonic() - 1,
            },
        )


def test_agent_repair_records_actual_call_telemetry_and_model(monkeypatch):
    class _RepairLLM:
        def chat_with_metadata(self, **_kwargs):
            return LLMCallResult(
                text=(
                    "GLP-1 therapy reduced cardiovascular events by 20% "
                    "in adults [1]."
                ),
                model="repair-model",
                model_revision="revision-2",
                tokens_in=7,
                tokens_out=8,
                cost_usd=0.02,
                latency_sec=0.05,
                finish_reason="stop",
                provider_metadata={"provider": "mock"},
            )

    repair_module = importlib.import_module("runtime_verification.repair")
    monkeypatch.setattr(repair_module, "LLMClient", lambda: _RepairLLM())
    agent = _SequenceAgent(
        [
            _agent_output(
                answer=(
                    "GLP-1 therapy reduced cardiovascular events by 20% "
                    "in adults [9]."
                )
            )
        ]
    )

    result = asyncio.run(
        invoke_agent_with_timeout(
            tool_name="search_pubmed",
            agent_graph=agent,
            query=QUERY,
            context={"top_k": 5, "model_id": "generation-model"},
            timeout_sec=1.0,
            trace_id="trace-repair-telemetry",
            verifier=RuntimeVerifier(),
            max_retries=0,
            max_synthesis_repairs=1,
            deadline_at=time.monotonic() + 5,
        )
    )
    repair_trace = result["evaluation_traces"][-1]

    assert repair_trace["exact_model"] == "repair-model"
    assert repair_trace["model_revision"] == "revision-2"
    assert repair_trace["token_usage"] == {
        "input": 7,
        "output": 8,
        "total": 15,
    }
    assert repair_trace["cost_breakdown_usd"]["repair"] == pytest.approx(0.02)
    assert repair_trace["stage_latency_sec"]["synthesis"] == pytest.approx(0.05)
    assert result["tokens_used"] == 27
    assert result["cost"] == pytest.approx(0.05)


def test_retrieval_retry_does_not_consume_agent_synthesis_repair(monkeypatch):
    class _RepairLLM:
        def chat(self, _messages=None, **_kwargs):
            return (
                "GLP-1 therapy reduced cardiovascular events by 20% "
                "in adults [1]."
            )

    repair_module = importlib.import_module("runtime_verification.repair")
    monkeypatch.setattr(repair_module, "LLMClient", lambda: _RepairLLM())
    agent = _SequenceAgent(
        [
            _agent_output(evidence=None, answer="No evidence was found."),
            _agent_output(
                answer=(
                    "GLP-1 therapy reduced cardiovascular events by 20% "
                    "in adults [9]."
                ),
                attempt_id="attempt-2",
                parent_attempt_id="attempt-1",
            ),
        ]
    )

    result = asyncio.run(
        invoke_agent_with_timeout(
            tool_name="search_pubmed",
            agent_graph=agent,
            query=QUERY,
            context={"top_k": 5, "model_id": "test-model"},
            timeout_sec=1.0,
            trace_id="trace-combined-repair",
            verifier=RuntimeVerifier(),
            max_retries=1,
            max_synthesis_repairs=1,
            deadline_at=time.monotonic() + 5,
        )
    )

    assert len(agent.calls) == 2
    assert result["verification_decision"]["status"] == "accept"
    assert [event["target_stage"] for event in result["repair_history"]] == [
        "retrieval",
        "synthesis",
    ]
    assert len(result["evaluation_traces"]) == 3
    attempt_ids = [trace["attempt_id"] for trace in result["evaluation_traces"]]
    assert len(attempt_ids) == len(set(attempt_ids))
    assert result["evaluation_traces"][2]["parent_attempt_id"] == attempt_ids[1]


def test_retry_policy_applies_allowlisted_changes_and_records_rejections():
    trace = _agent_output().evaluation_trace
    decision = VerificationDecision(
        status="retry_retrieval",
        component_scores={},
        failed_checks=["missing_query_components"],
        structured_feedback=[],
        target_stage="retrieval",
        target_agent="search_pubmed",
        recommended_retry_changes={
            "top_k": 999,
            "query_additions": ["pregnancy", "randomized trial"],
            "retrieval_method": "hybrid_with_sparse_fallback",
            "preserve_original_query": True,
            "authorization_header": "secret",
        },
        verifier_confidence=1.0,
    )

    retry_query, retry_context, event = build_retry_request(
        original_query=QUERY,
        context={"top_k": 5, "retrieval_method": "dense"},
        decision=decision,
        trace=trace,
        attempt_number=1,
    )

    assert "pregnancy" in retry_query
    assert retry_context["top_k"] == 20
    assert retry_context["retrieval_method"] == "hybrid_with_sparse_fallback"
    assert event["recommended_changes"] == decision.recommended_retry_changes
    assert event["applied_changes"]["top_k"] == 20
    assert event["applied_changes"]["retrieval_method"] == (
        "hybrid_with_sparse_fallback"
    )
    assert event["rejected_changes"]["authorization_header"] == "unsupported_key"
    assert event["actual_configuration"] == {
        "query": retry_query,
        "original_query_preserved": True,
        "top_k": 20,
        "retrieval_method": "hybrid_with_sparse_fallback",
    }


def test_retry_policy_rejects_unsupported_retrieval_method():
    trace = _agent_output().evaluation_trace
    decision = VerificationDecision(
        status="retry_retrieval",
        component_scores={},
        failed_checks=["empty_retrieval"],
        structured_feedback=[],
        target_stage="retrieval",
        target_agent="search_pubmed",
        recommended_retry_changes={"retrieval_method": "execute-arbitrary-plugin"},
        verifier_confidence=1.0,
    )

    _, retry_context, event = build_retry_request(
        original_query=QUERY,
        context={"retrieval_method": "dense"},
        decision=decision,
        trace=trace,
        attempt_number=1,
    )

    assert retry_context["retrieval_method"] == "dense"
    assert event["applied_changes"] == {}
    assert event["rejected_changes"]["retrieval_method"] == "unsupported_value"
    assert event["actual_configuration"]["retrieval_method"] == "dense"
    assert event["actual_configuration"]["query"] == QUERY


@pytest.mark.parametrize(
    "query_additions",
    [
        "pregnancy",
        ["x" * 200],
        [f"term-{index}" for index in range(20)],
    ],
)
def test_retry_policy_bounds_and_validates_query_additions(query_additions):
    trace = _agent_output().evaluation_trace
    decision = VerificationDecision(
        status="retry_retrieval",
        component_scores={},
        failed_checks=["missing_query_components"],
        structured_feedback=[],
        target_stage="retrieval",
        target_agent="search_pubmed",
        recommended_retry_changes={"query_additions": query_additions},
        verifier_confidence=1.0,
    )

    retry_query, _, event = build_retry_request(
        original_query=QUERY,
        context={},
        decision=decision,
        trace=trace,
        attempt_number=1,
    )

    assert len(retry_query) <= len(QUERY) + 8 * 81
    assert "query_additions" in event["rejected_changes"]


@pytest.mark.parametrize(
    ("decision", "coherence", "expected"),
    [
        (None, 0.2, "fallback_regen"),
        ({}, 0.2, "fallback_regen"),
        ({"status": "malformed", "valid": True}, 0.2, "fallback_regen"),
        ({"status": "accept", "valid": True}, 0.2, "format_response"),
        (
            {"status": "retry_synthesis", "valid": True},
            0.9,
            "fallback_regen",
        ),
        (
            {"status": "evidence_limited", "valid": True},
            0.2,
            "format_response",
        ),
    ],
)
def test_coherence_routing_only_trusts_recognized_valid_decisions(
    decision, coherence, expected
):
    state = _top_level_state("Supported answer [1].")
    state["coherence_score"] = coherence
    if decision is None:
        state.pop("verification_decision", None)
    else:
        state["verification_decision"] = decision

    assert after_evaluate_coherence(state) == expected


def test_format_response_appends_terminal_trace_matching_output_answer():
    state = _top_level_state(
        "GLP-1 therapy reduced cardiovascular events by 20% in adults [1]."
    )
    state.update(
        {
            "is_medical_query": True,
            "timestamp_start": datetime.utcnow(),
            "token_usage": {"input": 8, "output": 5, "total": 13},
            "cost_estimate": 0.12,
            "verification_decision": {
                "status": "accept",
                "component_scores": {},
                "failed_checks": [],
                "structured_feedback": [],
                "target_stage": "none",
                "target_agent": "orchestrator",
                "recommended_retry_changes": {},
                "verifier_confidence": 1.0,
                "valid": True,
                "error": None,
                "verifier_model": "deterministic",
                "verifier_model_revision": "",
                "prompt_version": "runtime-verifier-v1",
                "raw_decision": {"terminal_reason": "accepted"},
            },
        }
    )
    state = evaluate_coherence(state)

    result = format_response(state)
    final_trace = result["evaluation_traces"][-1]

    assert final_trace["answer"] == result["output_answer"]
    assert final_trace["verification_decisions"][-1]["status"] == "accept"
    assert final_trace["parent_attempt_id"]
    assert final_trace["attempt_id"].endswith(":terminal")
    assert final_trace["trace_role"] == "terminal_delivery"
    assert final_trace["token_usage"]["total"] == 13
    assert final_trace["cost_breakdown_usd"] == {"request_total": 0.12}


def test_terminal_trace_adapter_failure_cannot_break_response(monkeypatch):
    state = _top_level_state(
        "GLP-1 therapy reduced cardiovascular events by 20% in adults [1]."
    )
    state.update(
        {
            "is_medical_query": True,
            "timestamp_start": datetime.utcnow(),
            "token_usage": {"input": 2, "output": 1, "total": 3},
        }
    )
    state = evaluate_coherence(state)
    format_module = importlib.import_module("nodes.format_response")
    monkeypatch.setattr(
        format_module,
        "build_orchestrator_evaluation_trace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("trace adapter failed")
        ),
    )

    result = format_module.format_response(state)

    assert result["output_answer"]
    assert result["evaluation_traces"][-1]["answer"] == result["output_answer"]
    assert result["evaluation_traces"][-1]["trace_role"] == "terminal_delivery"
    assert result["evaluation_traces"][-1]["partial_response"] is True


def test_formatted_references_follow_ordered_synthesis_manifest():
    state = _top_level_state("Supported answer [1].")
    state["retrieval_results"]["search_pubmed"]["results"] = [
        {
            "title": f"Study {index}",
            "abstract": f"Evidence {index}.",
            "authors": ["Author A"],
            "year": 2024,
            "pmid": str(index),
        }
        for index in range(1, 5)
    ]
    _, sources, manifest = build_evidence_context(state)
    state.update(
        {
            "synthesis_context": manifest,
            "intermediate_sources": sources,
            "is_medical_query": True,
            "timestamp_start": datetime.utcnow(),
            "fallback_triggered": False,
            "is_partial_response": False,
        }
    )

    result = format_response(state)

    assert len(manifest) == 3
    assert len(result["output_citations"]) == len(manifest)
    assert "Study 4" not in result["output_answer"]


def test_missing_synthesis_manifest_never_substitutes_retrieved_references():
    state = _top_level_state(
        "GLP-1 therapy reduced cardiovascular events [1]."
    )
    state.pop("synthesis_context", None)
    state.update(
        {
            "is_medical_query": True,
            "timestamp_start": datetime.utcnow(),
        }
    )

    result = format_response(state)

    assert result["verification_decision"]["status"] == "evidence_limited"
    assert result["output_citations"] == []


def test_formatting_exception_path_is_terminal_and_evidence_limited():
    state = _top_level_state(
        "An unverified medical claim should not be delivered [1]."
    )
    state["is_medical_query"] = True
    state.pop("timestamp_start", None)

    result = format_response(state)

    assert result["verification_decision"]["status"] == "evidence_limited"
    assert result["evidence_limited"] is True
    assert "unverified medical claim" not in result["output_answer"].lower()
    assert result["evaluation_traces"][-1]["trace_role"] == "terminal_delivery"


def test_tool_task_creation_failure_still_emits_verified_trace(monkeypatch):
    parallel_module = importlib.import_module("nodes.parallel_retrieve")
    monkeypatch.setattr(parallel_module, "_get_agent_graph", lambda _name: None)
    monkeypatch.setattr(
        parallel_module.mcp_registry,
        "get_tool",
        lambda _name: (_ for _ in ()).throw(KeyError("not registered")),
    )
    state = {
        "trace_id": "trace-missing-tool",
        "input_query": QUERY,
        "context": {"runtime_verification_deadline_sec": 5},
        "discovered_skills": ["missing_tool"],
        "retrieval_results": {},
        "tokens_used": {},
        "retrieval_time_sec": {},
        "error_messages": [],
        "error_occurred": False,
        "is_partial_response": False,
    }

    result = parallel_retrieve(state)
    tool_result = result["retrieval_results"]["missing_tool"]

    assert tool_result["evidence_limited"] is True
    assert tool_result["verification_decision"]["valid"] is False
    assert len(tool_result["evaluation_traces"]) == 1


def test_agent_output_token_total_does_not_double_count_usage_keys():
    output = _agent_output(tokens=150)
    output.evaluation_trace.token_usage = {
        "input": 100,
        "output": 50,
        "total": 150,
        "calls": 3,
    }

    result = _agent_output_to_mcp_result(output, trace=output.evaluation_trace)

    assert result["tokens_used"] == 150


def test_failed_agent_repair_records_provider_telemetry(monkeypatch):
    class _FailingRepairLLM:
        def chat_with_metadata(self, **_kwargs):
            raise RuntimeError("provider unavailable")

        def chat(self, **_kwargs):
            raise RuntimeError("provider unavailable")

    repair_module = importlib.import_module("runtime_verification.repair")
    monkeypatch.setattr(repair_module, "LLMClient", lambda: _FailingRepairLLM())
    agent = _SequenceAgent(
        [
            _agent_output(
                answer=(
                    "GLP-1 therapy reduced cardiovascular events by 20% "
                    "in adults [9]."
                )
            )
        ]
    )

    result = asyncio.run(
        invoke_agent_with_timeout(
            tool_name="search_pubmed",
            agent_graph=agent,
            query=QUERY,
            context={"top_k": 5, "model_id": "generation-model"},
            timeout_sec=1.0,
            trace_id="trace-failed-repair",
            verifier=RuntimeVerifier(),
            max_retries=0,
            max_synthesis_repairs=1,
            deadline_at=time.monotonic() + 5,
        )
    )

    repair_events = [
        event
        for trace in result["evaluation_traces"]
        for event in (trace.get("attempt_events") or [])
        if event.get("stage") == "agent_synthesis_repair"
    ]

    assert repair_events
    assert repair_events[0]["status"] == "error"
    assert repair_events[0]["error_type"] == "RuntimeError"


def test_repair_budget_exhaustion_replaces_answer_before_format():
    state = _top_level_state(
        "Unsupported answer must not be delivered unchanged [1]."
    )
    state["verification_decision"] = {
        "status": "retry_synthesis",
        "valid": True,
    }
    state["fallback_count"] = 1
    state["context"]["max_synthesis_repairs"] = 1

    after_evaluate_coherence(state)

    assert state["evidence_limited"] is True
    assert "Unsupported answer must not be delivered unchanged" not in (
        state["intermediate_answer"]
    )
    assert "insufficient evidence" in state["intermediate_answer"].lower()
