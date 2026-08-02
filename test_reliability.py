import logging
import time

import pytest

from agents.clinical_trials_agent.data_fetcher import (
    ClinicalTrialsFetcher,
    SearchTerms,
)
from agents.clinical_trials_agent.graph import ClinicalTrialsAgentGraph
from agents.fda_agent.data_fetcher import FDAFetcher, FDASearchTerms
from agents.fda_agent.graph import FDAAgentGraph
from agents.pubmed_agent.data_fetcher import (
    PubMedFetcher,
    PubMedSearchTerms,
)
from agents.pubmed_agent.graph import PubMedAgentGraph
from llm_client import LLMCallResult, LLMClient, _call_with_retry
from nodes.classify_intent import _record_classifier_call
from rag_engine.hybrid_retriever import HybridRetriever
from rag_engine.mcp_rag_tool import RAGTool
from rag_engine.sparse_index import SparseResult
from runtime_verification.deadline import RuntimeDeadlineExceeded
from tools.clinical_trials_tool import ClinicalTrialsTool
from tools.fda_tool import FDATool
from tools.pubmed_tool import PubMedTool


class _FakeLLM:
    default_model = "test-model"


def test_pubmed_no_results_returns_valid_graph_state():
    agent = PubMedAgentGraph.__new__(PubMedAgentGraph)
    agent._llm = _FakeLLM()

    result = agent._synthesise_node(
        {
            "input_query": "question",
            "retrieval_results": [],
            "reranked_results": [],
            "fetched_papers": {},
        }
    )

    assert isinstance(result, dict)
    assert result["confidence"] == 0.0
    assert result["citations"] == []
    assert "No relevant PubMed" in result["answer"]


def test_hybrid_retrieval_falls_back_to_sparse_when_dense_fails():
    class BrokenEmbedder:
        def embed(self, query):
            raise RuntimeError("embedding service unavailable")

    class SparseIndex:
        def search(self, query, top_k):
            return [
                SparseResult(
                    doc_id=1,
                    score=2.0,
                    text="relevant evidence",
                    metadata={"source": "test"},
                )
            ]

    retriever = HybridRetriever(
        dense_index=object(),
        sparse_index=SparseIndex(),
        embedder=BrokenEmbedder(),
    )

    results = retriever.retrieve("query", top_k=1)

    assert len(results) == 1
    assert results[0].text == "relevant evidence"
    assert results[0].sparse_rank == 1


def test_transient_provider_errors_are_retried(monkeypatch):
    attempts = []
    sleeps = []

    def flaky_call():
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED retry in 0.01s")
        return "ok"

    monkeypatch.setattr("llm_client.time.sleep", sleeps.append)

    result = _call_with_retry(flaky_call, operation="test", max_attempts=3)

    assert result == "ok"
    assert len(attempts) == 3
    assert len(sleeps) == 2


def test_ollama_batch_embedding_uses_native_batch_endpoint(monkeypatch):
    observed = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embeddings": [[1.0, 0.0], [0.0, 1.0]]}

    def fake_post(url, json, timeout):
        observed.update(url=url, json=json, timeout=timeout)
        return Response()

    monkeypatch.setattr("requests.post", fake_post)
    client = LLMClient()

    vectors = client.embed_batch(
        ["first", "second"], model="ollama/nomic-embed-text"
    )

    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert observed["url"].endswith("/api/embed")
    assert observed["json"]["input"] == ["first", "second"]


def test_litellm_embedding_receives_remaining_provider_timeout(monkeypatch):
    observed = {}

    class _Usage:
        total_tokens = 2

    class _Response:
        data = [{"embedding": [1.0, 0.0]}]
        usage = _Usage()

    def fake_embedding(**kwargs):
        observed.update(kwargs)
        return _Response()

    monkeypatch.setattr("llm_client.litellm.embedding", fake_embedding)
    deadline_at = time.monotonic() + 2.0
    client = LLMClient()
    history_start = len(client.thread_call_history())

    vector = client.embed(
        "medical query",
        model="openai/test-embedding",
        deadline_at=deadline_at,
        client_max_attempts=1,
    )

    assert vector == [1.0, 0.0]
    assert 0 < observed["timeout"] <= 2.0
    call = client.thread_call_history()[history_start:][-1]
    assert call.model == "openai/test-embedding"
    assert call.tokens_in == 2
    assert call.tokens_out == 0
    assert call.finish_reason == "embedded"
    assert call.provider_metadata["telemetry_stage"] == "embedding"


def test_embedding_retry_records_each_physical_provider_attempt(
    monkeypatch,
):
    calls = {"count": 0}

    class _Usage:
        total_tokens = 2

    class _Response:
        data = [{"embedding": [1.0, 0.0]}]
        usage = _Usage()

    def flaky_embedding(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("503 temporarily unavailable")
        return _Response()

    monkeypatch.setattr("llm_client.litellm.embedding", flaky_embedding)
    monkeypatch.setattr("llm_client.time.sleep", lambda _delay: None)
    client = LLMClient()
    history_start = len(client.thread_call_history())

    assert client.embed(
        "medical query",
        model="openai/test-embedding",
        client_max_attempts=2,
    ) == [1.0, 0.0]

    attempts = client.thread_call_history()[history_start:]
    assert [attempt.status for attempt in attempts] == [
        "error",
        "success",
    ]
    assert [
        attempt.provider_metadata["provider_attempt"]
        for attempt in attempts
    ] == [1, 2]


def test_retry_telemetry_keeps_repeated_exception_instance_attempts(
    monkeypatch,
):
    calls = {"count": 0}
    shared_error = RuntimeError("503 temporarily unavailable")

    class _Response:
        data = [{"embedding": [1.0]}]
        usage = None

    def flaky_embedding(**_kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise shared_error
        return _Response()

    monkeypatch.setattr("llm_client.litellm.embedding", flaky_embedding)
    monkeypatch.setattr("llm_client.time.sleep", lambda _delay: None)
    client = LLMClient()
    history_start = len(client.thread_call_history())

    client.embed(
        "medical query",
        model="openai/test-embedding",
        client_max_attempts=3,
    )

    attempts = client.thread_call_history()[history_start:]
    assert [attempt.status for attempt in attempts] == [
        "error",
        "error",
        "success",
    ]
    assert [
        attempt.provider_metadata["provider_attempt"]
        for attempt in attempts
    ] == [1, 2, 3]


def test_embedding_retry_recomputes_remaining_provider_timeout(
    monkeypatch,
):
    clock = {"now": 100.0}
    observed_timeouts = []

    class _Response:
        data = [{"embedding": [1.0]}]
        usage = None

    def fake_monotonic():
        return clock["now"]

    def fake_sleep(delay):
        clock["now"] += delay

    def flaky_embedding(**kwargs):
        observed_timeouts.append(kwargs["timeout"])
        if len(observed_timeouts) == 1:
            raise RuntimeError("503 temporarily unavailable")
        return _Response()

    monkeypatch.setattr("llm_client.time.monotonic", fake_monotonic)
    monkeypatch.setattr("llm_client.time.sleep", fake_sleep)
    monkeypatch.setattr("llm_client.litellm.embedding", flaky_embedding)

    LLMClient().embed(
        "medical query",
        model="openai/test-embedding",
        deadline_at=110.0,
        client_max_attempts=2,
    )

    assert observed_timeouts[1] < observed_timeouts[0]
    assert observed_timeouts == pytest.approx([10.0, 9.0])


def test_chat_retry_records_each_physical_provider_attempt(monkeypatch):
    calls = {"count": 0}

    class _Message:
        content = "grounded answer"

    class _Choice:
        message = _Message()
        finish_reason = "stop"

    class _Usage:
        prompt_tokens = 3
        completion_tokens = 2

    class _Response:
        choices = [_Choice()]
        usage = _Usage()
        model = "openai/test-chat@r1"
        id = "safe-response-id"

    def flaky_completion(**_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("503 temporarily unavailable")
        return _Response()

    monkeypatch.setattr("llm_client.litellm.completion", flaky_completion)
    monkeypatch.setattr("llm_client.time.sleep", lambda _delay: None)
    client = LLMClient()
    history_start = len(client.thread_call_history())

    assert client.chat(
        [{"role": "user", "content": "medical query"}],
        model="openai/test-chat",
        client_max_attempts=2,
    ) == "grounded answer"

    attempts = client.thread_call_history()[history_start:]
    assert [attempt.status for attempt in attempts] == [
        "error",
        "success",
    ]
    assert [
        attempt.provider_metadata["provider_attempt"]
        for attempt in attempts
    ] == [1, 2]


def test_ollama_chat_uses_native_endpoint(monkeypatch):
    observed = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "message": {"content": "ok"},
                "prompt_eval_count": 2,
                "eval_count": 1,
            }

    def fake_post(url, json, timeout):
        observed.update(url=url, json=json, timeout=timeout)
        return Response()

    monkeypatch.setattr("requests.post", fake_post)
    client = LLMClient()

    result = client.chat(
        [{"role": "user", "content": "test"}],
        model="ollama/llama3",
        temperature=0.0,
        max_tokens=128,
        timeout=1.25,
        client_max_attempts=1,
    )

    assert result == "ok"
    assert observed["url"].endswith("/api/chat")
    assert observed["json"]["model"] == "llama3"
    assert observed["json"]["options"]["num_predict"] == 128
    assert observed["timeout"] == 1.25


def test_nested_fetcher_llm_calls_receive_runtime_deadline():
    observed = []

    class Fetcher:
        def analyze_user_query(self, _query, **kwargs):
            observed.append(kwargs["llm_kwargs"])
            return {"success": False, "error": "offline"}

    state = {
        "input_query": "medical query",
        "expanded_query": "medical query",
        "context": {
            "_runtime_deadline_at_monotonic": time.monotonic() + 5,
            "max_papers": 1,
            "max_records": 1,
            "max_trials": 1,
        },
    }
    for agent_class in (
        PubMedAgentGraph,
        FDAAgentGraph,
        ClinicalTrialsAgentGraph,
    ):
        agent = agent_class.__new__(agent_class)
        agent._fetcher = Fetcher()
        agent._fetch_node(dict(state))

    assert len(observed) == 3
    assert all(0 < item["timeout"] <= 5 for item in observed)
    assert all(item["client_max_attempts"] == 1 for item in observed)
    assert all("deadline_at" in item for item in observed)


def test_provider_retry_loop_does_not_start_after_deadline(monkeypatch):
    attempts = []
    clock = iter([0.0, 0.0, 0.0])
    monkeypatch.setattr("llm_client.time.monotonic", lambda: next(clock))
    monkeypatch.setattr(
        "llm_client.time.sleep",
        lambda _seconds: pytest.fail("deadline must prevent retry sleep"),
    )

    def always_transient():
        attempts.append(1)
        raise RuntimeError("429 rate limit")

    with pytest.raises(RuntimeDeadlineExceeded):
        _call_with_retry(
            always_transient,
            operation="deadline-test",
            max_attempts=5,
            deadline_at=0.5,
        )

    assert len(attempts) == 1


def test_classifier_telemetry_replay_is_idempotent():
    state = {
        "token_usage": {"input": 1, "output": 1, "total": 2},
        "cost_estimate": 0.1,
        "attempt_telemetry": [],
    }
    call = LLMCallResult(
        text="medical",
        model="classifier-model",
        model_revision="r1",
        tokens_in=3,
        tokens_out=2,
        cost_usd=0.01,
        latency_sec=0.1,
        finish_reason="stop",
        provider_metadata={"provider": "mock"},
    )

    _record_classifier_call(state, "trace-classifier", call)
    _record_classifier_call(state, "trace-classifier", call)

    assert state["token_usage"] == {"input": 4, "output": 3, "total": 7}
    assert state["cost_estimate"] == pytest.approx(0.11)
    assert len(state["attempt_telemetry"]) == 1


def test_rag_tool_expired_deadline_stops_before_embedding():
    tool = RAGTool()
    tool._ensure_embedder = lambda: pytest.fail(
        "expired RAG work must not initialize an embedder"
    )

    result = tool.call(
        {
            "query": "private medical query",
            "documents": [{"text": "Evidence."}],
            "_runtime_deadline_at_monotonic": time.monotonic() - 1,
        }
    )

    assert result["error"] == "rag_retrieve_failed:RuntimeDeadlineExceeded"


@pytest.mark.parametrize(
    ("tool_class", "args", "expected_error"),
    [
        (
            PubMedTool,
            (1, "index"),
            "pubmed_pipeline_exact_context_unavailable",
        ),
        (
            FDATool,
            (1, 10),
            "fda_pipeline_exact_context_unavailable",
        ),
        (
            ClinicalTrialsTool,
            (1, 10),
            "clinical_trials_pipeline_exact_context_unavailable",
        ),
    ],
)
def test_legacy_pipeline_does_not_run_without_exact_context_contract(
    tool_class, args, expected_error
):
    tool = tool_class.__new__(tool_class)
    tool._ensure_pipeline = lambda *_args: pytest.fail(
        "legacy generation must not run when exact context is unavailable"
    )

    result = tool._call_pipeline(
        "private medical query",
        *args,
        time.time(),
    )

    assert result["error"] == expected_error


@pytest.mark.parametrize(
    "fetcher_class",
    [PubMedFetcher, FDAFetcher, ClinicalTrialsFetcher],
)
def test_fetchers_fail_distinctly_before_work_when_deadline_expired(
    fetcher_class,
):
    class _ForbiddenLLM:
        def chat(self, *_args, **_kwargs):
            raise AssertionError("expired work must not reach the provider")

    fetcher = fetcher_class.__new__(fetcher_class)
    fetcher._llm = _ForbiddenLLM()
    result = fetcher.analyze_user_query(
        "distinctive private medical query",
        deadline_at=time.monotonic() - 1,
    )

    assert result["success"] is False
    assert result["error_type"] == "runtime_deadline_exhausted"


def test_fda_broad_fallback_does_not_start_after_deadline(monkeypatch):
    fetcher = FDAFetcher.__new__(FDAFetcher)
    fetcher._llm = _FakeLLM()
    fetcher.max_urls = 4
    fetcher.page_size = 10
    fetch_calls = []

    class _Terms:
        drug_names = ["aspirin"]
        conditions = []
        safety_terms = []
        recall_terms = []
        all_terms = ["aspirin"]

        def model_dump(self):
            return {"drug_names": ["aspirin"]}

    monkeypatch.setattr(
        fetcher, "extract_search_terms", lambda *_args, **_kwargs: _Terms()
    )
    monkeypatch.setattr(
        fetcher,
        "fetch_fda_data",
        lambda urls, **_kwargs: (fetch_calls.append(list(urls)) or ({}, [])),
    )
    clock = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(
            "evaluation_core.deadline.time.monotonic",
        lambda: next(clock),
    )

    result = fetcher.analyze_user_query(
        "private query", retry_broad=True, deadline_at=1.0
    )

    assert result["success"] is False
    assert result["error_type"] == "runtime_deadline_exhausted"
    assert len(fetch_calls) == 1


def test_fetcher_logs_never_include_raw_medical_query(monkeypatch, caplog):
    query = "Patient ZQ-771 has a private vamorolone exposure history."
    caplog.set_level(logging.INFO)

    pubmed = PubMedFetcher.__new__(PubMedFetcher)
    pubmed.include_fulltext = False
    monkeypatch.setattr(
        pubmed,
        "extract_search_terms",
        lambda *_args, **_kwargs: PubMedSearchTerms(),
    )
    monkeypatch.setattr(pubmed, "build_urls", lambda _terms: [])
    monkeypatch.setattr(
        pubmed, "search_pmids", lambda *_args, **_kwargs: ([], [])
    )
    monkeypatch.setattr(
        pubmed, "fetch_papers", lambda *_args, **_kwargs: {}
    )
    pubmed.analyze_user_query(query)

    fda = FDAFetcher.__new__(FDAFetcher)
    monkeypatch.setattr(
        fda,
        "extract_search_terms",
        lambda *_args, **_kwargs: FDASearchTerms(),
    )
    monkeypatch.setattr(fda, "build_urls", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        fda, "fetch_fda_data", lambda *_args, **_kwargs: ({}, [])
    )
    fda.analyze_user_query(query)

    trials = ClinicalTrialsFetcher.__new__(ClinicalTrialsFetcher)
    monkeypatch.setattr(
        trials,
        "extract_search_terms",
        lambda *_args, **_kwargs: SearchTerms(),
    )
    monkeypatch.setattr(
        trials, "build_urls", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        trials,
        "fetch_clinical_trials_data",
        lambda *_args, **_kwargs: ({}, []),
    )
    trials.analyze_user_query(query)

    assert query not in caplog.text
    assert "ZQ-771" not in caplog.text


@pytest.mark.parametrize(
    "fetcher_class",
    [PubMedFetcher, FDAFetcher, ClinicalTrialsFetcher],
)
def test_fetcher_exception_text_cannot_expose_medical_query(
    fetcher_class, monkeypatch, caplog
):
    query = "Patient PHI-ZQ-994 has private vamorolone history."
    fetcher = fetcher_class.__new__(fetcher_class)

    def fail_with_query(*_args, **_kwargs):
        raise RuntimeError(query)

    monkeypatch.setattr(
        fetcher, "extract_search_terms", fail_with_query
    )
    caplog.set_level(logging.ERROR)

    result = fetcher.analyze_user_query(query)

    assert query not in caplog.text
    assert "PHI-ZQ-994" not in caplog.text
    assert query not in str(result)
    assert "PHI-ZQ-994" not in str(result)
