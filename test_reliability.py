from agents.pubmed_agent.graph import PubMedAgentGraph
from llm_client import LLMClient, _call_with_retry
from rag_engine.hybrid_retriever import HybridRetriever
from rag_engine.sparse_index import SparseResult


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
    )

    assert result == "ok"
    assert observed["url"].endswith("/api/chat")
    assert observed["json"]["model"] == "llama3"
    assert observed["json"]["options"]["num_predict"] == 128
