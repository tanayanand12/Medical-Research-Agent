import pytest

from eval import metrics, rejudge_results
from eval.metrics import (
    EvalSample,
    MetricResult,
    _parse_judge_response,
    answer_correctness,
    citation_fidelity,
)
from eval import run_eval
from eval.run_eval import (
    _aggregate_results,
    _bootstrap_mean_ci,
    _select_questions,
    _validate_generation_models,
)


def test_parse_judge_response_accepts_fenced_json():
    parsed = _parse_judge_response(
        '```json\n{"score": 0.85, "reason": "Supported by the context."}\n```'
    )

    assert parsed["score"] == pytest.approx(0.85)
    assert parsed["reason"] == "Supported by the context."


def test_parse_judge_response_rejects_missing_score():
    with pytest.raises(ValueError, match="score"):
        _parse_judge_response('{"reason": "No numeric score was returned."}')


def test_citation_fidelity_penalises_missing_citations():
    sample = EvalSample(
        question="What is recommended?",
        answer="The guideline recommends treatment.",
        retrieved_contexts=["The guideline recommends treatment."],
    )

    result = citation_fidelity(sample)

    assert result.score == 0.0
    assert "no citation" in result.reason.lower()


def test_citation_fidelity_uses_actual_citation_inventory_not_chunk_count():
    sample = EvalSample(
        question="What is recommended?",
        answer="The recommendation is supported [3].",
        retrieved_contexts=["chunk"] * 5,
        citations=["Source 1", "Source 2"],
    )

    result = citation_fidelity(sample)

    assert result.score == 0.0


def test_aggregate_excludes_invalid_metrics_and_reports_counts():
    results = [
        {
            "model": "model-a",
            "agent": "local",
            "error": None,
            "metrics": {
                "faithfulness": {"score": 1.0, "valid": True},
                "answer_relevancy": {"score": 0.8, "valid": True},
                "citation_fidelity": {"score": 1.0, "valid": True},
                "hallucination_rate": {"score": 0.0, "valid": True},
            },
        },
        {
            "model": "model-a",
            "agent": "local",
            "error": None,
            "metrics": {
                "faithfulness": {"score": 0.0, "valid": False},
                "answer_relevancy": {"score": 0.6, "valid": True},
                "citation_fidelity": {"score": 0.0, "valid": True},
                "hallucination_rate": {"score": 0.2, "valid": True},
            },
        },
    ]

    summary = _aggregate_results(results, ["model-a"], ["local"])

    assert summary["overall"]["scores"]["faithfulness"] == 1.0
    assert summary["overall"]["valid_counts"]["faithfulness"] == 1
    assert summary["overall"]["scores"]["answer_relevancy"] == 0.7
    assert summary["overall"]["sample_count"] == 2
    assert summary["overall"]["latency_sec"]["mean"] == 0.0


def test_bootstrap_mean_ci_is_exact_for_constant_values():
    assert _bootstrap_mean_ci([0.75, 0.75, 0.75], seed=7) == (0.75, 0.75)


def test_evaluate_sample_uses_independent_judge_and_preserves_context(monkeypatch):
    observed = {}

    class Output:
        answer = "Supported answer [1]."
        confidence = 0.8
        model_used = "generator"
        citations = ["1. Source."]
        sources = [
            {
                "text": "Supporting passage.",
                "score": 0.9,
                "dense_rank": 1,
                "sparse_rank": 2,
                "metadata": {"id": "x"},
            },
            {
                "text": "Second passage.",
                "score": 0.6,
                "dense_rank": 3,
                "sparse_rank": None,
                "metadata": {"id": "y"},
            },
        ]
        error = None

    class Agent:
        def invoke(self, query, context):
            return Output()

    def fake_metrics(sample, model):
        observed["model"] = model
        return []

    monkeypatch.setattr(run_eval, "compute_all_metrics", fake_metrics)
    result = run_eval._evaluate_sample(
        "Question?",
        "Reference.",
        Agent(),
        model="generator",
        judge_model="independent-judge",
    )

    assert observed["model"] == "independent-judge"
    assert result["retrieved_contexts"] == [
        "Supporting passage.",
        "Second passage.",
    ]
    assert result["retrieval_diagnostics"]["score_top1"] == 0.9
    assert result["retrieval_diagnostics"]["score_margin"] == pytest.approx(0.3)
    assert result["retrieval_diagnostics"]["hybrid_overlap_fraction"] == 0.5


def test_llm_judge_allows_reasoning_models_enough_output_tokens(monkeypatch):
    observed = {}

    class FakeLLM:
        def chat(self, **kwargs):
            observed.update(kwargs)
            return '{"score": 1.0, "reason": "ok"}'

    monkeypatch.setattr(metrics, "_get_llm", lambda: FakeLLM())

    metrics._llm_judge("Evaluate this.", model="judge")

    assert observed["max_tokens"] >= 2048


def test_answer_correctness_compares_answer_with_reference(monkeypatch):
    observed = {}

    def fake_judge(prompt, model=None):
        observed["prompt"] = prompt
        return {"score": 0.9, "reason": "Substantially correct."}

    monkeypatch.setattr(metrics, "_llm_judge", fake_judge)
    sample = EvalSample(
        question="What is the threshold?",
        answer="The threshold is 126 mg/dL.",
        expected_answer="Fasting glucose at least 126 mg/dL.",
    )

    result = answer_correctness(sample, model="judge")

    assert result.name == "answer_correctness"
    assert result.score == 0.9
    assert "REFERENCE ANSWER" in observed["prompt"]


def test_compute_all_metrics_separates_grounding_from_reference_judging(monkeypatch):
    calls = []

    class FakeLLM:
        def chat(self, **kwargs):
            calls.append(kwargs)
            prompt = kwargs["messages"][0]["content"]
            if "faithfulness" in prompt:
                return (
                    '{"faithfulness":{"score":0.8,"reason":"grounded"},'
                    '"hallucination_rate":{"score":0.2,'
                    '"reason":"one unsupported claim"}}'
                )
            return (
                '{"answer_relevancy":{"score":0.9,"reason":"direct"},'
                '"answer_correctness":{"score":0.7,"reason":"mostly correct"}}'
            )

    monkeypatch.setattr(metrics, "_get_llm", lambda: FakeLLM())
    sample = EvalSample(
        question="Question?",
        answer="Answer [1].",
        expected_answer="Reference.",
        retrieved_contexts=["Evidence."],
        citations=["Citation"],
    )

    results = metrics.compute_all_metrics(sample, model="judge")

    assert len(calls) == 2
    grounding_prompt = calls[0]["messages"][0]["content"]
    quality_prompt = calls[1]["messages"][0]["content"]
    assert "REFERENCE ANSWER" not in grounding_prompt
    assert "RETRIEVED CONTEXT" not in quality_prompt
    assert {result.name for result in results} == {
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "citation_fidelity",
        "hallucination_rate",
    }


def test_rejudge_payload_preserves_answers_and_replaces_metrics(monkeypatch):
    source = {
        "config": {"judge_model": "old"},
        "results": [
            {
                "question": "Question?",
                "expected_answer": "Reference.",
                "agent_answer": "Answer [1].",
                "retrieved_contexts": ["Evidence."],
                "citations": ["Source"],
                "metrics": {"faithfulness": {"score": 0.0}},
            }
        ],
    }
    monkeypatch.setattr(
        rejudge_results,
        "compute_all_metrics",
        lambda sample, model: [
            MetricResult("faithfulness", 0.8, "grounded", valid=True)
        ],
    )
    monkeypatch.setattr(
        rejudge_results,
        "_aggregate_results",
        lambda rows, models, agents: {"overall": {"sample_count": len(rows)}},
    )

    output = rejudge_results.rejudge_payload(source, "new-judge")

    assert output["results"][0]["agent_answer"] == "Answer [1]."
    assert output["results"][0]["metrics"]["faithfulness"]["score"] == 0.8
    assert output["config"]["judge_model"] == "new-judge"
    assert output["config"]["rejudged_from_preserved_outputs"] is True


def test_question_sampling_happens_after_target_agent_filtering():
    class Dataset:
        def get_target_agents(self, question):
            return [question.split("_")[0]]

    selected = _select_questions(
        ["pubmed_1", "pubmed_2", "local_1", "local_2"],
        Dataset(),
        agent_names=["local"],
        n_samples=1,
        seed=42,
    )

    assert len(selected) == 1
    assert selected[0].startswith("local_")


def test_evaluation_rejects_mislabeled_generation_model():
    class LLM:
        default_model = "actual-model"

    class Agent:
        llm = LLM()

    with pytest.raises(ValueError, match="DEFAULT_LLM_MODEL"):
        _validate_generation_models({"pubmed": Agent()}, ["requested-model"])
