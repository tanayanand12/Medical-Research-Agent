from eval.finalize_paper_results import (
    _recompute_deterministic_metrics,
    build_results_markdown,
)


def test_build_results_markdown_reports_direction_and_denominators():
    summary = {
        "overall": {
            "scores": {
                "faithfulness": 0.9,
                "answer_relevancy": 0.8,
                "answer_correctness": 0.8525,
                "citation_fidelity": 1.0,
                "hallucination_rate": 0.1,
            },
            "ci95": {
                "faithfulness": [0.8, 1.0],
                "answer_relevancy": [0.7, 0.9],
                "answer_correctness": [0.7525, 0.9525],
                "citation_fidelity": [1.0, 1.0],
                "hallucination_rate": [0.0, 0.2],
            },
            "valid_counts": {
                "faithfulness": 20,
                "answer_relevancy": 20,
                "answer_correctness": 20,
                "citation_fidelity": 20,
                "hallucination_rate": 19,
            },
            "sample_count": 20,
            "error_count": 1,
            "latency_sec": {"mean": 2.0, "median": 1.5, "p95": 4.0},
        },
        "by_agent": {},
    }

    markdown = build_results_markdown(
        summary,
        {
            "n": 20,
            "top1_accuracy": 0.85,
            "top3_accuracy": 1.0,
            "mean_reciprocal_rank": 0.9083,
        },
        {
            "scores": {
                "context_relevance": 0.7,
                "context_sufficiency": 0.6,
                "citation_claim_support": 0.8,
                "sentence_citation_coverage": 0.5,
            },
            "ci95": {
                "context_relevance": [0.5, 0.9],
                "context_sufficiency": [0.4, 0.8],
                "citation_claim_support": [0.6, 1.0],
                "sentence_citation_coverage": [0.3, 0.7],
            },
            "valid_counts": {
                "context_relevance": 20,
                "context_sufficiency": 20,
                "citation_claim_support": 20,
                "sentence_citation_coverage": 20,
            },
            "sample_count": 20,
        },
    )

    assert "Hallucination rate (lower is better)" in markdown
    assert "19/20" in markdown
    assert "1 execution errors" in markdown
    assert "Top-1 accuracy | 0.850" in markdown
    assert "0.853 (0.753–0.953)" in markdown
    assert "Context sufficiency | 0.600 (0.400–0.800); 20/20" in markdown


def test_finalize_recomputes_citation_fidelity_from_actual_citations():
    rows = [
        {
            "question": "Question",
            "agent_answer": "Claim [3].",
            "expected_answer": "Reference",
            "retrieved_contexts": ["chunk"] * 5,
            "citations": ["Source 1", "Source 2"],
            "metrics": {
                "citation_fidelity": {
                    "score": 1.0,
                    "reason": "stale",
                    "valid": True,
                }
            },
        }
    ]

    corrected = _recompute_deterministic_metrics(rows)

    assert corrected[0]["metrics"]["citation_fidelity"]["score"] == 0.0
