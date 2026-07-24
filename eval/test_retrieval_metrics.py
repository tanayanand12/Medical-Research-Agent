import pytest

from eval.evaluate_retrieval_grounding import aggregate_grounding_rows
from eval.metrics import EvalSample
from eval.retrieval_metrics import (
    _parse_retrieval_judge_response,
    sentence_citation_coverage,
)


def test_retrieval_judge_parser_validates_three_stage_metrics():
    parsed = _parse_retrieval_judge_response(
        """
        {
          "context_relevance": {"score": 0.75, "reason": "3/4 relevant"},
          "context_sufficiency": {"score": 0.5, "reason": "partial coverage"},
          "citation_claim_support": {"score": 1.0, "reason": "all supported"}
        }
        """
    )

    assert parsed["context_relevance"]["score"] == 0.75
    assert parsed["context_sufficiency"]["score"] == 0.5
    assert parsed["citation_claim_support"]["score"] == 1.0


def test_retrieval_judge_parser_rejects_out_of_range_score():
    with pytest.raises(ValueError, match="context_relevance"):
        _parse_retrieval_judge_response(
            """
            {
              "context_relevance": {"score": 1.2},
              "context_sufficiency": {"score": 0.5},
              "citation_claim_support": {"score": 0.5}
            }
            """
        )


def test_sentence_citation_coverage_reports_cited_factual_sentences():
    sample = EvalSample(
        question="Question?",
        answer=(
            "Treatment reduced admissions [1]. "
            "Mortality was unchanged. "
            "Evidence remains limited [2]."
        ),
        citations=["Source 1", "Source 2"],
    )

    result = sentence_citation_coverage(sample)

    assert result.valid is True
    assert result.score == pytest.approx(2 / 3)
    assert result.raw["cited_sentences"] == 2
    assert result.raw["total_sentences"] == 3


def test_grounding_aggregation_reports_valid_denominators_by_agent():
    rows = [
        {
            "agent": "pubmed",
            "metrics": {
                "context_relevance": {"score": 0.8, "valid": True},
                "context_sufficiency": {"score": 0.6, "valid": True},
            },
        },
        {
            "agent": "pubmed",
            "metrics": {
                "context_relevance": {"score": 0.0, "valid": False},
                "context_sufficiency": {"score": 0.4, "valid": True},
            },
        },
    ]

    summary = aggregate_grounding_rows(rows)

    assert summary["overall"]["scores"]["context_relevance"] == 0.8
    assert summary["overall"]["valid_counts"]["context_relevance"] == 1
    assert summary["by_agent"]["pubmed"]["scores"]["context_sufficiency"] == 0.5
