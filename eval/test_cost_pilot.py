"""Offline tests for cost pilot aggregation and projection math."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.cost_pilot import (
    COST_PILOT_VERSION,
    QuestionCostRecord,
    aggregate_records,
    build_matrix_planning_projections,
    compute_mid_token_cost_per_question,
    extract_question_metrics,
    percentile,
    project_costs,
    recommend_credit_purchase_for_cost,
    recommend_credit_purchase_usd,
    render_markdown_summary,
    serialize_pilot_result,
    synthesize_mock_final_state,
    normalize_agent_names,
    load_pilot_questions,
    DEFAULT_FULL_SPLIT_SIZES,
)


def test_normalize_agent_names():
    assert normalize_agent_names(["pubmed", "fda"]) == [
        "search_pubmed",
        "search_fda",
    ]
    assert normalize_agent_names(None) is None


def test_percentile_empty_and_single():
    assert percentile([], 50) == 0.0
    assert percentile([4.2], 95) == 4.2


def test_percentile_p95():
    values = [1.0, 2.0, 3.0, 4.0, 10.0]
    assert percentile(values, 50) == 3.0
    assert percentile(values, 95) == 10.0


def test_extract_question_metrics_from_mock_state():
    state = synthesize_mock_final_state(
        question_id="q1",
        question="What is metformin used for?",
        model_id="gpt-4o",
        index=3,
    )
    record = extract_question_metrics(
        question_id="q1",
        final_state=state,
        query_text="What is metformin used for?",
    )
    assert record.total_tokens > 0
    assert record.cost_usd > 0
    assert record.verifier_calls >= 1
    assert record.query_fingerprint
    assert "metformin" not in json.dumps(record.__dict__)


def test_aggregate_records_means():
    records = [
        QuestionCostRecord(
            question_id="a",
            trace_id="t1",
            query_fingerprint="abc",
            query_length=10,
            tokens_in=100,
            tokens_out=50,
            total_tokens=150,
            cost_usd=0.01,
            latency_sec=2.0,
            retry_count=0,
            repair_count=0,
            verifier_calls=1,
            evidence_limited=False,
            fallback_triggered=False,
            error_occurred=False,
        ),
        QuestionCostRecord(
            question_id="b",
            trace_id="t2",
            query_fingerprint="def",
            query_length=12,
            tokens_in=200,
            tokens_out=100,
            total_tokens=300,
            cost_usd=0.03,
            latency_sec=4.0,
            retry_count=1,
            repair_count=1,
            verifier_calls=2,
            evidence_limited=True,
            fallback_triggered=True,
            error_occurred=False,
        ),
    ]
    agg = aggregate_records(records)
    assert agg.n_samples == 2
    assert agg.cost_usd_mean == pytest.approx(0.02)
    assert agg.cost_usd_total == pytest.approx(0.04)
    assert agg.tokens_total == 450
    assert agg.latency_sec_p50 == pytest.approx(2.0)  # nearest-rank p50 on n=2
    assert agg.retry_count_total == 1
    assert agg.repair_count_total == 1
    assert agg.verifier_calls_total == 3
    assert agg.evidence_limited_rate == 0.5
    assert agg.fallback_triggered_rate == 0.5


def test_project_costs_linear():
    records = [
        QuestionCostRecord(
            question_id="a",
            trace_id="t1",
            query_fingerprint="x",
            query_length=5,
            tokens_in=1000,
            tokens_out=200,
            total_tokens=1200,
            cost_usd=0.05,
            latency_sec=5.0,
            retry_count=0,
            repair_count=0,
            verifier_calls=1,
            evidence_limited=False,
            fallback_triggered=False,
            error_occurred=False,
        )
    ]
    agg = aggregate_records(records)
    projections = project_costs(agg, [100, 500])
    assert projections[0].projected_cost_usd == pytest.approx(5.0)
    assert projections[0].projected_tokens == 120000
    assert projections[1].projected_cost_usd == pytest.approx(25.0)


def test_recommend_credit_purchase_formula():
    agg = aggregate_records(
        [
            QuestionCostRecord(
                question_id="a",
                trace_id="t",
                query_fingerprint="fp",
                query_length=8,
                tokens_in=100,
                tokens_out=50,
                total_tokens=150,
                cost_usd=0.04,
                latency_sec=3.0,
                retry_count=0,
                repair_count=0,
                verifier_calls=1,
                evidence_limited=False,
                fallback_triggered=False,
                error_occurred=False,
            )
        ]
    )
    rec = recommend_credit_purchase_usd(agg, target_n=100, variance_buffer=0.25, fixed_buffer_usd=5.0)
    assert rec["base_projected_usd"] == pytest.approx(4.0)
    assert rec["recommended_purchase_usd"] == pytest.approx(10.0)


def test_render_markdown_no_raw_query():
    agg = aggregate_records([])
    md = render_markdown_summary(
        run_meta={"run_id": "abc", "mode": "dry_run", "model_id": "gpt-4o"},
        aggregate=agg,
        projections=[],
        recommendations={
            "100_question_pilot": recommend_credit_purchase_usd(agg, target_n=100),
        },
        per_question=[],
    )
    assert "metformin" not in md.lower()
    assert "Cost Pilot Summary" in md
    assert "recommended_purchase_usd" not in md  # human summary uses dollar amounts


def test_serialize_pilot_result_schema():
    agg = aggregate_records([])
    payload = serialize_pilot_result(
        run_meta={"run_id": "x"},
        records=[],
        aggregate=agg,
        projections=[],
        recommendations={},
    )
    assert payload["schema"]["evaluation_trace_version"] == "1.0.0"
    assert payload["schema"]["cost_pilot_version"] == COST_PILOT_VERSION


def test_compute_mid_token_cost_luna():
    cost = compute_mid_token_cost_per_question(0.0002, 0.0012, tokens_in=55_000, tokens_out=4_000)
    assert cost == pytest.approx(0.0158)


def test_matrix_planning_862_includes_luna():
    planning = build_matrix_planning_projections(target_n=862)
    assert planning["target_n"] == 862
    assert planning["tokens_in"] == 55_000
    assert planning["tokens_out"] == 4_000
    rows = {row["model_id"]: row for row in planning["rows"]}
    assert "gpt-5.6-luna" in rows
    luna = rows["gpt-5.6-luna"]
    assert luna["cost_per_question_usd"] == pytest.approx(0.0158)
    assert luna["projected_cost_usd"] == pytest.approx(round(0.0158 * 862, 2))
    rec = recommend_credit_purchase_for_cost(0.0158, target_n=862)
    assert luna["recommended_purchase_usd"] == rec["recommended_purchase_usd"]


def test_matrix_planning_sonnet_intro_and_standard_rows():
    planning = build_matrix_planning_projections(target_n=862)
    sonnet_rows = [
        row for row in planning["rows"] if row["model_id"].startswith("anthropic/claude-sonnet-5")
    ]
    assert len(sonnet_rows) == 2
    labels = {row["pricing_label"] for row in sonnet_rows}
    assert labels == {"introductory", "standard"}


def test_matrix_planning_gpt5_not_verified():
    planning = build_matrix_planning_projections(target_n=862)
    gpt5 = next(row for row in planning["rows"] if row["model_id"] == "gpt-5")
    assert gpt5["pricing_verified"] is False
    assert gpt5["target_n"] == DEFAULT_FULL_SPLIT_SIZES["medagentsbench_test_hard"]


def test_project_costs_includes_862():
    records = [
        QuestionCostRecord(
            question_id="a",
            trace_id="t1",
            query_fingerprint="x",
            query_length=5,
            tokens_in=59_000,
            tokens_out=4_000,
            total_tokens=63_000,
            cost_usd=0.05,
            latency_sec=5.0,
            retry_count=0,
            repair_count=0,
            verifier_calls=1,
            evidence_limited=False,
            fallback_triggered=False,
            error_occurred=False,
        )
    ]
    agg = aggregate_records(records)
    projections = project_costs(agg, [100, 500, 862])
    by_n = {p.target_n: p for p in projections}
    assert 862 in by_n
    assert by_n[862].projected_cost_usd == pytest.approx(43.1)


def test_load_pilot_questions_truncates(tmp_path):
    sample = [
        {"id": "1", "question": "Q1"},
        {"id": "2", "question": "Q2"},
        {"id": "3", "question": "Q3"},
    ]
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(sample), encoding="utf-8")
    rows = load_pilot_questions(path, n_samples=2)
    assert len(rows) == 2
    assert rows[0]["id"] == "1"


def test_dry_run_cli_integration(tmp_path):
    """Invoke run_cost_pilot main in dry-run without network."""
    from eval.run_cost_pilot import main
    import sys

    dataset = Path(__file__).resolve().parent / "data" / "medical_benchmark.json"
    out_json = tmp_path / "pilot.json"
    argv = [
        "run_cost_pilot",
        "--dry-run",
        "--n_samples",
        "3",
        "--dataset",
        str(dataset),
        "--output",
        str(out_json),
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        assert main() == 0
    finally:
        sys.argv = old_argv

    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["run_meta"]["mode"] == "dry_run"
    assert len(data["per_question"]) == 3
    assert data["aggregate"]["n_samples"] == 3
    assert "matrix_planning" in data
    assert data["matrix_planning"]["target_n"] == 862
    assert len(data["matrix_planning"]["rows"]) >= 5
    md_path = out_json.with_suffix(".md")
    assert md_path.exists()
    md_text = md_path.read_text(encoding="utf-8")
    assert "Cost Pilot Summary" in md_text
    assert "Matrix planning" in md_text
