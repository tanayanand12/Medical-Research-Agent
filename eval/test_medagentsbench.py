"""Offline tests for MedAgentsBench adapter stub (no network)."""

from __future__ import annotations

import pytest

from eval.datasets import DATASET_REGISTRY, MedAgentsBenchDataset, MedQADataset, get_dataset
from eval.medagentsbench import (
    MEDAGENTSBENCH_PROVENANCE,
    MedAgentsBenchNotDownloadedError,
    expected_source_total,
    normalize_row,
    provenance_metadata,
    stratified_sample_by_source,
)


def _fake_slice() -> list[dict]:
    """Minimal in-memory slice spanning four source benchmarks."""
    rows = []
    sources = ["MedQA", "MedMCQA", "PubMedQA", "MedExQA"]
    for source in sources:
        for i in range(5):
            rows.append(
                {
                    "id": f"{source.lower()}_{i}",
                    "question": f"[{source}] Question {i}?",
                    "answer": "A",
                    "source_benchmark": source,
                }
            )
    return rows


def test_provenance_metadata():
    meta = provenance_metadata(loaded_n=862)
    assert meta["arxiv"] == "2503.07459"
    assert meta["split"] == "test_hard"
    assert meta["official_n"] == 862
    assert meta["huggingface_canonical"] == "super-dainiu/MedicalAgentsBench"
    breakdown = meta["source_benchmark_breakdown"]
    assert breakdown["MedQA"] == 100
    assert sum(v for k, v in breakdown.items() if k != "AfriMedQA") == 862
    assert sum(breakdown.values()) == 894
    assert meta["loaded_n"] == 862
    assert meta["n_discrepancy"] is False


def test_expected_source_total():
    assert expected_source_total() == 862


def test_normalize_row_adds_benchmark_fields():
    row = normalize_row(
        {"question": "Q?", "answer": "B", "options": {"A": "x"}},
        source_benchmark="MedQA",
    )
    assert row["benchmark"] == "medagentsbench_test_hard"
    assert row["split"] == "test_hard"
    assert row["source_benchmark"] == "MedQA"


def test_stratified_sample_n20():
    items = _fake_slice()
    sample = stratified_sample_by_source(items, 20, seed=42)
    assert len(sample) == 20
    counts = {}
    for item in sample:
        src = item["source_benchmark"]
        counts[src] = counts.get(src, 0) + 1
    assert set(counts) == {"MedQA", "MedMCQA", "PubMedQA", "MedExQA"}
    assert sum(counts.values()) == 20


def test_stratified_sample_n100_on_small_pool_returns_all():
    items = _fake_slice()
    sample = stratified_sample_by_source(items, 100, seed=7)
    assert len(sample) == len(items)


def test_medagentsbench_dataset_injected_items():
    ds = MedAgentsBenchDataset(n_samples=20, seed=1, items=_fake_slice())
    assert len(ds) == 20
    provenance = ds.get_provenance()
    assert provenance["benchmark"] == "MedAgentsBench"
    assert provenance["split"] == "test_hard"
    q = ds.get_questions()[0]
    assert ds.get_source_benchmark(q) in {"MedQA", "MedMCQA", "PubMedQA", "MedExQA"}
    assert ds.source_counts()


def test_not_downloaded_raises_without_cache():
    with pytest.raises(MedAgentsBenchNotDownloadedError) as exc:
        MedAgentsBenchDataset(allow_download=False)
    msg = str(exc.value)
    assert "medagentsbench_test_hard.json" in msg
    assert "super-dainiu/MedicalAgentsBench" in msg
    assert "does not auto-download" in msg


def test_separate_registry_from_medqa():
    assert "medagentsbench_test_hard" in DATASET_REGISTRY
    assert "medqa" in DATASET_REGISTRY
    assert DATASET_REGISTRY["medagentsbench_test_hard"] is MedAgentsBenchDataset
    assert DATASET_REGISTRY["medqa"] is MedQADataset
    assert MedAgentsBenchDataset.name == "medagentsbench_test_hard"
    assert MedQADataset.name == "medqa"


def test_get_dataset_resolves_medagentsbench_class():
    cls = DATASET_REGISTRY["medagentsbench_test_hard"]
    assert cls is MedAgentsBenchDataset


def test_get_dataset_without_cache_raises():
    with pytest.raises(MedAgentsBenchNotDownloadedError):
        get_dataset("medagentsbench_test_hard", n_samples=8, seed=0)
