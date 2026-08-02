"""
MedAgentsBench test_hard adapter — provenance, stratified sampling, lazy load stub.

Separate from standalone MedQA (eval/datasets.py::MedQADataset). The 100 MedQA-hard
items inside MedAgentsBench must never be merged into MedQADataset output.

Paper: Tang et al., 2025 — arXiv:2503.07459
Split: test_hard only, N=862 (authoritative)
HF: super-dainiu/MedicalAgentsBench (alias: super-dainiu/medagents-benchmark)
GitHub: gersteinlab/MedicalAgentsBench
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

_EVAL_DATA_DIR = Path(__file__).resolve().parent / "data"
_DEFAULT_LOCAL_CACHE = _EVAL_DATA_DIR / "medagentsbench_test_hard.json"

MEDAGENTSBENCH_PROVENANCE: Dict[str, Any] = {
    "benchmark": "MedAgentsBench",
    "paper": "Tang et al., 2025",
    "arxiv": "2503.07459",
    "arxiv_url": "https://arxiv.org/abs/2503.07459",
    "split": "test_hard",
    "official_n": 862,
    "github": "gersteinlab/MedicalAgentsBench",
    "github_url": "https://github.com/gersteinlab/MedicalAgentsBench",
    "huggingface_canonical": "super-dainiu/MedicalAgentsBench",
    "huggingface_alias": "super-dainiu/medagents-benchmark",
    "avg_question_tokens": 147,
    "options_per_item": "3-10 depending on source benchmark",
    "source_benchmark_breakdown": {
        "MedQA": 100,
        "MedMCQA": 100,
        "PubMedQA": 100,
        "MedExQA": 100,
        "MMLU-Pro": 100,
        "MedXpertQA-Reasoning": 100,
        "MedXpertQA-Understanding": 100,
        "MedBullets": 89,
        "MMLU": 73,
        "AfriMedQA": 32,
    },
    "source_benchmark_notes": (
        "Official N=862 matches GitHub/HF test_hard when AfriMedQA (32) is absent "
        "from the loaded shard; all ten sources sum to 894."
    ),
    "rules": {
        "separate_from_standalone_medqa": True,
        "no_unfiltered_pool": True,
        "official_n_authoritative": True,
    },
}


class MedAgentsBenchNotDownloadedError(FileNotFoundError):
    """Raised when test_hard is not cached locally and auto-download is disabled."""

    def __init__(self, path: Optional[Path] = None) -> None:
        cache = path or _DEFAULT_LOCAL_CACHE
        message = (
            "MedAgentsBench test_hard split is not available locally.\n"
            f"Expected cache: {cache}\n\n"
            "This adapter stub does not auto-download the full 862-item corpus.\n"
            "To materialize the cache (one-time, requires network + `datasets`):\n\n"
            "  pip install datasets\n"
            "  python -m eval.medagentsbench --materialize-cache\n\n"
            "Or export eval/data/medagentsbench_test_hard.json manually from:\n"
            "  HF: super-dainiu/MedicalAgentsBench (per-source test_hard parquet shards)\n"
            "  GitHub: gersteinlab/MedicalAgentsBench\n\n"
            "For offline tests, pass items=[...] to MedAgentsBenchDataset."
        )
        super().__init__(message)


def expected_source_total() -> int:
    return int(MEDAGENTSBENCH_PROVENANCE["official_n"])


def normalize_row(raw: Mapping[str, Any], *, source_benchmark: Optional[str] = None) -> Dict[str, Any]:
    """Normalize a raw MedAgentsBench row to eval harness shape."""
    question = str(
        raw.get("question")
        or raw.get("input")
        or raw.get("prompt")
        or ""
    ).strip()
    answer = str(
        raw.get("answer")
        or raw.get("label")
        or raw.get("correct")
        or raw.get("gold")
        or ""
    ).strip()
    source = str(
        source_benchmark
        or raw.get("source_benchmark")
        or raw.get("source")
        or raw.get("dataset")
        or "unknown"
    )
    item: Dict[str, Any] = {
        "id": str(raw.get("id") or raw.get("question_id") or ""),
        "question": question,
        "answer": answer,
        "source_benchmark": source,
        "benchmark": "medagentsbench_test_hard",
        "split": "test_hard",
    }
    if raw.get("options") is not None:
        item["options"] = raw.get("options")
    if raw.get("meta_info") is not None:
        item["meta_info"] = raw.get("meta_info")
    return item


def stratified_sample_by_source(
    items: Sequence[Mapping[str, Any]],
    n: int,
    *,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Stratified sample by ``source_benchmark`` for n=20 / n=100 pilots.

    Uses largest-remainder allocation so counts sum exactly to n.
    """
    if n <= 0:
        return []
    pool = [dict(item) for item in items if item.get("question")]
    if n >= len(pool):
        return pool

    rng = random.Random(seed)
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in pool:
        by_source[str(item.get("source_benchmark") or "unknown")].append(item)

    total = len(pool)
    allocations: List[tuple[str, int, float]] = []
    assigned = 0
    for source, group in sorted(by_source.items()):
        exact = n * len(group) / total
        base = int(exact)
        allocations.append((source, base, exact - base))
        assigned += base

    # Distribute remainder to sources with highest fractional parts.
    remainder = n - assigned
    allocations.sort(key=lambda row: row[2], reverse=True)
    final_counts: Dict[str, int] = {}
    for index, (source, base, _) in enumerate(allocations):
        final_counts[source] = base + (1 if index < remainder else 0)

    sampled: List[Dict[str, Any]] = []
    for source, group in by_source.items():
        k = min(final_counts.get(source, 0), len(group))
        if k > 0:
            sampled.extend(rng.sample(group, k))

    rng.shuffle(sampled)
    return sampled[:n]


def load_local_cache(path: Path) -> List[Dict[str, Any]]:
    """Load normalized items from a local JSON cache file."""
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict) and "items" in raw:
        rows = raw["items"]
    elif isinstance(raw, dict) and "data" in raw:
        rows = raw["data"]
    else:
        raise ValueError(f"Unexpected MedAgentsBench JSON structure in {path}")

    items = [normalize_row(row) for row in rows]
    logger.info(
        "MedAgentsBench: loaded %d items from %s",
        len(items),
        path,
    )
    return items


def load_test_hard_items(
    *,
    local_path: Optional[Path] = None,
    allow_download: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load test_hard items from local cache only (default).

    Set allow_download=True to fetch from HuggingFace (not used by stub default).
    """
    path = local_path or _DEFAULT_LOCAL_CACHE
    if path.exists():
        return load_local_cache(path)

    if allow_download:
        return _load_huggingface_test_hard()

    raise MedAgentsBenchNotDownloadedError(path)


def _load_huggingface_test_hard() -> List[Dict[str, Any]]:
    """Optional HF loader — only when explicitly enabled (not stub default)."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise MedAgentsBenchNotDownloadedError() from exc

    hf_id = MEDAGENTSBENCH_PROVENANCE["huggingface_canonical"]
    source_names = list(MEDAGENTSBENCH_PROVENANCE["source_benchmark_breakdown"].keys())
    items: List[Dict[str, Any]] = []
    for source in source_names:
        try:
            ds = load_dataset(hf_id, source, split="test_hard", trust_remote_code=True)
        except Exception as exc:
            logger.warning("MedAgentsBench HF shard skipped (%s): %s", source, exc)
            continue
        for row in ds:
            items.append(normalize_row(row, source_benchmark=source))

    if not items:
        raise MedAgentsBenchNotDownloadedError()

    logger.info("MedAgentsBench: loaded %d items from HuggingFace %s", len(items), hf_id)
    return items


def provenance_metadata(*, hf_id_used: Optional[str] = None, loaded_n: Optional[int] = None) -> Dict[str, Any]:
    """Return provenance block suitable for run_meta / registry."""
    meta = dict(MEDAGENTSBENCH_PROVENANCE)
    meta["hf_id_used"] = hf_id_used or MEDAGENTSBENCH_PROVENANCE["huggingface_canonical"]
    if loaded_n is not None:
        meta["loaded_n"] = loaded_n
        meta["n_discrepancy"] = loaded_n != expected_source_total()
    return meta
