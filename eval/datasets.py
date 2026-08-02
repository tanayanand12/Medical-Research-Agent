"""
datasets.py — Phase 8: Benchmark dataset loaders.

Provides two loaders plus a generic CSV/JSON loader:

* :class:`MedQADataset`  — USMLE-style multiple-choice QA (Jin et al., 2021)
* :class:`BioASQDataset` — BioASQ biomedical QA (Tsatsaronis et al., 2015)
* :class:`MedAgentsBenchDataset` — MedAgentsBench test_hard (Tang et al., 2025; separate from MedQA)
* :class:`CustomDataset`  — load any CSV/JSON with ``question`` and ``answer`` columns

Common interface
----------------
Every dataset implements:

    dataset[i]              → {"question": str, "answer": str, ...}
    dataset.get_questions() → List[str]
    dataset.get_expected_answer(question) → str
    len(dataset)            → int

Data sources
------------
Loaders look for local files first (``eval/data/<name>.json``), then fall
back to HuggingFace ``datasets`` library when available.
"""

import csv
import json
import logging
import random
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------- #
# Abstract base
# ---------------------------------------------------------------------- #


class BaseDataset(ABC):
    """Abstract benchmark dataset."""

    name: str = ""

    def __init__(
        self,
        n_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self._items: List[Dict[str, Any]] = []
        self._load()
        if n_samples is not None and n_samples < len(self._items):
            rng = random.Random(seed)
            self._items = rng.sample(self._items, n_samples)
        self._question_idx: Dict[str, int] = {
            item["question"]: i for i, item in enumerate(self._items)
        }

    @abstractmethod
    def _load(self) -> None:
        """Populate ``self._items`` — each item is a dict with at least
        ``question`` and ``answer`` keys."""

    # ---- public interface ------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._items[idx]

    def get_questions(self) -> List[str]:
        """Return all questions in the dataset."""
        return [item["question"] for item in self._items]

    def get_expected_answer(self, question: str) -> str:
        """Return the reference answer for *question*."""
        idx = self._question_idx.get(question)
        if idx is None:
            return ""
        return self._items[idx].get("answer", "")

    def get_context(self, question: str) -> Dict[str, Any]:
        """Return optional per-question context (documents, top_k, etc.)."""
        idx = self._question_idx.get(question)
        if idx is None:
            return {}
        item = self._items[idx]
        ctx = item.get("context", {})
        if not isinstance(ctx, dict):
            ctx = {}
        if "documents" in item and "documents" not in ctx:
            ctx = {**ctx, "documents": item["documents"]}
        if "top_k" in item and "top_k" not in ctx:
            ctx = {**ctx, "top_k": item["top_k"]}
        return ctx

    def get_target_agents(self, question: str) -> List[str]:
        """Return agents this question is designed for (empty = all)."""
        idx = self._question_idx.get(question)
        if idx is None:
            return []
        agents = self._items[idx].get("target_agents", [])
        return agents if isinstance(agents, list) else []


# ---------------------------------------------------------------------- #
# MedQA
# ---------------------------------------------------------------------- #


class MedQADataset(BaseDataset):
    """MedQA USMLE-style biomedical QA dataset.

    Looks for ``eval/data/medqa.json`` — a JSON array of objects with
    ``question``, ``answer``, and optional ``options`` / ``meta_info`` keys.

    Falls back to HuggingFace ``datasets`` (``GBaker/MedQA-USMLE-4-options``)
    if the local file is absent and the library is installed.
    """

    name = "medqa"

    def _load(self) -> None:
        local = _DATA_DIR / "medqa.json"
        if local.exists():
            self._load_local(local)
            return
        self._load_huggingface()

    def _load_local(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, list):
            self._items = raw
        elif isinstance(raw, dict) and "data" in raw:
            self._items = raw["data"]
        else:
            raise ValueError(f"Unexpected MedQA JSON structure in {path}")
        logger.info("MedQA: loaded %d items from %s", len(self._items), path)

    def _load_huggingface(self) -> None:
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]

            ds = load_dataset(
                "GBaker/MedQA-USMLE-4-options", split="test", trust_remote_code=True
            )
            for row in ds:
                options = row.get("options", {})
                answer_key = row.get("answer_idx", row.get("answer", ""))
                answer_text = options.get(answer_key, str(answer_key))
                self._items.append(
                    {
                        "question": row["question"],
                        "answer": answer_text,
                        "options": options,
                        "meta_info": row.get("meta_info", ""),
                    }
                )
            logger.info(
                "MedQA: loaded %d items from HuggingFace", len(self._items)
            )
        except ImportError:
            logger.warning(
                "MedQA: neither local file nor HuggingFace datasets available. "
                "Place eval/data/medqa.json or install `datasets` package."
            )
        except Exception as exc:
            logger.warning("MedQA: HuggingFace load failed: %s", exc)


# ---------------------------------------------------------------------- #
# BioASQ
# ---------------------------------------------------------------------- #


class BioASQDataset(BaseDataset):
    """BioASQ biomedical QA dataset.

    Looks for ``eval/data/bioasq.json`` — a JSON array (or ``{"questions": [...]}``)
    with ``question`` (or ``body``) and ``answer`` (or ``ideal_answer``) keys.

    Falls back to HuggingFace ``rag-datasets/rag-mini-bioasq`` if available.
    """

    name = "bioasq"

    def _load(self) -> None:
        local = _DATA_DIR / "bioasq.json"
        if local.exists():
            self._load_local(local)
            return
        self._load_huggingface()

    def _load_local(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        items = raw if isinstance(raw, list) else raw.get("questions", [])
        for item in items:
            q = item.get("question", item.get("body", ""))
            a = item.get("answer", "")
            if not a:
                ideal = item.get("ideal_answer", [])
                a = ideal[0] if isinstance(ideal, list) and ideal else str(ideal)
            if q:
                self._items.append({"question": q, "answer": a})
        logger.info("BioASQ: loaded %d items from %s", len(self._items), path)

    def _load_huggingface(self) -> None:
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]

            ds = load_dataset(
                "rag-datasets/rag-mini-bioasq",
                "question-answer-passages",
                split="test",
                trust_remote_code=True,
            )
            for row in ds:
                q = row.get("question", "")
                a = row.get("answer", "")
                if q:
                    self._items.append({"question": q, "answer": a})
            logger.info(
                "BioASQ: loaded %d items from HuggingFace", len(self._items)
            )
        except ImportError:
            logger.warning(
                "BioASQ: neither local file nor HuggingFace datasets available. "
                "Place eval/data/bioasq.json or install `datasets` package."
            )
        except Exception as exc:
            logger.warning("BioASQ: HuggingFace load failed: %s", exc)


# ---------------------------------------------------------------------- #
# MedAgentsBench (test_hard — separate from standalone MedQA)
# ---------------------------------------------------------------------- #


class MedAgentsBenchDataset(BaseDataset):
    """MedAgentsBench ``test_hard`` split (N=862).

    **Not merged with** :class:`MedQADataset`. The 100 MedQA-hard items here are
    already inside the 862; standalone MedQA (~1,273) remains a separate benchmark.

    Loads from ``eval/data/medagentsbench_test_hard.json`` when present.
    Otherwise raises :class:`eval.medagentsbench.MedAgentsBenchNotDownloadedError`
    with materialization instructions (no auto-download in stub mode).

    Supports stratified sampling by ``source_benchmark`` for pilot sizes (n=20, 100).
    """

    name = "medagentsbench_test_hard"

    def __init__(
        self,
        n_samples: Optional[int] = None,
        seed: int = 42,
        *,
        items: Optional[List[Dict[str, Any]]] = None,
        local_path: Optional[str] = None,
        allow_download: bool = False,
        stratified: bool = True,
    ) -> None:
        self._injected_items = items
        self._local_path = Path(local_path) if local_path else None
        self._allow_download = allow_download
        self._stratified = stratified
        self._provenance: Dict[str, Any] = {}
        self._items: List[Dict[str, Any]] = []
        self._load()
        if n_samples is not None and n_samples < len(self._items):
            if self._stratified and self._has_source_labels():
                from eval.medagentsbench import stratified_sample_by_source

                self._items = stratified_sample_by_source(
                    self._items, n_samples, seed=seed
                )
            else:
                rng = random.Random(seed)
                self._items = rng.sample(self._items, n_samples)
        self._question_idx: Dict[str, int] = {
            item["question"]: i for i, item in enumerate(self._items)
        }

    def _has_source_labels(self) -> bool:
        return any(item.get("source_benchmark") for item in self._items)

    def _load(self) -> None:
        from eval.medagentsbench import (
            load_test_hard_items,
            provenance_metadata,
        )

        if self._injected_items is not None:
            self._items = list(self._injected_items)
            self._provenance = provenance_metadata(loaded_n=len(self._items))
            logger.info(
                "MedAgentsBench: loaded %d injected items", len(self._items)
            )
            return

        self._items = load_test_hard_items(
            local_path=self._local_path,
            allow_download=self._allow_download,
        )
        self._provenance = provenance_metadata(loaded_n=len(self._items))

    def get_provenance(self) -> Dict[str, Any]:
        """Return provenance metadata (paper, arXiv, HF id, split, N, source breakdown)."""
        return dict(self._provenance)

    def get_source_benchmark(self, question: str) -> str:
        idx = self._question_idx.get(question)
        if idx is None:
            return ""
        return str(self._items[idx].get("source_benchmark") or "")

    def source_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in self._items:
            source = str(item.get("source_benchmark") or "unknown")
            counts[source] = counts.get(source, 0) + 1
        return counts


# ---------------------------------------------------------------------- #
# Custom (CSV / JSON)
# ---------------------------------------------------------------------- #


class CustomDataset(BaseDataset):
    """Load a custom evaluation dataset from a CSV or JSON file.

    Expected columns / keys: ``question``, ``answer``.
    """

    name = "custom"

    def __init__(
        self,
        path: str,
        n_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self._path = Path(path)
        super().__init__(n_samples=n_samples, seed=seed)

    def _load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Custom dataset not found: {self._path}")
        suffix = self._path.suffix.lower()
        if suffix == ".json":
            self._load_json()
        elif suffix == ".csv":
            self._load_csv()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def _load_json(self) -> None:
        with open(self._path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        items = raw if isinstance(raw, list) else raw.get("data", [])
        for item in items:
            if "question" in item:
                self._items.append(item)
        logger.info(
            "Custom: loaded %d items from %s", len(self._items), self._path
        )

    def _load_csv(self) -> None:
        with open(self._path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if "question" in row:
                    self._items.append(dict(row))
        logger.info(
            "Custom: loaded %d items from %s", len(self._items), self._path
        )


# ---------------------------------------------------------------------- #
# Registry
# ---------------------------------------------------------------------- #

DATASET_REGISTRY: Dict[str, type] = {
    "medqa": MedQADataset,
    "bioasq": BioASQDataset,
    "medagentsbench_test_hard": MedAgentsBenchDataset,
}


def get_dataset(
    name: str,
    n_samples: Optional[int] = None,
    seed: int = 42,
    path: Optional[str] = None,
) -> BaseDataset:
    """Instantiate a dataset by name.

    Parameters
    ----------
    name : str
        ``"medqa"``, ``"bioasq"``, ``"medagentsbench_test_hard"``, or ``"custom"``.
    n_samples : int, optional
        Sub-sample the dataset to this size.
    seed : int
        Random seed for reproducible sub-sampling.
    path : str, optional
        File path (required for ``"custom"``).
    """
    if name == "custom":
        if path is None:
            raise ValueError("--path is required for custom datasets")
        return CustomDataset(path=path, n_samples=n_samples, seed=seed)
    cls = DATASET_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(DATASET_REGISTRY.keys()) + ['custom']}"
        )
    return cls(n_samples=n_samples, seed=seed)
