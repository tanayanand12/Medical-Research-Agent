"""Transparent runtime confidence aggregation (not clinical calibration)."""

from __future__ import annotations

from typing import Dict, Tuple


CONFIDENCE_WEIGHTS = {
    "retrieval_coverage": 0.15,
    "evidence_sufficiency": 0.15,
    "claim_grounding": 0.30,
    "citation_support": 0.15,
    "query_coverage": 0.15,
    "verifier_confidence": 0.10,
}


def calculate_combined_confidence(
    components: Dict[str, float],
) -> Tuple[float, str]:
    """Return a bounded weighted quality indicator and its explicit formula."""
    score = sum(
        CONFIDENCE_WEIGHTS[name]
        * max(0.0, min(1.0, float(components.get(name, 0.0))))
        for name in CONFIDENCE_WEIGHTS
    )
    formula = " + ".join(
        f"{weight:.2f}*{name}" for name, weight in CONFIDENCE_WEIGHTS.items()
    )
    explanation = (
        f"Combined runtime quality score = {formula}. "
        "This is not a clinically calibrated probability."
    )
    return round(max(0.0, min(1.0, score)), 6), explanation
