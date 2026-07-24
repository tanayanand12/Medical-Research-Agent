"""Retrieval-stage and grounding metrics for post-hoc technical evaluation."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from eval.metrics import EvalSample, MetricResult, _get_llm

_RETRIEVAL_METRICS = (
    "context_relevance",
    "context_sufficiency",
    "citation_claim_support",
)
_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _parse_retrieval_judge_response(raw: str) -> Dict[str, Dict[str, Any]]:
    """Parse a structured retrieval-stage judgment."""
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.I)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    candidates = [text]
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    parsed: Optional[Dict[str, Any]] = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed = value
            break
    if parsed is None:
        raise ValueError("Retrieval judge response did not contain valid JSON")

    validated: Dict[str, Dict[str, Any]] = {}
    for name in _RETRIEVAL_METRICS:
        value = parsed.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"Retrieval judge response missing {name}")
        score = value.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"Retrieval metric {name} has invalid score")
        if not 0.0 <= float(score) <= 1.0:
            raise ValueError(f"Retrieval metric {name} is out of range")
        validated[name] = {
            **value,
            "score": float(score),
            "reason": str(value.get("reason", "")),
        }
    return validated


def sentence_citation_coverage(sample: EvalSample) -> MetricResult:
    """Fraction of factual answer sentences containing a valid citation marker."""
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_RE.split(sample.answer.strip())
        if len(re.findall(r"[A-Za-z]+", sentence)) >= 3
    ]
    if not sentences:
        return MetricResult(
            name="sentence_citation_coverage",
            score=0.0,
            reason="No factual sentences were detected.",
            valid=False,
        )

    citation_count = len(sample.citations)
    cited = 0
    for sentence in sentences:
        markers = [int(value) for value in _CITATION_RE.findall(sentence)]
        if any(1 <= marker <= citation_count for marker in markers):
            cited += 1
    score = cited / len(sentences)
    raw = {"cited_sentences": cited, "total_sentences": len(sentences)}
    return MetricResult(
        name="sentence_citation_coverage",
        score=score,
        reason=f"{cited}/{len(sentences)} factual sentences contain valid citations.",
        raw=raw,
    )


def compute_retrieval_grounding_metrics(
    sample: EvalSample,
    model: Optional[str] = None,
) -> List[MetricResult]:
    """Evaluate retrieval relevance, sufficiency, and cited-claim support."""
    coverage = sentence_citation_coverage(sample)
    if not sample.retrieved_contexts:
        invalid = [
            MetricResult(
                name=name,
                score=0.0,
                reason="No retrieved context was available.",
                valid=False,
            )
            for name in _RETRIEVAL_METRICS
        ]
        return [*invalid, coverage]

    contexts = "\n\n".join(
        f"[CONTEXT {index + 1}]\n{text}"
        for index, text in enumerate(sample.retrieved_contexts[:10])
    )
    prompt = (
        "You are an impartial evaluator of a medical retrieval-augmented "
        "generation system. Score three retrieval-stage properties from 0 to 1.\n"
        "1. context_relevance: fraction of retrieved passages that materially "
        "help answer the QUESTION.\n"
        "2. context_sufficiency: fraction of clinically important facts in the "
        "REFERENCE ANSWER supported by the RETRIEVED CONTEXTS.\n"
        "3. citation_claim_support: fraction of factual claims in the CANDIDATE "
        "ANSWER that contain [N] markers and are supported by at least one "
        "retrieved passage. This tests support by the retrieved set, not strict "
        "entailment by the individually numbered bibliography entry.\n\n"
        f"QUESTION:\n{sample.question}\n\n"
        f"REFERENCE ANSWER:\n{sample.expected_answer}\n\n"
        f"RETRIEVED CONTEXTS:\n{contexts}\n\n"
        f"CANDIDATE ANSWER:\n{sample.answer}\n\n"
        "Respond ONLY with JSON in this exact shape:\n"
        '{"context_relevance":{"score":0.0,"reason":""},'
        '"context_sufficiency":{"score":0.0,"reason":""},'
        '"citation_claim_support":{"score":0.0,"reason":""}}'
    )
    raw = _get_llm().chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        max_tokens=1536,
    )
    judged = _parse_retrieval_judge_response(raw)
    results = [
        MetricResult(
            name=name,
            score=judged[name]["score"],
            reason=judged[name]["reason"],
            raw=judged[name],
        )
        for name in _RETRIEVAL_METRICS
    ]
    return [*results, coverage]
