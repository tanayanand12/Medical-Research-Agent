"""
metrics.py — Phase 8: Evaluation metrics for Medical Research Agent.

Implements five metrics, each returning a score in [0, 1]:

1. **faithfulness**      — answer stays true to retrieved context (LLM-as-judge)
2. **answer_relevancy**  — answer directly addresses the question (LLM-as-judge)
3. **answer_correctness** — factual agreement with the reference answer
4. **citation_fidelity** — citations in the answer trace back to retrieved sources
5. **hallucination_rate** — fraction of answer claims *not* grounded in context

All LLM calls route through ``LLMClient`` (zero hardcoded providers).
"""

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow imports when run from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------- #
# Data containers
# ---------------------------------------------------------------------- #


@dataclass
class EvalSample:
    """A single evaluation sample fed to the metric functions."""

    question: str
    answer: str
    expected_answer: str = ""
    retrieved_contexts: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)


@dataclass
class MetricResult:
    """Result of a single metric computation."""

    name: str
    score: float  # [0, 1]
    reason: str = ""
    raw: Optional[Dict[str, Any]] = None
    valid: bool = True


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

_LLM: Optional[LLMClient] = None


def _get_llm() -> LLMClient:
    global _LLM
    if _LLM is None:
        _LLM = LLMClient()
    return _LLM


def _parse_judge_response(raw: str) -> Dict[str, Any]:
    """Parse and validate a judge response without silently inventing zeroes."""
    text = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.I)
    if fenced:
        text = fenced.group(1).strip()

    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
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
        raise ValueError("Judge response did not contain valid JSON")

    score = parsed.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("Judge response did not contain a numeric score")
    if not 0.0 <= float(score) <= 1.0:
        raise ValueError("Judge score must be between 0 and 1")

    parsed["score"] = float(score)
    parsed["reason"] = str(parsed.get("reason", ""))
    return parsed


def _parse_metric_group_response(
    raw: str, metric_names: tuple[str, ...]
) -> Dict[str, Dict[str, Any]]:
    """Validate a structured response containing a named metric group."""
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
        raise ValueError("Combined judge response did not contain valid JSON")

    validated: Dict[str, Dict[str, Any]] = {}
    for name in metric_names:
        value = parsed.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"Combined judge response missing {name}")
        score = value.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"Combined judge metric {name} has invalid score")
        if not 0.0 <= float(score) <= 1.0:
            raise ValueError(f"Combined judge metric {name} is out of range")
        validated[name] = {
            **value,
            "score": float(score),
            "reason": str(value.get("reason", "")),
        }
    return validated


def _parse_combined_judge_response(raw: str) -> Dict[str, Dict[str, Any]]:
    """Backward-compatible parser for the original four-metric response."""
    return _parse_metric_group_response(
        raw,
        (
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
            "hallucination_rate",
        ),
    )


def _grounding_llm_judge(
    sample: EvalSample, model: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """Judge context grounding without exposing the reference answer."""
    context_block = (
        "\n---\n".join(sample.retrieved_contexts[:10])
        if sample.retrieved_contexts
        else "[NO RETRIEVED CONTEXT]"
    )
    prompt = (
        "You are an impartial medical question-answering evaluator. Assess "
        "the candidate answer using only the retrieved context. Faithfulness "
        "is support by retrieved context. "
        "Hallucination rate is the fraction of atomic factual claims not "
        "supported by retrieved context (lower is better).\n\n"
        f"QUESTION:\n{sample.question}\n\n"
        f"RETRIEVED CONTEXT:\n{context_block}\n\n"
        f"CANDIDATE ANSWER:\n{sample.answer}\n\n"
        "Respond ONLY with JSON in this exact shape, using scores from 0 to 1:\n"
        '{"faithfulness":{"score":0.0,"reason":""},'
        '"hallucination_rate":{"score":0.0,"reason":""}}'
    )
    raw = _get_llm().chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        max_tokens=2048,
    )
    return _parse_metric_group_response(
        raw, ("faithfulness", "hallucination_rate")
    )


def _quality_llm_judge(
    sample: EvalSample, model: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """Judge question relevance and reference-answer correctness."""
    reference = sample.expected_answer or "[NO REFERENCE ANSWER]"
    prompt = (
        "You are an impartial medical question-answering evaluator. Assess "
        "the candidate answer on two dimensions. Answer relevancy is "
        "directness and completeness for the question. Answer correctness is "
        "factual agreement with the reference without requiring identical "
        "wording.\n\n"
        f"QUESTION:\n{sample.question}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"CANDIDATE ANSWER:\n{sample.answer}\n\n"
        "Respond ONLY with JSON in this exact shape, using scores from 0 to 1:\n"
        '{"answer_relevancy":{"score":0.0,"reason":""},'
        '"answer_correctness":{"score":0.0,"reason":""}}'
    )
    raw = _get_llm().chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        max_tokens=2048,
    )
    return _parse_metric_group_response(
        raw, ("answer_relevancy", "answer_correctness")
    )


def _llm_judge(prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Ask the LLM for a validated JSON score and explanation."""
    llm = _get_llm()
    raw = llm.chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        max_tokens=2048,
    )
    return _parse_judge_response(raw)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------- #
# Metric 1: Faithfulness
# ---------------------------------------------------------------------- #


def faithfulness(
    sample: EvalSample, model: Optional[str] = None
) -> MetricResult:
    """Score whether the answer is faithful to the retrieved context.

    Uses an LLM judge to check that every claim in the answer can be
    traced to the provided context passages.  Returns 0 when the answer
    fabricates information; 1 when fully grounded.
    """
    if not sample.retrieved_contexts:
        return MetricResult(
            name="faithfulness",
            score=0.0,
            reason="No retrieved context provided — cannot assess faithfulness.",
            valid=False,
        )

    context_block = "\n---\n".join(sample.retrieved_contexts[:10])
    prompt = (
        "You are an impartial evaluator.  Given the CONTEXT passages and "
        "the ANSWER, score how faithful the answer is to the context.\n\n"
        "A faithful answer only contains claims that are supported by the "
        "context.  Penalise fabricated facts, unsupported statistics, and "
        "hallucinated citations.\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"ANSWER:\n{sample.answer}\n\n"
        "Respond ONLY with a JSON object: "
        '{"score": <float 0-1>, "reason": "<brief explanation>"}'
    )

    result = _llm_judge(prompt, model=model)
    return MetricResult(
        name="faithfulness",
        score=_clamp(result.get("score", 0.0)),
        reason=result.get("reason", ""),
        raw=result,
    )


# ---------------------------------------------------------------------- #
# Metric 2: Answer relevancy
# ---------------------------------------------------------------------- #


def answer_relevancy(
    sample: EvalSample, model: Optional[str] = None
) -> MetricResult:
    """Score whether the answer directly addresses the question.

    Uses an LLM judge.  Returns 1 when the answer is a precise, complete
    response to the question; 0 when off-topic or vacuous.
    """
    prompt = (
        "You are an impartial evaluator.  Given the QUESTION and the "
        "ANSWER, score how relevant and complete the answer is.\n\n"
        "A relevant answer directly addresses the question, is specific, "
        "and does not contain excessive tangential information.\n\n"
        f"QUESTION:\n{sample.question}\n\n"
        f"ANSWER:\n{sample.answer}\n\n"
    )
    if sample.expected_answer:
        prompt += (
            "REFERENCE ANSWER (for comparison — do not penalise the answer "
            "for phrasing differences, only factual coverage):\n"
            f"{sample.expected_answer}\n\n"
        )
    prompt += (
        "Respond ONLY with a JSON object: "
        '{"score": <float 0-1>, "reason": "<brief explanation>"}'
    )

    result = _llm_judge(prompt, model=model)
    return MetricResult(
        name="answer_relevancy",
        score=_clamp(result.get("score", 0.0)),
        reason=result.get("reason", ""),
        raw=result,
    )


# ---------------------------------------------------------------------- #
# Metric 3: Answer correctness
# ---------------------------------------------------------------------- #


def answer_correctness(
    sample: EvalSample, model: Optional[str] = None
) -> MetricResult:
    """Score factual agreement with an author-provided reference answer."""
    if not sample.expected_answer:
        return MetricResult(
            name="answer_correctness",
            score=0.0,
            reason="No reference answer provided.",
            valid=False,
        )

    prompt = (
        "You are an impartial medical question-answering evaluator. "
        "Compare the CANDIDATE ANSWER with the REFERENCE ANSWER for the "
        "given QUESTION. Score factual correctness and coverage, not style. "
        "Penalize contradictions, clinically important omissions, invented "
        "facts, and incorrect certainty. Do not require verbatim wording.\n\n"
        f"QUESTION:\n{sample.question}\n\n"
        f"REFERENCE ANSWER:\n{sample.expected_answer}\n\n"
        f"CANDIDATE ANSWER:\n{sample.answer}\n\n"
        "Respond ONLY with a JSON object: "
        '{"score": <float 0-1>, "reason": "<brief explanation>"}'
    )
    result = _llm_judge(prompt, model=model)
    return MetricResult(
        name="answer_correctness",
        score=_clamp(result.get("score", 0.0)),
        reason=result.get("reason", ""),
        raw=result,
    )


# ---------------------------------------------------------------------- #
# Metric 4: Citation fidelity
# ---------------------------------------------------------------------- #

_CITATION_RE = re.compile(r"\[(\d+)\]")


def citation_fidelity(sample: EvalSample, **_: Any) -> MetricResult:
    """Score whether citations in the answer reference real retrieved sources.

    This is a deterministic metric (no LLM call).

    * Extracts ``[N]`` citation markers from the answer.
    * Checks each N against the actual citation inventory returned by the agent.
    * Score = valid_citations / total_citations.
    """
    cited_ids = _CITATION_RE.findall(sample.answer)
    if not cited_ids:
        return MetricResult(
            name="citation_fidelity",
            score=0.0,
            reason="Answer contains no citation markers despite a retrieval context.",
        )

    n_sources = len(sample.citations)
    valid = sum(1 for c in cited_ids if 1 <= int(c) <= n_sources)
    total = len(cited_ids)
    score = valid / total if total else 1.0

    return MetricResult(
        name="citation_fidelity",
        score=_clamp(score),
        reason=f"{valid}/{total} citation markers reference valid sources "
        f"(max source index = {n_sources}).",
    )


# ---------------------------------------------------------------------- #
# Metric 5: Hallucination rate
# ---------------------------------------------------------------------- #


def hallucination_rate(
    sample: EvalSample, model: Optional[str] = None
) -> MetricResult:
    """Estimate the fraction of claims in the answer that are hallucinated.

    Uses an LLM judge to decompose the answer into atomic claims, then
    check each against the retrieved context. Lower is better; 0.0 means
    every claim is supported and 1.0 means no claims are supported.
    """
    if not sample.retrieved_contexts:
        return MetricResult(
            name="hallucination_rate",
            score=0.0,
            reason="No retrieved context — cannot verify claims.",
            valid=False,
        )

    context_block = "\n---\n".join(sample.retrieved_contexts[:10])
    prompt = (
        "You are an impartial evaluator.\n\n"
        "1. Decompose the ANSWER into a list of atomic factual claims.\n"
        "2. For each claim, decide if it is SUPPORTED or NOT SUPPORTED "
        "by the CONTEXT.\n"
        "3. Compute hallucination_rate = (NOT SUPPORTED) / (total claims).\n"
        "4. Return score = hallucination_rate "
        "(so 0.0 means no hallucinations).\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"ANSWER:\n{sample.answer}\n\n"
        "Respond ONLY with a JSON object: "
        '{"score": <float 0-1>, "total_claims": <int>, '
        '"supported_claims": <int>, "reason": "<brief explanation>"}'
    )

    result = _llm_judge(prompt, model=model)
    return MetricResult(
        name="hallucination_rate",
        score=_clamp(result.get("score", 0.0)),
        reason=result.get("reason", ""),
        raw=result,
    )


# ---------------------------------------------------------------------- #
# Convenience: run all metrics on a sample
# ---------------------------------------------------------------------- #

ALL_METRICS = [
    faithfulness,
    answer_relevancy,
    answer_correctness,
    citation_fidelity,
    hallucination_rate,
]


def compute_all_metrics(
    sample: EvalSample, model: Optional[str] = None
) -> List[MetricResult]:
    """Run every metric on *sample* and return the list of results.

    LLM-based metrics use *model* (or default from LLMClient).
    Deterministic metrics ignore the model parameter.
    """
    results_by_name: Dict[str, MetricResult] = {}
    judge_groups = (
        (
            ("faithfulness", "hallucination_rate"),
            _grounding_llm_judge,
        ),
        (
            ("answer_relevancy", "answer_correctness"),
            _quality_llm_judge,
        ),
    )
    for names, judge in judge_groups:
        if names[0] == "faithfulness" and not sample.retrieved_contexts:
            for name in names:
                results_by_name[name] = MetricResult(
                    name=name,
                    score=0.0,
                    reason="No retrieved context provided.",
                    valid=False,
                )
            continue
        try:
            judged = judge(sample, model=model)
            for name in names:
                value = judged[name]
                valid = not (
                    name == "answer_correctness" and not sample.expected_answer
                )
                reason = (
                    "No reference answer provided."
                    if not valid
                    else value["reason"]
                )
                results_by_name[name] = MetricResult(
                    name=name,
                    score=_clamp(value["score"]),
                    reason=reason,
                    raw=value,
                    valid=valid,
                )
        except Exception as exc:
            logger.warning("%s judge failed: %s", names[0], exc)
            for name in names:
                results_by_name[name] = MetricResult(
                    name=name,
                    score=0.0,
                    reason=f"Metric computation failed: {exc}",
                    valid=False,
                )

    try:
        results_by_name["citation_fidelity"] = citation_fidelity(sample)
    except Exception as exc:
        results_by_name["citation_fidelity"] = MetricResult(
            name="citation_fidelity",
            score=0.0,
            reason=f"Metric computation failed: {exc}",
            valid=False,
        )

    return [results_by_name[metric.__name__] for metric in ALL_METRICS]
