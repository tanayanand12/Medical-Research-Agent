"""Adapters from heterogeneous agent/orchestrator states to EvaluationTrace."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .schemas import (
    EVALUATION_TRACE_SCHEMA_VERSION,
    RUNTIME_SYNTHESIS_PROMPT_VERSION,
    AtomicClaim,
    CitationResolution,
    ContextSpan,
    EvaluationTrace,
    RerankedDocument,
    RetrievedDocument,
)


_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:,\s-]+)\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_CONFIG_KEYS = {
    "top_k",
    "max_papers",
    "max_records",
    "max_trials",
    "include_fulltext",
    "db_name",
    "index_path",
    "date_from",
    "date_to",
    "status",
    "indication",
    "population",
    "study_design",
    "retrieval_method",
    "embedding_model",
    "reranker_model",
    "reranker_revision",
    "model_id",
    "retrieval_only",
}
_METADATA_KEYS = {
    "title",
    "authors",
    "journal",
    "year",
    "date",
    "publication_date",
    "publication_type",
    "study_type",
    "record_type",
    "source",
    "source_type",
    "authority",
    "provenance",
    "doi",
    "pmid",
    "pmcid",
    "nct_id",
    "record_id",
    "url",
    "status",
    "indication",
    "population",
}


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_document_id(
    document: Dict[str, Any], domain: str, fallback_rank: int
) -> str:
    metadata = document.get("metadata", {}) or {}
    explicit_id = document.get("document_id")
    if explicit_id is not None and str(explicit_id).strip():
        return str(explicit_id).strip()
    identifiers = (
        ("pmid", metadata.get("pmid") or document.get("pmid")),
        ("pmcid", metadata.get("pmcid") or document.get("pmcid")),
        ("doi", metadata.get("doi") or document.get("doi")),
        ("nct", metadata.get("nct_id") or document.get("nct_id")),
        ("fda", metadata.get("record_id") or document.get("record_id")),
        ("doc", document.get("doc_id")),
    )
    for prefix, value in identifiers:
        if value is not None and str(value).strip():
            record_id = f"{domain}:{prefix}:{str(value).strip()}"
            text = document_text(document)
            if text:
                return f"{record_id}:chunk:{content_hash(text)[:12]}"
            return record_id

    text = document_text(document)
    digest = content_hash(
        json.dumps(
            {
                "domain": domain,
                "title": metadata.get("title") or document.get("title"),
                "text": text,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )[:20]
    return f"{domain}:sha256:{digest}"


def document_text(document: Dict[str, Any]) -> str:
    """Return the canonical evidence text used by prompts and traces."""
    return str(
        document.get("text")
        or document.get("abstract")
        or document.get("content")
        or ""
    )


def _safe_metadata(document: Dict[str, Any], domain: str) -> Dict[str, Any]:
    raw = dict(document.get("metadata", {}) or {})
    for key in _METADATA_KEYS:
        if key in document and key not in raw:
            raw[key] = document[key]
    raw.setdefault("source", domain)
    raw.setdefault("provenance", raw.get("url") or raw.get("source") or domain)
    raw.setdefault("source_type", raw.get("record_type") or domain)
    return {
        key: value
        for key, value in raw.items()
        if key in _METADATA_KEYS and value is not None
    }


def _retrieved_documents(
    values: Sequence[Dict[str, Any]], domain: str
) -> List[RetrievedDocument]:
    documents: List[RetrievedDocument] = []
    for index, value in enumerate(values, 1):
        text = document_text(value)
        metadata = _safe_metadata(value, domain)
        score = value.get("raw_score", value.get("score"))
        documents.append(
            RetrievedDocument(
                document_id=stable_document_id(value, domain, index),
                source=str(metadata.get("source", domain)),
                provenance=str(metadata.get("provenance", metadata.get("source", domain))),
                retrieval_method=str(
                    value.get("retrieval_method")
                    or metadata.get("retrieval_method")
                    or "hybrid"
                ),
                original_rank=int(value.get("original_rank") or index),
                raw_score=float(score) if isinstance(score, (int, float)) else None,
                score_type=str(value.get("score_type") or "retrieval_score"),
                content_hash=content_hash(text),
                text=text,
                metadata=metadata,
            )
        )
    return documents


def _reranked_documents(
    values: Sequence[Dict[str, Any]],
    domain: str,
    context: Dict[str, Any],
) -> List[RerankedDocument]:
    model = str(context.get("reranker_model") or "ncbi/MedCPT-Cross-Encoder")
    revision = str(context.get("reranker_revision") or "")
    documents: List[RerankedDocument] = []
    for index, value in enumerate(values, 1):
        score = value.get("reranker_score", value.get("score"))
        documents.append(
            RerankedDocument(
                document_id=stable_document_id(value, domain, index),
                rank=index,
                reranker_score=(
                    float(score) if isinstance(score, (int, float)) else None
                ),
                reranker_model=model,
                reranker_revision=revision,
            )
        )
    return documents


def _split_model(value: str, context: Dict[str, Any]) -> Tuple[str, str]:
    explicit_revision = str(context.get("model_revision") or "")
    if explicit_revision:
        return value, explicit_revision
    if "@" in value:
        model, revision = value.rsplit("@", 1)
        return model, revision
    return value, ""


def _atomic_claims(
    answer: str,
    ordered_document_ids: Sequence[str],
    citation_marker_map: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[AtomicClaim], List[CitationResolution]]:
    claims: List[AtomicClaim] = []
    resolutions: List[CitationResolution] = []
    seen_citations = set()
    for sentence in _SENTENCE_RE.split(answer.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        cited_ids: List[str] = []
        for citation_group in _CITATION_RE.findall(sentence):
            for marker in _expand_citation_group(citation_group):
                document_id: Optional[str] = None
                mapped_ids = [
                    item
                    for item in (citation_marker_map or {}).get(marker, [])
                    if item in ordered_document_ids
                ]
                if mapped_ids:
                    document_id = mapped_ids[0]
                    cited_ids.extend(mapped_ids)
                elif marker.isdigit() and not citation_marker_map:
                    position = int(marker) - 1
                    if 0 <= position < len(ordered_document_ids):
                        document_id = ordered_document_ids[position]
                elif marker in ordered_document_ids:
                    document_id = marker
                if document_id and document_id not in cited_ids:
                    cited_ids.append(document_id)
                citation_key = (marker, document_id)
                if citation_key not in seen_citations:
                    seen_citations.add(citation_key)
                    resolutions.append(
                        CitationResolution(
                            citation=f"[{marker}]",
                            document_id=document_id,
                            resolved=document_id is not None,
                            reason="" if document_id else "citation_not_in_final_context",
                        )
                    )
        claim_text = re.sub(
            r"\s+([.,;:!?])",
            r"\1",
            _CITATION_RE.sub("", sentence).strip(),
        )
        if len(claim_text.split()) < 3:
            continue
        claim_id = "claim:" + content_hash(claim_text.lower())[:16]
        claims.append(
            AtomicClaim(
                claim_id=claim_id,
                text=claim_text,
                cited_document_ids=[
                    value for value in cited_ids if value in ordered_document_ids
                ],
            )
        )
    return claims, resolutions


def _expand_citation_group(value: str) -> List[str]:
    markers: List[str] = []
    for part in (item.strip() for item in value.split(",")):
        if not part:
            continue
        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)
        if range_match:
            start, end = (int(item) for item in range_match.groups())
            if start <= end and end - start <= 20:
                markers.extend(str(item) for item in range(start, end + 1))
                continue
        markers.append(part)
    return markers


def _normalize_int_dict(value: Any) -> Dict[str, int]:
    if isinstance(value, dict):
        return {
            str(key): int(item)
            for key, item in value.items()
            if isinstance(item, (int, float))
        }
    if isinstance(value, (int, float)):
        return {"total": int(value)}
    return {}


def _normalize_float_dict(value: Any) -> Dict[str, float]:
    if isinstance(value, dict):
        return {
            str(key): float(item)
            for key, item in value.items()
            if isinstance(item, (int, float))
        }
    if isinstance(value, (int, float)):
        return {"total": float(value)}
    return {}


def build_agent_evaluation_trace(
    *,
    agent_name: str,
    domain: str,
    original_query: str,
    state: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> EvaluationTrace:
    """Build the sidecar without changing AgentOutput's existing fields."""
    context = dict(context or {})
    trace_id = str(context.get("trace_id") or uuid.uuid4())
    attempt_id = str(context.get("attempt_id") or f"{trace_id}:{agent_name}:1")
    parent_attempt_id = context.get("parent_attempt_id")
    query = str(context.get("original_query") or original_query)

    raw_retrieved = list(state.get("retrieval_results") or [])
    raw_reranked = list(state.get("reranked_results") or raw_retrieved)
    retrieved = _retrieved_documents(raw_retrieved or raw_reranked, domain)
    reranked = _reranked_documents(raw_reranked, domain, context)
    source_by_id = {item.document_id: item for item in retrieved}
    spans: List[ContextSpan] = []
    citation_marker_map: Dict[str, List[str]] = {}
    synthesis_manifest_present = "synthesis_context" in state
    if synthesis_manifest_present:
        for included in list(state.get("synthesis_context") or []):
            document_id = str(included.get("document_id") or "")
            source = source_by_id.get(document_id)
            span_text = str(included.get("text") or "")
            start_char = int(included.get("start_char") or 0)
            source_length = (
                len(source.text)
                if source is not None
                else int(included.get("original_length") or len(span_text))
            )
            spans.append(
                ContextSpan(
                    document_id=document_id,
                    start_char=start_char,
                    end_char=start_char + len(span_text),
                    text=span_text,
                    content_hash=content_hash(span_text),
                    truncated=bool(included.get("truncated", False)),
                    original_length=int(
                        included.get("original_length") or source_length
                    ),
                )
            )
            marker = included.get("citation_marker")
            if marker is None:
                marker = included.get("citation_index")
            if marker is not None:
                citation_marker_map.setdefault(str(marker), []).append(
                    document_id
                )
    ordered_ids = [span.document_id for span in spans]

    answer = str(state.get("answer") or "")
    retrieval_only = bool(
        context.get("retrieval_only") or state.get("retrieval_only")
    )
    answer_origin = str(
        state.get("answer_origin")
        or (
            "retrieval_only"
            if retrieval_only and not answer.strip()
            else "generated"
        )
    )
    claims, citation_resolutions = _atomic_claims(
        answer, ordered_ids, citation_marker_map
    )
    for index, citation in enumerate(state.get("citations") or [], 1):
        if any(item.citation == f"[{index}]" for item in citation_resolutions):
            continue
        mapped_ids = citation_marker_map.get(str(index), [])
        document_id = (
            mapped_ids[0]
            if mapped_ids
            else (ordered_ids[index - 1] if index <= len(ordered_ids) else None)
        )
        citation_resolutions.append(
            CitationResolution(
                citation=str(citation),
                document_id=document_id,
                resolved=document_id is not None,
                reason="" if document_id else "citation_without_context_document",
            )
        )

    exact_model, revision = _split_model(str(state.get("model_used") or ""), context)
    errors: List[str] = []
    for value in (
        state.get("error"),
        *(state.get("errors") or []),
        *(state.get("error_messages") or []),
    ):
        if value and str(value) not in errors:
            errors.append(str(value))
    retry_feedback = context.get("verification_feedback") or []
    if isinstance(retry_feedback, dict):
        retry_feedback = [retry_feedback]

    stage_latency = {
        key: float(value)
        for key, value in {
            "retrieval": state.get("retrieval_time_sec"),
            "total": state.get("execution_time_sec"),
        }.items()
        if isinstance(value, (int, float))
    }
    stage_latency.update(
        {
            str(key): float(value)
            for key, value in dict(state.get("stage_latency_sec") or {}).items()
            if isinstance(value, (int, float))
        }
    )

    return EvaluationTrace(
        schema_version=EVALUATION_TRACE_SCHEMA_VERSION,
        trace_id=trace_id,
        attempt_id=attempt_id,
        parent_attempt_id=str(parent_attempt_id) if parent_attempt_id else None,
        agent_name=agent_name,
        domain=domain,
        original_query=query,
        expanded_queries=[
            str(state["expanded_query"])
        ]
        if state.get("expanded_query")
        else [],
        retrieval_configuration={
            key: value for key, value in context.items() if key in _CONFIG_KEYS
        },
        retrieved_documents=retrieved,
        reranked_documents=reranked,
        final_context_document_ids=ordered_ids,
        final_context_spans=spans,
        answer=answer,
        atomic_claims=claims,
        citations=citation_resolutions,
        stage_latency_sec=stage_latency,
        token_usage=_normalize_int_dict(state.get("token_usage")),
        cost_breakdown_usd=_normalize_float_dict(state.get("cost_breakdown_usd")),
        exact_model=exact_model or str(context.get("model_id") or "unknown"),
        model_revision=revision,
        prompt_version=str(
            context.get("prompt_version") or RUNTIME_SYNTHESIS_PROMPT_VERSION
        ),
        config_version=str(context.get("config_version") or "runtime-v1"),
        errors=errors,
        partial_response=bool(
            errors
            or state.get("partial_response")
            or state.get("is_partial_response")
        ),
        retry_feedback=list(retry_feedback),
        repair_history=list(context.get("repair_history") or []),
        synthesis_manifest_present=synthesis_manifest_present,
        answer_origin=answer_origin,
        attempt_events=list(
            state.get("attempt_events")
            or state.get("attempt_telemetry")
            or []
        ),
    )


def build_orchestrator_evaluation_trace(
    state: Dict[str, Any],
    *,
    answer: Optional[str] = None,
    attempt_id: Optional[str] = None,
    parent_attempt_id: Optional[str] = None,
) -> EvaluationTrace:
    """Adapt the top-level synthesis state using only production evidence."""
    domain = "multi_source"
    flattened: List[Dict[str, Any]] = []
    for tool_name in state.get("discovered_skills", []):
        result = state.get("retrieval_results", {}).get(tool_name, {})
        for source_rank, document in enumerate(result.get("results", []), 1):
            item = dict(document)
            item.setdefault("metadata", {})
            item["metadata"] = dict(item["metadata"])
            item["metadata"].setdefault("source", tool_name)
            item.setdefault(
                "document_id",
                stable_document_id(item, tool_name, source_rank),
            )
            flattened.append(item)

    context = dict(state.get("context") or {})
    context.update(
        {
            "trace_id": state.get("trace_id"),
            "attempt_id": attempt_id
            or f"{state.get('trace_id', 'trace')}:orchestrator:1",
            "parent_attempt_id": parent_attempt_id,
        }
    )
    context_documents = []
    flattened_by_id = {
        str(item.get("document_id")): item for item in flattened
    }
    for included in state.get("synthesis_context", []):
        document_id = str(included.get("document_id") or "")
        item = dict(flattened_by_id.get(document_id, {}))
        item["document_id"] = document_id
        item["text"] = str(included.get("text") or "")
        context_documents.append(item)

    synthetic_state = {
        "expanded_query": state.get("input_query", ""),
        "retrieval_results": flattened,
        "reranked_results": context_documents or flattened,
        "answer": answer
        if answer is not None
        else state.get("intermediate_answer", ""),
        "citations": [],
        "model_used": state.get("intermediate_model_used", ""),
        "retrieval_time_sec": state.get("total_retrieval_time_sec", 0.0),
        "execution_time_sec": state.get("synthesis_time_sec", 0.0),
        "token_usage": {
            "input": state.get("synthesis_tokens_in", 0),
            "output": state.get("synthesis_tokens_out", 0),
            "total": int(state.get("synthesis_tokens_in", 0) or 0)
            + int(state.get("synthesis_tokens_out", 0) or 0),
        },
        "cost_breakdown_usd": {
            "synthesis": state.get(
                "last_synthesis_cost_usd",
                state.get("cost_estimate", 0.0),
            )
        },
        "errors": list(state.get("error_messages") or []),
        "partial_response": bool(
            state.get("is_partial_response") or state.get("error_occurred")
        ),
        "answer_origin": state.get("answer_origin", "generated"),
        "attempt_events": list(state.get("attempt_telemetry") or []),
    }
    if "synthesis_context" in state:
        synthetic_state["synthesis_context"] = list(
            state.get("synthesis_context") or []
        )
    trace = build_agent_evaluation_trace(
        agent_name="orchestrator",
        domain=domain,
        original_query=str(state.get("input_query") or ""),
        state=synthetic_state,
        context=context,
    )
    if state.get("synthesis_context"):
        trace.final_context_document_ids = [
            str(item.get("document_id"))
            for item in state["synthesis_context"]
            if item.get("document_id")
        ]
        trace.final_context_spans = [
            ContextSpan(
                document_id=str(item["document_id"]),
                start_char=int(item.get("start_char") or 0),
                end_char=(
                    int(item.get("start_char") or 0)
                    + len(str(item.get("text") or ""))
                ),
                text=str(item.get("text") or ""),
                content_hash=content_hash(str(item.get("text") or "")),
                truncated=bool(item.get("truncated", False)),
                original_length=int(
                    item.get("original_length")
                    or len(str(item.get("text") or ""))
                ),
            )
            for item in state["synthesis_context"]
            if item.get("document_id")
        ]
    return trace
