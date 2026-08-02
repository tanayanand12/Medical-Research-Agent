"""Deterministic evidence formatting shared by synthesis and repair."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from evaluation_core import document_text, stable_document_id


def build_evidence_context(
    state: Dict[str, Any],
    *,
    max_documents_per_source: int = 3,
    max_characters_per_document: int = 2000,
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    blocks: List[str] = []
    sources: List[str] = []
    included: List[Dict[str, Any]] = []
    citation_index = 0

    for tool_name in state.get("discovered_skills", []):
        tool_result = state.get("retrieval_results", {}).get(tool_name, {})
        if tool_result.get("error"):
            continue
        decision = tool_result.get("verification_decision")
        if tool_result.get("evidence_limited") or (
            isinstance(decision, dict)
            and (
                decision.get("valid") is False
                or decision.get("status") == "evidence_limited"
            )
        ):
            continue
        documents = list(tool_result.get("results") or [])
        if not documents:
            continue
        sources.append(tool_name)
        for source_rank, document in enumerate(
            documents[:max_documents_per_source], 1
        ):
            citation_index += 1
            raw_text = document_text(document)
            text = raw_text[:max_characters_per_document]
            truncated = len(text) < len(raw_text)
            document_id = str(
                document.get("document_id")
                or stable_document_id(document, tool_name, source_rank)
            )
            title = str(document.get("title") or "Untitled evidence")
            provenance = (
                document.get("provenance")
                or document.get("doi")
                or document.get("pmid")
                or document.get("nct_id")
                or document.get("record_id")
                or tool_name
            )
            nested_metadata = dict(document.get("metadata") or {})
            citation_metadata = {
                key: document.get(key, nested_metadata.get(key))
                for key in (
                    "title",
                    "authors",
                    "year",
                    "journal",
                    "volume",
                    "issue",
                    "pages",
                    "doi",
                    "pmid",
                )
            }
            blocks.append(
                f"[{citation_index}] Document ID: {document_id}\n"
                f"Source: {tool_name}; Provenance: {provenance}\n"
                f"Title: {title}\n"
                f"Evidence: {text}"
                + ("\n[Context truncated]" if truncated else "")
            )
            included.append(
                {
                    "citation_index": citation_index,
                    "document_id": document_id,
                    "tool_name": tool_name,
                    "text": text,
                    "original_length": len(raw_text),
                    "truncated": truncated,
                    "citation_metadata": citation_metadata,
                }
            )
    return "\n\n".join(blocks), sources, included


def evidence_limited_answer(reason: str) -> str:
    return (
        "The available sources provide insufficient evidence to answer this "
        f"question reliably. {reason.strip()} "
        "No unsupported answer was generated."
    ).strip()
