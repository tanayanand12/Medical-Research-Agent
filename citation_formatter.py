"""
citation_formatter.py — AMA-style citation formatting utility.

Reused from legacy codebase (no LLM calls, pure formatting).
Updated to handle both single-dict and list inputs for Phase 4 compatibility.
"""

import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


def format_citations_to_ama(citations: Union[Dict[str, Any], List[Any]]) -> str:
    """Format citation(s) into AMA-style reference string.

    Parameters
    ----------
    citations : dict or list
        A single citation dict or a list of citation objects.

    Returns
    -------
    str
        Formatted AMA-style reference(s).
    """
    # Handle single dict input (Phase 4 format_response node passes one at a time)
    if isinstance(citations, dict):
        return _format_single(citations)

    # Handle list input (legacy interface)
    if not citations:
        return ""

    formatted: List[str] = []
    seen: set = set()

    for citation in citations:
        try:
            if isinstance(citation, dict):
                if "metadata" in citation:
                    text = _format_metadata(citation)
                else:
                    text = _format_single(citation)
            elif isinstance(citation, str):
                text = citation
            else:
                continue

            if text and text not in seen:
                seen.add(text)
                formatted.append(text)
        except Exception as exc:
            logger.warning("Error formatting citation: %s", exc)

    if not formatted:
        return ""

    refs = "\n".join(f"- {i+1}. {ref}" for i, ref in enumerate(formatted))
    return f"\n\n## References\n\n{refs}"


def _format_single(citation: Dict[str, Any]) -> str:
    """Format a single citation dict into AMA style."""
    authors = citation.get("authors", [])
    title = citation.get("title", "Untitled")
    journal = citation.get("journal", "")
    year = citation.get("year", "")
    volume = citation.get("volume", "")
    issue = citation.get("issue", "")
    pages = citation.get("pages", "")
    doi = citation.get("doi", "")
    pmid = citation.get("pmid", "")

    # Format authors (AMA: LastName Initials, ...)
    if isinstance(authors, list) and authors:
        if len(authors) > 3:
            author_str = f"{authors[0]}, et al"
        else:
            author_str = ", ".join(str(a) for a in authors)
    elif isinstance(authors, str):
        author_str = authors
    else:
        author_str = "Unknown"

    parts = [f"{author_str}. {title}."]

    if journal:
        journal_part = f" {journal}."
        if year:
            journal_part += f" {year}"
        if volume:
            journal_part += f";{volume}"
        if issue:
            journal_part += f"({issue})"
        if pages:
            journal_part += f":{pages}"
        parts.append(journal_part + ".")
    elif year:
        parts.append(f" {year}.")

    if doi:
        parts.append(f" doi:{doi}")
    if pmid:
        parts.append(f" PMID:{pmid}")

    return "".join(parts)


def _format_metadata(citation: Dict[str, Any]) -> str:
    """Format a metadata-style citation dict (legacy local agent format)."""
    metadata = citation.get("metadata", {})
    pdf_name = metadata.get("pdf_name", "Unknown Source")
    page_number = metadata.get("page_number", "")
    topic = metadata.get("topic", "")

    parts = [pdf_name]
    if page_number:
        parts.append(f"Page {page_number}")
    if topic:
        parts.append(f"Topic: {topic}")

    return ". ".join(parts) + "."
