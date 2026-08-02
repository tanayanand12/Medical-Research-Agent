"""Safe serialization helpers for response-visible evaluation traces."""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Any, Dict, Iterable, List


_SENSITIVE_KEY_PARTS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
}
_SENSITIVE_TOKEN_KEYS = {
    "access_token",
    "auth_token",
    "id_token",
    "refresh_token",
    "token",
}
_QUERY_VALUE_KEYS = {
    "expanded_queries",
    "expanded_query",
    "input_query",
    "original_query",
    "query",
    "user_query",
}
_CONTENT_VALUE_KEYS = {
    "evidence",
    "messages",
    "prompt",
    "raw_prompt",
}
_ERROR_PAYLOAD_KEYS = {
    "body",
    "detail",
    "details",
    "error",
    "errors",
    "message",
    "reason",
}
_URL_VALUE_KEYS = {
    "all_source_urls",
    "attempted_urls",
    "failed_urls",
    "source_url",
    "url",
}
_SECRET_PATTERNS = (
    re.compile(
        r"(?i)\bAuthorization\s*[:=]\s*(?:Bearer|Basic)\s+"
        r"[A-Za-z0-9._~+/=-]+"
    ),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\bBasic\s+[A-Za-z0-9+/=-]+"),
    re.compile(r"(?i)\bCookie\s*[:=]\s*[^\r\n]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bls__[A-Za-z0-9_-]{8,}\b"),
    re.compile(
        r"(?i)\b(api[_-]?key|authorization|password|secret|token)"
        r"\s*[:=]\s*[^\s,;]+"
    ),
)


def sanitize_sensitive_text(value: Any) -> str:
    """Remove common credential patterns from a string representation."""
    text = str(value)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text


def stable_query_fingerprint(value: Any) -> str:
    """Return a stable, non-reversible identifier for operational logs."""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]


def safe_error_type(value: Any) -> str:
    """Return only an exception class name for default operational logs."""
    return type(value).__name__


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().lower()
    return (
        normalized in _SENSITIVE_TOKEN_KEYS
        or any(part in normalized for part in _SENSITIVE_KEY_PARTS)
        or (
            normalized.endswith("_token")
            and normalized not in {"input_token", "output_token"}
        )
    )


def redact_sensitive_values(value: Any) -> Any:
    """Recursively scrub credential-bearing keys and string values."""
    if isinstance(value, dict):
        return {
            str(key): (
                {"response_redacted": True}
                if str(key).strip().lower() == "raw_decision"
                else "[REDACTED_QUERY]"
                if str(key).strip().lower() in _QUERY_VALUE_KEYS
                else "[REDACTED_CONTENT]"
                if str(key).strip().lower() in _CONTENT_VALUE_KEYS
                else "[REDACTED_ERROR]"
                if str(key).strip().lower() in _ERROR_PAYLOAD_KEYS
                else "[REDACTED_URL]"
                if str(key).strip().lower() in _URL_VALUE_KEYS
                else "[REDACTED_HEADERS]"
                if str(key).strip().lower() in {"header", "headers"}
                else "[REDACTED]"
                if _is_sensitive_key(key)
                else redact_sensitive_values(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_values(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive_values(item) for item in value]
    if isinstance(value, str):
        return sanitize_sensitive_text(value)
    return value


def redact_trace_for_response(trace: Dict[str, Any]) -> Dict[str, Any]:
    """Return a public trace view without full evidence or credentials."""
    redacted = redact_sensitive_values(copy.deepcopy(trace))
    if "original_query" in redacted:
        redacted["original_query"] = "[REDACTED]"
        redacted["original_query_redacted"] = True
    if "expanded_queries" in redacted:
        redacted["expanded_queries"] = []
        redacted["expanded_queries_redacted"] = True
    for document in redacted.get("retrieved_documents", []) or []:
        if isinstance(document, dict) and "text" in document:
            document["text"] = "[REDACTED]"
            document["text_redacted"] = True
    for span in redacted.get("final_context_spans", []) or []:
        if isinstance(span, dict) and "text" in span:
            span["text"] = "[REDACTED]"
            span["text_redacted"] = True
    redacted["response_redaction"] = {
        "full_evidence_removed": True,
        "credentials_scrubbed": True,
        "internal_trace_unchanged": True,
    }
    return redacted


def redact_traces_for_response(
    traces: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [redact_trace_for_response(trace) for trace in traces]
