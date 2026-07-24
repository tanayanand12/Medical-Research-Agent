"""
Structured JSON logging for Medical Research Agent.

Replaces the plain-text log format from Phase 1-4 with machine-parseable
JSON lines.  Every log record includes:

- ``timestamp``  — ISO 8601
- ``level``      — DEBUG / INFO / WARNING / ERROR / CRITICAL
- ``logger``     — module logger name
- ``message``    — human-readable message
- ``trace_id``   — correlation ID (from ``extra`` or ``""``)
- ``node``       — graph node name (from ``extra`` or ``""``)

Usage
-----
Call ``configure_structured_logging()`` once at application startup.
Then use the standard ``logging`` module as normal — the JSON formatter
is applied to all handlers.

Individual log calls can attach trace context via ``extra``::

    logger.info(
        "Node completed",
        extra={"trace_id": state["trace_id"], "node": "classify_intent"},
    )
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", ""),
            "node": getattr(record, "node", ""),
        }

        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False, default=str)


def configure_structured_logging(
    level: Optional[str] = None,
    log_dir: str = "logs",
    log_file: str = "observability.log",
) -> None:
    """Configure structured JSON logging for the application.

    Parameters
    ----------
    level : str, optional
        Log level.  Falls back to ``LOG_LEVEL`` env var, then ``"INFO"``.
    log_dir : str
        Directory for log files (created if absent).
    log_file : str
        Filename for the observability log.
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = JSONFormatter()

    # Console handler (stderr, UTF-8 safe)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)

    # File handler
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric_level)

    # Apply to root logger
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicate output
    # (only remove non-JSON handlers to be safe with prior config)
    for handler in root.handlers[:]:
        if not isinstance(handler.formatter, JSONFormatter):
            root.removeHandler(handler)

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    logging.getLogger(__name__).info(
        "Structured JSON logging configured — level=%s, file=%s",
        level.upper(),
        file_path,
    )
