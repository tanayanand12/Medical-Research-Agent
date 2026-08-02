"""Shared absolute-deadline controls for bounded runtime work."""

from evaluation_core.deadline import (
    RuntimeDeadlineExceeded,
    ensure_deadline,
    remaining_seconds,
    sleep_with_deadline,
)

__all__ = [
    "RuntimeDeadlineExceeded",
    "ensure_deadline",
    "remaining_seconds",
    "sleep_with_deadline",
]
