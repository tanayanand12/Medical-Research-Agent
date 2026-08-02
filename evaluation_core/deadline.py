"""Cross-layer runtime deadline exception."""

from __future__ import annotations

import time
from typing import Optional


class RuntimeDeadlineExceeded(TimeoutError):
    """The request's absolute runtime deadline has been exhausted."""


def remaining_seconds(
    deadline_at: Optional[float],
    *,
    default: Optional[float] = None,
) -> Optional[float]:
    """Return bounded remaining seconds or raise before work starts."""
    if deadline_at is None:
        return default
    remaining = float(deadline_at) - time.monotonic()
    if remaining <= 0:
        raise RuntimeDeadlineExceeded("runtime request deadline exhausted")
    if default is None:
        return remaining
    return min(float(default), remaining)


def ensure_deadline(deadline_at: Optional[float]) -> None:
    """Fail distinctly when an absolute deadline has expired."""
    remaining_seconds(deadline_at)


def sleep_with_deadline(
    seconds: float, deadline_at: Optional[float]
) -> None:
    """Sleep only when the entire delay fits inside the request budget."""
    delay = max(0.0, float(seconds))
    if deadline_at is not None:
        remaining = remaining_seconds(deadline_at)
        if remaining is None or delay >= remaining:
            raise RuntimeDeadlineExceeded(
                "runtime request deadline exhausted before retry"
            )
    time.sleep(delay)
