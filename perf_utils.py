# perf_utils.py
import logging
import time
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger("perf")

def timed(label: str | None = None):
    """Function-level wall-clock timing. Use on node entry points."""
    def deco(fn):
        name = label or f"{fn.__module__}.{fn.__qualname__}"
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                logger.info(f"[PERF] {name} took {time.perf_counter()-t0:.3f}s")
        return wrapper
    return deco

@contextmanager
def time_block(label: str, extra: dict | None = None):
    """Sub-step timing. Use inside functions for granular breakdowns."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        msg = f"[PERF] {label} took {time.perf_counter()-t0:.3f}s"
        if extra:
            msg += f" | {extra}"
        logger.info(msg)