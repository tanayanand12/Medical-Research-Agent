# medical_research_agent/utils/perf.py
import logging, time
from contextlib import contextmanager
from functools import wraps

logger = logging.getLogger("perf")

def timed_node(name: str):
    """Wraps a LangGraph node function with wall-clock timing."""
    def deco(fn):
        @wraps(fn)
        def wrapper(state, *a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(state, *a, **kw)
            finally:
                logger.info(f"[NODE] {name:<18s} {time.perf_counter()-t0:7.2f}s")
        return wrapper
    return deco

@contextmanager
def time_block(label: str, **extra):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        tail = " | " + " ".join(f"{k}={v}" for k,v in extra.items()) if extra else ""
        logger.info(f"[STEP] {label:<28s} {time.perf_counter()-t0:7.2f}s{tail}")