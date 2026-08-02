"""Process-bounded executor for blocking retrieval and repair operations."""

from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional


class ExecutorSaturatedError(RuntimeError):
    """Raised when all worker and bounded queue slots are occupied."""


class BoundedExecutor:
    """Thread pool whose running plus queued work has a hard upper bound."""

    def __init__(
        self,
        *,
        max_workers: int,
        max_queue: int,
        thread_name_prefix: str = "mra-runtime",
    ) -> None:
        self.max_workers = max(1, int(max_workers))
        self.max_queue = max(0, int(max_queue))
        self._capacity = self.max_workers + self.max_queue
        self._slots = threading.BoundedSemaphore(self._capacity)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._in_flight = 0
        self._submitted = 0
        self._completed = 0
        self._rejected = 0
        self._timed_out = 0
        self._shutdown = False

    def submit(self, call: Callable[[], Any]) -> Future[Any]:
        with self._lock:
            if self._shutdown:
                self._rejected += 1
                raise ExecutorSaturatedError("runtime executor is shut down")
        if not self._slots.acquire(blocking=False):
            with self._lock:
                self._rejected += 1
            raise ExecutorSaturatedError(
                "runtime executor is saturated; no bounded queue slot is available"
            )
        with self._lock:
            self._submitted += 1
            self._in_flight += 1
        try:
            future = self._executor.submit(call)
        except Exception:
            self._release_slot()
            raise
        future.add_done_callback(lambda _future: self._release_slot())
        return future

    def mark_timeout(self) -> None:
        with self._lock:
            self._timed_out += 1

    def metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "max_workers": self.max_workers,
                "max_queue": self.max_queue,
                "capacity": self._capacity,
                "in_flight": self._in_flight,
                "submitted": self._submitted,
                "completed": self._completed,
                "rejected": self._rejected,
                "timed_out": self._timed_out,
                "shutdown": self._shutdown,
            }

    def shutdown(
        self, *, wait: bool = False, cancel_futures: bool = True
    ) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def _release_slot(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._completed += 1
        self._slots.release()


_EXECUTOR_LOCK = threading.Lock()
_EXECUTOR: Optional[BoundedExecutor] = None


def get_runtime_executor() -> BoundedExecutor:
    """Return a lazily restartable process-wide bounded executor."""
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None or _EXECUTOR.metrics()["shutdown"]:
            _EXECUTOR = BoundedExecutor(
                max_workers=int(os.getenv("RUNTIME_EXECUTOR_MAX_WORKERS", "4")),
                max_queue=int(os.getenv("RUNTIME_EXECUTOR_MAX_QUEUE", "4")),
            )
        return _EXECUTOR


def shutdown_runtime_executor(
    *, wait: bool = False, cancel_futures: bool = True
) -> None:
    """Stop accepting work and cancel queued operations during shutdown."""
    with _EXECUTOR_LOCK:
        executor = _EXECUTOR
    if executor is not None:
        executor.shutdown(wait=wait, cancel_futures=cancel_futures)
