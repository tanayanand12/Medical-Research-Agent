"""Conditional structured verification for ambiguous high-risk medical claims."""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, Optional, Sequence, Tuple

from evaluation_core import RuntimeDeadlineExceeded
from runtime_verification.telemetry import call_llm_with_metadata


HIGH_RISK_CLAIM_PROMPT_VERSION = "high-risk-claim-v2"


def _model_parts(model: str) -> Tuple[str, str]:
    if "@" not in model:
        return model, ""
    name, revision = model.rsplit("@", 1)
    return name, revision


class ConditionalClaimVerifier:
    """Use one bounded LLM call to classify a claim against cited evidence."""

    def __init__(
        self,
        *,
        llm_client: Optional[Any] = None,
        model: Optional[str] = None,
        timeout_sec: float = 8.0,
        deadline_at: Optional[float] = None,
        prompt_version: str = HIGH_RISK_CLAIM_PROMPT_VERSION,
        minimum_supported_confidence: Optional[float] = None,
    ) -> None:
        if llm_client is None:
            from llm_client import LLMClient

            llm_client = LLMClient()
        configured_model = (
            model
            or os.getenv("RUNTIME_CLAIM_VERIFIER_MODEL")
            or getattr(llm_client, "default_model", "")
            or "unknown"
        )
        self.llm_client = llm_client
        self.model, self.model_revision = _model_parts(str(configured_model))
        self.timeout_sec = max(0.1, float(timeout_sec))
        self.deadline_at = deadline_at
        self.prompt_version = prompt_version
        configured_minimum = (
            minimum_supported_confidence
            if minimum_supported_confidence is not None
            else os.getenv("RUNTIME_CLAIM_VERIFIER_MIN_CONFIDENCE", "0.70")
        )
        self.minimum_supported_confidence = float(configured_minimum)
        if not 0.0 <= self.minimum_supported_confidence <= 1.0:
            raise ValueError(
                "minimum_supported_confidence must be between 0 and 1"
            )

    def __call__(
        self, claim: str, cited_evidence: Sequence[str]
    ) -> Dict[str, Any]:
        """Return supported/unsupported/uncertain without consulting qrels."""
        started_at = time.monotonic()
        remaining = self.timeout_sec
        if self.deadline_at is not None:
            remaining = min(remaining, self.deadline_at - time.monotonic())
        if remaining <= 0:
            return self._failure(
                RuntimeDeadlineExceeded(
                    "runtime deadline expired before claim verification"
                ),
                started_at,
            )
        if not cited_evidence:
            return {
                "decision": "uncertain",
                "confidence": 0.0,
                "valid": True,
                "error": "no cited final-context evidence",
                **self._metadata(started_at),
            }

        evidence = "\n\n".join(
            f"[Evidence {index}] {text}"
            for index, text in enumerate(cited_evidence, 1)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify whether one medical claim is supported by only the "
                    "cited evidence supplied below. Return strict JSON with keys "
                    "decision (supported|unsupported|uncertain), confidence "
                    "(0..1), and rationale. Unsupported includes contradiction or "
                    "material overstatement. Use uncertain when the cited text is "
                    "ambiguous. Do not use outside knowledge."
                ),
            },
            {
                "role": "user",
                "content": f"Claim:\n{claim}\n\nCited evidence only:\n{evidence}",
            },
        ]
        call_result = None
        try:
            call_result = call_llm_with_metadata(
                self.llm_client,
                messages=messages,
                model=self.model,
                temperature=0.0,
                max_tokens=220,
                timeout=max(0.1, remaining),
                client_max_attempts=1,
                deadline_at=self.deadline_at,
            )
            payload = json.loads(call_result.text.strip())
            if not isinstance(payload, dict):
                raise ValueError("claim verifier response must be a JSON object")
            decision = str(payload.get("decision") or "").strip().lower()
            confidence = payload.get("confidence")
            raw_rationale = payload.get("rationale")
            if not isinstance(raw_rationale, str):
                raise ValueError("claim verifier rationale must be a string")
            rationale = raw_rationale.strip()
            if decision not in {"supported", "unsupported", "uncertain"}:
                raise ValueError("claim verifier returned an invalid decision")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or not 0.0 <= float(confidence) <= 1.0
            ):
                raise ValueError("claim verifier returned invalid confidence")
            if not rationale:
                raise ValueError("claim verifier returned an empty rationale")
            confidence_value = float(confidence)
            if confidence_value < self.minimum_supported_confidence:
                if decision == "supported":
                    return {
                        "decision": "uncertain",
                        "confidence": confidence_value,
                        "rationale": rationale,
                        "valid": True,
                        "raw_decision": payload,
                        "downgrade_reason": "supported_confidence_below_minimum",
                        **self._metadata(started_at, call_result=call_result),
                    }
                raise ValueError(
                    "claim verifier returned a definitive decision below "
                    "minimum confidence"
                )
            return {
                "decision": decision,
                "confidence": confidence_value,
                "rationale": rationale,
                "valid": True,
                "raw_decision": payload,
                **self._metadata(started_at, call_result=call_result),
            }
        except Exception as exc:
            if call_result is None:
                history_reader = getattr(
                    self.llm_client, "thread_call_history", None
                )
                if callable(history_reader):
                    history = list(history_reader() or [])
                    if history:
                        call_result = history[-1]
            return self._failure(
                exc,
                started_at,
                call_result=call_result,
                raw_decision=locals().get("payload"),
            )

    def _metadata(
        self, started_at: float, *, call_result: Optional[Any] = None
    ) -> Dict[str, Any]:
        if call_result is None:
            return {
                "model": self.model,
                "model_revision": self.model_revision,
                "prompt_version": self.prompt_version,
                "latency_sec": max(0.0, time.monotonic() - started_at),
                "token_usage": {"input": 0, "output": 0, "total": 0},
                "cost_usd": 0.0,
            }
        return {
            "model": str(call_result.model or self.model),
            "model_revision": str(
                call_result.model_revision or self.model_revision
            ),
            "prompt_version": self.prompt_version,
            "latency_sec": float(call_result.latency_sec),
            "token_usage": {
                "input": int(call_result.tokens_in),
                "output": int(call_result.tokens_out),
                "total": int(call_result.tokens_in) + int(call_result.tokens_out),
            },
            "cost_usd": float(call_result.cost_usd),
            "finish_reason": str(call_result.finish_reason or ""),
            "provider_metadata": dict(
                getattr(call_result, "provider_metadata", {}) or {}
            ),
        }

    def _failure(
        self,
        exc: Exception,
        started_at: float,
        *,
        call_result: Optional[Any] = None,
        raw_decision: Optional[Any] = None,
    ) -> Dict[str, Any]:
        result = {
            "decision": "uncertain",
            "confidence": 0.0,
            "valid": False,
            "error": f"claim_verifier_failed:{type(exc).__name__}",
            "error_type": type(exc).__name__,
            **self._metadata(started_at, call_result=call_result),
        }
        if raw_decision is not None:
            result["raw_decision"] = raw_decision
        return result
