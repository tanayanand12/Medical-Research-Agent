"""Construction of consistently configured production runtime verifiers."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from runtime_verification.claim_verifier import ConditionalClaimVerifier
from runtime_verification.verifier import ClaimVerifier, RuntimeVerifier


def build_runtime_verifier(
    context: Optional[Dict[str, Any]] = None,
    *,
    llm_client: Optional[Any] = None,
    claim_verifier: Optional[ClaimVerifier] = None,
) -> RuntimeVerifier:
    """Build the shared verifier without benchmark-dependent inputs."""
    settings = dict(context or {})
    if claim_verifier is None and settings.get(
        "enable_high_risk_claim_verifier", True
    ):
        deadline_at = settings.get("_runtime_deadline_at_monotonic")
        timeout_sec = float(settings.get("claim_verifier_timeout_sec", 8.0))
        if deadline_at is not None:
            timeout_sec = min(
                timeout_sec, max(0.1, float(deadline_at) - time.monotonic())
            )
        claim_verifier = ConditionalClaimVerifier(
            llm_client=llm_client,
            model=settings.get("claim_verifier_model"),
            timeout_sec=timeout_sec,
            deadline_at=float(deadline_at) if deadline_at is not None else None,
        )
    return RuntimeVerifier(claim_verifier=claim_verifier)
